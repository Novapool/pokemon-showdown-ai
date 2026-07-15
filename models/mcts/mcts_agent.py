"""
mcts_agent.py — Determinized UCT search over the PPO policy/value net (M4).

Layers Monte Carlo Tree Search on top of a trained MLP-PPO checkpoint at
inference time. No training happens here: the policy head provides the PUCT
prior, the value head evaluates leaves, and the BattleSim forward model
(sim_clone/sim_step/sim_fork through the gym bridge) provides transitions.

Imperfect information is handled by root determinization: the search runs
`n_determinizations` independent trees, each on a clone of the live battle
whose unrevealed opponent Pokémon were resampled from the format's team
generator (`sim_clone(determinize=True)`), and root visit counts are summed
across trees before picking the action.

Inside a tree, nodes are OUR decision points. The opponent is part of the
environment: whenever the simulated battle needs a p2 action, one is sampled
from the same policy network applied to p2's (reveal-tracked) observation.
Transitions are stochastic (damage rolls, crits, the sampled opponent action);
each edge's outcome is sampled once at expansion and reused — standard
single-sample determinized UCT.

PUCT selection:  score(s, a) = Q(s, a) + c_puct * P(a|s) * sqrt(N(s)) / (1 + N(s, a))
Backup value:    sum of gym-shaped step rewards along the path (win/loss ±1
                 included) plus the value head at non-terminal leaves.
"""

import math

import numpy as np
import torch


class MCTSNode:
    """One of OUR decision points in a simulated battle (or a terminal state)."""

    __slots__ = (
        "sim_id", "obs", "mask", "done", "reward_from_parent",
        "priors", "children", "child_visits", "child_values", "expanded",
    )

    def __init__(self, sim_id, obs, mask, done, reward_from_parent):
        self.sim_id = sim_id            # bridge sim holding this node's state
        self.obs = obs                  # p1 obs (flattened), None if terminal
        self.mask = mask                # (9,) bool legality mask at this node
        self.done = done
        self.reward_from_parent = reward_from_parent  # shaped reward accumulated on the way here
        self.priors = None              # (9,) masked policy prior, set on expand
        self.children = {}              # action -> MCTSNode
        self.child_visits = np.zeros(9, dtype=np.int64)
        self.child_values = np.zeros(9, dtype=np.float64)  # summed backup returns W(s,a)
        self.expanded = False


class MCTSAgent:
    """Inference-time MCTS wrapper around a PPOAgent checkpoint.

    Drives simulations through a GymClient's sim_* commands, so act() must be
    given the same client that hosts the live battle being played.
    """

    def __init__(
        self,
        agent,
        n_sims: int = 100,
        n_determinizations: int = 4,
        c_puct: float = 1.5,
        determinize: bool = True,
        seed: int | None = None,
        seat: str = "p1",
    ):
        """
        Args:
            agent:               a loaded PPOAgent (policy prior + leaf value).
            n_sims:              total simulation budget per move decision,
                                 split evenly across determinizations.
            n_determinizations:  independent opponent-team samples per decision.
            c_puct:              PUCT exploration constant.
            determinize:         False keeps the true hidden opponent team in
                                 the clones (omniscient search — diagnostic only).
            seed:                seeds determinization sampling for reproducibility.
            seat:                which live-battle seat this searcher occupies
                                 ('p1' default; 'p2' for seat-balanced h2h runs).
                                 Tree nodes are this seat's decision points and
                                 backups negate the p1-perspective reward for p2.
        """
        if seat not in ("p1", "p2"):
            raise ValueError(f"seat must be 'p1' or 'p2', got {seat!r}")
        self.agent = agent
        self.n_sims = n_sims
        self.n_determinizations = max(1, n_determinizations)
        self.c_puct = c_puct
        self.determinize = determinize
        self.seat = seat
        self._opp_seat = "p2" if seat == "p1" else "p1"
        self._reward_sign = 1.0 if seat == "p1" else -1.0
        self._rng = np.random.default_rng(seed)
        self._det_counter = 0
        # Stats for latency reporting (evaluate.py --model mcts)
        self.last_search_sims = 0

    # ------------------------------------------------------------------
    # Network helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _policy_value(self, obs: np.ndarray, mask) -> tuple[np.ndarray, float]:
        """Masked policy distribution and state value for one observation."""
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.agent.device)
        mask_t = torch.tensor(np.asarray(mask, dtype=bool)).unsqueeze(0).to(self.agent.device)
        features = self.agent.trunk(obs_t)
        logits = self.agent.policy_head(features).masked_fill(~mask_t, -1e9)
        probs = torch.softmax(logits, dim=-1)
        value = self.agent.value_head(features).squeeze()
        return probs.squeeze(0).cpu().numpy(), float(value.item())

    def _sample_opponent_action(self, seat_state: dict) -> int:
        """Sample the opponent seat's action from the policy on its
        (reveal-tracked) observation."""
        obs = np.asarray(seat_state["obs"], dtype=np.float32).reshape(-1)
        obs = self._fit_obs(obs)
        probs, _ = self._policy_value(obs, seat_state["mask"])
        total = probs.sum()
        if not np.isfinite(total) or total <= 0:
            legal = np.flatnonzero(np.asarray(seat_state["mask"], dtype=bool))
            return int(self._rng.choice(legal))
        return int(self._rng.choice(9, p=probs / total))

    def _fit_obs(self, obs_flat: np.ndarray) -> np.ndarray:
        """Slice a structured obs down to the agent's obs_size (v2 env → v1 net)."""
        expected = self.agent._hparams["obs_size"]
        if obs_flat.shape[-1] == expected:
            return obs_flat
        from gym_client import slice_structured_obs

        return slice_structured_obs(obs_flat[None, :], expected)[0]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def act(self, client, obs, valid_mask) -> int:
        """Pick the action for the live battle hosted by `client`.

        Runs n_determinizations root-parallel trees of n_sims//n_determinizations
        simulations each and returns the action with the highest total root
        visit count (Q as tiebreaker). `obs` is the live flattened observation,
        used only for the policy fallback when no search could run.
        """
        legal = np.flatnonzero(np.asarray(valid_mask, dtype=bool))
        if len(legal) == 1:
            self.last_search_sims = 0
            return int(legal[0])

        sims_per_det = max(1, self.n_sims // self.n_determinizations)
        total_visits = np.zeros(9, dtype=np.int64)
        total_value = np.zeros(9, dtype=np.float64)
        ran_sims = 0

        for _ in range(self.n_determinizations):
            self._det_counter += 1
            root_state = client.sim_clone(
                determinize=self.determinize,
                seed=int(self._rng.integers(0, 2**31 - 1)),
                perspective=self.seat,
            )
            if root_state["done"] or not root_state[self.seat]["needs"]:
                # Locked states (sleep/recharge) can auto-complete our choice in
                # the sim even though the live battle wants an explicit action.
                client.sim_free(root_state["sim"])
                continue
            root = self._make_node(root_state["sim"], root_state, reward=0.0)
            for _ in range(sims_per_det):
                self._simulate(client, root)
                ran_sims += 1
            total_visits += root.child_visits
            with np.errstate(invalid="ignore"):
                total_value += root.child_values
            client.sim_free_all()

        self.last_search_sims = ran_sims
        if total_visits.sum() == 0:
            # Search couldn't run (e.g. every determinization auto-completed
            # our locked choice) — fall back to the raw policy.
            probs, _ = self._policy_value(
                self._fit_obs(np.asarray(obs, dtype=np.float32).reshape(-1)), valid_mask,
            )
            return int(np.argmax(probs))

        # Mask to live-battle-legal actions: the live mask is authoritative.
        visits = np.where(np.asarray(valid_mask, dtype=bool), total_visits, -1)
        best = np.flatnonzero(visits == visits.max())
        if len(best) > 1:
            q = np.full(9, -np.inf)
            for a in best:
                if total_visits[a] > 0:
                    q[a] = total_value[a] / total_visits[a]
            if np.isfinite(q).any():
                return int(np.argmax(q))
        return int(best[0])

    def _make_node(self, sim_id: int, state: dict, reward: float) -> MCTSNode:
        done = bool(state["done"])
        if done:
            return MCTSNode(sim_id, None, None, True, reward)
        seat = state[self.seat]
        obs = self._fit_obs(np.asarray(seat["obs"], dtype=np.float32).reshape(-1))
        return MCTSNode(sim_id, obs, np.asarray(seat["mask"], dtype=bool), False, reward)

    def _expand(self, node: MCTSNode) -> float:
        """Set the node's policy prior; return its leaf value estimate."""
        priors, value = self._policy_value(node.obs, node.mask)
        node.priors = priors
        node.expanded = True
        return value

    def _select_action(self, node: MCTSNode) -> int:
        sqrt_n = math.sqrt(max(1, node.child_visits.sum()))
        best_action, best_score = -1, -np.inf
        for a in np.flatnonzero(node.mask):
            n_a = node.child_visits[a]
            q = node.child_values[a] / n_a if n_a > 0 else 0.0
            score = q + self.c_puct * node.priors[a] * sqrt_n / (1 + n_a)
            if score > best_score:
                best_action, best_score = int(a), score
        return best_action

    def _advance(self, client, sim_id: int, state: dict, our_action) -> tuple[dict, float]:
        """Step a sim to OUR next decision point (or terminal).

        `our_action` is consumed at the first step that needs it; any further
        decision points that belong only to the opponent (their force-switches)
        are auto-played by the opponent model. Returns (state, summed reward)
        with the reward already flipped to this searcher's perspective.
        """
        reward = 0.0
        action = our_action
        guard = 50
        while not state["done"] and guard > 0:
            we_need = state[self.seat]["needs"]
            opp_needs = state[self._opp_seat]["needs"]
            if we_need and action is None:
                break  # our next decision point
            ours = action if we_need else None
            theirs = self._sample_opponent_action(state[self._opp_seat]) if opp_needs else None
            action = None
            if self.seat == "p1":
                state = client.sim_step(sim_id, ours, theirs)
            else:
                state = client.sim_step(sim_id, theirs, ours)
            reward += self._reward_sign * state["reward"]
            guard -= 1
        if guard == 0:
            raise RuntimeError("MCTS _advance: no decision point reached in 50 sim steps")
        return state, reward

    def _simulate(self, client, root: MCTSNode) -> None:
        """One PUCT simulation: select to a leaf, expand, back up."""
        node = root
        path = []  # (parent, action) edges walked

        if not root.expanded:
            self._expand(root)

        # --- selection
        while True:
            action = self._select_action(node)
            path.append((node, action))
            child = node.children.get(action)
            if child is None:
                # --- expansion: sample this edge's outcome once
                fork = client.sim_fork(node.sim_id)
                state, reward = self._advance(client, fork["sim"], fork, action)
                child = self._make_node(fork["sim"], state, reward)
                node.children[action] = child
                if child.done:
                    leaf_value = 0.0
                else:
                    leaf_value = self._expand(child)
                break
            node = child
            if node.done:
                leaf_value = 0.0
                break
            if not node.expanded:
                leaf_value = self._expand(node)
                break

        # --- backup: G accumulates shaped rewards from leaf toward the root
        g = leaf_value
        for parent, action in reversed(path):
            child = parent.children[action]
            g += child.reward_from_parent
            parent.child_visits[action] += 1
            parent.child_values[action] += g

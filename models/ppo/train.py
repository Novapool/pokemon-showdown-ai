"""
train.py — Rollout-based PPO training loop for Pokemon Showdown.

Usage:
    python models/ppo/train.py
    python models/ppo/train.py --steps 500000 --rollout-steps 512 --checkpoint-every 50000

    # M2 verification: MLP PPO on flattened (12,65)->780 structured obs.
    # Checkpoints land in models/ppo/checkpoints/structured/ so they never
    # collide with the flat-obs (M1) baseline checkpoints.
    python models/ppo/train.py --structured --steps 2600000 --checkpoint-every 250000

The loop collects rollout_steps environment steps per update cycle regardless
of episode boundaries.  When a 'done' signal arrives mid-rollout, the
environment is auto-reset so collection continues seamlessly.

Rollouts are collected from --num-envs parallel battle simulations (M3.1):
each env is its own gym_bridge.js subprocess, and inference runs batched over
all envs' observations. --num-envs 1 uses the same code path.

Note: --steps counts steps, not battles. At ~50 steps/battle, hitting a
target battle count (e.g. the M2/M3 milestone success criteria, which are
specified in battles) means multiplying by the average steps/battle, not
using the battle count directly.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

# Resolve models/ directory so vec_gym_client and ppo modules are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from gym_client import slice_structured_obs  # noqa: E402
from vec_gym_client import VecGymClient   # noqa: E402

# ppo_agent and trajectory_buffer live alongside this script
sys.path.insert(0, str(Path(__file__).parent))
from ppo_agent import PPOAgent            # noqa: E402
from trajectory_buffer import TrajectoryBuffer, merge_buffers  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO agent on Pokemon Showdown")
    parser.add_argument(
        "--steps",
        type=int,
        default=100000,
        help="Total environment steps to train for (default: 100000)",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=512,
        help="Number of steps per rollout before a PPO update (default: 512)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25000,
        help="Save a checkpoint every N steps (default: 25000)",
    )
    parser.add_argument(
        "--structured",
        action="store_true",
        help=(
            "Train on the M2 (12,65)->780 flattened structured observation "
            "instead of the legacy flat 100-dim vector. Checkpoints go to "
            "checkpoints/structured/ to avoid colliding with the M1 baseline."
        ),
    )
    parser.add_argument(
        "--obs-v2",
        action="store_true",
        help=(
            "M3.4: train on the (12,77)->924 schema-v2 observation (boost "
            "stages + volatile flags appended to each token). Implies "
            "--structured. Checkpoints go to checkpoints/v2/ — incompatible "
            "with v1 checkpoints by design."
        ),
    )
    parser.add_argument(
        "--obs-v3",
        action="store_true",
        help=(
            "M7: train on the (12,86)->1032 schema-v3 observation (type "
            "effectiveness + move-effect flags + Sleep Clause appended to "
            "each token). Implies --structured. obs_size is auto-inferred "
            "(1032). Checkpoints go to checkpoints/v3/ — v1/v2 checkpoints "
            "still play as pool/h2h opponents (sliced per token)."
        ),
    )
    parser.add_argument(
        "--obs-v3-extended",
        action="store_true",
        help=(
            "M8 Phase 1A: train on the (12,87)->1044 schema-v3-extended "
            "observation (v3 plus the own/opp active base-speed ratio at "
            "dim 86). Implies --structured. obs_size is auto-inferred (1044). "
            "Checkpoints go to checkpoints/v3-extended/ — v1/v2/v3 checkpoints "
            "still play as pool/h2h opponents (sliced per token)."
        ),
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="Number of parallel battle environments (default: 8)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        default=None,
        help="Torch device override (default: auto-detect cuda > mps > cpu)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help=(
            "Override the checkpoint directory (default: checkpoints/ or "
            "checkpoints/structured next to this script). Use for runs that "
            "must not overwrite an earlier run's checkpoints at the same step "
            "counts (e.g. a self-play run vs the M2 baseline)."
        ),
    )
    parser.add_argument(
        "--opponent",
        choices=["random", "damagefirst", "selfplay"],
        default="random",
        help=(
            "M3.3: who sits in the p2 seat. 'random' = RandomPlayerAI (legacy), "
            "'damagefirst' = highest-base-power heuristic, 'selfplay' = frozen "
            "past checkpoints of this agent (see --selfplay-pool)."
        ),
    )
    parser.add_argument(
        "--selfplay-pool",
        type=str,
        default=None,
        help=(
            "Directory of .pt checkpoints to sample self-play opponents from "
            "(default: this run's own checkpoint directory). Each rollout picks "
            "the newest checkpoint 50%% of the time, else uniform over the pool; "
            "until any checkpoint exists the opponent is a frozen copy of the "
            "current policy. Pool checkpoints may use the v1 obs schema even "
            "in an --obs-v2 run (their view is sliced down per token)."
        ),
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help=(
            "Trunk width for a from-scratch run. IGNORED when --resume or "
            "--pretrain-checkpoint is given: those inherit the width stored in "
            "the checkpoint's own hparams. To train a wider agent from a BC "
            "warm-start, run bc_pretrain_mlp.py --hidden-size N first."
        ),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=(
            "Adam learning rate. Default None keeps whatever the agent was "
            "built or loaded with (3e-4 for every run M2-M9), so omitting this "
            "reproduces the historical recipe exactly. Set it to sweep LR — "
            "needed when changing --hidden-size, since an LR tuned at one "
            "width need not hold at another, and a null result at a fixed LR "
            "cannot distinguish capacity from step-size."
        ),
    )
    parser.add_argument(
        "--opp-coef",
        type=float,
        default=0.1,
        help=(
            "M5: coefficient λ for the auxiliary opponent-action-prediction "
            "cross-entropy loss on the opp_head, computed over the gym's "
            "per-step opp_action labels (masked steps contribute nothing). "
            "0 disables the aux loss and reproduces the exact pre-M5 loss "
            "path. Headed runs (opp-coef != 0) checkpoint to checkpoints/opp/ "
            "by default (see --checkpoint-dir to override)."
        ),
    )
    parser.add_argument(
        "--opponent-mix",
        type=str,
        default=None,
        help=(
            "M3.4: per-rollout opponent-family sampling instead of a fixed "
            "--opponent, e.g. 'selfplay=0.5,damagefirst=0.3,random=0.2' "
            "(weights are normalized). Each rollout trains against one "
            "family; envs are reset when the family changes (in-flight "
            "episodes are abandoned and bootstrapped, standard PPO practice). "
            "Mutually exclusive with --opponent."
        ),
    )
    parser.add_argument(
        "--pretrain-checkpoint",
        type=str,
        default=None,
        help=(
            "M5.5: warm-start from a BC checkpoint (models/bc_pretrain_mlp.py "
            "output — a normal PPOAgent checkpoint). Unlike --resume, training "
            "starts at step 0 with this run's flags."
        ),
    )
    parser.add_argument(
        "--value-warmup-steps",
        type=int,
        default=0,
        help=(
            "M3.2 fix (ported for M5.5): freeze trunk + policy/opp heads for "
            "the first N steps, training only the value head, so the value "
            "function fits the warm-started policy before full PPO gradients "
            "flow through the shared trunk. 0 = off."
        ),
    )
    parser.add_argument(
        "--bc-anchor",
        type=str,
        default=None,
        help=(
            "M3.2 fix (ported for M5.5): path to a BC checkpoint kept frozen "
            "as a KL anchor — adds coef*KL(pi||pi_BC) to the loss so PPO "
            "can't drift far from human-cloned play. Held CONSTANT (not "
            "annealed): M3.2 showed decay resumes when the anchor anneals away."
        ),
    )
    parser.add_argument(
        "--bc-anchor-coef",
        type=float,
        default=0.05,
        help="KL-anchor coefficient (with --bc-anchor; default 0.05, the M3.2 value)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint (e.g. checkpoints/opp/ppo_step_2925396.pt) "
            "to resume training from. Restores full agent + optimizer state, "
            "parses the starting step count from the filename, and keeps "
            "appending checkpoints to the same directory (overrides "
            "--checkpoint-dir and the opp/v2/structured routing)."
        ),
    )
    args = parser.parse_args()
    if sum([args.obs_v2, args.obs_v3, args.obs_v3_extended]) > 1:
        parser.error("--obs-v2, --obs-v3, and --obs-v3-extended are mutually exclusive")
    if args.obs_v2 or args.obs_v3 or args.obs_v3_extended:
        args.structured = True
    if args.opponent_mix and args.opponent != "random":
        parser.error("--opponent-mix and --opponent are mutually exclusive")
    return args


def _parse_opponent_mix(spec: str) -> dict:
    """'selfplay=0.5,damagefirst=0.3,random=0.2' -> normalized weight dict."""
    mix = {}
    for part in spec.split(","):
        name, sep, weight = part.partition("=")
        name = name.strip()
        if not sep or name not in ("selfplay", "damagefirst", "random"):
            raise ValueError(
                f"bad --opponent-mix entry {part!r} "
                "(expected name=weight with name in selfplay/damagefirst/random)"
            )
        mix[name] = float(weight)
    total = sum(mix.values())
    if total <= 0:
        raise ValueError("--opponent-mix weights must sum to > 0")
    return {name: weight / total for name, weight in mix.items()}


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"step_(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def _sample_opponent(pool_dir: Path, agent: PPOAgent, device, rng) -> PPOAgent:
    """Pick a frozen self-play opponent for one rollout.

    50% the newest pool checkpoint, 50% uniform over the whole pool. Before
    any checkpoint exists, a frozen copy of the current policy is used.
    """
    checkpoints = sorted(pool_dir.glob("ppo_step_*.pt"), key=_checkpoint_step)
    if checkpoints:
        path = checkpoints[-1] if rng.random() < 0.5 else rng.choice(checkpoints)
        opponent = PPOAgent.load(str(path), device=device)
    else:
        opponent = PPOAgent(**agent._hparams, device=device)
        opponent.load_state_dict(agent.state_dict())
    opponent.eval()  # frozen: act_batch is already no_grad
    return opponent


def _push_pending(buffer: TrajectoryBuffer, pending: dict, done: bool) -> None:
    buffer.push(
        pending["obs"],
        pending["action"],
        pending["reward"],
        done,
        pending["value"],
        pending["log_prob"],
        valid_mask=pending["mask"],
        opp_action=pending.get("opp_action", -1),
    )


def main() -> None:
    args = parse_args()
    total_budget = args.steps
    rollout_steps = args.rollout_steps
    checkpoint_every = args.checkpoint_every
    num_envs = args.num_envs

    if args.resume:
        # Same directory as the checkpoint being resumed, so this run keeps
        # appending to the original sequence.
        checkpoint_dir = Path(args.resume).parent
    elif args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    elif args.opp_coef != 0.0:
        # M5: headed runs (aux opponent-prediction loss active) get their own
        # checkpoint dir so they never collide with the λ=0 control runs.
        checkpoint_dir = Path(__file__).parent / "checkpoints" / "opp"
    else:
        checkpoint_dir = (
            Path(__file__).parent / "checkpoints"
            / ("v3-extended" if args.obs_v3_extended else "v3" if args.obs_v3
               else "v2" if args.obs_v2
               else "structured" if args.structured else ".")
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mix = _parse_opponent_mix(args.opponent_mix) if args.opponent_mix else None
    mix_names = list(mix) if mix else None
    mix_weights = [mix[name] for name in mix_names] if mix else None

    # One buffer per env — GAE must never cross env streams. merge_buffers()
    # combines them (and normalizes advantages globally) before each update.
    buffers = [TrajectoryBuffer() for _ in range(num_envs)]
    # The opponent family is set per reset by _switch_family() below, so the
    # constructor doesn't need spawn-time opponent flags.
    env = VecGymClient(num_envs, structured=args.structured, obs_v2=args.obs_v2,
                       obs_v3=args.obs_v3, obs_v3_extended=args.obs_v3_extended)

    def _flatten(o):
        # The MLP consumes flat vectors: (N, 12, D) -> (N, 12*D) in structured
        # mode (D=65 v1, 77 v2, 86 v3, 87 v3-extended); flat mode is (N, 100).
        return o.reshape(num_envs, -1) if args.structured else o

    def _flatten_seat(state):
        state["obs"] = _flatten(state["obs"])
        return state

    rng = np.random.default_rng()

    obs_batch = masks = None      # single-seat rollout state
    p1_state = p2_state = None    # dual-seat (selfplay) rollout state
    # Open p1 transitions, one per env (selfplay only): a transition closes at
    # p1's NEXT decision point (or terminal), accumulating rewards from any
    # opponent-only steps (e.g. p2 force-switches) in between.
    pending: list = [None] * num_envs
    current_family = None

    def _sample_family() -> str:
        if mix is None:
            return args.opponent
        return str(rng.choice(mix_names, p=mix_weights))

    def _switch_family(family: str) -> None:
        """Reset every env into `family`, abandoning in-flight episodes.

        Only called at rollout boundaries: the abandoned episodes' last
        transitions were stored with done=False and bootstrapped from their
        value estimates — the same truncation PPO already applies at every
        rollout end.
        """
        nonlocal obs_batch, masks, p1_state, p2_state, pending, current_family
        current_family = family
        if family == "selfplay":
            p1_state, p2_state = env.reset_all_dual()
            _flatten_seat(p1_state)
            _flatten_seat(p2_state)
            pending = [None] * num_envs
        else:
            obs_batch, masks = env.reset_all(opponent=family)
            obs_batch = _flatten(obs_batch)

    _switch_family(_sample_family())
    obs_size = (p1_state["obs"] if current_family == "selfplay" else obs_batch).shape[1]

    # obs_size is derived from the actual observation rather than hardcoded,
    # so this loop works for the M1 flat vector (100), the M2 structured
    # flatten (780), and schema v2 (924) without duplicating the PPOAgent class.
    if args.resume:
        agent = PPOAgent.load(args.resume, device=args.device)
        ckpt_obs_size = agent.trunk[0].in_features
        if ckpt_obs_size != obs_size:
            raise ValueError(
                f"--resume checkpoint obs_size={ckpt_obs_size} does not match "
                f"the environment's obs_size={obs_size} — check --obs-v2/--structured"
            )
    elif args.pretrain_checkpoint:
        agent = PPOAgent.load(args.pretrain_checkpoint, device=args.device)
        ckpt_obs_size = agent.trunk[0].in_features
        if ckpt_obs_size != obs_size:
            raise ValueError(
                f"--pretrain-checkpoint obs_size={ckpt_obs_size} does not match "
                f"the environment's obs_size={obs_size} — check --obs-v2/--structured"
            )
        agent.opp_coef = args.opp_coef  # this run's flag, not the BC default
        print(f"Warm-started from BC checkpoint: {args.pretrain_checkpoint}", flush=True)
    else:
        agent = PPOAgent(obs_size=obs_size, hidden_size=args.hidden_size,
                         device=args.device, opp_coef=args.opp_coef)

    # --hidden-size cannot override a loaded checkpoint (load_state_dict needs
    # exact shapes), so a mismatch means the run is silently narrower/wider than
    # asked for. Fail loudly rather than train the wrong-sized agent for 5M steps.
    ckpt_hidden = agent.trunk[0].out_features
    if (args.resume or args.pretrain_checkpoint) and ckpt_hidden != args.hidden_size:
        raise ValueError(
            f"--hidden-size={args.hidden_size} but the loaded checkpoint has "
            f"hidden_size={ckpt_hidden}. Width comes from the checkpoint; drop "
            f"the flag to accept {ckpt_hidden}, or build a matching warm-start "
            f"with bc_pretrain_mlp.py --hidden-size {args.hidden_size}."
        )

    # Must come AFTER any PPOAgent.load() above: load_state_dict on the
    # optimizer restores the checkpoint's param_groups, lr included, so an lr
    # passed to the constructor would be silently reverted on every warm-start
    # and --resume. Overriding the live param_groups here is the only point
    # that survives both paths.
    effective_lr = agent.optimizer.param_groups[0]["lr"]
    if args.lr is not None and args.lr != effective_lr:
        for group in agent.optimizer.param_groups:
            group["lr"] = args.lr
        agent._hparams["lr"] = args.lr  # so the next load() reports it honestly
        print(f"Learning rate overridden: {effective_lr:g} -> {args.lr:g}", flush=True)
        effective_lr = args.lr

    # M3.2 fixes (M5.5 port): KL anchor to the BC policy + value-head warmup.
    if args.bc_anchor:
        agent.set_bc_anchor(args.bc_anchor, args.bc_anchor_coef)
        print(f"BC KL-anchor attached: {args.bc_anchor} (coef {args.bc_anchor_coef}, constant)", flush=True)
    warmup_active = False  # decided below, once the starting step is known
    # Self-play opponents come from this run's own checkpoints unless a pool
    # is given explicitly.
    pool_dir = Path(args.selfplay_pool) if args.selfplay_pool else checkpoint_dir
    uses_selfplay = args.opponent == "selfplay" or bool(mix and mix.get("selfplay"))

    print(
        f"Starting PPO training: {total_budget} steps | "
        f"obs_mode={'v3-extended' if args.obs_v3_extended else 'v3' if args.obs_v3 else 'v2' if args.obs_v2 else 'structured' if args.structured else 'flat'} "
        f"(obs_size={obs_size}) | "
        f"rollout={rollout_steps} steps | "
        f"hidden={agent.trunk[0].out_features} "
        f"(params={sum(p.numel() for p in agent.parameters()):,}) | "
        f"lr={effective_lr:g} | "
        f"num_envs={num_envs} | device={agent.device} | "
        f"opponent={args.opponent_mix or args.opponent}"
        + (f" (pool: {pool_dir})" if uses_selfplay else "")
        + " | "
        f"checkpoint every {checkpoint_every} steps | "
        f"opp_coef={args.opp_coef}",
        flush=True,
    )

    if args.resume:
        match = re.search(r"step_(\d+)", Path(args.resume).stem)
        if not match:
            raise ValueError(f"Cannot parse step count from --resume filename: {args.resume}")
        total_steps = int(match.group(1))
        last_checkpoint_step = total_steps
    else:
        total_steps = 0
        last_checkpoint_step = 0

    if args.value_warmup_steps > 0 and total_steps < args.value_warmup_steps:
        warmup_active = True
        agent.set_policy_frozen(True)
        print(f"Value warmup: policy frozen until step {args.value_warmup_steps}", flush=True)

    # Episode tracking within each rollout window
    rollout_wins = 0
    rollout_episodes = 0

    try:
        while total_steps < total_budget:
            # ----------------------------------------------------------------
            # Collect one rollout across all envs
            # ----------------------------------------------------------------
            rollout_wins = 0
            rollout_episodes = 0
            steps_this_rollout = 0

            # One opponent family per rollout (fixed --opponent, or sampled
            # from --opponent-mix); envs reset when the family changes.
            family = _sample_family()
            if family != current_family:
                _switch_family(family)

            if family == "selfplay":
                # One frozen opponent per rollout, shared by all envs so its
                # inference stays batched. Re-sampled every rollout. Pool
                # checkpoints may predate the current obs schema — their view
                # of a structured observation is sliced down per token.
                opponent_agent = _sample_opponent(pool_dir, agent, args.device, rng)
                opp_obs_size = opponent_agent._hparams["obs_size"]

                def _opp_view(obs):
                    return slice_structured_obs(obs, opp_obs_size) if args.structured else obs

                while steps_this_rollout < rollout_steps and total_steps < total_budget:
                    # p1 (the learner): a new decision point closes the
                    # previous pending transition, whose reward has accumulated
                    # any opponent-only steps since.
                    a1: list = [None] * num_envs
                    p1_idx = np.flatnonzero(p1_state["needs"])
                    if len(p1_idx):
                        acts, lps, vals = agent.act_batch(
                            p1_state["obs"][p1_idx], p1_state["mask"][p1_idx]
                        )
                        for k, i in enumerate(p1_idx):
                            if pending[i] is not None:
                                _push_pending(buffers[i], pending[i], done=False)
                            pending[i] = {
                                "obs": p1_state["obs"][i].copy(),
                                "action": int(acts[k]),
                                "log_prob": float(lps[k]),
                                "value": float(vals[k]),
                                "mask": p1_state["mask"][i].copy(),
                                "reward": 0.0,
                                "opp_action": -1,
                            }
                            a1[i] = int(acts[k])
                        total_steps += len(p1_idx)
                        steps_this_rollout += len(p1_idx)

                    # p2 (frozen opponent)
                    a2: list = [None] * num_envs
                    p2_idx = np.flatnonzero(p2_state["needs"])
                    if len(p2_idx):
                        opp_acts, _, _ = opponent_agent.act_batch(
                            _opp_view(p2_state["obs"][p2_idx]), p2_state["mask"][p2_idx]
                        )
                        for k, i in enumerate(p2_idx):
                            a2[i] = int(opp_acts[k])

                    p1_state, p2_state, rewards, dones, infos = env.step_dual(a1, a2)
                    _flatten_seat(p1_state)
                    _flatten_seat(p2_state)

                    for i in range(num_envs):
                        if "error" in infos[i]:
                            print(
                                f"  [step {total_steps}] env {i} error ({infos[i]['error']}) — reset",
                                flush=True,
                            )
                            pending[i] = None
                            continue
                        if pending[i] is not None:
                            pending[i]["reward"] += float(rewards[i])
                            if a1[i] is not None:
                                # This is the same step_dual() call that just
                                # submitted a1[i] — its info["opp_action"] is
                                # p1's label (the p2 seat's simultaneous
                                # choice), i.e. the label for the transition
                                # just opened above, not any older one still
                                # accumulating reward from a prior round.
                                pending[i]["opp_action"] = int(infos[i].get("opp_action", -1))
                        if dones[i]:
                            if pending[i] is not None:
                                _push_pending(buffers[i], pending[i], done=True)
                                pending[i] = None
                            rollout_episodes += 1
                            if infos[i].get("winner") == "Gym":
                                rollout_wins += 1

                    # Checkpoint on step boundary
                    if total_steps - last_checkpoint_step >= checkpoint_every:
                        ckpt_path = checkpoint_dir / f"ppo_step_{total_steps}.pt"
                        agent.save(str(ckpt_path))
                        last_checkpoint_step = total_steps
                        print(f"[checkpoint] Saved {ckpt_path}")

                # Close any still-open transitions so the buffers are complete;
                # their bootstrap comes from the current obs below.
                for i in range(num_envs):
                    if pending[i] is not None:
                        _push_pending(buffers[i], pending[i], done=False)
                        pending[i] = None
                bootstrap_obs, bootstrap_masks = p1_state["obs"], p1_state["mask"]

            else:
                while steps_this_rollout < rollout_steps and total_steps < total_budget:
                    actions, log_probs, values = agent.act_batch(obs_batch, masks)
                    # The masks in force when the actions were chosen — stored per
                    # step so PPO updates re-mask with real legality (M3.2).
                    step_masks = masks
                    next_obs, rewards, dones, infos, masks = env.step(actions)
                    next_obs = _flatten(next_obs)

                    for i in range(num_envs):
                        if "error" in infos[i]:
                            # Env was reset in place by VecGymClient; drop this slot's
                            # transition (same reset-and-continue handling as before).
                            print(
                                f"  [step {total_steps}] env {i} error ({infos[i]['error']}) — reset",
                                flush=True,
                            )
                            continue
                        buffers[i].push(
                            obs_batch[i],
                            int(actions[i]),
                            float(rewards[i]),
                            bool(dones[i]),
                            float(values[i]),
                            float(log_probs[i]),
                            valid_mask=step_masks[i],
                            opp_action=int(infos[i].get("opp_action", -1)),
                        )
                        if dones[i]:
                            rollout_episodes += 1
                            if infos[i].get("winner") == "Gym":
                                rollout_wins += 1

                    obs_batch = next_obs
                    total_steps += num_envs
                    steps_this_rollout += num_envs

                    # Checkpoint on step boundary
                    if total_steps - last_checkpoint_step >= checkpoint_every:
                        ckpt_path = checkpoint_dir / f"ppo_step_{total_steps}.pt"
                        agent.save(str(ckpt_path))
                        last_checkpoint_step = total_steps
                        print(f"[checkpoint] Saved {ckpt_path}")

                bootstrap_obs, bootstrap_masks = obs_batch, masks

            # ----------------------------------------------------------------
            # Compute advantages and update
            # ----------------------------------------------------------------
            if all(len(b) == 0 for b in buffers):
                break

            # Bootstrap each env from the value of its current obs. Where an
            # env's last stored transition was terminal, GAE's not_done factor
            # zeroes the bootstrap, so passing the fresh-episode value is safe.
            _, _, last_values = agent.act_batch(bootstrap_obs, bootstrap_masks)
            for i in range(num_envs):
                if len(buffers[i]) > 0:
                    buffers[i].compute_advantages(
                        last_value=float(last_values[i]), normalize=False
                    )

            merged = merge_buffers(buffers)
            loss = agent.update(merged)
            for b in buffers:
                b.clear()

            if warmup_active and total_steps >= args.value_warmup_steps:
                warmup_active = False
                agent.set_policy_frozen(False)
                print(f"[value-warmup] complete at step {total_steps} — policy unfrozen", flush=True)

            # ----------------------------------------------------------------
            # Log rollout summary
            # ----------------------------------------------------------------
            win_rate = (
                rollout_wins / rollout_episodes if rollout_episodes > 0 else float("nan")
            )
            log_line = (
                f"Step {total_steps}/{total_budget} | "
                f"Win rate (rollout): {win_rate:.2f} | "
                f"Loss: {loss:.3f}"
            )
            if args.opp_coef != 0.0:
                # M5: fraction of this rollout's pushed steps with a valid
                # (unmasked) opp_action label — the aux CE only trains on
                # these. ppo_agent.update() doesn't expose the CE term
                # separately from the combined loss, so coverage is the only
                # additional diagnostic available here.
                opp_actions_t = merged.get("opp_actions")
                if opp_actions_t is not None and opp_actions_t.numel() > 0:
                    coverage = float((opp_actions_t >= 0).float().mean())
                    log_line += f" | Opp label coverage: {coverage:.2f}"
            print(log_line)

    finally:
        env.close()

    # Final checkpoint
    final_path = checkpoint_dir / f"ppo_step_{total_steps}_final.pt"
    agent.save(str(final_path))
    print(f"\nTraining complete. Final model saved to {final_path}")
    print(f"Total steps: {total_steps}")


if __name__ == "__main__":
    main()

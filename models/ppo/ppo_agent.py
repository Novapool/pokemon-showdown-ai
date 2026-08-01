"""
ppo_agent.py — PPO Actor-Critic agent for Pokemon Showdown.

Architecture (H = hidden_size, default 128):
  Shared trunk: Linear(obs_size, H) → ReLU → Linear(H, H) → ReLU
  Policy head:  Linear(H, 9)  — outputs action logits
  Value head:   Linear(H, 1)  — outputs state value scalar
  Opp head:     Linear(H, 9)  — outputs opponent-action logits (M5 aux head)

Parameter count is h**2 + (obs_size + 21)*h + 19; at the v3 obs_size=1032 that
is 151,187 at H=128 and 801,299 at H=512. Note the input layer dominates (87%
of params at H=128), so H mostly buys capacity right at the obs bottleneck.

hidden_size is stored in _hparams, so load() reconstructs the right width from
the checkpoint itself and checkpoints written before this knob existed (which
carry no hidden_size key) fall back to the 128 default unchanged.
"""

import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PPOAgent(nn.Module):
    """Actor-Critic agent trained with Proximal Policy Optimization."""

    # Set True to make act()/act_batch() play the masked argmax instead of
    # sampling (evaluate.py --greedy, infer_server.py --greedy). Rollouts must
    # sample, so the trainer never touches this and the default stays False.
    greedy = False

    def __init__(
        self,
        obs_size: int = 100,
        n_actions: int = 9,
        hidden_size: int = 128,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        opp_coef: float = 0.0,
        device: str | None = None,
    ):
        super().__init__()

        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        # Default coefficient for the M5 auxiliary opponent-prediction CE loss.
        # 0.0 reproduces the pre-M5 loss path bit-for-bit; the trainer overrides
        # it per-update via update(..., opp_coef=λ).
        self.opp_coef = opp_coef

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Policy head
        self.policy_head = nn.Linear(hidden_size, n_actions)

        # Value head
        self.value_head = nn.Linear(hidden_size, 1)

        # Opponent-prediction head (M5): predicts the opponent's simultaneous
        # action in the opponent's own 9-way action frame. Off the same shared
        # trunk as policy/value.
        self.opp_head = nn.Linear(hidden_size, n_actions)
        # Overwritten by load() to False when the loaded checkpoint predates
        # the opp head (M5) and the head above is a fresh init, not trained.
        self.has_opp_head = True

        # device is a runtime choice, not a model hyperparameter — it is
        # deliberately excluded from _hparams so checkpoints stay portable
        # across machines (Mac/MPS ↔ CUDA box).
        self.device = torch.device(device) if device else _pick_device()
        self.to(self.device)

        # Single optimizer over all parameters
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # M3.2-style fine-tune knobs, ported for M5.5 (BC→PPO): configured by
        # train.py via set_bc_anchor()/set_policy_frozen(), never saved into
        # checkpoints. The anchor lives in a plain list so nn.Module does not
        # register it (keeps it out of state_dict/parameters()); the optimizer
        # above was already created, so anchor params are never optimized.
        self._bc_anchor: list["PPOAgent"] = []
        self.bc_anchor_coef = 0.0
        self._policy_frozen = False

        # Store hparams for save/load
        self._hparams = dict(
            obs_size=obs_size,
            n_actions=n_actions,
            hidden_size=hidden_size,
            lr=lr,
            clip_eps=clip_eps,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            ppo_epochs=ppo_epochs,
            batch_size=batch_size,
            opp_coef=opp_coef,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self, obs: np.ndarray, valid_mask: list
    ) -> tuple[int, float, float]:
        """Pick an action from the policy — sampled, or argmax when self.greedy.

        Args:
            obs:        Observation array, shape (100,), float32.
            valid_mask: List of bool, length 9.  True = action is legal.

        Returns:
            (action, log_prob, value) — all as Python scalars.
        """
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        mask_t = torch.tensor(valid_mask, dtype=torch.bool).unsqueeze(0).to(self.device)

        features = self.trunk(obs_t)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)

        # Mask out invalid actions
        logits = logits.masked_fill(~mask_t, -1e9)

        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if self.greedy else dist.sample()
        log_prob = dist.log_prob(action)

        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.item()),
        )

    @torch.no_grad()
    def act_batch(
        self, obs_batch: np.ndarray, valid_masks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pick actions for a batch of observations (one per parallel env) —
        sampled, or argmax when self.greedy.

        Args:
            obs_batch:   float32 array, shape (N, obs_size).
            valid_masks: bool array, shape (N, 9). True = action is legal.

        Returns:
            (actions, log_probs, values) — numpy arrays of shape (N,), dtypes
            int64/float32/float32.
        """
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32).to(self.device)
        mask_t = torch.as_tensor(np.asarray(valid_masks, dtype=bool)).to(self.device)

        features = self.trunk(obs_t)
        logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)

        logits = logits.masked_fill(~mask_t, -1e9)

        dist = Categorical(logits=logits)
        actions = logits.argmax(dim=-1) if self.greedy else dist.sample()
        log_probs = dist.log_prob(actions)

        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy().astype(np.float32),
            values.cpu().numpy().astype(np.float32),
        )

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # M3.2/M5.5 fine-tune configuration (value-head warmup, BC KL-anchor)
    # ------------------------------------------------------------------

    @property
    def bc_policy(self) -> "PPOAgent | None":
        return self._bc_anchor[0] if self._bc_anchor else None

    def set_policy_frozen(self, frozen: bool) -> None:
        """Freeze/unfreeze the trunk + policy/opp heads (value-head warmup).

        Training only the value head for the first N steps lets the value
        function fit the BC policy before full PPO gradients flow through the
        shared trunk and scramble the BC-learned features (the M3 failure
        mode, diagnosed and fixed in M3.2 for the transformer).
        """
        for module in (self.trunk, self.policy_head, self.opp_head):
            for p in module.parameters():
                p.requires_grad = not frozen
        self._policy_frozen = frozen

    def set_bc_anchor(self, checkpoint_path: str, coef: float) -> None:
        """Attach a frozen copy of a BC checkpoint as a KL anchor.

        update() adds `coef × KL(π_θ ‖ π_BC)` to the loss so PPO fine-tuning
        can't drift arbitrarily far from human-cloned play. The anchor is a
        separate frozen agent; it is never saved into training checkpoints
        (train.py re-attaches it on --resume) and its parameters are not in
        the optimizer.
        """
        anchor = PPOAgent.load(checkpoint_path, device=str(self.device))
        anchor.eval()
        for p in anchor.parameters():
            p.requires_grad = False
        self._bc_anchor = [anchor]
        self.bc_anchor_coef = coef

    def evaluate_actions(
        self,
        obs_batch: torch.Tensor,
        actions_batch: torch.Tensor,
        valid_mask_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute log-probs, values, entropy, and opponent logits for a batch
        of (obs, action) pairs.

        Args:
            obs_batch:        float32 tensor, shape (B, obs_size)
            actions_batch:    long tensor,    shape (B,)
            valid_mask_batch: bool tensor,    shape (B, n_actions)

        Returns:
            (log_probs, values, entropy, opp_logits)
              log_probs  — float32 tensor, shape (B,)
              values     — float32 tensor, shape (B,)
              entropy    — scalar float32 tensor (mean entropy)
              opp_logits — float32 tensor, shape (B, n_actions); raw (unmasked)
                           opponent-prediction logits off the shared trunk
        """
        features = self.trunk(obs_batch)
        logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        opp_logits = self.opp_head(features)

        logits = logits.masked_fill(~valid_mask_batch, -1e9)

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_batch)
        entropy = dist.entropy().mean()

        return log_probs, values, entropy, opp_logits

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, data, opp_coef: float | None = None) -> float:
        """Run PPO_EPOCHS passes of minibatch updates over one rollout batch.

        Args:
            data: Either a tensor dict as produced by
                  TrajectoryBuffer.get_tensors() / merge_buffers(), or a
                  TrajectoryBuffer with compute_advantages() already called.
            opp_coef: Coefficient λ for the M5 auxiliary opponent-prediction
                  cross-entropy term. None uses self.opp_coef. λ = 0.0 skips
                  the term entirely and reproduces the pre-M5 loss path
                  bit-for-bit (the opp head then receives no gradient).

        Returns:
            Mean total loss (float) across all minibatch updates.
        """
        if hasattr(data, "get_tensors"):
            data = data.get_tensors()
        obs = data["obs"].to(self.device)
        actions = data["actions"].to(self.device)
        returns = data["returns"].to(self.device)
        advantages = data["advantages"].to(self.device)
        old_log_probs = data["log_probs"].to(self.device)

        n = obs.shape[0]
        # Real per-step action-legality masks (M3.2). Older buffers without a
        # "masks" key fall back to all-legal, the pre-M3.2 behavior.
        if "masks" in data:
            full_mask = data["masks"].to(self.device)
        else:
            full_mask = torch.ones(
                n, self._hparams["n_actions"], dtype=torch.bool, device=self.device
            )

        # M5 opponent-action labels (-1 = masked). Buffers without the key
        # (pre-M5 data) are treated as fully masked so the aux term is a no-op.
        if "opp_actions" in data:
            opp_actions = data["opp_actions"].to(self.device)
        else:
            opp_actions = torch.full(
                (n,), -1, dtype=torch.long, device=self.device
            )

        opp_coef = self.opp_coef if opp_coef is None else opp_coef

        total_loss_sum = 0.0
        num_updates = 0

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                idx = indices[start : start + self.batch_size]
                if len(idx) == 0:
                    continue

                obs_b = obs[idx]
                actions_b = actions[idx]
                returns_b = returns[idx]
                advantages_b = advantages[idx]
                old_log_probs_b = old_log_probs[idx]
                mask_b = full_mask[idx]

                new_log_probs, values_b, entropy, opp_logits_b = (
                    self.evaluate_actions(obs_b, actions_b, mask_b)
                )

                # PPO clipped surrogate objective
                ratio = torch.exp(new_log_probs - old_log_probs_b)
                surrogate1 = ratio * advantages_b
                surrogate2 = (
                    torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    * advantages_b
                )
                surrogate_loss = torch.min(surrogate1, surrogate2).mean()

                # Value loss (MSE)
                value_loss = nn.functional.mse_loss(values_b, returns_b)

                # Total loss
                loss = (
                    -surrogate_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                # M5 auxiliary opponent-prediction CE, computed only over steps
                # with a valid (>= 0) label. λ = 0 skips it entirely (pre-M5
                # path); an all-masked minibatch also skips it, so the CE term
                # is never NaN.
                if opp_coef != 0.0:
                    opp_labels_b = opp_actions[idx]
                    valid_opp = opp_labels_b >= 0
                    if valid_opp.any():
                        opp_ce = nn.functional.cross_entropy(
                            opp_logits_b[valid_opp], opp_labels_b[valid_opp]
                        )
                        loss = loss + opp_coef * opp_ce

                # BC KL-anchor (M3.2 recipe, M5.5 port): penalize divergence
                # from the frozen human-cloned policy on the same minibatch.
                if self.bc_policy is not None and self.bc_anchor_coef > 0:
                    features_b = self.trunk(obs_b)
                    logits_b = self.policy_head(features_b).masked_fill(~mask_b, -1e9)
                    dist_b = Categorical(logits=logits_b)
                    with torch.no_grad():
                        bc_features = self.bc_policy.trunk(obs_b)
                        bc_logits = self.bc_policy.policy_head(bc_features).masked_fill(~mask_b, -1e9)
                    bc_dist = Categorical(logits=bc_logits)
                    anchor_kl = torch.distributions.kl_divergence(dist_b, bc_dist).mean()
                    loss = loss + self.bc_anchor_coef * anchor_kl

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss_sum += loss.item()
                num_updates += 1

        return total_loss_sum / max(num_updates, 1)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save agent state dicts and hyperparameters to path."""
        torch.save(
            {
                "trunk": self.trunk.state_dict(),
                "policy_head": self.policy_head.state_dict(),
                "value_head": self.value_head.state_dict(),
                "opp_head": self.opp_head.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "hparams": self._hparams,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "PPOAgent":
        """Reconstruct an agent from a checkpoint file.

        Args:
            path:   Path to a .pt file written by save().
            device: Optional device override ("cpu"/"mps"/"cuda"); auto-detected
                    when omitted.

        Returns:
            Fully restored PPOAgent instance.
        """
        checkpoint = torch.load(path, map_location="cpu")
        # Unknown hparam keys (e.g. opp_coef missing from pre-M5 checkpoints)
        # fall back to the __init__ defaults — the target_kl-style backward-
        # compatible pattern used by the transformer agent.
        hparams = checkpoint["hparams"]
        agent = cls(**hparams, device=device)  # device auto-detected when None
        agent.trunk.load_state_dict(checkpoint["trunk"])
        agent.policy_head.load_state_dict(checkpoint["policy_head"])
        agent.value_head.load_state_dict(checkpoint["value_head"])

        # M5 opp head: present in headed checkpoints, absent in pre-M5 ones.
        # Missing → keep the freshly-initialized head and warn (not a failure).
        agent.has_opp_head = "opp_head" in checkpoint
        if agent.has_opp_head:
            agent.opp_head.load_state_dict(checkpoint["opp_head"])
        else:
            warnings.warn(
                "checkpoint has no 'opp_head'; using a freshly initialized "
                "opponent-prediction head (expected for pre-M5 checkpoints)",
                stacklevel=2,
            )

        # Pre-M5 optimizer state omits the opp_head params, so its param-group
        # size won't match the headed optimizer and load_state_dict raises.
        # Fall back to a fresh optimizer with a warning rather than failing.
        try:
            agent.optimizer.load_state_dict(checkpoint["optimizer"])
        except (ValueError, KeyError) as err:
            warnings.warn(
                f"could not restore optimizer state ({err}); using a fresh "
                "optimizer (expected when loading a pre-M5 checkpoint into the "
                "headed agent)",
                stacklevel=2,
            )
        return agent

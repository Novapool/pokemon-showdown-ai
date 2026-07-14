"""
transformer_agent.py — PPO Actor-Critic agent wrapping TransformerPolicy.

Composes TransformerPolicy rather than subclassing it, so the policy's
state-dict keys (embed.*, encoder.*, policy_head.*, value_head.*) stay
unprefixed and load_pretrain_checkpoint() — which matches by exact key
name — can load the BC checkpoint into agent.policy directly.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from transformer_policy import (
    TransformerPolicy,
    load_pretrain_checkpoint,
    N_ACTIONS,
    TOKEN_DIM,
)


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TransformerAgent(nn.Module):
    """Actor-Critic agent trained with PPO over a transformer token encoder."""

    def __init__(
        self,
        token_dim: int = TOKEN_DIM,
        n_actions: int = N_ACTIONS,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        target_kl: float = 0.02,
        device: str | None = None,
    ):
        super().__init__()

        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.target_kl = target_kl
        self.lr = lr

        self.policy = TransformerPolicy(
            token_dim=token_dim,
            n_actions=n_actions,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

        # device is a runtime choice, not a model hyperparameter — it is
        # deliberately excluded from _hparams so checkpoints stay portable
        # across machines (Mac/MPS ↔ CUDA box).
        self.device = torch.device(device) if device else _pick_device()
        self.to(self.device)

        # M3.2: optional frozen BC policy used as a KL anchor during PPO
        # updates, and a policy-freeze flag for value-head warmup. Both are
        # configured by train.py (set_bc_anchor / set_policy_frozen), never
        # persisted in checkpoints.
        self.bc_policy: TransformerPolicy | None = None
        self.bc_anchor_coef = 0.0
        self._policy_frozen = False

        # Single optimizer over all parameters. Created before set_bc_anchor()
        # can ever run, so the frozen anchor's parameters are never optimized.
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Store hparams for save/load
        self._hparams = dict(
            token_dim=token_dim,
            n_actions=n_actions,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            lr=lr,
            clip_eps=clip_eps,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            ppo_epochs=ppo_epochs,
            batch_size=batch_size,
            target_kl=target_kl,
        )

    # ------------------------------------------------------------------
    # M3.2 configuration (value-head warmup, BC KL-anchor)
    # ------------------------------------------------------------------

    def set_policy_frozen(self, frozen: bool) -> None:
        """Freeze/unfreeze everything except the value head.

        Used for value-head warmup: BC pretraining trains the policy head
        only, so the value head starts random. Freezing embed/encoder/
        policy_head for the first N steps lets the value function fit the BC
        policy before full PPO gradients flow through the shared encoder and
        scramble the BC-learned features (the M3 failure mode).
        """
        for module in (self.policy.embed, self.policy.encoder, self.policy.policy_head):
            for p in module.parameters():
                p.requires_grad = not frozen
        self._policy_frozen = frozen

    def set_bc_anchor(self, checkpoint_path: str, coef: float) -> None:
        """Attach a frozen copy of the BC policy as a KL anchor.

        update() adds `coef × KL(π_θ ‖ π_BC)` to the loss so PPO fine-tuning
        can't drift arbitrarily far from human-cloned play. The anchor is a
        separate frozen TransformerPolicy loaded from the BC checkpoint; it is
        never saved into training checkpoints (train.py re-attaches it on
        --resume), and its parameters are not in the optimizer.
        """
        policy_keys = ("token_dim", "n_actions", "d_model", "nhead", "num_layers", "d_ff", "dropout")
        self.bc_policy = TransformerPolicy(**{k: self._hparams[k] for k in policy_keys})
        load_pretrain_checkpoint(self.bc_policy, checkpoint_path)
        self.bc_policy.to(self.device).eval()
        for p in self.bc_policy.parameters():
            p.requires_grad = False
        self.bc_anchor_coef = coef

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self, obs: np.ndarray, valid_mask: list
    ) -> tuple[int, float, float]:
        """Sample an action from the policy.

        Args:
            obs:        Observation array, shape (12, 65), float32.
            valid_mask: List of bool, length 9.  True = action is legal.

        Returns:
            (action, log_prob, value) — all as Python scalars.
        """
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        mask_t = torch.tensor(valid_mask, dtype=torch.bool).unsqueeze(0).to(self.device)

        logits, value = self.policy(obs_t)

        # Mask out invalid actions
        logits = logits.masked_fill(~mask_t, -1e9)

        dist = Categorical(logits=logits)
        action = dist.sample()
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
        """Sample actions for a batch of observations (one per parallel env).

        Args:
            obs_batch:   float32 array, shape (N, 12, 65).
            valid_masks: bool array, shape (N, 9). True = action is legal.

        Returns:
            (actions, log_probs, values) — numpy arrays of shape (N,), dtypes
            int64/float32/float32.
        """
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32).to(self.device)
        mask_t = torch.as_tensor(np.asarray(valid_masks, dtype=bool)).to(self.device)

        logits, values = self.policy(obs_t)
        logits = logits.masked_fill(~mask_t, -1e9)

        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy().astype(np.float32),
            values.cpu().numpy().astype(np.float32),
        )

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        obs_batch: torch.Tensor,
        actions_batch: torch.Tensor,
        valid_mask_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute log-probs, values, and entropy for a batch of (obs, action) pairs.

        Args:
            obs_batch:        float32 tensor, shape (B, 12, 65)
            actions_batch:    long tensor,    shape (B,)
            valid_mask_batch: bool tensor,    shape (B, n_actions)

        Returns:
            (log_probs, values, entropy, dist)
              log_probs — float32 tensor, shape (B,)
              values    — float32 tensor, shape (B,)
              entropy   — scalar float32 tensor (mean entropy)
              dist      — the Categorical over masked logits (used by the
                          BC KL-anchor term in update())
        """
        logits, values = self.policy(obs_batch)
        logits = logits.masked_fill(~valid_mask_batch, -1e9)

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_batch)
        entropy = dist.entropy().mean()

        return log_probs, values, entropy, dist

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, data) -> tuple[float, bool]:
        """Run PPO_EPOCHS passes of minibatch updates over one rollout batch.

        Args:
            data: Either a tensor dict as produced by
                  TrajectoryBuffer.get_tensors() / merge_buffers()
                  (models/ppo/trajectory_buffer.py), or a TrajectoryBuffer
                  with compute_advantages() already called. obs tensors here
                  are (T, 12, 65) rather than (T, obs_size), but
                  evaluate_actions() already handles that shape via
                  TransformerPolicy.forward().

        Returns:
            (mean total loss, whether the approx-KL early-stop triggered).
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

        total_loss_sum = 0.0
        num_updates = 0
        kl_early_stop = False

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(n)
            stop_early = False
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

                new_log_probs, values_b, entropy, dist = self.evaluate_actions(
                    obs_b, actions_b, mask_b
                )

                # PPO clipped surrogate objective
                log_ratio = new_log_probs - old_log_probs_b
                ratio = torch.exp(log_ratio)
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

                # BC KL-anchor (M3.2): penalize divergence from the frozen BC
                # policy so PPO can't wander far from human-cloned play.
                # Illegal actions are masked to -1e9 in both distributions, so
                # their ~0 probabilities contribute nothing to the KL.
                if self.bc_policy is not None and self.bc_anchor_coef > 0:
                    with torch.no_grad():
                        bc_logits, _ = self.bc_policy(obs_b)
                        bc_logits = bc_logits.masked_fill(~mask_b, -1e9)
                    bc_dist = Categorical(logits=bc_logits)
                    anchor_kl = torch.distributions.kl_divergence(dist, bc_dist).mean()
                    loss = loss + self.bc_anchor_coef * anchor_kl

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss_sum += loss.item()
                num_updates += 1

                # Approximate-KL early stopping (http://joschu.net/blog/kl-approx.html):
                # a single minibatch pushing the policy this far from the rollout
                # policy is the failure mode behind the mid-training win-rate
                # collapses observed on long runs — stop this rollout's updates
                # rather than let the policy move further on a bad batch.
                # While the policy is frozen (value warmup) the only ratio
                # movement is dropout noise — skip the early-stop so value
                # epochs aren't cut short spuriously.
                with torch.no_grad():
                    approx_kl = (ratio - 1 - log_ratio).mean().item()
                if not self._policy_frozen and approx_kl > 1.5 * self.target_kl:
                    stop_early = True
                    kl_early_stop = True
                    break

            if stop_early:
                break

        return total_loss_sum / max(num_updates, 1), kl_early_stop

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save agent state dicts and hyperparameters to path."""
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "hparams": self._hparams,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "TransformerAgent":
        """Reconstruct an agent from a checkpoint file.

        Args:
            path:   Path to a .pt file written by save().
            device: Optional device override ("cpu"/"mps"/"cuda"); auto-detected
                    when omitted.

        Returns:
            Fully restored TransformerAgent instance.
        """
        checkpoint = torch.load(path, map_location="cpu")
        hparams = checkpoint["hparams"]
        agent = cls(**hparams, device=device)  # device auto-detected when None
        agent.policy.load_state_dict(checkpoint["policy"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        return agent

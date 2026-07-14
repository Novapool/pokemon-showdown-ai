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

from transformer_policy import TransformerPolicy, N_ACTIONS, TOKEN_DIM


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

        # Single optimizer over all parameters
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
            (log_probs, values, entropy)
              log_probs — float32 tensor, shape (B,)
              values    — float32 tensor, shape (B,)
              entropy   — scalar float32 tensor (mean entropy)
        """
        logits, values = self.policy(obs_batch)
        logits = logits.masked_fill(~valid_mask_batch, -1e9)

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_batch)
        entropy = dist.entropy().mean()

        return log_probs, values, entropy

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
        # Build a uniform valid mask (all actions valid) for minibatch evaluation.
        # The actual per-step masks are not stored in the buffer; PPO uses the
        # importance-ratio clipping rather than re-masking during updates.
        full_mask = torch.ones(n, self._hparams["n_actions"], dtype=torch.bool, device=self.device)

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

                new_log_probs, values_b, entropy = self.evaluate_actions(
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
                with torch.no_grad():
                    approx_kl = (ratio - 1 - log_ratio).mean().item()
                if approx_kl > 1.5 * self.target_kl:
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

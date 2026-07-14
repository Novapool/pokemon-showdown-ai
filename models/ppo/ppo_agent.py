"""
ppo_agent.py — PPO Actor-Critic agent for Pokemon Showdown.

Architecture:
  Shared trunk: Linear(100, 128) → ReLU → Linear(128, 128) → ReLU
  Policy head:  Linear(128, 9)  — outputs action logits
  Value head:   Linear(128, 1)  — outputs state value scalar
"""

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

    def __init__(
        self,
        obs_size: int = 100,
        n_actions: int = 9,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        device: str | None = None,
    ):
        super().__init__()

        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Policy head
        self.policy_head = nn.Linear(128, n_actions)

        # Value head
        self.value_head = nn.Linear(128, 1)

        # device is a runtime choice, not a model hyperparameter — it is
        # deliberately excluded from _hparams so checkpoints stay portable
        # across machines (Mac/MPS ↔ CUDA box).
        self.device = torch.device(device) if device else _pick_device()
        self.to(self.device)

        # Single optimizer over all parameters
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Store hparams for save/load
        self._hparams = dict(
            obs_size=obs_size,
            n_actions=n_actions,
            lr=lr,
            clip_eps=clip_eps,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            ppo_epochs=ppo_epochs,
            batch_size=batch_size,
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
            obs_batch:        float32 tensor, shape (B, obs_size)
            actions_batch:    long tensor,    shape (B,)
            valid_mask_batch: bool tensor,    shape (B, n_actions)

        Returns:
            (log_probs, values, entropy)
              log_probs — float32 tensor, shape (B,)
              values    — float32 tensor, shape (B,)
              entropy   — scalar float32 tensor (mean entropy)
        """
        features = self.trunk(obs_batch)
        logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)

        logits = logits.masked_fill(~valid_mask_batch, -1e9)

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_batch)
        entropy = dist.entropy().mean()

        return log_probs, values, entropy

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, data) -> float:
        """Run PPO_EPOCHS passes of minibatch updates over one rollout batch.

        Args:
            data: Either a tensor dict as produced by
                  TrajectoryBuffer.get_tensors() / merge_buffers(), or a
                  TrajectoryBuffer with compute_advantages() already called.

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
        # Build a uniform valid mask (all actions valid) for minibatch evaluation.
        # The actual per-step masks are not stored in the buffer; PPO uses the
        # importance-ratio clipping rather than re-masking during updates.
        full_mask = torch.ones(n, self._hparams["n_actions"], dtype=torch.bool, device=self.device)

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

                new_log_probs, values_b, entropy = self.evaluate_actions(
                    obs_b, actions_b, mask_b
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
        hparams = checkpoint["hparams"]
        agent = cls(**hparams, device=device)  # device auto-detected when None
        agent.trunk.load_state_dict(checkpoint["trunk"])
        agent.policy_head.load_state_dict(checkpoint["policy_head"])
        agent.value_head.load_state_dict(checkpoint["value_head"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        return agent

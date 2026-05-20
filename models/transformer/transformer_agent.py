"""
transformer_agent.py — Transformer-based PPO Actor-Critic agent for Pokemon Showdown.

Architecture:
  token_proj:   Linear(73, 128)                          — project each of 12 tokens
  pos_embed:    Embedding(12, 128)                       — learnable per-slot positional bias
  encoder:      TransformerEncoder(2 × EncoderLayer)     — batch_first=True, no dropout
  mean pool:    features.mean(dim=1)                     — (B, 12, 128) → (B, 128)
  policy_head:  Linear(128, 9)                           — action logits
  value_head:   Linear(128, 1)                           — state value
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))          # models/
sys.path.insert(0, str(Path(__file__).parent))                 # models/transformer/
sys.path.insert(0, str(Path(__file__).parent.parent / "ppo"))  # models/ppo/ for TrajectoryBuffer

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from trajectory_buffer import TrajectoryBuffer


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TransformerAgent(nn.Module):
    """Transformer-based Actor-Critic agent trained with Proximal Policy Optimization.

    Obs shape: (12, 73) — 12 tokens, each with 73 features.
    Action space: 9 discrete actions (4 moves + 5 switches).
    """

    def __init__(
        self,
        token_dim: int = 73,
        n_tokens: int = 12,
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 256,
        n_layers: int = 2,
        n_actions: int = 9,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.05,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 2,
        batch_size: int = 64,
    ):
        super().__init__()

        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.n_tokens = n_tokens
        self.d_model = d_model

        # Project each token from token_dim → d_model
        self.token_proj = nn.Linear(token_dim, d_model)

        # Learnable positional embedding: one vector per token slot
        self.pos_embed = nn.Embedding(n_tokens, d_model)

        # Register position indices as a buffer so .to(device) moves them correctly
        self.register_buffer("pos_indices", torch.arange(n_tokens))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Policy and value heads
        self.policy_head = nn.Linear(d_model, n_actions)
        self.value_head = nn.Linear(d_model, 1)

        self.device = _pick_device()
        self.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Store all constructor hparams for save/load round-trip
        self._hparams = dict(
            token_dim=token_dim,
            n_tokens=n_tokens,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
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
    # Forward (shared backbone)
    # ------------------------------------------------------------------

    def _encode(self, obs_t: torch.Tensor) -> torch.Tensor:
        """Project tokens, add positional embeddings, run encoder, mean-pool.

        Args:
            obs_t: float32 tensor, shape (B, n_tokens, token_dim)

        Returns:
            features: float32 tensor, shape (B, d_model)
        """
        x = self.token_proj(obs_t)                   # (B, 12, 128)
        x = x + self.pos_embed(self.pos_indices)      # broadcast (12, 128) → (B, 12, 128)
        x = self.encoder(x)                           # (B, 12, 128)
        return x.mean(dim=1)                          # (B, 128)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self, obs: np.ndarray, valid_mask: list
    ) -> tuple[int, float, float]:
        """Sample an action from the policy.

        Args:
            obs:        Observation array, shape (12, 69), float32.
            valid_mask: List of bool, length 9.  True = action is legal.

        Returns:
            (action, log_prob, value) — all as Python scalars.
        """
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        # obs_t shape: (1, 12, 69)

        mask_t = torch.tensor(valid_mask, dtype=torch.bool).unsqueeze(0).to(self.device)

        features = self._encode(obs_t)                # (1, 128)
        logits = self.policy_head(features)           # (1, 9)
        value = self.value_head(features).squeeze(-1) # (1,)

        logits = logits.masked_fill(~mask_t, -1e9)

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.item()),
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
            obs_batch:        float32 tensor, shape (B, 12, 69)
            actions_batch:    long tensor,    shape (B,)
            valid_mask_batch: bool tensor,    shape (B, n_actions)

        Returns:
            (log_probs, values, entropy)
              log_probs — float32 tensor, shape (B,)
              values    — float32 tensor, shape (B,)
              entropy   — scalar float32 tensor (mean entropy)
        """
        features = self._encode(obs_batch)            # (B, 128)
        logits = self.policy_head(features)           # (B, 9)
        values = self.value_head(features).squeeze(-1) # (B,)

        logits = logits.masked_fill(~valid_mask_batch, -1e9)

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_batch)
        entropy = dist.entropy().mean()

        return log_probs, values, entropy

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, buffer: TrajectoryBuffer) -> float:
        """Run ppo_epochs passes of minibatch updates over the rollout buffer.

        Args:
            buffer: A TrajectoryBuffer with compute_advantages() already called.

        Returns:
            Mean total loss (float) across all minibatch updates.
        """
        data = buffer.get_tensors()
        obs = data["obs"].to(self.device)             # (T, 12, 69) stacked by buffer
        actions = data["actions"].to(self.device)
        returns = data["returns"].to(self.device)
        advantages = data["advantages"].to(self.device)
        old_log_probs = data["log_probs"].to(self.device)

        n = obs.shape[0]
        # Build a uniform valid mask — per-step masks are not stored in the buffer.
        # PPO uses importance-ratio clipping instead of re-masking during updates.
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
        """Save model state dicts, optimizer state, and hyperparameters to path."""
        torch.save(
            {
                "token_proj": self.token_proj.state_dict(),
                "pos_embed": self.pos_embed.state_dict(),
                "encoder": self.encoder.state_dict(),
                "policy_head": self.policy_head.state_dict(),
                "value_head": self.value_head.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "hparams": self._hparams,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "TransformerAgent":
        """Reconstruct an agent from a checkpoint file.

        Args:
            path: Path to a .pt file written by save().

        Returns:
            Fully restored TransformerAgent instance.
        """
        checkpoint = torch.load(path, map_location="cpu")
        hparams = checkpoint["hparams"]
        agent = cls(**hparams)  # device auto-detected in __init__
        agent.token_proj.load_state_dict(checkpoint["token_proj"])
        agent.pos_embed.load_state_dict(checkpoint["pos_embed"])
        agent.encoder.load_state_dict(checkpoint["encoder"])
        agent.policy_head.load_state_dict(checkpoint["policy_head"])
        agent.value_head.load_state_dict(checkpoint["value_head"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        return agent

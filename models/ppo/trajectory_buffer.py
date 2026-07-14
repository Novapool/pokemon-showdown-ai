"""
trajectory_buffer.py — Rollout buffer for PPO with GAE advantage estimation.

Stores per-step transitions during a rollout, computes Generalized Advantage
Estimation (GAE), and exposes tensors for the PPO update.
"""

import numpy as np
import torch


class TrajectoryBuffer:
    """Stores one rollout of transitions and computes GAE advantages."""

    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        self.gamma = gamma
        self.lam = lam

        self._obs: list = []
        self._actions: list = []
        self._rewards: list = []
        self._dones: list = []
        self._values: list = []
        self._log_probs: list = []

        self.returns: torch.Tensor | None = None
        self.advantages: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        """Append one step of data to the buffer."""
        self._obs.append(obs.copy())
        self._actions.append(action)
        self._rewards.append(reward)
        self._dones.append(done)
        self._values.append(value)
        self._log_probs.append(log_prob)

    # ------------------------------------------------------------------
    # Advantage computation
    # ------------------------------------------------------------------

    def compute_advantages(self, last_value: float, normalize: bool = True) -> None:
        """Compute GAE advantages and discounted returns.

        Args:
            last_value: Bootstrap value for the state after the final step in
                        the rollout.  Pass 0.0 if the rollout ended on a
                        terminal state.
            normalize:  Whether to normalize advantages in place. Pass False
                        when this buffer is one of several per-env buffers
                        that will be combined with merge_buffers(), which
                        normalizes globally — normalizing each small per-env
                        buffer separately skews the update.
        """
        n = len(self._rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        gae = 0.0
        next_value = last_value

        for t in reversed(range(n)):
            not_done = 1.0 - float(self._dones[t])
            delta = (
                self._rewards[t]
                + self.gamma * next_value * not_done
                - self._values[t]
            )
            gae = delta + self.gamma * self.lam * not_done * gae
            advantages[t] = gae
            returns[t] = advantages[t] + self._values[t]
            next_value = self._values[t]

        if normalize:
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        self.advantages = torch.tensor(advantages, dtype=torch.float32)
        self.returns = torch.tensor(returns, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Tensor retrieval
    # ------------------------------------------------------------------

    def get_tensors(self) -> dict:
        """Return a dict of torch tensors for all stored transitions.

        Returns:
            dict with keys:
                "obs"        — float32 tensor, shape (T, obs_size)
                "actions"    — long tensor, shape (T,)
                "returns"    — float32 tensor, shape (T,)
                "advantages" — float32 tensor, shape (T,)
                "log_probs"  — float32 tensor, shape (T,)
        """
        if self.returns is None or self.advantages is None:
            raise RuntimeError(
                "compute_advantages() must be called before get_tensors()"
            )

        obs_array = np.stack(self._obs, axis=0)
        return {
            "obs": torch.tensor(obs_array, dtype=torch.float32),
            "actions": torch.tensor(self._actions, dtype=torch.long),
            "returns": self.returns,
            "advantages": self.advantages,
            "log_probs": torch.tensor(self._log_probs, dtype=torch.float32),
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Reset all internal lists and delete stored tensors."""
        self._obs = []
        self._actions = []
        self._rewards = []
        self._dones = []
        self._values = []
        self._log_probs = []

        if hasattr(self, "returns"):
            del self.returns
        if hasattr(self, "advantages"):
            del self.advantages
        self.returns = None
        self.advantages = None

    def __len__(self) -> int:
        return len(self._rewards)


def merge_buffers(buffers: list) -> dict:
    """Combine per-env TrajectoryBuffers into one update batch.

    Each buffer must have had compute_advantages(..., normalize=False) called
    already (per-env, with that env's own bootstrap value — GAE never crosses
    env streams). Advantages are normalized here, globally over the combined
    batch. Empty buffers are skipped.

    Returns:
        The same tensor dict shape as TrajectoryBuffer.get_tensors().
    """
    buffers = [b for b in buffers if len(b) > 0]
    if not buffers:
        raise RuntimeError("merge_buffers() called with no non-empty buffers")

    parts = [b.get_tensors() for b in buffers]
    merged = {key: torch.cat([p[key] for p in parts], dim=0) for key in parts[0]}

    adv = merged["advantages"]
    merged["advantages"] = (adv - adv.mean()) / (adv.std() + 1e-8)
    return merged

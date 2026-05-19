"""
dqn_agent.py — Deep Q-Network agent for Pokemon Showdown Gen 1 battles.

Architecture:
  - Two-hidden-layer MLP (100 → 128 → 128 → 9)
  - Policy network updated every step; target network updated every
    target_update_freq steps via hard copy
  - ε-greedy action selection; always respects valid_mask
"""

import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from replay_buffer import ReplayBuffer


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class QNetwork(nn.Module):
    """Fully-connected Q-value network: obs_size → 128 → 128 → n_actions."""

    def __init__(self, obs_size: int = 100, n_actions: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """DQN agent with experience replay and a periodically-synced target network.

    Args:
        obs_size:           Dimension of the observation vector (default 100).
        n_actions:          Number of discrete actions (default 9).
        lr:                 Adam learning rate (default 1e-3).
        gamma:              Discount factor (default 0.99).
        epsilon:            Initial exploration probability (default 1.0).
        epsilon_min:        Minimum exploration probability (default 0.05).
        epsilon_decay:      Multiplicative decay applied each episode (default 0.9995).
        target_update_freq: Steps between hard target-network syncs (default 1000).
        batch_size:         Mini-batch size for learning updates (default 64).
    """

    def __init__(
        self,
        obs_size: int = 100,
        n_actions: int = 9,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
        target_update_freq: int = 1000,
        batch_size: int = 64,
    ):
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size

        self.device = _pick_device()
        self.policy_net = QNetwork(obs_size, n_actions).to(self.device)
        self.target_net = QNetwork(obs_size, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Target network is never updated by the optimizer
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self._loss_fn = nn.MSELoss()

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray, valid_mask: list) -> int:
        """Select an action using ε-greedy policy, respecting valid_mask.

        Args:
            obs:        Observation array, shape (100,), float32.
            valid_mask: List of bool, length 9; True = legal action.

        Returns:
            Integer action index.
        """
        valid_indices = [i for i, v in enumerate(valid_mask) if v]
        if not valid_indices:
            return 0  # Fallback — should not occur in a healthy environment

        if random.random() < self.epsilon:
            return random.choice(valid_indices)

        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(obs_tensor).squeeze(0)  # shape (9,)

        # Mask invalid actions to -inf so argmax always picks a legal action
        for i in range(self.n_actions):
            if not valid_mask[i]:
                q_values[i] = float("-inf")

        return int(q_values.argmax().item())

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn(self, replay_buffer: ReplayBuffer) -> float | None:
        """Sample a mini-batch and apply one gradient step.

        Args:
            replay_buffer: Buffer to sample from.

        Returns:
            Scalar loss (float), or None if the buffer is too small.
        """
        if len(replay_buffer) < self.batch_size:
            return None

        obs_b, act_b, rew_b, next_obs_b, done_b = replay_buffer.sample(self.batch_size)
        obs_b = obs_b.to(self.device)
        act_b = act_b.to(self.device)
        rew_b = rew_b.to(self.device)
        next_obs_b = next_obs_b.to(self.device)
        done_b = done_b.to(self.device)

        # Target Q-values (no gradient through target_net)
        with torch.no_grad():
            max_next_q = self.target_net(next_obs_b).max(dim=1).values
            target_q = rew_b + self.gamma * max_next_q * (1.0 - done_b)

        # Predicted Q-values for the actions actually taken
        predicted_q = (
            self.policy_net(obs_b).gather(1, act_b.unsqueeze(1)).squeeze(1)
        )

        loss = self._loss_fn(predicted_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    # ------------------------------------------------------------------
    # Target network & exploration
    # ------------------------------------------------------------------

    def update_target(self) -> None:
        """Hard-copy policy network weights into the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self) -> None:
        """Apply one multiplicative decay step to epsilon."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save policy network weights and hyperparameters.

        Args:
            path: Destination file path (e.g. 'dqn_step_10000.pt').
        """
        torch.save(
            {
                "state_dict": self.policy_net.state_dict(),
                "hparams": {
                    "obs_size": self.obs_size,
                    "n_actions": self.n_actions,
                    "lr": self.lr,
                    "gamma": self.gamma,
                    "epsilon": self.epsilon,
                    "epsilon_min": self.epsilon_min,
                    "epsilon_decay": self.epsilon_decay,
                    "target_update_freq": self.target_update_freq,
                    "batch_size": self.batch_size,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "DQNAgent":
        """Reconstruct a DQNAgent from a checkpoint file.

        Args:
            path: Path to a file saved by :meth:`save`.

        Returns:
            DQNAgent with loaded weights and hyperparameters.
        """
        checkpoint = torch.load(path, map_location="cpu")
        hp = checkpoint["hparams"]
        agent = cls(
            obs_size=hp["obs_size"],
            n_actions=hp["n_actions"],
            lr=hp["lr"],
            gamma=hp["gamma"],
            epsilon=hp["epsilon"],
            epsilon_min=hp["epsilon_min"],
            epsilon_decay=hp["epsilon_decay"],
            target_update_freq=hp["target_update_freq"],
            batch_size=hp["batch_size"],
        )
        agent.policy_net.load_state_dict(checkpoint["state_dict"])
        agent.update_target()
        return agent

"""
replay_buffer.py — Experience replay buffer for DQN training.

Stores (obs, action, reward, next_obs, done) transitions in a circular
deque and provides random mini-batch sampling as torch tensors.
"""

import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-capacity circular experience replay buffer.

    Args:
        capacity: Maximum number of transitions to store (oldest entries
                  are evicted automatically once capacity is reached).
    """

    def __init__(self, capacity: int = 10000):
        self._buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single transition.

        All array-like inputs are converted to numpy arrays before storage
        so the buffer is always CPU-resident and framework-agnostic.

        Args:
            obs:      Current observation, shape (100,).
            action:   Integer action index.
            reward:   Scalar reward.
            next_obs: Next observation, shape (100,).
            done:     True if the episode ended after this transition.
        """
        self._buffer.append(
            (
                np.asarray(obs, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_obs, dtype=np.float32),
                float(done),
            )
        )

    def sample(self, batch_size: int) -> tuple:
        """Draw a random mini-batch without replacement.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of five CPU torch tensors:
              obs_batch      — float32,  shape (batch_size, obs_size)
              action_batch   — long,     shape (batch_size,)
              reward_batch   — float32,  shape (batch_size,)
              next_obs_batch — float32,  shape (batch_size, obs_size)
              done_batch     — float32,  shape (batch_size,)
        """
        transitions = random.sample(self._buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*transitions)

        obs_batch = torch.tensor(np.stack(obs), dtype=torch.float32)
        action_batch = torch.tensor(actions, dtype=torch.long)
        reward_batch = torch.tensor(rewards, dtype=torch.float32)
        next_obs_batch = torch.tensor(np.stack(next_obs), dtype=torch.float32)
        done_batch = torch.tensor(dones, dtype=torch.float32)

        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

    def __len__(self) -> int:
        return len(self._buffer)

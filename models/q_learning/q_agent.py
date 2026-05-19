"""
q_agent.py — Tabular Q-learning agent for Pokemon Showdown Gen 1 battles.

State discretization maps the 100-feature observation vector to a small
hashable tuple so a lookup table (defaultdict) can serve as the Q-function.
"""

import pickle
import random
from collections import defaultdict

import numpy as np


class QAgent:
    """Tabular Q-learning agent with epsilon-greedy exploration.

    Q-table keys are discretized state tuples derived from the 100-float
    observation vector produced by feature-extractor.ts.

    Feature layout reference:
        [0]     own active HP (normalized 0-1)
        [4]     own active type1 (normalized, type_index/20)
        [15:19] move availability flags (boolean-like 0/1)
        [55:60] switch mask (boolean-like, one per bench slot)
        [60]    opponent active HP (normalized 0-1)
    """

    def __init__(
        self,
        n_actions: int = 9,
        lr: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
    ):
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: state tuple -> numpy array of shape (n_actions,)
        self.q_table: dict = defaultdict(lambda: np.zeros(self.n_actions))

    # ------------------------------------------------------------------
    # State discretization
    # ------------------------------------------------------------------

    def _discretize(self, obs: np.ndarray) -> tuple:
        """Convert a raw observation vector into a hashable state tuple.

        Discretization scheme:
          - own_hp_bucket  : int in [0, 4]   (obs[0] * 4 bucketed)
          - type_index     : int in [0, 20]  (obs[4] * 20 bucketed)
          - switch_mask    : 5-tuple of bool (obs[55:60] > 0.5)
          - opp_hp_bucket  : int in [0, 4]   (obs[60] * 4 bucketed)
          - n_valid_moves  : int in [0, 4]   (count of obs[15:19] > 0.5)
        """
        own_hp_bucket = int(obs[0] * 4)
        type_index = int(obs[4] * 20)
        switch_mask = tuple(obs[55:60] > 0.5)
        opp_hp_bucket = int(obs[60] * 4)
        n_valid_moves = int(sum(obs[15:19] > 0.5))
        return (own_hp_bucket, type_index, switch_mask, opp_hp_bucket, n_valid_moves)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray, valid_mask: list) -> int:
        """Select an action using epsilon-greedy policy.

        Always respects valid_mask — never selects an invalid action.

        Args:
            obs:        observation array, shape (100,), float32
            valid_mask: list of bool, length 9; True means action is legal

        Returns:
            action index (int)
        """
        valid_indices = [i for i, v in enumerate(valid_mask) if v]
        if not valid_indices:
            # Fallback — should not happen in a healthy env
            return 0

        if random.random() < self.epsilon:
            return random.choice(valid_indices)

        state = self._discretize(obs)
        q_values = self.q_table[state].copy()

        # Mask invalid actions to negative infinity
        for i in range(self.n_actions):
            if not valid_mask[i]:
                q_values[i] = -np.inf

        return int(np.argmax(q_values))

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Apply a standard Q-learning (TD(0)) update.

        Q[s][a] += lr * (reward + gamma * max(Q[s']) * (1 - done) - Q[s][a])

        Args:
            obs:      current observation, shape (100,)
            action:   action taken
            reward:   scalar reward received
            next_obs: observation after action, shape (100,)
            done:     True if episode ended
        """
        state = self._discretize(obs)
        next_state = self._discretize(next_obs)

        current_q = self.q_table[state][action]
        max_next_q = float(np.max(self.q_table[next_state]))
        td_target = reward + self.gamma * max_next_q * (1.0 - float(done))
        self.q_table[state][action] += self.lr * (td_target - current_q)

    def decay_epsilon(self) -> None:
        """Decay exploration rate by one step."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Pickle the Q-table and hyperparameters to disk.

        Args:
            path: file path to write (e.g. 'qtable.pkl')
        """
        payload = {
            "q_table": dict(self.q_table),
            "hyperparams": {
                "n_actions": self.n_actions,
                "lr": self.lr,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "QAgent":
        """Load a QAgent from a pickle file.

        Args:
            path: file path to read

        Returns:
            reconstructed QAgent with loaded Q-table and hyperparameters
        """
        with open(path, "rb") as f:
            payload = pickle.load(f)

        hp = payload["hyperparams"]
        agent = cls(
            n_actions=hp["n_actions"],
            lr=hp["lr"],
            gamma=hp["gamma"],
            epsilon=hp["epsilon"],
            epsilon_min=hp["epsilon_min"],
            epsilon_decay=hp["epsilon_decay"],
        )
        # Restore Q-table as defaultdict
        loaded = payload["q_table"]
        agent.q_table = defaultdict(lambda: np.zeros(agent.n_actions), loaded)
        return agent

"""
vec_gym_client.py — Synchronous vectorized wrapper around N GymClient instances.

Spawns N independent gym_bridge.js subprocesses and exposes a batched
Gym-like interface. The training loop is responsible for resetting individual
done envs by calling env.envs[i].reset() directly — VecGymClient.step() does
NOT auto-reset.

API:
    env = VecGymClient(n_envs=4, flat_mode=False, opponent='random')
    obs_batch, masks = env.reset()      # obs_batch: (N, 12, 69) or (N, 828)
    obs_batch, rewards, dones, infos, masks = env.step(actions)
    env.close()
    env.envs[i].reset()                 # reset individual env i
"""

import numpy as np

from gym_client import GymClient


class VecGymClient:
    """Synchronous vectorized wrapper around N independent GymClient instances.

    Each env owns its own gym_bridge.js subprocess, so all envs are fully
    independent — no shared state, no locking required.
    """

    def __init__(self, n_envs: int = 4, flat_mode: bool = False, opponent: str = "random"):
        self.n_envs = n_envs
        self._flat_mode = flat_mode
        self.envs = [GymClient(flat_mode=flat_mode, opponent=opponent) for _ in range(n_envs)]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> tuple:
        """Reset all environments and return batched observations.

        Returns:
            obs_batch:  np.ndarray shape (N, 12, 69) float32 (or (N, 828) in flat mode)
            masks_batch: list of N valid-action masks (each a list of bool, length 9)
        """
        results = [env.reset() for env in self.envs]
        obs_batch = np.stack([r[0] for r in results], axis=0)
        masks_batch = [r[1] for r in results]
        return obs_batch, masks_batch

    def step(self, actions: list) -> tuple:
        """Step each environment with the corresponding action.

        Does NOT auto-reset done environments. The caller must call
        env.envs[i].reset() for any env where dones[i] is True.

        Args:
            actions: list of int, length N — one action per environment

        Returns:
            obs_batch:   np.ndarray shape (N, 12, 69) float32 (or (N, 828) flat)
            rewards:     list of float, length N
            dones:       list of bool, length N
            infos:       list of dict, length N
            masks_batch: list of valid-action masks, length N
        """
        results = [self.envs[i].step(actions[i]) for i in range(self.n_envs)]
        obs_batch = np.stack([r[0] for r in results], axis=0)
        rewards = [r[1] for r in results]
        dones = [r[2] for r in results]
        infos = [r[3] for r in results]
        masks_batch = [r[4] for r in results]
        return obs_batch, rewards, dones, infos, masks_batch

    def close(self):
        """Close all environments and their bridge subprocesses."""
        for env in self.envs:
            env.close()


# ---------------------------------------------------------------------------
# Smoke test — run with: python models/vec_gym_client.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = VecGymClient(n_envs=2)

    obs_batch, masks_batch = env.reset()
    print(f"reset obs_batch.shape: {obs_batch.shape}")
    print(f"reset len(masks_batch): {len(masks_batch)}")

    obs_batch, rewards, dones, infos, masks_batch = env.step([0, 0])
    print(f"step obs_batch.shape: {obs_batch.shape}")
    print(f"step rewards: {rewards}")
    print(f"step dones: {dones}")
    print(f"step len(masks_batch): {len(masks_batch)}")

    env.close()
    print("VecGymClient smoke test: OK")

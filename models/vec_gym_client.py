"""
vec_gym_client.py — Vectorized wrapper over N GymClient bridge subprocesses.

Each env is its own `node gym_bridge.js` process (its own event loop, its own
CPU core), so N battles simulate genuinely in parallel. step() pipelines the
round trip: all N step commands are written first, then all N responses are
read — Python only ever waits for the slowest env, not the sum of all of them.

Auto-reset: when an env's battle ends (done=True), VecGymClient immediately
resets it and returns the fresh episode's first observation in that slot. The
terminal step's reward/done/info are still reported, so the trainer sees the
episode boundary; it just never has to call reset itself. reset_all() is only
needed once, at startup.

Per-env errors (the bridge's {"error": ...} responses, which GymClient raises
as RuntimeError) don't kill the batch: the failed env is reset in place and
its info dict carries {"error": "<msg>"} so the caller can skip that slot's
transition — mirroring the reset-and-continue handling the serial trainers
already did.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gym_client import GymClient  # noqa: E402

N_ACTIONS = 9


class VecGymClient:
    """N parallel PokemonGymEnv bridges with a batched Gym-like interface."""

    def __init__(
        self,
        n_envs: int,
        structured: bool = True,
        opponent: str = "random",
        selfplay: bool = False,
    ):
        if n_envs < 1:
            raise ValueError(f"n_envs must be >= 1, got {n_envs}")
        self.n_envs = n_envs
        self._structured = structured
        self._selfplay = selfplay
        self._clients = [
            GymClient(structured=structured, opponent=opponent, selfplay=selfplay)
            for _ in range(n_envs)
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_obs_mask(self, response: dict) -> tuple:
        client = self._clients[0]
        obs = client._reshape(np.array(response["obs"], dtype=np.float32))
        mask = np.array(response["mask"], dtype=bool)
        return obs, mask

    def _reset_indices(self, indices: list) -> dict:
        """Pipelined reset of the given env indices; returns {i: response}."""
        for i in indices:
            self._clients[i]._write({"cmd": "reset"})
        return {i: self._clients[i]._read() for i in indices}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset_all(self) -> tuple:
        """Reset every env. Returns (obs (N, ...), masks (N, 9) bool array)."""
        responses = self._reset_indices(list(range(self.n_envs)))
        obs_list, mask_list = [], []
        for i in range(self.n_envs):
            obs, mask = self._parse_obs_mask(responses[i])
            obs_list.append(obs)
            mask_list.append(mask)
        return np.stack(obs_list), np.stack(mask_list)

    def step(self, actions) -> tuple:
        """Step every env with its action; auto-resets finished/errored envs.

        Args:
            actions: sequence of N ints.

        Returns:
            (obs, rewards, dones, infos, masks)
              obs     — (N, 12, 65) or (N, 100) float32; fresh-episode obs in
                        slots where dones is True (auto-reset)
              rewards — (N,) float32; 0.0 in errored slots
              dones   — (N,) bool; False in errored slots
              infos   — list of N dicts; the terminal step's info where done,
                        {"error": "<msg>"} where the env errored and was reset
              masks   — (N, 9) bool
        """
        if len(actions) != self.n_envs:
            raise ValueError(f"expected {self.n_envs} actions, got {len(actions)}")

        for client, action in zip(self._clients, actions):
            client._write({"cmd": "step", "action": int(action)})

        obs_list = [None] * self.n_envs
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=bool)
        infos = [None] * self.n_envs
        mask_list = [None] * self.n_envs
        needs_reset = []

        for i, client in enumerate(self._clients):
            try:
                response = client._read()
            except RuntimeError as e:
                infos[i] = {"error": str(e)}
                needs_reset.append(i)
                continue
            obs, mask = self._parse_obs_mask(response)
            obs_list[i] = obs
            mask_list[i] = mask
            rewards[i] = float(response["reward"])
            dones[i] = bool(response["done"])
            infos[i] = response["info"]
            if dones[i]:
                needs_reset.append(i)

        # Auto-reset finished + errored envs in one pipelined round trip; the
        # returned obs/mask for those slots come from the fresh episode.
        if needs_reset:
            responses = self._reset_indices(needs_reset)
            for i in needs_reset:
                obs, mask = self._parse_obs_mask(responses[i])
                obs_list[i] = obs
                mask_list[i] = mask

        return np.stack(obs_list), rewards, dones, infos, np.stack(mask_list)

    # ------------------------------------------------------------------
    # Dual-seat (self-play) interface — requires selfplay=True
    # ------------------------------------------------------------------

    def _stack_seats(self, seats: list) -> dict:
        """[{obs, mask, needs} × N] -> {"obs": (N,...), "mask": (N,9), "needs": (N,)}."""
        return {
            "obs": np.stack([s["obs"] for s in seats]),
            "mask": np.stack([s["mask"] for s in seats]),
            "needs": np.array([s["needs"] for s in seats], dtype=bool),
        }

    def reset_all_dual(self) -> tuple:
        """Reset every env in dual-seat mode. Returns (p1, p2) stacked dicts."""
        for client in self._clients:
            client._write({"cmd": "reset"})
        responses = [client._read() for client in self._clients]
        p1 = [self._clients[i]._parse_seat(responses[i]["p1"]) for i in range(self.n_envs)]
        p2 = [self._clients[i]._parse_seat(responses[i]["p2"]) for i in range(self.n_envs)]
        return self._stack_seats(p1), self._stack_seats(p2)

    def step_dual(self, actions, opp_actions) -> tuple:
        """Step every env with per-seat actions (None where a seat doesn't act).

        Auto-resets finished/errored envs like step(). Returns
        (p1, p2, rewards, dones, infos) where p1/p2 are stacked seat dicts
        holding fresh-episode state in slots where dones is True, rewards are
        from p1's perspective, and errored slots carry infos[i]["error"].
        """
        if len(actions) != self.n_envs or len(opp_actions) != self.n_envs:
            raise ValueError(f"expected {self.n_envs} actions for each seat")

        for client, a1, a2 in zip(self._clients, actions, opp_actions):
            client._write({
                "cmd": "step",
                "action": None if a1 is None else int(a1),
                "opp_action": None if a2 is None else int(a2),
            })

        p1_seats = [None] * self.n_envs
        p2_seats = [None] * self.n_envs
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=bool)
        infos = [None] * self.n_envs
        needs_reset = []

        for i, client in enumerate(self._clients):
            try:
                response = client._read()
            except RuntimeError as e:
                infos[i] = {"error": str(e)}
                needs_reset.append(i)
                continue
            p1_seats[i] = client._parse_seat(response["p1"])
            p2_seats[i] = client._parse_seat(response["p2"])
            rewards[i] = float(response["reward"])
            dones[i] = bool(response["done"])
            infos[i] = response["info"]
            if dones[i]:
                needs_reset.append(i)

        if needs_reset:
            for i in needs_reset:
                self._clients[i]._write({"cmd": "reset"})
            for i in needs_reset:
                response = self._clients[i]._read()
                p1_seats[i] = self._clients[i]._parse_seat(response["p1"])
                p2_seats[i] = self._clients[i]._parse_seat(response["p2"])

        return self._stack_seats(p1_seats), self._stack_seats(p2_seats), rewards, dones, infos

    def close(self) -> None:
        for client in self._clients:
            client.close()


# ---------------------------------------------------------------------------
# Smoke test — run with: python models/vec_gym_client.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N = 4
    STEPS = 200

    vec = VecGymClient(N)
    obs, masks = vec.reset_all()
    assert obs.shape == (N, 12, 65), obs.shape
    assert masks.shape == (N, 9), masks.shape
    print(f"reset_all: obs {obs.shape}, masks {masks.shape}")

    rng = np.random.default_rng(0)
    episodes = 0
    for t in range(STEPS):
        # Pick a random valid action per env
        actions = [int(rng.choice(np.flatnonzero(masks[i]))) for i in range(N)]
        obs, rewards, dones, infos, masks = vec.step(actions)
        assert obs.shape == (N, 12, 65), obs.shape
        assert not np.isnan(obs).any(), "NaN in observations"
        for i in range(N):
            if "error" in (infos[i] or {}):
                print(f"  [env {i}] error at t={t}: {infos[i]['error']}")
            elif dones[i]:
                episodes += 1
                print(f"  [env {i}] episode done at t={t}, winner={infos[i].get('winner')}")

    vec.close()
    print(f"smoke OK: {STEPS} vec steps ({STEPS * N} env steps), {episodes} episodes completed")

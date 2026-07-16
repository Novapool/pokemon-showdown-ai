"""
gym_client.py — Python wrapper that spawns gym_bridge.js as a subprocess and
communicates via line-delimited JSON over stdin/stdout.

Protocol (one JSON object per line):
  reset        → {"cmd":"reset"}
  step         → {"cmd":"step","action":<int>}
  valid_actions→ {"cmd":"valid_actions"}
  close        → {"cmd":"close"}
  sim_clone / sim_step / sim_fork / sim_free / sim_free_all — forward-model
  simulation of the current battle for MCTS (M4); see the sim_* methods.

By default the bridge returns the M2 structured (12, 65) token observation
as a flat 780-element array; GymClient reshapes it back to (12, 65). Pass
structured=False to run the bridge with --flat and get the legacy 100-dim
vector instead (M1 MLP baseline regression checks), or obs_v2=True (M3.4)
for the (12, 77) schema-v2 observation (boost stages + volatile flags).
"""

import json
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

N_TOKENS = 12
TOKEN_DIM = 65
TOKEN_DIM_V2 = 77

# Set to True to print every send/receive for hang debugging
DEBUG = False


def with_opp_action(info: dict) -> dict:
    """Surface the M5 opponent-action label as `opp_action` (int, -1 = masked).

    The bridge forwards the gym's raw camelCase `info` object as-is (`oppAction`
    for single-seat steps and the p1-perspective dual-seat label; `oppActionP2`
    is the p2-perspective label, unused by the p1-only trainer). This adds the
    snake_case `opp_action` key callers actually consume, defaulting to -1 when
    the field is absent (e.g. sim_* responses, which predate M5 labeling).
    Returns a shallow copy — the original response dict is left untouched.
    """
    info = dict(info)
    info["opp_action"] = int(info.get("oppAction", -1))
    return info


def slice_structured_obs(obs_flat: np.ndarray, agent_obs_size: int, n_tokens: int = N_TOKENS) -> np.ndarray:
    """Adapt a flattened structured observation batch to an agent's obs size.

    Schema v2 tokens are v1 tokens with extra dims APPENDED, so a v1 view of a
    v2 observation is exactly the first 65 dims of each 77-dim token. This
    lets v1 checkpoints (the M2 baseline, the M3.3 self-play best) act inside
    a v2 env — self-play pool seeding and cross-schema head-to-head (M3.4).

    Args:
        obs_flat:       (N, n_tokens * d_env) float32 batch.
        agent_obs_size: the agent's expected flat obs size (n_tokens * d_agent).

    Returns:
        (N, agent_obs_size) — the input unchanged if sizes already match.
    """
    if obs_flat.shape[-1] == agent_obs_size:
        return obs_flat
    d_env = obs_flat.shape[-1] // n_tokens
    d_agent = agent_obs_size // n_tokens
    if n_tokens * d_env != obs_flat.shape[-1] or n_tokens * d_agent != agent_obs_size or d_agent > d_env:
        raise ValueError(
            f"cannot adapt obs of size {obs_flat.shape[-1]} to agent obs_size {agent_obs_size}"
        )
    n = obs_flat.shape[0]
    return obs_flat.reshape(n, n_tokens, d_env)[:, :, :d_agent].reshape(n, agent_obs_size)


class GymClient:
    """Spawns gym_bridge.js and exposes a Gym-like interface over stdio JSON."""

    def __init__(
        self,
        bridge_path: str = None,
        structured: bool = True,
        opponent: str = "random",
        selfplay: bool = False,
        obs_v2: bool = False,
    ):
        """opponent: 'random' (legacy default) or 'damagefirst' (M3.3 heuristic).
        selfplay=True runs the bridge in dual-seat mode (opponent seat driven
        externally) — use reset_dual()/step_dual() instead of reset()/step().
        obs_v2=True (M3.4) selects the (12, 77) schema-v2 observation.
        set_opponent() switches the opponent (and single/dual protocol) for
        subsequent resets — used by --opponent-mix training."""
        if bridge_path is None:
            bridge_path = str(Path(__file__).parent / "gym_bridge.js")
        if obs_v2 and not structured:
            raise ValueError("obs_v2=True requires structured=True")
        self._bridge_path = bridge_path
        self._structured = structured
        self._token_dim = TOKEN_DIM_V2 if obs_v2 else TOKEN_DIM
        self._selfplay = selfplay
        self._opponent = "self" if selfplay else opponent
        args = ["node", bridge_path]
        if not structured:
            args.append("--flat")
        if obs_v2:
            args.append("--obs-v2")
        if selfplay:
            args.append("--selfplay")
        elif opponent != "random":
            args.extend(["--opponent", opponent])
        self._proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Drain stderr in background so bridge errors are visible immediately
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True
        )
        self._stderr_thread.start()

    def _reshape(self, obs: np.ndarray) -> np.ndarray:
        return obs.reshape(N_TOKENS, self._token_dim) if self._structured else obs

    def set_opponent(self, opponent: str) -> None:
        """Switch the opponent used by subsequent resets (M3.4 --opponent-mix).

        'self' switches this client to the dual-seat protocol (use
        reset_dual()/step_dual()); anything else back to single-seat."""
        if opponent not in ("random", "damagefirst", "self"):
            raise ValueError(f"unknown opponent: {opponent!r}")
        self._opponent = opponent
        self._selfplay = opponent == "self"

    def _reset_cmd(self) -> dict:
        return {"cmd": "reset", "opponent": self._opponent}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _drain_stderr(self):
        """Read bridge stderr and forward to Python stderr."""
        for line in self._proc.stderr:
            sys.stderr.write("[bridge] " + line.decode(errors="replace"))
            sys.stderr.flush()

    def _write(self, cmd: dict) -> None:
        """Write one command line without waiting for the response.

        Split out from _send so VecGymClient can pipeline: write commands to
        all N bridge processes first, then read all N responses — the sims
        run concurrently while Python waits.
        """
        if DEBUG:
            print(f"  [send] {cmd}", flush=True)

        if self._proc.poll() is not None:
            raise RuntimeError(f"bridge process has exited (code {self._proc.returncode})")

        line = json.dumps(cmd) + "\n"
        self._proc.stdin.write(line.encode())
        self._proc.stdin.flush()

    def _read(self) -> dict:
        """Read and parse one response line; raises on bridge errors."""
        raw = self._proc.stdout.readline()

        if not raw:
            raise RuntimeError("bridge stdout closed unexpectedly (bridge may have crashed)")

        if DEBUG:
            print(f"  [recv] {raw.decode().strip()}", flush=True)

        response = json.loads(raw.decode())
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    def _send(self, cmd: dict) -> dict:
        """Send one command and return the parsed response."""
        self._write(cmd)
        return self._read()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> tuple:
        """Reset the environment and return (obs, valid_mask).

        Returns:
            obs:        np.ndarray shape (12, 65) structured, or (100,) if
                        constructed with structured=False. dtype float32
            valid_mask: list of bool, length 9
        """
        response = self._send(self._reset_cmd())
        obs = self._reshape(np.array(response["obs"], dtype=np.float32))
        mask = list(response["mask"])
        return obs, mask

    def step(self, action: int) -> tuple:
        """Take one step in the environment.

        Args:
            action: integer action index (0–8)

        Returns:
            (obs, reward, done, info, valid_mask)
              obs        — np.ndarray shape (12, 65) structured, or (100,)
                           if constructed with structured=False. float32
              reward     — float
              done       — bool
              info       — dict; includes `opp_action` (M5, int, -1 = masked)
              valid_mask — list of bool, length 9
        """
        response = self._send({"cmd": "step", "action": action})
        obs = self._reshape(np.array(response["obs"], dtype=np.float32))
        reward = float(response["reward"])
        done = bool(response["done"])
        info = with_opp_action(response["info"])
        mask = list(response["mask"])
        return obs, reward, done, info, mask

    # ------------------------------------------------------------------
    # Dual-seat (self-play) interface — requires selfplay=True
    # ------------------------------------------------------------------

    def _parse_seat(self, seat: dict) -> dict:
        return {
            "obs": self._reshape(np.array(seat["obs"], dtype=np.float32)),
            "mask": np.array(seat["mask"], dtype=bool),
            "needs": bool(seat["needs"]),
        }

    def reset_dual(self) -> tuple:
        """Reset in dual-seat mode. Returns (p1_state, p2_state) dicts, each
        {"obs": ndarray, "mask": (9,) bool ndarray, "needs": bool}."""
        response = self._send(self._reset_cmd())
        return self._parse_seat(response["p1"]), self._parse_seat(response["p2"])

    def step_dual(self, action, opp_action) -> tuple:
        """Step both seats. Pass an int for exactly the seats whose "needs"
        was True in the previous state, None otherwise.

        Returns (p1_state, p2_state, reward, done, info) — reward is from
        p1's perspective (self-play trains the p1 seat only). info's
        `opp_action` (M5, int, -1 = masked) is p1's label — the p2 seat's
        simultaneous choice this step, in p2's own 9-way action frame."""
        response = self._send({
            "cmd": "step",
            "action": None if action is None else int(action),
            "opp_action": None if opp_action is None else int(opp_action),
        })
        return (
            self._parse_seat(response["p1"]),
            self._parse_seat(response["p2"]),
            float(response["reward"]),
            bool(response["done"]),
            with_opp_action(response["info"]),
        )

    def valid_actions(self) -> list:
        """Return a mask of valid actions.

        Returns:
            list of bool, length 9
        """
        response = self._send({"cmd": "valid_actions"})
        return list(response["mask"])

    # ------------------------------------------------------------------
    # Simulation / forward-model interface (M4, MCTS)
    # ------------------------------------------------------------------

    def _parse_sim_state(self, response: dict) -> dict:
        state = {
            "p1": self._parse_seat(response["p1"]),
            "p2": self._parse_seat(response["p2"]),
            "done": bool(response["done"]),
            "info": response.get("info", {}),
        }
        if "sim" in response:
            state["sim"] = int(response["sim"])
        return state

    def sim_clone(self, determinize: bool = True, seed=None, perspective: str = "p1") -> dict:
        """Clone the CURRENT live battle into an independent simulator.

        determinize=True replaces the unrevealed Pokémon of the seat OPPOSITE
        `perspective` with sets sampled from the format's generator
        (reproducible via `seed`), so the searcher never sees hidden
        information. determinize=False keeps the true hidden team
        (omniscient search — diagnostic only).

        Returns {"sim": id, "p1": seat, "p2": seat, "done": bool, "info": {}}
        where each seat is {"obs": ndarray, "mask": (9,) bool, "needs": bool}.
        """
        cmd = {
            "cmd": "sim_clone",
            "determinize": bool(determinize),
            "perspective": perspective,
        }
        if seed is not None:
            cmd["seed"] = int(seed)
        return self._parse_sim_state(self._send(cmd))

    def sim_step(self, sim_id: int, action, opp_action) -> dict:
        """Step a simulator (dual-seat semantics: int for seats whose "needs"
        was True, None otherwise). Reward is p1-perspective, gym-identical.

        Returns {"p1": seat, "p2": seat, "reward": float, "done": bool, "info": {}}.
        """
        response = self._send({
            "cmd": "sim_step",
            "sim": int(sim_id),
            "action": None if action is None else int(action),
            "opp_action": None if opp_action is None else int(opp_action),
        })
        state = self._parse_sim_state(response)
        state["reward"] = float(response["reward"])
        return state

    def sim_fork(self, sim_id: int) -> dict:
        """Independent copy of a sim's current state (tree branching).
        Same return shape as sim_clone()."""
        return self._parse_sim_state(self._send({"cmd": "sim_fork", "sim": int(sim_id)}))

    def sim_free(self, sim_id: int) -> None:
        """Release one simulator."""
        self._send({"cmd": "sim_free", "sim": int(sim_id)})

    def sim_free_all(self) -> None:
        """Release every live simulator (call after each real decision)."""
        self._send({"cmd": "sim_free_all"})

    def close(self):
        """Send close command, wait for the bridge process to exit."""
        if self._proc.poll() is not None:
            return  # already dead
        try:
            self._send({"cmd": "close"})
        except Exception:
            pass  # bridge may have already exited
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()


# ---------------------------------------------------------------------------
# Smoke test — run with: python models/gym_client.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    client = GymClient()

    obs, mask = client.reset()
    print(f"reset obs shape: {obs.shape}")
    print(f"valid_actions mask: {mask}")

    obs, reward, done, info, mask = client.step(0)
    print(f"step(0) → obs.shape={obs.shape}, reward={reward}, done={done}, info={info}")

    client.close()
    print("close: OK")

    flat_client = GymClient(structured=False)
    obs, mask = flat_client.reset()
    print(f"[--flat] reset obs shape: {obs.shape}")
    flat_client.close()
    print("[--flat] close: OK")

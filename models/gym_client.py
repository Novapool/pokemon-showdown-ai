"""
gym_client.py — Python wrapper that spawns gym_bridge.js as a subprocess and
communicates via line-delimited JSON over stdin/stdout.

Protocol (one JSON object per line):
  reset        → {"cmd":"reset"}
  step         → {"cmd":"step","action":<int>}
  valid_actions→ {"cmd":"valid_actions"}
  close        → {"cmd":"close"}
"""

import json
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

# Set to True to print every send/receive for hang debugging
DEBUG = False


class GymClient:
    """Spawns gym_bridge.js and exposes a Gym-like interface over stdio JSON."""

    def __init__(self, bridge_path: str = None):
        if bridge_path is None:
            bridge_path = str(Path(__file__).parent / "gym_bridge.js")
        self._bridge_path = bridge_path
        self._proc = subprocess.Popen(
            ["node", bridge_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Drain stderr in background so bridge errors are visible immediately
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True
        )
        self._stderr_thread.start()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _drain_stderr(self):
        """Read bridge stderr and forward to Python stderr."""
        for line in self._proc.stderr:
            sys.stderr.write("[bridge] " + line.decode(errors="replace"))
            sys.stderr.flush()

    def _send(self, cmd: dict) -> dict:
        """Send one command and return the parsed response."""
        if DEBUG:
            print(f"  [send] {cmd}", flush=True)

        if self._proc.poll() is not None:
            raise RuntimeError(f"bridge process has exited (code {self._proc.returncode})")

        line = json.dumps(cmd) + "\n"
        self._proc.stdin.write(line.encode())
        self._proc.stdin.flush()

        raw = self._proc.stdout.readline()

        if not raw:
            raise RuntimeError("bridge stdout closed unexpectedly (bridge may have crashed)")

        if DEBUG:
            print(f"  [recv] {raw.decode().strip()}", flush=True)

        response = json.loads(raw.decode())
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> tuple:
        """Reset the environment and return (obs, valid_mask).

        Returns:
            obs:        np.ndarray shape (100,), dtype float32
            valid_mask: list of bool, length 9
        """
        response = self._send({"cmd": "reset"})
        obs = np.array(response["obs"], dtype=np.float32)
        mask = list(response["mask"])
        return obs, mask

    def step(self, action: int) -> tuple:
        """Take one step in the environment.

        Args:
            action: integer action index (0–8)

        Returns:
            (obs, reward, done, info, valid_mask)
              obs        — np.ndarray shape (100,) float32
              reward     — float
              done       — bool
              info       — dict
              valid_mask — list of bool, length 9
        """
        response = self._send({"cmd": "step", "action": action})
        obs = np.array(response["obs"], dtype=np.float32)
        reward = float(response["reward"])
        done = bool(response["done"])
        info = response["info"]
        mask = list(response["mask"])
        return obs, reward, done, info, mask

    def valid_actions(self) -> list:
        """Return a mask of valid actions.

        Returns:
            list of bool, length 9
        """
        response = self._send({"cmd": "valid_actions"})
        return list(response["mask"])

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

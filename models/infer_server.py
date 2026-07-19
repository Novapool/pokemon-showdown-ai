"""
infer_server.py — Minimal stdio inference server for the ladder bot (M6).

The reverse of gym_bridge.js: Node (tools/ladder-bot) speaks line-delimited
JSON over stdin/stdout to this process, which loads a PPO checkpoint once and
answers action queries. Keeps PyTorch out of Node and reuses PPOAgent's
checkpoint loading verbatim.

Protocol (one JSON object per line):
  in : {"cmd": "act", "obs": [<obs_size> floats], "mask": [9 bools]}
  out: {"action": <int 0-8>}
  in : {"cmd": "act_tracked", "obs": [...], "mask": [...],
        "formatid": "gen1randombattle", "seat": "p1"|"p2",
        "request": <move request JSON>, "trackers": <TrackerSnapshot>,
        "turn": <int>, "obs_mode": "structured-v3"|"structured-v2"|"structured"}
  out: {"action": <int 0-8>, "sims": <int>, "fallback": <str?>}
       (M6 P2 — only in --mcts mode: determinized UCT over a battle
        reconstructed from the client's request + tracker snapshot via the
        bridge's sim_from_tracked; falls back to the raw policy, with the
        reason reported, when search can't run)
  in : {"cmd": "ping"}         out: {"ok": true, "obs_size": <int>, "mcts": bool}
  in : {"cmd": "close"}        out: {"ok": true}  (then exits)

Usage: python models/infer_server.py --checkpoint <path> [--device cpu]
           [--mcts] [--sims 100] [--determinizations 1] [--c-puct 0.5]
Action selection matches evaluate.py: PPOAgent.act() (masked policy sampling)
for "act"; MCTSAgent with the tuned defaults for "act_tracked" under --mcts.

Obs schema is inferred from the checkpoint: obs_size comes straight from the
loaded PPOAgent's hparams (780 v1 / 924 v2 / 1032 v3, M7), and act/act_tracked
validate/pass the obs at whatever width the checkpoint expects — no per-schema
code path here. The ladder client picks the matching obs_mode string.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODELS_DIR))
sys.path.insert(0, str(MODELS_DIR / "ppo"))

from ppo_agent import PPOAgent  # noqa: E402


class TrackedSimClient:
    """Minimal sim-protocol client for MCTS over a remote (ladder) battle.

    Speaks the gym bridge's sim_* commands to a dedicated Node subprocess,
    substituting sim_clone() with sim_from_tracked over the current payload
    (the ladder client's request + tracker snapshot) — so MCTSAgent runs
    unchanged against a battle that only exists on the remote server.
    """

    def __init__(self):
        self.proc = subprocess.Popen(
            ["node", str(MODELS_DIR / "gym_bridge.js")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self.payload = None  # set before each act_tracked search

    def _send(self, cmd: dict) -> dict:
        self.proc.stdin.write(json.dumps(cmd) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("sim host exited")
        resp = json.loads(line)
        if "error" in resp:
            raise RuntimeError(f"sim host: {resp['error']}")
        return resp

    # MCTSAgent's client interface -----------------------------------------
    def sim_clone(self, determinize: bool = True, seed=None, perspective: str = "p1") -> dict:
        cmd = dict(self.payload, cmd="sim_from_tracked", determinize=bool(determinize))
        if seed is not None:
            cmd["seed"] = int(seed)
        return self._send(cmd)

    def sim_step(self, sim_id: int, action, opp_action) -> dict:
        return self._send({
            "cmd": "sim_step", "sim": int(sim_id),
            "action": None if action is None else int(action),
            "opp_action": None if opp_action is None else int(opp_action),
        })

    def sim_fork(self, sim_id: int) -> dict:
        return self._send({"cmd": "sim_fork", "sim": int(sim_id)})

    def sim_free(self, sim_id: int) -> None:
        self._send({"cmd": "sim_free", "sim": int(sim_id)})

    def sim_free_all(self) -> None:
        self._send({"cmd": "sim_free_all"})

    def close(self) -> None:
        try:
            self._send({"cmd": "close"})
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mcts", action="store_true",
                        help="enable act_tracked (determinized UCT on ladder battles)")
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--determinizations", type=int, default=1)
    parser.add_argument("--c-puct", type=float, default=0.5)
    args = parser.parse_args()

    agent = PPOAgent.load(args.checkpoint, device=args.device)
    obs_size = agent._hparams["obs_size"]

    sim_client = None
    mcts_cls = None
    if args.mcts:
        sys.path.insert(0, str(MODELS_DIR / "mcts"))
        from mcts_agent import MCTSAgent
        mcts_cls = MCTSAgent
        sim_client = TrackedSimClient()

    def act_tracked(msg: dict) -> dict:
        obs = np.asarray(msg["obs"], dtype=np.float32)
        mask = msg["mask"]
        if not args.mcts:
            action, _logp, _value = agent.act(obs, mask)
            return {"action": int(action), "sims": 0, "fallback": "no --mcts"}
        sim_client.payload = {
            "formatid": msg["formatid"],
            "seat": msg["seat"],
            "request": msg["request"],
            "trackers": msg["trackers"],
            "turn": msg.get("turn", 0),
            "obs_mode": msg.get("obs_mode"),
        }
        mcts = mcts_cls(
            agent, n_sims=args.sims, n_determinizations=args.determinizations,
            c_puct=args.c_puct, seat=msg["seat"],
        )
        try:
            action = mcts.act(sim_client, obs, mask)
            return {"action": int(action), "sims": mcts.last_search_sims}
        except Exception as err:
            # Search must never lose a ladder battle to an exception — fall
            # back to the raw policy and report why.
            print(f"act_tracked search failed: {err}", file=sys.stderr, flush=True)
            action, _logp, _value = agent.act(obs, mask)
            return {"action": int(action), "sims": 0, "fallback": str(err)}

    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            cmd = msg.get("cmd")
            if cmd == "act":
                obs = np.asarray(msg["obs"], dtype=np.float32)
                if obs.shape != (obs_size,):
                    raise ValueError(f"obs size {obs.shape} != expected ({obs_size},)")
                action, _logp, _value = agent.act(obs, msg["mask"])
                resp = {"action": int(action)}
            elif cmd == "act_tracked":
                resp = act_tracked(msg)
            elif cmd == "ping":
                resp = {"ok": True, "obs_size": obs_size, "mcts": bool(args.mcts)}
            elif cmd == "close":
                if sim_client:
                    sim_client.close()
                print(json.dumps({"ok": True}), file=out, flush=True)
                return
            else:
                resp = {"error": f"unknown cmd: {cmd}"}
        except Exception as err:  # surface, never crash mid-battle
            resp = {"error": str(err)}
        print(json.dumps(resp), file=out, flush=True)


if __name__ == "__main__":
    main()

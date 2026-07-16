"""
infer_server.py — Minimal stdio inference server for the ladder bot (M6).

The reverse of gym_bridge.js: Node (tools/ladder-bot) speaks line-delimited
JSON over stdin/stdout to this process, which loads a PPO checkpoint once and
answers action queries. Keeps PyTorch out of Node and reuses PPOAgent's
checkpoint loading verbatim.

Protocol (one JSON object per line):
  in : {"cmd": "act", "obs": [<obs_size> floats], "mask": [9 bools]}
  out: {"action": <int 0-8>}
  in : {"cmd": "ping"}         out: {"ok": true, "obs_size": <int>}
  in : {"cmd": "close"}        out: {"ok": true}  (then exits)

Usage: python models/infer_server.py --checkpoint <path> [--device cpu]
Action selection matches evaluate.py's single-opponent path: PPOAgent.act()
(masked policy sampling).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODELS_DIR))
sys.path.insert(0, str(MODELS_DIR / "ppo"))

from ppo_agent import PPOAgent  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    agent = PPOAgent.load(args.checkpoint, device=args.device)
    obs_size = agent._hparams["obs_size"]

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
            elif cmd == "ping":
                resp = {"ok": True, "obs_size": obs_size}
            elif cmd == "close":
                print(json.dumps({"ok": True}), file=out, flush=True)
                return
            else:
                resp = {"error": f"unknown cmd: {cmd}"}
        except Exception as err:  # surface, never crash mid-battle
            resp = {"error": str(err)}
        print(json.dumps(resp), file=out, flush=True)


if __name__ == "__main__":
    main()

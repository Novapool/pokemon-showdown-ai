#!/usr/bin/env python3
"""Phase 0 — non-greedy decision probe (experiment/hierarchical-action-policy).

Measures whether our agent plays greedily, *before* any architecture work.

Corpus-derived "the greedy answer was wrong" set: positions from held-out
validation shards where a rated human declined a >=90 BP damaging move and
instead chose a 0-BP status move or a switch. On that subset we ask two things:

  1. top-1 agreement  — does the agent match the human less often here than on
     the full set? (the accuracy gap)
  2. greediness rate  — when the human declined the big attack, how often does
     the agent take it anyway? (the direct degeneracy measurement)

Everything is read out of the observation itself, so this needs no schema
change, no damage calculator and no new training. See
docs/experimental/MILESTONES-EXPERIMENTAL.md Phase 0 for the pre-registered
gate and kill criterion.

Usage:
    .venv/bin/python scripts/phase0_nongreedy_probe.py
    .venv/bin/python scripts/phase0_nongreedy_probe.py --big-bp 90 --limit 50000
"""
from __future__ import annotations

import argparse
import base64
import csv
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.ppo.ppo_agent import PPOAgent  # noqa: E402

# --- v3 observation layout (sim/tools/feature-extractor.ts) ------------------
TOKEN_DIM = 86
N_TOKENS = 12
T_HP = 0
T_FAINTED_FLAG = 40
T_MOVES = 41
T_MOVE_DIM = 6
BP_SCALE = 250.0          # obs stores min(1, basePower/250)
CAT_STATUS = 2            # categoryToIndex: Physical 0, Special 1, Status 2
N_MOVE_SLOTS = 4
N_ACTIONS = 9             # 4 moves + 5 switches; switch a -> own-bench token a-3


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — correct near 0/1 where normal approx fails."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def diff_ci(k1: int, n1: int, k2: int, n2: int, z: float = 1.96) -> tuple[float, float, float]:
    """Difference of two independent proportions with a normal-approx CI."""
    p1, p2 = k1 / n1, k2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    d = p1 - p2
    return d, d - z * se, d + z * se


def cluster_bootstrap_gap(correct: np.ndarray, sub: np.ndarray, battles: np.ndarray,
                          n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile CI for the full-vs-subset accuracy gap, resampling *battles*.

    Decisions within one battle are highly correlated, so treating them as
    independent (Wilson/normal) understates the interval. We resample whole
    battles with replacement instead.
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(battles, return_inverse=True)
    idx_by_battle = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    gaps = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by_battle[i] for i in pick])
        c, s = correct[idx], sub[idx]
        if s.sum() == 0 or len(c) == 0:
            continue
        gaps.append(c.mean() - c[s].mean())
    if not gaps:
        return (float("nan"), float("nan"))
    return (float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5)))


def parse_moves(obs: np.ndarray) -> list[dict]:
    """Decode the own-active move slots from token 0.

    An absent slot is all-zero, which is indistinguishable from a 0-BP
    *Physical* move on the basePower/category dims alone. The accuracy dim is
    the presence test: the extractor writes accuracy >= 0.01 for every real
    move (1.0 when never-miss) and never touches an unused slot.
    """
    out = []
    for i in range(N_MOVE_SLOTS):
        b = T_MOVES + i * T_MOVE_DIM
        acc = float(obs[b + 1])
        if acc <= 0.0:
            continue  # empty slot, not a move
        out.append({
            "slot": i,
            "bp": float(obs[b + 0]) * BP_SCALE,
            "acc": acc,
            "pp": float(obs[b + 2]),
            "cat": int(round(float(obs[b + 4]) * 2)),
            "disabled": float(obs[b + 5]) > 0.5,
        })
    return out


def legal_mask(obs: np.ndarray) -> np.ndarray:
    """Legal-action mask, derived from the observation.

    Records store no mask, and scoring an unmasked argmax would let the model
    "choose" impossible actions and deflate accuracy artificially.
    """
    mask = np.zeros(N_ACTIONS, dtype=bool)
    for m in parse_moves(obs):
        if not m["disabled"] and m["pp"] > 0.0:
            mask[m["slot"]] = True
    for a in range(4, N_ACTIONS):
        tok = (a - 3) * TOKEN_DIM          # own bench occupies tokens 1..5
        if float(obs[tok + T_HP]) > 0.0 and float(obs[tok + T_FAINTED_FLAG]) < 0.5:
            mask[a] = True
    return mask


def keep(rec: dict, fmt: str, min_rating: int, tournaments: bool) -> bool:
    """Mirror ReplayShardDataset._keep so the probe scores the same population
    the BC checkpoint was validated on."""
    rating = rec.get("r")
    if rating is not None:
        return rating >= min_rating
    return tournaments and not rec["b"].startswith(fmt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", default="data/replay_trajs/v3")
    ap.add_argument("--formats", default="gen1randombattle,gen1ou")
    ap.add_argument("--val-shards", type=int, default=1,
                    help="held-out shards, matching bc_pretrain_mlp.py's split")
    ap.add_argument("--min-rating", type=int, default=1300)
    ap.add_argument("--no-tournaments", action="store_true")
    ap.add_argument("--big-bp", type=float, default=90.0,
                    help="threshold defining a 'big' damaging move")
    ap.add_argument("--limit", type=int, default=0, help="0 = no cap")
    ap.add_argument("--checkpoints", default=(
        "bc=models/checkpoints/bc_mlp_gen1_v3_h512.pt,"
        "ppo=models/ppo/checkpoints/v3/ppo_step_5000002_final.pt"))
    ap.add_argument("--out", default="docs/experimental/phase0_results.csv")
    args = ap.parse_args()

    agents = {}
    for spec in args.checkpoints.split(","):
        name, path = spec.split("=", 1)
        if not Path(path).exists():
            print(f"[skip] {name}: {path} not found")
            continue
        agents[name] = PPOAgent.load(path, device="cpu")
        agents[name].eval()
        print(f"[load] {name}: {path} (H={agents[name]._hparams['hidden_size']})")
    if not agents:
        sys.exit("no checkpoints loaded")

    rows = []
    for fmt in args.formats.split(","):
        shards = sorted((Path(args.traj_dir) / fmt).glob("shard-*.jsonl.gz"))
        val = shards[:args.val_shards]
        if not val:
            print(f"[skip] {fmt}: no shards")
            continue
        print(f"\n=== {fmt} — val shards: {[p.name for p in val]} ===")

        obs_all, act_all, sub_all, big_all, bat_all = [], [], [], [], []
        n_seen = n_kept = 0
        for path in val:
            with gzip.open(path, "rt") as f:
                for line in f:
                    rec = json.loads(line)
                    n_seen += 1
                    if not keep(rec, fmt, args.min_rating, not args.no_tournaments):
                        continue
                    obs = np.frombuffer(base64.b64decode(rec["o"]), dtype=np.float32)
                    if obs.shape[0] != N_TOKENS * TOKEN_DIM:
                        continue
                    a = rec["a"]
                    moves = parse_moves(obs)
                    avail = [m for m in moves if not m["disabled"] and m["pp"] > 0.0]
                    big = [m for m in avail if m["bp"] >= args.big_bp]
                    if not big:
                        declined = False
                        best_big = -1
                    else:
                        best_big = max(big, key=lambda m: m["bp"])["slot"]
                        if a >= 4:
                            declined = True                      # switched away
                        else:
                            chosen = next((m for m in moves if m["slot"] == a), None)
                            declined = (chosen is not None
                                        and chosen["bp"] == 0.0
                                        and chosen["cat"] == CAT_STATUS)
                    obs_all.append(obs)
                    act_all.append(a)
                    sub_all.append(declined)
                    big_all.append(best_big)
                    bat_all.append(rec["b"])
                    n_kept += 1
                    if args.limit and n_kept >= args.limit:
                        break
            if args.limit and n_kept >= args.limit:
                break

        if not obs_all:
            print("  no records kept")
            continue
        X = np.stack(obs_all)
        y = np.array(act_all)
        sub = np.array(sub_all)
        bigs = np.array(big_all)
        battles = np.array(bat_all)
        masks = np.stack([legal_mask(o) for o in obs_all])
        print(f"  records: seen={n_seen} kept={n_kept} non-greedy subset={int(sub.sum())} "
              f"({100 * sub.mean():.1f}%)")

        for name, agent in agents.items():
            preds = []
            with torch.no_grad():
                for i in range(0, len(X), 4096):
                    xb = torch.from_numpy(X[i:i + 4096]).to(agent.device)
                    logits = agent.policy_head(agent.trunk(xb)).cpu().numpy()
                    mb = masks[i:i + 4096]
                    logits[~mb] = -np.inf
                    preds.append(logits.argmax(axis=1))
            pred = np.concatenate(preds)

            correct = pred == y
            n_full, k_full = len(y), int(correct.sum())
            n_sub, k_sub = int(sub.sum()), int(correct[sub].sum())
            acc_full, acc_sub = k_full / n_full, (k_sub / n_sub if n_sub else float("nan"))
            lo_f, hi_f = wilson(k_full, n_full)
            lo_s, hi_s = wilson(k_sub, n_sub)

            # Greediness: on the subset, how often does the agent take the very
            # high-BP move the human passed up?
            greedy_hits = int((pred[sub] == bigs[sub]).sum())
            g_rate = greedy_hits / n_sub if n_sub else float("nan")
            lo_g, hi_g = wilson(greedy_hits, n_sub)

            d, d_lo, d_hi = diff_ci(k_full, n_full, k_sub, n_sub) if n_sub else (float("nan"),) * 3
            cb_lo, cb_hi = cluster_bootstrap_gap(correct, sub, battles) if n_sub else (float("nan"),) * 2

            print(f"  [{name}] full  top-1 {100*acc_full:5.2f}%  n={n_full}  "
                  f"CI [{100*lo_f:.2f}, {100*hi_f:.2f}]")
            print(f"  [{name}] sub   top-1 {100*acc_sub:5.2f}%  n={n_sub}  "
                  f"CI [{100*lo_s:.2f}, {100*hi_s:.2f}]")
            print(f"  [{name}] gap   {100*d:+.2f}pp  CI [{100*d_lo:+.2f}, {100*d_hi:+.2f}]"
                  f"{'  <- excludes 0' if d_lo > 0 else ''}")
            print(f"  [{name}] gap   {100*d:+.2f}pp  battle-clustered CI "
                  f"[{100*cb_lo:+.2f}, {100*cb_hi:+.2f}]"
                  f"{'  <- excludes 0' if cb_lo > 0 else ''}")
            print(f"  [{name}] greediness {100*g_rate:5.2f}%  "
                  f"CI [{100*lo_g:.2f}, {100*hi_g:.2f}]")

            rows.append({
                "format": fmt, "checkpoint": name,
                "n_full": n_full, "acc_full": round(acc_full, 5),
                "acc_full_lo": round(lo_f, 5), "acc_full_hi": round(hi_f, 5),
                "n_sub": n_sub, "acc_sub": round(acc_sub, 5),
                "acc_sub_lo": round(lo_s, 5), "acc_sub_hi": round(hi_s, 5),
                "gap_pp": round(100 * d, 3),
                "gap_lo_pp": round(100 * d_lo, 3), "gap_hi_pp": round(100 * d_hi, 3),
                "gap_clust_lo_pp": round(100 * cb_lo, 3),
                "gap_clust_hi_pp": round(100 * cb_hi, 3),
                "n_battles": int(len(np.unique(battles))),
                "greediness": round(g_rate, 5),
                "greediness_lo": round(lo_g, 5), "greediness_hi": round(hi_g, 5),
            })

    if rows:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

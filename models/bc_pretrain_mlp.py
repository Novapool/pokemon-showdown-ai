"""
bc_pretrain_mlp.py — Behavior-cloning pretraining of the MLP-PPO agent on
human Showdown replays (M5.5).

Trains PPOAgent's shared trunk + policy head (and the M5 opp head, using the
cross-linked simultaneous-opponent labels) with cross-entropy on records
produced by the replay adapter (models/replay_adapter_cli.js). The value head
is left at initialization for PPO to learn. Checkpoints are written with
PPOAgent.save(), so evaluate.py / mcts / train.py --resume consume them
unchanged.

Multi-format training (project decision 2026-07-16): the competitive target is
gen1randombattle, but high-level gen1ou games ("how pros play with real
teams") are mixed in with per-format sampling weights. Per-format accuracy is
reported separately so the OU weight can be tuned if it hurts randbats play.

Record filtering: rated games need rating >= --min-rating; unrated games are
kept only when their battle id carries a tournament prefix (e.g.
"bigbang-gen1ou-798" — Smogon tour games are unrated but high-level) unless
--no-tournaments.

Usage:
    python models/bc_pretrain_mlp.py                       # both formats, defaults
    python models/bc_pretrain_mlp.py --formats gen1randombattle
    python models/bc_pretrain_mlp.py --format-weights "gen1randombattle=0.5,gen1ou=0.5"
    python models/bc_pretrain_mlp.py --max-shards 2 --epochs 1   # smoke test

Prerequisites: scripts/scrape_replays.py (+ scripts/bootstrap_gen1ou_replays.py),
then models/replay_adapter_cli.js for each format.
"""

import argparse
import base64
import gzip
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MODELS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODELS_DIR))
sys.path.insert(0, str(MODELS_DIR / "ppo"))

from ppo_agent import PPOAgent  # noqa: E402

OBS_V2_SIZE = 12 * 77  # flattened schema-v2 observation
SHUFFLE_BUFFER = 50_000
LOG_EVERY = 200  # batches


class ReplayShardDataset:
    """Streams (obs, action, opp_action, format_idx) from one format's shards."""

    def __init__(self, fmt: str, traj_dir: Path, min_rating: int, tournaments: bool,
                 format_idx: int, val_shards: int):
        self.fmt = fmt
        self.format_idx = format_idx
        self.min_rating = min_rating
        self.tournaments = tournaments
        shards = sorted((traj_dir / fmt).glob("shard-*.jsonl.gz"))
        if not shards:
            raise FileNotFoundError(
                f"no trajectory shards under {traj_dir / fmt} — run "
                f"`node models/replay_adapter_cli.js --format {fmt}` first"
            )
        # First shard(s) held out for validation (never trained on).
        self.val_files = shards[:val_shards]
        self.files = shards[val_shards:] or shards  # tiny smoke datasets: train=val

    def _keep(self, rec: dict) -> bool:
        rating = rec.get("r")
        if rating is not None:
            return rating >= self.min_rating
        # Unrated: keep tournament games (battle id has a tour prefix, e.g.
        # "smogtours-gen1ou-..." / "bigbang-gen1ou-..."), drop casual unrated.
        return self.tournaments and not rec["b"].startswith(self.fmt)

    def _iter_files(self, files, seed: int | None):
        if seed is not None:
            files = list(files)
            random.Random(seed).shuffle(files)
        for path in files:
            with gzip.open(path, "rt") as f:
                for line in f:
                    rec = json.loads(line)
                    if not self._keep(rec):
                        continue
                    obs = np.frombuffer(base64.b64decode(rec["o"]), dtype=np.float32)
                    # Value target: the trajectory's final outcome in [-1, 1]
                    # (win +1 / loss -1) — the supervised value-net recipe, so
                    # the checkpoint is searchable (MCTS evaluates leaves with
                    # the value head, which plain policy-BC leaves at init).
                    yield obs, rec["a"], rec["oa"], 2.0 * rec["w"] - 1.0, self.format_idx

    def train_samples(self, seed: int):
        yield from self._iter_files(self.files, seed)

    def val_samples(self):
        yield from self._iter_files(self.val_files, None)


def weighted_interleave(streams: list, weights: list[float], seed: int):
    """Draw the next sample from stream i with probability weights[i],
    renormalizing as streams exhaust."""
    rng = random.Random(seed)
    iters = [iter(s) for s in streams]
    live = list(range(len(iters)))
    w = list(weights)
    while live:
        total = sum(w[i] for i in live)
        r = rng.random() * total
        acc = 0.0
        pick = live[-1]
        for i in live:
            acc += w[i]
            if r <= acc:
                pick = i
                break
        try:
            yield next(iters[pick])
        except StopIteration:
            live.remove(pick)


def shuffled(sample_iter, seed: int):
    rng = random.Random(seed)
    buffer = []
    for sample in sample_iter:
        buffer.append(sample)
        if len(buffer) >= SHUFFLE_BUFFER:
            i = rng.randrange(len(buffer))
            buffer[i], buffer[-1] = buffer[-1], buffer[i]
            yield buffer.pop()
    rng.shuffle(buffer)
    yield from buffer


def batches(sample_iter, batch_size: int):
    obs_buf, act_buf, opp_buf, val_buf, fmt_buf = [], [], [], [], []
    for obs, action, opp_action, value, fmt_idx in sample_iter:
        obs_buf.append(obs)
        act_buf.append(action)
        opp_buf.append(opp_action)
        val_buf.append(value)
        fmt_buf.append(fmt_idx)
        if len(obs_buf) == batch_size:
            yield (np.stack(obs_buf), np.array(act_buf, dtype=np.int64),
                   np.array(opp_buf, dtype=np.int64), np.array(val_buf, dtype=np.float32),
                   np.array(fmt_buf, dtype=np.int64))
            obs_buf, act_buf, opp_buf, val_buf, fmt_buf = [], [], [], [], []
    if obs_buf:
        yield (np.stack(obs_buf), np.array(act_buf, dtype=np.int64),
               np.array(opp_buf, dtype=np.int64), np.array(val_buf, dtype=np.float32),
               np.array(fmt_buf, dtype=np.int64))


def parse_weights(spec: str | None, formats: list[str]) -> list[float]:
    if not spec:
        return [1.0] * len(formats)
    given = {}
    for part in spec.split(","):
        name, val = part.split("=")
        given[name.strip()] = float(val)
    return [given.get(fmt, 1.0) for fmt in formats]


def evaluate(agent: PPOAgent, datasets: list[ReplayShardDataset], batch_size: int, device):
    """Per-format policy/opp-head top-1 accuracy on the held-out shards."""
    agent.eval()
    # correct, n, opp_correct, opp_n, value_sign_correct
    stats = {ds.fmt: [0, 0, 0, 0, 0] for ds in datasets}
    with torch.no_grad():
        for ds in datasets:
            for obs_np, act_np, opp_np, val_np, _ in batches(ds.val_samples(), batch_size):
                obs = torch.from_numpy(obs_np).to(device)
                features = agent.trunk(obs)
                pred = agent.policy_head(features).argmax(dim=-1).cpu().numpy()
                s = stats[ds.fmt]
                s[0] += int((pred == act_np).sum())
                s[1] += len(act_np)
                labeled = opp_np >= 0
                if labeled.any():
                    opp_pred = agent.opp_head(features).argmax(dim=-1).cpu().numpy()
                    s[2] += int((opp_pred[labeled] == opp_np[labeled]).sum())
                    s[3] += int(labeled.sum())
                value_pred = agent.value_head(features).squeeze(-1).cpu().numpy()
                s[4] += int((np.sign(value_pred) == np.sign(val_np)).sum())
    agent.train()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="BC pretraining of MLP-PPO on human replays")
    parser.add_argument("--formats", default="gen1randombattle,gen1ou")
    parser.add_argument("--format-weights", default=None,
                        help='per-format sampling weights, e.g. "gen1randombattle=0.6,gen1ou=0.4"')
    parser.add_argument("--traj-dir", default="data/replay_trajs")
    parser.add_argument("--min-rating", type=int, default=1300)
    parser.add_argument("--no-tournaments", action="store_true",
                        help="drop unrated tournament games instead of keeping them")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--opp-bc-coef", type=float, default=0.1,
                        help="weight of the opp-head CE term (M5 convention)")
    parser.add_argument("--value-bc-coef", type=float, default=0.5,
                        help="weight of the value-head MSE toward the game outcome "
                             "(0 disables — leaves the value head at init)")
    parser.add_argument("--val-shards", type=int, default=1,
                        help="shards per format held out for validation")
    parser.add_argument("--max-shards", type=int, default=None,
                        help="limit training shards per format (smoke tests)")
    parser.add_argument("--checkpoint-dir", default=str(MODELS_DIR / "checkpoints"))
    parser.add_argument("--out", default="bc_mlp_gen1.pt")
    parser.add_argument("--device", default=None, choices=[None, "cpu", "mps", "cuda"])
    args = parser.parse_args()

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    weights = parse_weights(args.format_weights, formats)
    datasets = []
    for i, fmt in enumerate(formats):
        ds = ReplayShardDataset(fmt, Path(args.traj_dir), args.min_rating,
                                not args.no_tournaments, i, args.val_shards)
        if args.max_shards:
            ds.files = ds.files[: args.max_shards]
        datasets.append(ds)
        print(f"dataset[{fmt}]: {len(ds.files)} train shards, "
              f"{len(ds.val_files)} val shards, weight {weights[i]:g}")

    agent = PPOAgent(obs_size=OBS_V2_SIZE, device=args.device)
    device = agent.device
    # BC's own optimizer (policy/opp/trunk only) — agent.optimizer stays the
    # untouched PPO Adam that gets checkpointed for a later fine-tune.
    params = (list(agent.trunk.parameters()) + list(agent.policy_head.parameters()) +
              list(agent.opp_head.parameters()))
    if args.value_bc_coef > 0:
        params += list(agent.value_head.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    print(f"device: {device} | params: {sum(p.numel() for p in agent.parameters()):,}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / args.out

    total = 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        agent.train()
        ep = {"loss": 0.0, "correct": 0, "n": 0, "opp_correct": 0, "opp_n": 0}
        win = {"loss": 0.0, "correct": 0, "n": 0}
        n_batches = 0

        stream = weighted_interleave(
            [ds.train_samples(seed=epoch * 31 + i) for i, ds in enumerate(datasets)],
            weights, seed=epoch,
        )
        for obs_np, act_np, opp_np, val_np, _fmt in batches(shuffled(stream, seed=epoch), args.batch_size):
            obs = torch.from_numpy(obs_np).to(device)
            actions = torch.from_numpy(act_np).to(device)
            opp_actions = torch.from_numpy(opp_np).to(device)

            features = agent.trunk(obs)
            logits = agent.policy_head(features)
            loss = F.cross_entropy(logits, actions)

            if args.value_bc_coef > 0:
                values = agent.value_head(features).squeeze(-1)
                targets = torch.from_numpy(val_np).to(device)
                loss = loss + args.value_bc_coef * F.mse_loss(values, targets)

            labeled = opp_actions >= 0
            if args.opp_bc_coef > 0 and bool(labeled.any()):
                opp_logits = agent.opp_head(features)
                opp_loss = F.cross_entropy(opp_logits[labeled], opp_actions[labeled])
                loss = loss + args.opp_bc_coef * opp_loss
                with torch.no_grad():
                    ep["opp_correct"] += int(
                        (opp_logits[labeled].argmax(dim=-1) == opp_actions[labeled]).sum().item())
                    ep["opp_n"] += int(labeled.sum().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n = len(actions)
            correct = int((logits.argmax(dim=-1) == actions).sum().item())
            for d in (ep, win):
                d["loss"] += loss.item() * n
                d["correct"] += correct
                d["n"] += n
            n_batches += 1
            if n_batches % LOG_EVERY == 0:
                print(f"epoch {epoch} batch {n_batches}: loss {win['loss'] / win['n']:.4f} "
                      f"acc {win['correct'] / win['n']:.3f} "
                      f"({ep['n']} samples, {time.time() - start:.0f}s)", flush=True)
                win = {"loss": 0.0, "correct": 0, "n": 0}

        total += ep["n"]
        opp_msg = (f" opp-acc {ep['opp_correct'] / ep['opp_n']:.3f}" if ep["opp_n"] else "")
        print(f"epoch {epoch}/{args.epochs} done: loss {ep['loss'] / max(ep['n'], 1):.4f} "
              f"acc {ep['correct'] / max(ep['n'], 1):.3f}{opp_msg} over {ep['n']} samples")

        val = evaluate(agent, datasets, args.batch_size, device)
        for fmt, (c, n, oc, on, vc) in val.items():
            if n:
                opp_msg = f" opp-acc {oc / on:.3f}" if on else ""
                print(f"  val[{fmt}]: acc {c / n:.3f}{opp_msg} "
                      f"value-sign-acc {vc / n:.3f} ({n} samples)")

        agent.save(str(checkpoint_path))
        print(f"checkpoint saved: {checkpoint_path}")

    print(f"finished: {total} samples in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()

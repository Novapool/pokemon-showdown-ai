"""
bc_pretrain.py — Behavior-cloning pretraining on Metamon human replays (M2.5).

Trains the shared TransformerPolicy on (obs, action) pairs streamed from
MetamonDataAdapter. Cross-entropy on the policy head only; the value head is
left at initialization for PPO to learn. Trajectories are streamed (not held
in memory) with a shuffle buffer, so the full gen1ou dataset fits any machine.

Usage:
    python models/bc_pretrain.py
    python models/bc_pretrain.py --epochs 5 --format gen1ou \
        --checkpoint_dir models/checkpoints
    python models/bc_pretrain.py --max_files 50   # quick smoke test

Prerequisite: bash scripts/download_metamon.sh
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

MODELS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODELS_DIR))
sys.path.insert(0, str(MODELS_DIR / "transformer"))

from metamon_adapter import MetamonDataAdapter  # noqa: E402
from transformer_policy import TransformerPolicy  # noqa: E402

BATCH_SIZE = 256
LR = 1e-3
SHUFFLE_BUFFER = 50_000
LOG_EVERY = 200  # batches


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def shuffled_samples(adapter: MetamonDataAdapter, epoch_seed: int):
    """Stream (obs, action) pairs in near-random order without loading
    the whole dataset: files are shuffled, then samples pass through a
    fixed-size shuffle buffer."""
    rng = random.Random(epoch_seed)
    buffer = []
    for obs, action, _done in adapter.samples(shuffle_files=True, seed=epoch_seed):
        buffer.append((obs, action))
        if len(buffer) >= SHUFFLE_BUFFER:
            i = rng.randrange(len(buffer))
            buffer[i], buffer[-1] = buffer[-1], buffer[i]
            yield buffer.pop()
    rng.shuffle(buffer)
    yield from buffer


def batches(sample_iter, batch_size: int):
    obs_buf, act_buf = [], []
    for obs, action in sample_iter:
        obs_buf.append(obs)
        act_buf.append(action)
        if len(obs_buf) == batch_size:
            yield np.stack(obs_buf), np.array(act_buf, dtype=np.int64)
            obs_buf, act_buf = [], []
    if obs_buf:
        yield np.stack(obs_buf), np.array(act_buf, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="BC pretraining on Metamon replays")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--format", default="gen1ou", dest="battle_format")
    parser.add_argument("--checkpoint_dir", default=str(MODELS_DIR / "checkpoints"))
    parser.add_argument(
        "--max_files", type=int, default=None,
        help="Limit number of replay files (for smoke tests)",
    )
    args = parser.parse_args()

    adapter = MetamonDataAdapter(battle_format=args.battle_format)
    if args.max_files:
        adapter.files = adapter.files[: args.max_files]
    print(f"dataset: {len(adapter)} {args.battle_format} replay files under {adapter.cache_dir}")

    device = _pick_device()
    model = TransformerPolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    print(f"device: {device} | params: {sum(p.numel() for p in model.parameters()):,}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"bc_pretrain_{args.battle_format}.pt"

    total_samples = 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, epoch_correct, epoch_n, n_batches = 0.0, 0, 0, 0
        window_loss, window_correct, window_n = 0.0, 0, 0

        for obs_np, act_np in batches(shuffled_samples(adapter, epoch_seed=epoch), BATCH_SIZE):
            obs = torch.from_numpy(obs_np).to(device)
            actions = torch.from_numpy(act_np).to(device)

            logits, _value = model(obs)  # value head unused: CE on policy only
            loss = criterion(logits, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n = len(actions)
            correct = int((logits.argmax(dim=-1) == actions).sum().item())
            epoch_loss += loss.item() * n
            epoch_correct += correct
            epoch_n += n
            window_loss += loss.item() * n
            window_correct += correct
            window_n += n
            n_batches += 1

            if n_batches % LOG_EVERY == 0:
                print(
                    f"epoch {epoch} batch {n_batches}: "
                    f"loss {window_loss / window_n:.4f} "
                    f"acc {window_correct / window_n:.3f} "
                    f"({epoch_n} samples, {time.time() - start:.0f}s)"
                )
                window_loss, window_correct, window_n = 0.0, 0, 0

        total_samples += epoch_n
        print(
            f"epoch {epoch}/{args.epochs} done: "
            f"loss {epoch_loss / max(epoch_n, 1):.4f} "
            f"acc {epoch_correct / max(epoch_n, 1):.3f} "
            f"over {epoch_n} samples"
        )

        torch.save(
            {
                "model": model.state_dict(),
                "hparams": model.hparams,
                "format": args.battle_format,
                "epoch": epoch,
                "total_samples": total_samples,
            },
            checkpoint_path,
        )
        print(f"checkpoint saved: {checkpoint_path}")

    print(f"finished: {total_samples} samples in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()

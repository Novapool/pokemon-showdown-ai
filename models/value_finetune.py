"""
value_finetune.py — AlphaZero-style value-head fine-tuning (M8 Phase 2).

Takes a dataset collected by `collect_value_data.py` and retrains the value
head of a PPO checkpoint on search/self-play targets instead of the PPO
bootstrapped returns it was originally trained on. The policy and opponent
heads are left bit-identical (and, by default, so is the shared trunk), so any
downstream change in MCTS strength is attributable to leaf evaluation alone —
the search prior is untouched.

Targets (`--target`):
  outcome   the searching seat's final game result (+1 / -1 / 0) at every
            state — the pure AlphaZero target, undiscounted.
  mc        discounted Monte-Carlo return of the shaped gym rewards actually
            received from that state onward: G_t = r_t + γ·G_{t+1} (γ from
            `--gamma`, default 0.99 to match the PPO training discount).
  root      the search's own root Q at that decision. States where no search
            ran (forced/locked choices) carry NaN and are dropped.

Example
-------
    python models/value_finetune.py \
        --checkpoint models/ppo/checkpoints/v3/ppo_step_5000002_final.pt \
        --data-dir data/value_targets/m8_v3 --target outcome \
        --out models/ppo/checkpoints/v3_valft/ppo_v3_valft.pt
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppo"))

from ppo_agent import PPOAgent  # noqa: E402


def load_dataset(data_dir: str) -> dict:
    """Concatenate every .npz shard in data_dir into one in-memory dataset."""
    paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not paths:
        raise SystemExit(f"no .npz shards found in {data_dir}")
    fields = ("obs", "visits", "root_value", "reward", "outcome", "seat",
              "step_idx", "game_id")
    chunks = {f: [] for f in fields}
    for path in paths:
        with np.load(path) as shard:
            for f in fields:
                chunks[f].append(shard[f])
    data = {f: np.concatenate(chunks[f]) for f in fields}
    print(f"loaded {len(paths)} shards | {len(data['obs'])} decisions | "
          f"{len(np.unique(data['game_id']))} games")
    return data


def compute_targets(data: dict, target: str, gamma: float) -> np.ndarray:
    """Per-decision value target, aligned with data['obs']."""
    if target == "outcome":
        return data["outcome"].astype(np.float32)
    if target == "root":
        return data["root_value"].astype(np.float32)

    # Discounted Monte-Carlo return, computed backwards within each game.
    rewards = data["reward"].astype(np.float64)
    game_id = data["game_id"]
    step_idx = data["step_idx"]
    order = np.lexsort((step_idx, game_id))
    returns = np.zeros_like(rewards)
    running = 0.0
    prev_game = None
    for i in order[::-1]:
        if game_id[i] != prev_game:
            running = 0.0
            prev_game = game_id[i]
        running = rewards[i] + gamma * running
        returns[i] = running
    return returns.astype(np.float32)


def split_by_game(game_id: np.ndarray, val_frac: float, seed: int) -> tuple:
    """Train/val masks split on whole games, so no game leaks across the split."""
    games = np.unique(game_id)
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    n_val = max(1, int(round(len(games) * val_frac))) if val_frac > 0 else 0
    val_games = set(games[:n_val].tolist())
    val_mask = np.fromiter((g in val_games for g in game_id), dtype=bool, count=len(game_id))
    return ~val_mask, val_mask


@torch.no_grad()
def evaluate_value(agent: PPOAgent, obs: torch.Tensor, targets: torch.Tensor,
                   batch_size: int = 4096) -> tuple:
    """(MSE, sign agreement) of the current value head on a held-out set."""
    agent.eval()
    se, sign_hits, n = 0.0, 0, 0
    for i in range(0, len(obs), batch_size):
        ob = obs[i:i + batch_size].to(agent.device)
        tg = targets[i:i + batch_size].to(agent.device)
        pred = agent.value_head(agent.trunk(ob)).squeeze(-1)
        se += float(((pred - tg) ** 2).sum().item())
        sign_hits += int(((pred > 0) == (tg > 0)).sum().item())
        n += len(tg)
    return se / max(1, n), sign_hits / max(1, n)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a PPO checkpoint's value head on self-play/search targets",
    )
    parser.add_argument("--checkpoint", required=True, help="PPO checkpoint to fine-tune")
    parser.add_argument("--data-dir", required=True, help="collect_value_data.py output dir")
    parser.add_argument("--out", required=True, help="path for the fine-tuned checkpoint")
    parser.add_argument("--target", choices=["outcome", "mc", "root"], default="outcome")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount for --target mc")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--unfreeze-trunk", action="store_true",
        help="also train the shared trunk. This CHANGES THE POLICY (the trunk "
             "feeds the policy head), so the Criterion C A/B is no longer a "
             "clean value-head-only comparison. Off by default.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    data = load_dataset(args.data_dir)
    targets_np = compute_targets(data, args.target, args.gamma)

    keep = np.isfinite(targets_np)
    dropped = int((~keep).sum())
    if dropped:
        print(f"dropping {dropped} rows with no target (no search ran at that decision)")
    obs_np = data["obs"][keep]
    targets_np = targets_np[keep]
    game_id = data["game_id"][keep]

    print(f"target={args.target} | n={len(targets_np)} | mean {targets_np.mean():+.3f} "
          f"| std {targets_np.std():.3f} | min {targets_np.min():+.3f} "
          f"| max {targets_np.max():+.3f}")

    agent = PPOAgent.load(args.checkpoint, device=args.device)
    obs_size = agent._hparams["obs_size"]
    if obs_np.shape[1] != obs_size:
        raise SystemExit(
            f"dataset obs width {obs_np.shape[1]} != checkpoint obs_size {obs_size} — "
            "the dataset was collected for a different checkpoint/obs schema"
        )

    obs_t = torch.from_numpy(np.ascontiguousarray(obs_np))
    targets_t = torch.from_numpy(np.ascontiguousarray(targets_np))
    train_mask, val_mask = split_by_game(game_id, args.val_frac, args.seed)
    train_obs, train_tg = obs_t[train_mask], targets_t[train_mask]
    val_obs, val_tg = obs_t[val_mask], targets_t[val_mask]
    print(f"train {len(train_tg)} decisions | val {len(val_tg)} decisions")

    if len(val_tg):
        mse0, sign0 = evaluate_value(agent, val_obs, val_tg)
        print(f"BEFORE fine-tune | val MSE {mse0:.4f} | sign agreement {sign0:.3f}")

    # Freeze everything, then re-enable exactly what we intend to train. The
    # policy and opp heads stay bit-identical either way.
    for param in agent.parameters():
        param.requires_grad_(False)
    trainable = list(agent.value_head.parameters())
    for param in trainable:
        param.requires_grad_(True)
    if args.unfreeze_trunk:
        for param in agent.trunk.parameters():
            param.requires_grad_(True)
        trainable += list(agent.trunk.parameters())
        print("WARNING: --unfreeze-trunk also changes the policy's features")
    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    n = len(train_tg)
    for epoch in range(1, args.epochs + 1):
        agent.train()
        perm = torch.randperm(n)
        total_loss, batches = 0.0, 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            ob = train_obs[idx].to(agent.device)
            tg = train_tg[idx].to(agent.device)
            pred = agent.value_head(agent.trunk(ob)).squeeze(-1)
            loss = torch.nn.functional.mse_loss(pred, tg)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, agent.max_grad_norm)
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        line = f"epoch {epoch}/{args.epochs} | train MSE {total_loss / max(1, batches):.4f}"
        if len(val_tg):
            mse, sign = evaluate_value(agent, val_obs, val_tg)
            line += f" | val MSE {mse:.4f} | sign agreement {sign:.3f}"
        print(line, flush=True)

    for param in agent.parameters():
        param.requires_grad_(True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    agent.save(args.out)
    print(f"saved fine-tuned checkpoint -> {args.out}")

    meta_path = os.path.splitext(args.out)[0] + "_valft.json"
    with open(meta_path, "w") as fh:
        json.dump({
            "base_checkpoint": args.checkpoint,
            "data_dir": args.data_dir,
            "target": args.target,
            "gamma": args.gamma,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "unfreeze_trunk": args.unfreeze_trunk,
            "n_train": int(len(train_tg)),
            "n_val": int(len(val_tg)),
        }, fh, indent=2)


if __name__ == "__main__":
    main()

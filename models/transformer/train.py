"""
train.py — Rollout-based PPO training loop for Pokemon Showdown (TransformerAgent).

Usage:
    python models/transformer/train.py
    python models/transformer/train.py --battles 500000 --rollout-steps 2048
"""

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np

# Resolve models/ directory so gym_client is importable
sys.path.insert(0, str(Path(__file__).parent.parent))        # models/
sys.path.insert(0, str(Path(__file__).parent))               # models/transformer/
sys.path.insert(0, str(Path(__file__).parent.parent / "ppo")) # models/ppo/ for TrajectoryBuffer

from vec_gym_client import VecGymClient                       # noqa: E402
from transformer_agent import TransformerAgent                # noqa: E402
from trajectory_buffer import TrajectoryBuffer               # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TransformerAgent on Pokemon Showdown")
    parser.add_argument(
        "--battles",
        type=int,
        default=200000,
        help="Total environment steps to train for (default: 200000)",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=2048,
        help="Number of steps per rollout before a PPO update (default: 2048)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25000,
        help="Save a checkpoint every N steps (default: 25000)",
    )
    parser.add_argument(
        "--win-rate-window",
        type=int,
        default=500,
        help="Rolling window size for win rate tracking across episodes (default: 500)",
    )
    parser.add_argument(
        "--simple-reward",
        action="store_true",
        help="Zero out intermediate rewards; use only terminal ±1.0 win/loss signal",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.05,
        help="Entropy bonus coefficient — higher keeps policy more exploratory (default: 0.05)",
    )
    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=2,
        help="PPO update epochs per rollout — fewer reduces drift on noisy advantages (default: 2)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Adam learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--lr-final",
        type=float,
        default=1e-5,
        help="Final learning rate after linear decay (default: 1e-5). Set to same as --lr to disable decay.",
    )
    parser.add_argument("--n-envs", type=int, default=4,
        help="Number of parallel gym environments (default: 4)")
    parser.add_argument("--opponent", type=str, default="random",
        choices=["random", "damage-first"],
        help="Opponent AI type (default: random)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_budget = args.battles
    rollout_steps = args.rollout_steps
    checkpoint_every = args.checkpoint_every

    checkpoint_dir = Path(__file__).parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    agent = TransformerAgent(
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        ppo_epochs=args.ppo_epochs,
    )
    buffer = TrajectoryBuffer()
    env = VecGymClient(n_envs=args.n_envs, flat_mode=False, opponent=args.opponent)

    reward_mode = "simple (terminal ±1 only)" if args.simple_reward else "full (KO + status + win/loss)"
    lr_decay = args.lr_final != args.lr
    print(
        f"Starting TransformerAgent training: {total_budget} steps | "
        f"rollout={rollout_steps} steps | "
        f"checkpoint every {checkpoint_every} steps | "
        f"reward={reward_mode} | "
        f"lr={args.lr}{'→'+str(args.lr_final) if lr_decay else ''} "
        f"entropy={args.entropy_coef} ppo_epochs={args.ppo_epochs} | "
        f"n_envs={args.n_envs} | opponent={args.opponent}",
        flush=True,
    )

    total_steps = 0
    last_checkpoint_step = 0

    # Rolling win rate across all episodes (spans rollout boundaries)
    win_history: deque = deque(maxlen=args.win_rate_window)

    # Initialize all envs
    obs_batch, masks_batch = env.reset()
    obs = list(obs_batch)    # list of n_envs arrays, each shape (12, 69)
    masks = list(masks_batch)
    dones = [False] * args.n_envs

    try:
        while total_steps < total_budget:
            rollout_wins = 0
            rollout_episodes = 0

            for _ in range(rollout_steps):
                if total_steps >= total_budget:
                    break

                # Act for all envs
                actions, log_probs, values = [], [], []
                for i in range(args.n_envs):
                    try:
                        a, lp, v = agent.act(obs[i], masks[i])
                    except Exception as e:
                        print(f"  [act error env {i}] {e} — using action 0", flush=True)
                        a, lp, v = 0, 0.0, 0.0
                    actions.append(a)
                    log_probs.append(lp)
                    values.append(v)

                prev_obs = list(obs)

                # Step all envs
                try:
                    obs_raw, rewards, dones, infos, masks_raw = env.step(actions)
                    obs = list(obs_raw)
                    masks = list(masks_raw)
                except RuntimeError as e:
                    print(f"  [step error] {e} — resetting all envs", flush=True)
                    obs_batch, masks_batch = env.reset()
                    obs = list(obs_batch)
                    masks = list(masks_batch)
                    dones = [False] * args.n_envs
                    continue

                # Push transitions and handle episode ends
                for i in range(args.n_envs):
                    if args.simple_reward and not dones[i]:
                        rewards[i] = 0.0
                    buffer.push(prev_obs[i], actions[i], rewards[i], dones[i], values[i], log_probs[i])

                total_steps += args.n_envs

                for i in range(args.n_envs):
                    if dones[i]:
                        rollout_episodes += 1
                        won = 1 if infos[i].get("winner") == "Gym" else 0
                        rollout_wins += won
                        win_history.append(won)
                        # Reset this env
                        try:
                            obs[i], masks[i] = env.envs[i].reset()
                        except RuntimeError as e:
                            print(f"  [reset error env {i}] {e}", flush=True)
                        dones[i] = False

                # Checkpoint on step boundary
                if total_steps - last_checkpoint_step >= checkpoint_every:
                    ckpt_path = checkpoint_dir / f"transformer_step_{total_steps}.pt"
                    agent.save(str(ckpt_path))
                    last_checkpoint_step = total_steps
                    print(f"[checkpoint] Saved {ckpt_path}", flush=True)

            # PPO update
            if len(buffer) == 0:
                break

            # Bootstrap value from first non-done env (approximation)
            first_not_done = next((i for i in range(args.n_envs) if not dones[i]), 0)
            _, _, last_value = agent.act(obs[first_not_done], masks[first_not_done])

            buffer.compute_advantages(last_value=last_value)
            loss = agent.update(buffer)
            buffer.clear()

            # LR decay
            if lr_decay:
                frac = min(1.0, total_steps / total_budget)
                current_lr = args.lr + frac * (args.lr_final - args.lr)
                for pg in agent.optimizer.param_groups:
                    pg["lr"] = current_lr

            # Log
            rollout_wr = rollout_wins / rollout_episodes if rollout_episodes > 0 else float("nan")
            rolling_wr = sum(win_history) / len(win_history) if win_history else float("nan")
            current_lr = agent.optimizer.param_groups[0]["lr"]
            print(
                f"Step {total_steps}/{total_budget} | "
                f"Eps: {rollout_episodes} | "
                f"WR(rollout): {rollout_wr:.2f} | "
                f"WR(rolling-{args.win_rate_window}): {rolling_wr:.2f} | "
                f"Loss: {loss:.3f} | LR: {current_lr:.2e}",
                flush=True,
            )

    finally:
        env.close()

    # Final checkpoint
    final_path = checkpoint_dir / f"transformer_step_{total_steps}_final.pt"
    agent.save(str(final_path))
    print(f"\nTraining complete. Final model saved to {final_path}")
    print(f"Total steps: {total_steps}")


if __name__ == "__main__":
    main()

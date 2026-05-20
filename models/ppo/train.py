"""
train.py — Rollout-based PPO training loop for Pokemon Showdown.

Usage:
    python models/ppo/train.py
    python models/ppo/train.py --battles 500000 --rollout-steps 2048 --checkpoint-every 50000
    python models/ppo/train.py --simple-reward   # terminal ±1.0 only, no intermediate rewards
"""

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np

# Resolve models/ directory so gym_client and ppo modules are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from gym_client import GymClient          # noqa: E402

# ppo_agent and trajectory_buffer live alongside this script
sys.path.insert(0, str(Path(__file__).parent))
from ppo_agent import PPOAgent            # noqa: E402
from trajectory_buffer import TrajectoryBuffer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO agent on Pokemon Showdown")
    parser.add_argument(
        "--battles",
        type=int,
        default=100000,
        help="Total environment steps to train for (default: 100000)",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_budget = args.battles
    rollout_steps = args.rollout_steps
    checkpoint_every = args.checkpoint_every

    checkpoint_dir = Path(__file__).parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    agent = PPOAgent(
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        ppo_epochs=args.ppo_epochs,
    )
    buffer = TrajectoryBuffer()
    env = GymClient(flat_mode=True)

    reward_mode = "simple (terminal ±1 only)" if args.simple_reward else "full (KO + status + win/loss)"
    print(
        f"Starting PPO training: {total_budget} steps | "
        f"rollout={rollout_steps} steps | "
        f"checkpoint every {checkpoint_every} steps | "
        f"reward={reward_mode} | "
        f"lr={args.lr} entropy={args.entropy_coef} ppo_epochs={args.ppo_epochs}",
        flush=True,
    )

    total_steps = 0
    last_checkpoint_step = 0

    # Rolling win rate across all episodes (spans rollout boundaries)
    win_history: deque = deque(maxlen=args.win_rate_window)

    try:
        obs, valid_mask = env.reset()
        done = False

        while total_steps < total_budget:
            # ----------------------------------------------------------------
            # Collect one rollout
            # ----------------------------------------------------------------
            rollout_wins = 0
            rollout_episodes = 0

            for _ in range(rollout_steps):
                if total_steps >= total_budget:
                    break

                action, log_prob, value = agent.act(obs, valid_mask)
                try:
                    next_obs, reward, done, info, valid_mask = env.step(action)
                except RuntimeError as e:
                    print(f"  [step {total_steps}] step error ({e}) — resetting", flush=True)
                    obs, valid_mask = env.reset()
                    done = False
                    continue

                if args.simple_reward and not done:
                    reward = 0.0

                buffer.push(obs, action, reward, done, value, log_prob)
                obs = next_obs
                total_steps += 1

                if done:
                    rollout_episodes += 1
                    won = 1 if info.get("winner") == "Gym" else 0
                    rollout_wins += won
                    win_history.append(won)
                    obs, valid_mask = env.reset()
                    done = False

                # Checkpoint on step boundary
                if total_steps - last_checkpoint_step >= checkpoint_every:
                    ckpt_path = checkpoint_dir / f"ppo_step_{total_steps}.pt"
                    agent.save(str(ckpt_path))
                    last_checkpoint_step = total_steps
                    print(f"[checkpoint] Saved {ckpt_path}", flush=True)

            # ----------------------------------------------------------------
            # Compute advantages and update
            # ----------------------------------------------------------------
            if len(buffer) == 0:
                break

            if done:
                last_value = 0.0
            else:
                _, _, last_value = agent.act(obs, valid_mask)

            buffer.compute_advantages(last_value=last_value)
            loss = agent.update(buffer)
            buffer.clear()

            # ----------------------------------------------------------------
            # Log rollout summary
            # ----------------------------------------------------------------
            rollout_wr = rollout_wins / rollout_episodes if rollout_episodes > 0 else float("nan")
            rolling_wr = sum(win_history) / len(win_history) if win_history else float("nan")
            print(
                f"Step {total_steps}/{total_budget} | "
                f"Eps: {rollout_episodes} | "
                f"WR(rollout): {rollout_wr:.2f} | "
                f"WR(rolling-{args.win_rate_window}): {rolling_wr:.2f} | "
                f"Loss: {loss:.3f}",
                flush=True,
            )

    finally:
        env.close()

    # Final checkpoint
    final_path = checkpoint_dir / f"ppo_step_{total_steps}_final.pt"
    agent.save(str(final_path))
    print(f"\nTraining complete. Final model saved to {final_path}")
    print(f"Total steps: {total_steps}")


if __name__ == "__main__":
    main()

"""
train.py — Rollout-based PPO training loop for Pokemon Showdown.

Usage:
    python models/ppo/train.py
    python models/ppo/train.py --battles 500000 --rollout-steps 512 --checkpoint-every 50000

The loop collects rollout_steps environment steps per update cycle regardless
of episode boundaries.  When a 'done' signal arrives mid-rollout, the
environment is reset immediately so collection continues seamlessly.
"""

import argparse
import sys
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
        default=512,
        help="Number of steps per rollout before a PPO update (default: 512)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25000,
        help="Save a checkpoint every N steps (default: 25000)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_budget = args.battles
    rollout_steps = args.rollout_steps
    checkpoint_every = args.checkpoint_every

    checkpoint_dir = Path(__file__).parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    agent = PPOAgent()
    buffer = TrajectoryBuffer()
    # This PPO trunk is hardcoded to the legacy flat 100-dim vector. The M2
    # verification run (MLP PPO on flattened structured obs) is a separate
    # trunk, not this baseline — see MILESTONES.md M2 success criteria.
    env = GymClient(structured=False)

    print(
        f"Starting PPO training: {total_budget} steps | "
        f"rollout={rollout_steps} steps | "
        f"checkpoint every {checkpoint_every} steps",
        flush=True,
    )

    total_steps = 0
    last_checkpoint_step = 0

    # Episode tracking within each rollout window
    rollout_wins = 0
    rollout_episodes = 0

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

                buffer.push(obs, action, reward, done, value, log_prob)

                obs = next_obs
                total_steps += 1

                if done:
                    rollout_episodes += 1
                    if info.get("winner") == "Gym":
                        rollout_wins += 1
                    # Start a new episode immediately
                    obs, valid_mask = env.reset()
                    done = False

                # Checkpoint on step boundary
                if total_steps - last_checkpoint_step >= checkpoint_every:
                    ckpt_path = checkpoint_dir / f"ppo_step_{total_steps}.pt"
                    agent.save(str(ckpt_path))
                    last_checkpoint_step = total_steps
                    print(f"[checkpoint] Saved {ckpt_path}")

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
            win_rate = (
                rollout_wins / rollout_episodes if rollout_episodes > 0 else float("nan")
            )
            print(
                f"Step {total_steps}/{total_budget} | "
                f"Win rate (rollout): {win_rate:.2f} | "
                f"Loss: {loss:.3f}"
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

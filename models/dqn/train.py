"""
train.py — Training loop for the DQN agent.

Usage:
    python models/dqn/train.py
    python models/dqn/train.py --steps 500000 --checkpoint-every 50000

The script counts total environment steps (not battles/episodes).  Training
continues until total_steps >= steps.  Checkpoints are written to
models/dqn/checkpoints/ every checkpoint_every steps.

Note: --steps counts steps, not battles. At ~50 steps/battle, hitting a
target battle count (e.g. the M2/M3 milestone success criteria, which are
specified in battles) means multiplying by the average steps/battle, not
using the battle count directly.
"""

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np

# Resolve models/gym_client.py regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from gym_client import GymClient  # noqa: E402
from dqn_agent import DQNAgent    # noqa: E402
from replay_buffer import ReplayBuffer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN agent on Pokemon Showdown")
    parser.add_argument(
        "--steps",
        type=int,
        default=100000,
        help="Total environment steps to train for (default: 100000)",
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
    total_budget = args.steps
    checkpoint_every = args.checkpoint_every
    # Auto-scale so short smoke-test runs still produce output
    log_every = min(500, max(1, total_budget // 200))

    checkpoint_dir = Path(__file__).parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    agent = DQNAgent()
    buffer = ReplayBuffer(capacity=10000)
    # DQN's QNetwork trunk is hardcoded to the legacy flat 100-dim vector.
    env = GymClient(structured=False)

    print(f"Starting DQN training: {total_budget} steps (logging every {log_every} episodes)", flush=True)

    total_steps = 0
    episode = 0
    last_checkpoint_step = 0

    # Rolling window for win-rate and loss tracking
    recent_wins: deque[bool] = deque(maxlen=log_every)
    recent_losses: deque[float] = deque(maxlen=log_every)

    try:
        while total_steps < total_budget:
            obs, valid_mask = env.reset()
            done = False
            episode_won = False
            episode += 1

            while not done:
                action = agent.act(obs, valid_mask)
                try:
                    next_obs, reward, done, info, valid_mask = env.step(action)
                except RuntimeError as e:
                    print(f"  [episode {episode}] step error ({e}) — resetting", flush=True)
                    break

                buffer.push(obs, action, reward, next_obs, done)

                loss = agent.learn(buffer)
                if loss is not None:
                    recent_losses.append(loss)

                # Hard sync target network every target_update_freq steps
                if total_steps > 0 and total_steps % agent.target_update_freq == 0:
                    agent.update_target()

                obs = next_obs
                total_steps += 1

                # Checkpoint on step boundary
                if total_steps - last_checkpoint_step >= checkpoint_every:
                    ckpt_path = checkpoint_dir / f"dqn_step_{total_steps}.pt"
                    agent.save(str(ckpt_path))
                    last_checkpoint_step = total_steps
                    print(f"[checkpoint] Saved {ckpt_path}")

                if total_steps >= total_budget:
                    break

            # Episode ended — record outcome and decay exploration
            if done and info.get("winner") == "Gym":
                episode_won = True

            agent.decay_epsilon()
            recent_wins.append(episode_won)

            if episode % log_every == 0:
                win_rate = sum(recent_wins) / len(recent_wins) if recent_wins else 0.0
                avg_loss = (
                    sum(recent_losses) / len(recent_losses) if recent_losses else float("nan")
                )
                print(
                    f"Step {total_steps}/{total_budget} (episode {episode}) | "
                    f"Win rate (last {log_every}): {win_rate:.2f} | "
                    f"Epsilon: {agent.epsilon:.4f} | "
                    f"Loss: {avg_loss:.4f}"
                )

    finally:
        env.close()

    # Final checkpoint
    final_path = checkpoint_dir / f"dqn_step_{total_steps}_final.pt"
    agent.save(str(final_path))
    print(f"\nTraining complete. Final model saved to {final_path}")
    print(f"Total steps: {total_steps} | Episodes: {episode}")


if __name__ == "__main__":
    main()

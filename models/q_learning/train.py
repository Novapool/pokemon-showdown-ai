"""
train.py — Training loop for the tabular Q-learning agent.

Usage:
    python models/q_learning/train.py
    python models/q_learning/train.py --episodes 50000

The script imports GymClient from models/gym_client.py using a sys.path
insertion so it can be run from any working directory.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow `from gym_client import GymClient` to resolve models/gym_client.py
# regardless of current working directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from gym_client import GymClient  # noqa: E402
from q_agent import QAgent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tabular Q-learning agent")
    parser.add_argument(
        "--episodes",
        type=int,
        default=10000,
        help="Number of training episodes (default: 10000)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_episodes = args.episodes
    log_every = min(500, max(1, n_episodes // 10))

    # This tabular baseline discretizes the legacy flat 100-dim vector, not
    # M2's (12, 65) structured tokens.
    env = GymClient(structured=False)
    agent = QAgent()

    print(f"Starting Q-Learning training: {n_episodes} episodes (logging every {log_every})")

    recent_wins: list[bool] = []
    total_wins = 0

    try:
        for episode in range(1, n_episodes + 1):
            obs, valid_mask = env.reset()
            done = False
            episode_won = False

            while not done:
                action = agent.act(obs, valid_mask)
                try:
                    next_obs, reward, done, info, valid_mask = env.step(action)
                except RuntimeError as e:
                    print(f"  [episode {episode}] step error ({e}) — resetting", flush=True)
                    break
                agent.update(obs, action, reward, next_obs, done)
                obs = next_obs

                if done and info.get("winner") == "Gym":
                    episode_won = True

            agent.decay_epsilon()

            recent_wins.append(episode_won)
            if episode_won:
                total_wins += 1

            # Keep only the last log_every results for the rolling window
            if len(recent_wins) > log_every:
                recent_wins.pop(0)

            if episode % log_every == 0:
                win_rate = sum(recent_wins) / len(recent_wins) if recent_wins else 0.0
                print(
                    f"Episode {episode}/{n_episodes} | "
                    f"Win rate: {win_rate:.2f} | "
                    f"Epsilon: {agent.epsilon:.4f} | "
                    f"States: {len(agent.q_table)}"
                )

    finally:
        env.close()

    # Save trained Q-table
    save_path = Path(__file__).parent / "qtable.pkl"
    agent.save(str(save_path))
    print(f"\nTraining complete. Q-table saved to {save_path}")
    print(f"Total episodes: {n_episodes} | Total wins: {total_wins} | States visited: {len(agent.q_table)}")


if __name__ == "__main__":
    main()

"""
evaluate.py — Standalone evaluation script for trained Pokemon Showdown RL agents.

Runs N greedy battles against RandomPlayerAI (the default gym opponent) and
reports win rate to stdout.

Usage:
    python models/evaluate.py --model dqn --checkpoint models/dqn/checkpoints/dqn_step_100000.pt --battles 200
    python models/evaluate.py --model q_learning --checkpoint models/q_learning/qtable.pkl --battles 200
    python models/evaluate.py --model ppo --checkpoint models/ppo/checkpoints/ppo_step_100000.pt --battles 200
"""

import argparse
import sys
from pathlib import Path

# ---- path setup so imports resolve regardless of cwd ----------------------
_MODELS_DIR = Path(__file__).parent
sys.path.insert(0, str(_MODELS_DIR))

from gym_client import GymClient  # noqa: E402 (after sys.path patch)


def _load_agent(model: str, checkpoint: str):
    """Load and return the appropriate agent from checkpoint."""
    if model == "q_learning":
        sys.path.insert(0, str(_MODELS_DIR / "q_learning"))
        from q_agent import QAgent

        agent = QAgent.load(checkpoint)
        agent.epsilon = 0.0  # fully greedy
        return agent

    elif model == "dqn":
        sys.path.insert(0, str(_MODELS_DIR / "dqn"))
        from dqn_agent import DQNAgent

        agent = DQNAgent.load(checkpoint)
        agent.epsilon = 0.0  # fully greedy
        return agent

    elif model == "ppo":
        sys.path.insert(0, str(_MODELS_DIR / "ppo"))
        from ppo_agent import PPOAgent

        agent = PPOAgent.load(checkpoint)
        return agent

    else:
        raise ValueError(f"Unknown model type: {model!r}")


def _run_battles(agent, n_battles: int) -> tuple:
    """Run n_battles greedy episodes; return (wins, total)."""
    # These M1 baselines (q_learning/dqn/ppo) all expect the legacy flat
    # 100-dim vector, not M2's (12, 65) structured tokens.
    env = GymClient(structured=False)
    wins = 0

    try:
        for _ in range(n_battles):
            obs, valid_mask = env.reset()
            done = False
            while not done:
                action = agent.act(obs, valid_mask)
                obs, _reward, done, info, valid_mask = env.step(action)
            if done and info.get("winner") == "Gym":
                wins += 1
    finally:
        env.close()

    return wins, n_battles


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Pokemon Showdown RL agent vs RandomPlayerAI."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["q_learning", "dqn", "ppo"],
        help="Model architecture to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the saved model checkpoint file.",
    )
    parser.add_argument(
        "--battles",
        type=int,
        default=200,
        help="Number of evaluation battles to run (default: 200).",
    )
    args = parser.parse_args()

    agent = _load_agent(args.model, args.checkpoint)
    wins, total = _run_battles(agent, args.battles)
    win_rate = wins / total if total > 0 else 0.0

    print(f"Model: {args.model} | Checkpoint: {args.checkpoint}")
    print(f"Battles: {total}")
    print(f"Win rate vs RandomPlayerAI: {win_rate:.2f} ({wins}/{total})")


if __name__ == "__main__":
    main()

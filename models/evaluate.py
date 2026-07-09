"""
evaluate.py — Standalone evaluation script for trained Pokemon Showdown RL agents.

Runs N greedy battles against RandomPlayerAI (the default gym opponent) and
reports win rate to stdout.

Usage:
    python models/evaluate.py --model dqn --checkpoint models/dqn/checkpoints/dqn_step_100000.pt --battles 200
    python models/evaluate.py --model q_learning --checkpoint models/q_learning/qtable.pkl --battles 200
    python models/evaluate.py --model ppo --checkpoint models/ppo/checkpoints/ppo_step_100000.pt --battles 200

    # M2 verification: a PPO checkpoint trained with --structured
    python models/evaluate.py --model ppo --structured \
        --checkpoint models/ppo/checkpoints/structured/ppo_step_2600000_final.pt --battles 200
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


def _run_battles(agent, n_battles: int, structured: bool = False) -> tuple:
    """Run n_battles greedy episodes; return (wins, total).

    structured=True flattens the (12, 65) M2 observation to match a PPO
    checkpoint trained with train.py --structured. All other baselines
    (q_learning/dqn/ppo without --structured) expect the legacy flat
    100-dim vector.
    """
    env = GymClient(structured=structured)
    wins = 0
    # Auto-scale so short smoke-test runs still produce output (same
    # approach as the train.py scripts' log_every).
    log_every = min(50, max(1, n_battles // 10))

    try:
        for i in range(1, n_battles + 1):
            obs, valid_mask = env.reset()
            if structured:
                obs = obs.reshape(-1)
            done = False
            while not done:
                # PPOAgent.act() returns (action, log_prob, value); QAgent/DQNAgent.act()
                # return a plain int. Handle both without the caller needing to know
                # which model type is loaded.
                act_result = agent.act(obs, valid_mask)
                action = act_result[0] if isinstance(act_result, tuple) else act_result
                obs, _reward, done, info, valid_mask = env.step(action)
                if structured:
                    obs = obs.reshape(-1)
            if done and info.get("winner") == "Gym":
                wins += 1

            if i % log_every == 0 or i == n_battles:
                print(f"Battle {i}/{n_battles} | running win rate: {wins / i:.2f} ({wins}/{i})", flush=True)
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
    parser.add_argument(
        "--structured",
        action="store_true",
        help="Evaluate a PPO checkpoint trained with train.py --structured (M2 verification).",
    )
    args = parser.parse_args()

    if args.structured and args.model != "ppo":
        parser.error("--structured is only meaningful with --model ppo")

    agent = _load_agent(args.model, args.checkpoint)
    wins, total = _run_battles(agent, args.battles, structured=args.structured)
    win_rate = wins / total if total > 0 else 0.0

    print(f"Model: {args.model} | Checkpoint: {args.checkpoint}")
    print(f"Battles: {total}")
    print(f"Win rate vs RandomPlayerAI: {win_rate:.2f} ({wins}/{total})")


if __name__ == "__main__":
    main()

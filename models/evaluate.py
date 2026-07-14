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

    # M3: transformer agent (always structured, unflattened (12,65) obs)
    python models/evaluate.py --model transformer \
        --checkpoint models/transformer/checkpoints/pretrained/transformer_step_2600000_final.pt --battles 200
"""

import argparse
import sys
from pathlib import Path

# ---- path setup so imports resolve regardless of cwd ----------------------
_MODELS_DIR = Path(__file__).parent
sys.path.insert(0, str(_MODELS_DIR))

from gym_client import GymClient  # noqa: E402 (after sys.path patch)
from vec_gym_client import VecGymClient  # noqa: E402


def _load_agent(model: str, checkpoint: str, device: str | None = None):
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

        agent = PPOAgent.load(checkpoint, device=device)
        return agent

    elif model == "transformer":
        # trajectory_buffer.py (models/ppo/) is a module-level import inside
        # transformer_agent.py, so ppo/ must be on sys.path before importing it.
        sys.path.insert(0, str(_MODELS_DIR / "ppo"))
        sys.path.insert(0, str(_MODELS_DIR / "transformer"))
        from transformer_agent import TransformerAgent

        agent = TransformerAgent.load(checkpoint, device=device)
        return agent

    else:
        raise ValueError(f"Unknown model type: {model!r}")


def _run_battles_vec(
    agent, n_battles: int, num_envs: int, structured: bool, flatten: bool,
    opponent: str = "random",
) -> tuple:
    """Run n_battles episodes across num_envs parallel envs; return (wins, total).

    Each env gets a fixed battle quota (n_battles split as evenly as possible)
    so the count is exact and unbiased by battle length. Envs that hit their
    quota keep being stepped (VecGymClient auto-resets) but their further
    episodes aren't counted; the leftover work is bounded by one battle/env.

    Agents with act_batch() (PPO/transformer) get batched inference; the
    q_learning/dqn agents fall back to per-env act() calls — the parallel
    simulation is the speedup either way.
    """
    num_envs = max(1, min(num_envs, n_battles))
    env = VecGymClient(num_envs, structured=structured, opponent=opponent)
    quotas = [n_battles // num_envs] * num_envs
    for i in range(n_battles % num_envs):
        quotas[i] += 1

    completed = [0] * num_envs
    wins = 0
    done_total = 0
    log_every = min(50, max(1, n_battles // 10))
    batched = hasattr(agent, "act_batch")

    def _prep(obs):
        return obs.reshape(num_envs, -1) if (structured and flatten) else obs

    try:
        obs, masks = env.reset_all()
        obs = _prep(obs)
        while any(completed[i] < quotas[i] for i in range(num_envs)):
            if batched:
                actions = agent.act_batch(obs, masks)[0]
            else:
                actions = [agent.act(obs[i], masks[i].tolist()) for i in range(num_envs)]
            obs, _rewards, dones, infos, masks = env.step(actions)
            obs = _prep(obs)
            for i in range(num_envs):
                if "error" in infos[i]:
                    print(f"  [env {i}] error ({infos[i]['error']}) — battle not counted", flush=True)
                    continue
                if dones[i] and completed[i] < quotas[i]:
                    completed[i] += 1
                    done_total += 1
                    if infos[i].get("winner") == "Gym":
                        wins += 1
                    if done_total % log_every == 0 or done_total == n_battles:
                        print(
                            f"Battle {done_total}/{n_battles} | running win rate: "
                            f"{wins / done_total:.2f} ({wins}/{done_total})",
                            flush=True,
                        )
    finally:
        env.close()

    return wins, n_battles


def _run_battles(
    agent, n_battles: int, structured: bool = False, flatten: bool = True,
    opponent: str = "random",
) -> tuple:
    """Run n_battles greedy episodes; return (wins, total).

    structured=True makes GymClient return the (12, 65) M2 observation
    instead of the legacy flat 100-dim vector. flatten=True additionally
    reshapes that to (780,), matching a PPO checkpoint trained with
    train.py --structured. The transformer consumes the (12, 65) observation
    directly, so its caller passes structured=True, flatten=False.
    """
    env = GymClient(structured=structured, opponent=opponent)
    wins = 0
    # Auto-scale so short smoke-test runs still produce output (same
    # approach as the train.py scripts' log_every).
    log_every = min(50, max(1, n_battles // 10))

    try:
        for i in range(1, n_battles + 1):
            obs, valid_mask = env.reset()
            if structured and flatten:
                obs = obs.reshape(-1)
            done = False
            while not done:
                # PPOAgent/TransformerAgent.act() returns (action, log_prob, value);
                # QAgent/DQNAgent.act() return a plain int. Handle both without the
                # caller needing to know which model type is loaded.
                act_result = agent.act(obs, valid_mask)
                action = act_result[0] if isinstance(act_result, tuple) else act_result
                obs, _reward, done, info, valid_mask = env.step(action)
                if structured and flatten:
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
        choices=["q_learning", "dqn", "ppo", "transformer"],
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
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="Number of parallel battle environments (default: 8; 1 = legacy serial path).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        default=None,
        help="Torch device override for ppo/transformer (default: auto-detect).",
    )
    parser.add_argument(
        "--opponent",
        choices=["random", "damagefirst"],
        default="random",
        help="Evaluation opponent (M3.3): RandomPlayerAI or the DamageFirst heuristic.",
    )
    args = parser.parse_args()

    if args.structured and args.model != "ppo":
        parser.error(
            "--structured is only meaningful with --model ppo "
            "(transformer always uses structured, unflattened observations; no flag needed)"
        )

    agent = _load_agent(args.model, args.checkpoint, device=args.device)
    if args.model == "transformer":
        structured, flatten = True, False
    else:
        structured, flatten = args.structured, True

    if args.num_envs > 1:
        wins, total = _run_battles_vec(
            agent, args.battles, args.num_envs, structured=structured, flatten=flatten,
            opponent=args.opponent,
        )
    else:
        wins, total = _run_battles(
            agent, args.battles, structured=structured, flatten=flatten,
            opponent=args.opponent,
        )
    win_rate = wins / total if total > 0 else 0.0

    opponent_name = "DamageFirstAI" if args.opponent == "damagefirst" else "RandomPlayerAI"
    print(f"Model: {args.model} | Checkpoint: {args.checkpoint}")
    print(f"Battles: {total}")
    print(f"Win rate vs {opponent_name}: {win_rate:.2f} ({wins}/{total})")


if __name__ == "__main__":
    main()

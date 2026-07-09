"""
train.py — Rollout-based PPO training loop for Pokemon Showdown.

Usage:
    python models/ppo/train.py
    python models/ppo/train.py --steps 500000 --rollout-steps 512 --checkpoint-every 50000

    # M2 verification: MLP PPO on flattened (12,65)->780 structured obs.
    # Checkpoints land in models/ppo/checkpoints/structured/ so they never
    # collide with the flat-obs (M1) baseline checkpoints.
    python models/ppo/train.py --structured --steps 2600000 --checkpoint-every 250000

The loop collects rollout_steps environment steps per update cycle regardless
of episode boundaries.  When a 'done' signal arrives mid-rollout, the
environment is reset immediately so collection continues seamlessly.

Note: --steps counts steps, not battles. At ~50 steps/battle, hitting a
target battle count (e.g. the M2/M3 milestone success criteria, which are
specified in battles) means multiplying by the average steps/battle, not
using the battle count directly.
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
        "--steps",
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
    parser.add_argument(
        "--structured",
        action="store_true",
        help=(
            "Train on the M2 (12,65)->780 flattened structured observation "
            "instead of the legacy flat 100-dim vector. Checkpoints go to "
            "checkpoints/structured/ to avoid colliding with the M1 baseline."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_budget = args.steps
    rollout_steps = args.rollout_steps
    checkpoint_every = args.checkpoint_every

    checkpoint_dir = Path(__file__).parent / "checkpoints" / ("structured" if args.structured else ".")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    buffer = TrajectoryBuffer()
    env = GymClient(structured=args.structured)

    def _reset():
        o, mask = env.reset()
        return (o.reshape(-1) if args.structured else o), mask

    def _step(action):
        o, r, d, info, mask = env.step(action)
        return (o.reshape(-1) if args.structured else o), r, d, info, mask

    obs, valid_mask = _reset()
    # obs_size is derived from the actual observation rather than hardcoded,
    # so this loop works for both the M1 flat vector (100) and the M2
    # structured flatten (780) without duplicating the PPOAgent class.
    agent = PPOAgent(obs_size=obs.shape[0])

    print(
        f"Starting PPO training: {total_budget} steps | "
        f"obs_mode={'structured' if args.structured else 'flat'} (obs_size={obs.shape[0]}) | "
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
                    next_obs, reward, done, info, valid_mask = _step(action)
                except RuntimeError as e:
                    print(f"  [step {total_steps}] step error ({e}) — resetting", flush=True)
                    obs, valid_mask = _reset()
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
                    obs, valid_mask = _reset()
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

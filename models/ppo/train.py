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
environment is auto-reset so collection continues seamlessly.

Rollouts are collected from --num-envs parallel battle simulations (M3.1):
each env is its own gym_bridge.js subprocess, and inference runs batched over
all envs' observations. --num-envs 1 uses the same code path.

Note: --steps counts steps, not battles. At ~50 steps/battle, hitting a
target battle count (e.g. the M2/M3 milestone success criteria, which are
specified in battles) means multiplying by the average steps/battle, not
using the battle count directly.
"""

import argparse
import sys
from pathlib import Path

# Resolve models/ directory so vec_gym_client and ppo modules are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from vec_gym_client import VecGymClient   # noqa: E402

# ppo_agent and trajectory_buffer live alongside this script
sys.path.insert(0, str(Path(__file__).parent))
from ppo_agent import PPOAgent            # noqa: E402
from trajectory_buffer import TrajectoryBuffer, merge_buffers  # noqa: E402


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
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="Number of parallel battle environments (default: 8)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        default=None,
        help="Torch device override (default: auto-detect cuda > mps > cpu)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_budget = args.steps
    rollout_steps = args.rollout_steps
    checkpoint_every = args.checkpoint_every
    num_envs = args.num_envs

    checkpoint_dir = Path(__file__).parent / "checkpoints" / ("structured" if args.structured else ".")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # One buffer per env — GAE must never cross env streams. merge_buffers()
    # combines them (and normalizes advantages globally) before each update.
    buffers = [TrajectoryBuffer() for _ in range(num_envs)]
    env = VecGymClient(num_envs, structured=args.structured)

    def _flatten(o):
        # The MLP consumes flat vectors: (N, 12, 65) -> (N, 780) in structured
        # mode; flat mode is already (N, 100).
        return o.reshape(num_envs, -1) if args.structured else o

    obs_batch, masks = env.reset_all()
    obs_batch = _flatten(obs_batch)
    # obs_size is derived from the actual observation rather than hardcoded,
    # so this loop works for both the M1 flat vector (100) and the M2
    # structured flatten (780) without duplicating the PPOAgent class.
    agent = PPOAgent(obs_size=obs_batch.shape[1], device=args.device)

    print(
        f"Starting PPO training: {total_budget} steps | "
        f"obs_mode={'structured' if args.structured else 'flat'} (obs_size={obs_batch.shape[1]}) | "
        f"rollout={rollout_steps} steps | "
        f"num_envs={num_envs} | device={agent.device} | "
        f"checkpoint every {checkpoint_every} steps",
        flush=True,
    )

    total_steps = 0
    last_checkpoint_step = 0

    # Episode tracking within each rollout window
    rollout_wins = 0
    rollout_episodes = 0

    try:
        while total_steps < total_budget:
            # ----------------------------------------------------------------
            # Collect one rollout across all envs
            # ----------------------------------------------------------------
            rollout_wins = 0
            rollout_episodes = 0
            steps_this_rollout = 0

            while steps_this_rollout < rollout_steps and total_steps < total_budget:
                actions, log_probs, values = agent.act_batch(obs_batch, masks)
                next_obs, rewards, dones, infos, masks = env.step(actions)
                next_obs = _flatten(next_obs)

                for i in range(num_envs):
                    if "error" in infos[i]:
                        # Env was reset in place by VecGymClient; drop this slot's
                        # transition (same reset-and-continue handling as before).
                        print(
                            f"  [step {total_steps}] env {i} error ({infos[i]['error']}) — reset",
                            flush=True,
                        )
                        continue
                    buffers[i].push(
                        obs_batch[i],
                        int(actions[i]),
                        float(rewards[i]),
                        bool(dones[i]),
                        float(values[i]),
                        float(log_probs[i]),
                    )
                    if dones[i]:
                        rollout_episodes += 1
                        if infos[i].get("winner") == "Gym":
                            rollout_wins += 1

                obs_batch = next_obs
                total_steps += num_envs
                steps_this_rollout += num_envs

                # Checkpoint on step boundary
                if total_steps - last_checkpoint_step >= checkpoint_every:
                    ckpt_path = checkpoint_dir / f"ppo_step_{total_steps}.pt"
                    agent.save(str(ckpt_path))
                    last_checkpoint_step = total_steps
                    print(f"[checkpoint] Saved {ckpt_path}")

            # ----------------------------------------------------------------
            # Compute advantages and update
            # ----------------------------------------------------------------
            if all(len(b) == 0 for b in buffers):
                break

            # Bootstrap each env from the value of its current obs. Where an
            # env's last stored transition was terminal, GAE's not_done factor
            # zeroes the bootstrap, so passing the fresh-episode value is safe.
            _, _, last_values = agent.act_batch(obs_batch, masks)
            for i in range(num_envs):
                if len(buffers[i]) > 0:
                    buffers[i].compute_advantages(
                        last_value=float(last_values[i]), normalize=False
                    )

            loss = agent.update(merge_buffers(buffers))
            for b in buffers:
                b.clear()

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

"""
train.py — Rollout-based PPO training loop for the M3 transformer agent.

Usage:
    # From scratch — checkpoints land in checkpoints/scratch/
    python models/transformer/train.py --steps 2600000 --checkpoint-every 250000

    # Warm-started from the M2.5 BC checkpoint — checkpoints land in
    # checkpoints/pretrained/, so the two runs never collide. Both are
    # required for the M3 comparison (MILESTONES.md).
    python models/transformer/train.py --steps 2600000 --checkpoint-every 250000 \
        --pretrain_checkpoint models/checkpoints/bc_pretrain_gen1ou.pt

    # Resume an interrupted run from its last checkpoint (full agent +
    # optimizer state restored; step count is parsed from the filename).
    # Mutually exclusive with --pretrain_checkpoint (resuming already
    # restores whatever weights the original run started from).
    python models/transformer/train.py --steps 2600000 --checkpoint-every 250000 \
        --resume models/transformer/checkpoints/scratch/transformer_step_500000.pt

Unlike models/ppo/train.py, this loop always uses the M2 structured (12, 65)
observation un-flattened — there is no --structured flag, since the
transformer consumes per-token structure directly and has no other obs mode.

Note: --steps counts environment steps, not battles (~50 steps/battle at the
M2 baseline). See models/ppo/train.py's docstring for the same caveat.
"""

import argparse
import re
import sys
from pathlib import Path

# Resolve models/ directory so gym_client is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
from gym_client import GymClient  # noqa: E402

# trajectory_buffer lives in models/ppo/; transformer_agent.py imports it too,
# so this path must be inserted before transformer_agent is imported below.
sys.path.insert(0, str(Path(__file__).parent.parent / "ppo"))
from trajectory_buffer import TrajectoryBuffer  # noqa: E402

# transformer_agent and transformer_policy live alongside this script
sys.path.insert(0, str(Path(__file__).parent))
from transformer_agent import TransformerAgent  # noqa: E402
from transformer_policy import load_pretrain_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Transformer PPO agent on Pokemon Showdown")
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
        "--pretrain_checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a BC checkpoint (e.g. models/checkpoints/bc_pretrain_gen1ou.pt) "
            "to warm-start the transformer's weights before PPO begins. Omit to "
            "train from scratch."
        ),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint saved by this script (e.g. "
            "checkpoints/scratch/transformer_step_500000.pt) to resume training "
            "from. Restores full agent + optimizer state; the starting step count "
            "is parsed from the filename. Mutually exclusive with "
            "--pretrain_checkpoint (resuming already restores whatever weights "
            "the original run started from)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume and args.pretrain_checkpoint:
        raise ValueError(
            "--resume and --pretrain_checkpoint are mutually exclusive — "
            "resuming already restores the full agent + optimizer state"
        )

    total_budget = args.steps
    rollout_steps = args.rollout_steps
    checkpoint_every = args.checkpoint_every

    buffer = TrajectoryBuffer()
    env = GymClient(structured=True)  # transformer always takes raw (12, 65) tokens

    obs, valid_mask = env.reset()

    if args.resume:
        # Same directory as the checkpoint being resumed, so this run keeps
        # appending to whichever scratch/pretrained sequence it came from.
        checkpoint_dir = Path(args.resume).parent
        agent = TransformerAgent.load(args.resume)
        match = re.search(r"step_(\d+)", Path(args.resume).stem)
        if not match:
            raise ValueError(f"Cannot parse step count from --resume filename: {args.resume}")
        total_steps = int(match.group(1))
        last_checkpoint_step = total_steps
    else:
        # Separate subdir per run mode so a from-scratch and a warm-started run
        # never overwrite each other's checkpoints at the same step count.
        run_mode = "pretrained" if args.pretrain_checkpoint else "scratch"
        checkpoint_dir = Path(__file__).parent / "checkpoints" / run_mode
        agent = TransformerAgent()
        if args.pretrain_checkpoint:
            # Must load into agent.policy, not agent — TransformerAgent composes
            # TransformerPolicy as self.policy, so agent.state_dict() keys are
            # "policy."-prefixed. Passing agent here would silently load 0/N
            # tensors (load_pretrain_checkpoint only warns, never raises).
            load_pretrain_checkpoint(agent.policy, args.pretrain_checkpoint)
        total_steps = 0
        last_checkpoint_step = 0

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Starting Transformer PPO training: {total_budget} steps | "
        f"obs_mode=structured (12,65) | rollout={rollout_steps} steps | "
        f"checkpoint every {checkpoint_every} steps | "
        f"pretrain={args.pretrain_checkpoint or 'none (from scratch)'} | "
        f"resumed_from={args.resume or 'none'} (starting at step {total_steps})",
        flush=True,
    )

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
                    ckpt_path = checkpoint_dir / f"transformer_step_{total_steps}.pt"
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
            loss, kl_early_stop = agent.update(buffer)
            buffer.clear()

            # Linear LR annealing toward 0 over the training budget — late-run
            # updates on a mostly-converged policy shouldn't be as aggressive
            # as early ones. Uses current progress against total_budget, so a
            # --resume run keeps decaying rather than jumping back to full lr.
            frac_remaining = max(0.0, 1.0 - total_steps / total_budget)
            current_lr = agent.lr * frac_remaining
            for param_group in agent.optimizer.param_groups:
                param_group["lr"] = current_lr

            # ----------------------------------------------------------------
            # Log rollout summary
            # ----------------------------------------------------------------
            win_rate = (
                rollout_wins / rollout_episodes if rollout_episodes > 0 else float("nan")
            )
            print(
                f"Step {total_steps}/{total_budget} | "
                f"Win rate (rollout): {win_rate:.2f} | "
                f"Loss: {loss:.3f} | "
                f"LR: {current_lr:.2e}"
                + (" | [kl early-stop]" if kl_early_stop else "")
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

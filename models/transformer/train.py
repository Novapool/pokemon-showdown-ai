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

Rollouts are collected from --num-envs parallel battle simulations (M3.1):
each env is its own gym_bridge.js subprocess, and inference runs batched over
all envs' observations. --num-envs 1 uses the same code path.

Note: --steps counts environment steps, not battles (~50 steps/battle at the
M2 baseline). See models/ppo/train.py's docstring for the same caveat.
"""

import argparse
import re
import sys
from pathlib import Path

# Resolve models/ directory so vec_gym_client is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
from vec_gym_client import VecGymClient  # noqa: E402

# trajectory_buffer lives in models/ppo/; transformer_agent.py imports it too,
# so this path must be inserted before transformer_agent is imported below.
sys.path.insert(0, str(Path(__file__).parent.parent / "ppo"))
from trajectory_buffer import TrajectoryBuffer, merge_buffers  # noqa: E402

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
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="Number of parallel battle environments (default: 8)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help=(
            "Override the checkpoint directory (default: checkpoints/scratch "
            "or checkpoints/pretrained next to this script). Use for runs that "
            "must not overwrite an earlier run's checkpoints at the same step "
            "counts (e.g. the M3.2 decision run vs the original M3 runs)."
        ),
    )
    parser.add_argument(
        "--value-warmup-steps",
        type=int,
        default=0,
        help=(
            "M3.2: freeze embed/encoder/policy_head for the first N steps so "
            "the (randomly initialized) value head fits the BC policy before "
            "full PPO gradients flow through the shared encoder. 0 = off."
        ),
    )
    parser.add_argument(
        "--bc-anchor",
        type=str,
        default=None,
        help=(
            "M3.2: path to a BC checkpoint to use as a frozen KL anchor — "
            "adds coef × KL(π‖π_BC) to the PPO loss. Usually the same file as "
            "--pretrain_checkpoint. Works with --resume (re-attached each run)."
        ),
    )
    parser.add_argument(
        "--bc-anchor-coef",
        type=float,
        default=0.05,
        help="Initial KL-anchor coefficient; annealed to 0 with the LR schedule (default: 0.05)",
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
    if args.resume and args.pretrain_checkpoint:
        raise ValueError(
            "--resume and --pretrain_checkpoint are mutually exclusive — "
            "resuming already restores the full agent + optimizer state"
        )

    total_budget = args.steps
    rollout_steps = args.rollout_steps
    checkpoint_every = args.checkpoint_every
    num_envs = args.num_envs

    # One buffer per env — GAE must never cross env streams. merge_buffers()
    # combines them (and normalizes advantages globally) before each update.
    buffers = [TrajectoryBuffer() for _ in range(num_envs)]
    env = VecGymClient(num_envs, structured=True)  # transformer always takes raw (12, 65) tokens

    obs_batch, masks = env.reset_all()

    if args.resume:
        # Same directory as the checkpoint being resumed, so this run keeps
        # appending to whichever scratch/pretrained sequence it came from.
        checkpoint_dir = Path(args.resume).parent
        agent = TransformerAgent.load(args.resume, device=args.device)
        match = re.search(r"step_(\d+)", Path(args.resume).stem)
        if not match:
            raise ValueError(f"Cannot parse step count from --resume filename: {args.resume}")
        total_steps = int(match.group(1))
        last_checkpoint_step = total_steps
    else:
        # Separate subdir per run mode so a from-scratch and a warm-started run
        # never overwrite each other's checkpoints at the same step count.
        run_mode = "pretrained" if args.pretrain_checkpoint else "scratch"
        checkpoint_dir = (
            Path(args.checkpoint_dir)
            if args.checkpoint_dir
            else Path(__file__).parent / "checkpoints" / run_mode
        )
        agent = TransformerAgent(device=args.device)
        if args.pretrain_checkpoint:
            # Must load into agent.policy, not agent — TransformerAgent composes
            # TransformerPolicy as self.policy, so agent.state_dict() keys are
            # "policy."-prefixed. Passing agent here would silently load 0/N
            # tensors (load_pretrain_checkpoint only warns, never raises).
            load_pretrain_checkpoint(agent.policy, args.pretrain_checkpoint)
        total_steps = 0
        last_checkpoint_step = 0

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # M3.2: BC KL-anchor (never persisted in checkpoints, so re-attach on
    # every run, resumed or not) and value-head warmup (keyed off absolute
    # total_steps, so a --resume past the threshold never re-freezes).
    if args.bc_anchor:
        agent.set_bc_anchor(args.bc_anchor, args.bc_anchor_coef)
    warmup_active = 0 <= total_steps < args.value_warmup_steps
    if warmup_active:
        agent.set_policy_frozen(True)

    print(
        f"Starting Transformer PPO training: {total_budget} steps | "
        f"obs_mode=structured (12,65) | rollout={rollout_steps} steps | "
        f"num_envs={num_envs} | device={agent.device} | "
        f"checkpoint every {checkpoint_every} steps | "
        f"pretrain={args.pretrain_checkpoint or 'none (from scratch)'} | "
        f"value_warmup={args.value_warmup_steps or 'off'} | "
        f"bc_anchor={args.bc_anchor or 'off'}"
        + (f" (coef {args.bc_anchor_coef})" if args.bc_anchor else "")
        + f" | resumed_from={args.resume or 'none'} (starting at step {total_steps})",
        flush=True,
    )

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
                # The masks in force when the actions were chosen — stored per
                # step so PPO updates re-mask with real legality (M3.2).
                step_masks = masks
                next_obs, rewards, dones, infos, masks = env.step(actions)

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
                        valid_mask=step_masks[i],
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
                    ckpt_path = checkpoint_dir / f"transformer_step_{total_steps}.pt"
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

            loss, kl_early_stop = agent.update(merge_buffers(buffers))
            for b in buffers:
                b.clear()

            # End of value warmup: unfreeze once total_steps crosses the
            # threshold (checked per rollout; the exact boundary step doesn't
            # matter at these scales).
            if warmup_active and total_steps >= args.value_warmup_steps:
                agent.set_policy_frozen(False)
                warmup_active = False
                print(f"[value-warmup] complete at step {total_steps} — policy unfrozen", flush=True)

            # Linear LR annealing toward 0 over the training budget — late-run
            # updates on a mostly-converged policy shouldn't be as aggressive
            # as early ones. Uses current progress against total_budget, so a
            # --resume run keeps decaying rather than jumping back to full lr.
            frac_remaining = max(0.0, 1.0 - total_steps / total_budget)
            current_lr = agent.lr * frac_remaining
            for param_group in agent.optimizer.param_groups:
                param_group["lr"] = current_lr

            # The BC anchor anneals with the same schedule: strong early (when
            # updates are most able to wreck the BC policy), fading to zero.
            if args.bc_anchor:
                agent.bc_anchor_coef = args.bc_anchor_coef * frac_remaining

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
                + (f" | anchor: {agent.bc_anchor_coef:.3f}" if args.bc_anchor else "")
                + (" | [value-warmup]" if warmup_active else "")
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

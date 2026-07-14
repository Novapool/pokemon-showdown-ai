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
import re
import sys
from pathlib import Path

import numpy as np

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
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help=(
            "Override the checkpoint directory (default: checkpoints/ or "
            "checkpoints/structured next to this script). Use for runs that "
            "must not overwrite an earlier run's checkpoints at the same step "
            "counts (e.g. a self-play run vs the M2 baseline)."
        ),
    )
    parser.add_argument(
        "--opponent",
        choices=["random", "damagefirst", "selfplay"],
        default="random",
        help=(
            "M3.3: who sits in the p2 seat. 'random' = RandomPlayerAI (legacy), "
            "'damagefirst' = highest-base-power heuristic, 'selfplay' = frozen "
            "past checkpoints of this agent (see --selfplay-pool)."
        ),
    )
    parser.add_argument(
        "--selfplay-pool",
        type=str,
        default=None,
        help=(
            "Directory of .pt checkpoints to sample self-play opponents from "
            "(default: this run's own checkpoint directory). Each rollout picks "
            "the newest checkpoint 50%% of the time, else uniform over the pool; "
            "until any checkpoint exists the opponent is a frozen copy of the "
            "current policy."
        ),
    )
    return parser.parse_args()


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"step_(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def _sample_opponent(pool_dir: Path, agent: PPOAgent, device, rng) -> PPOAgent:
    """Pick a frozen self-play opponent for one rollout.

    50% the newest pool checkpoint, 50% uniform over the whole pool. Before
    any checkpoint exists, a frozen copy of the current policy is used.
    """
    checkpoints = sorted(pool_dir.glob("ppo_step_*.pt"), key=_checkpoint_step)
    if checkpoints:
        path = checkpoints[-1] if rng.random() < 0.5 else rng.choice(checkpoints)
        opponent = PPOAgent.load(str(path), device=device)
    else:
        opponent = PPOAgent(**agent._hparams, device=device)
        opponent.load_state_dict(agent.state_dict())
    opponent.eval()  # frozen: act_batch is already no_grad
    return opponent


def _push_pending(buffer: TrajectoryBuffer, pending: dict, done: bool) -> None:
    buffer.push(
        pending["obs"],
        pending["action"],
        pending["reward"],
        done,
        pending["value"],
        pending["log_prob"],
        valid_mask=pending["mask"],
    )


def main() -> None:
    args = parse_args()
    total_budget = args.steps
    rollout_steps = args.rollout_steps
    checkpoint_every = args.checkpoint_every
    num_envs = args.num_envs

    checkpoint_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else Path(__file__).parent / "checkpoints" / ("structured" if args.structured else ".")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    selfplay = args.opponent == "selfplay"

    # One buffer per env — GAE must never cross env streams. merge_buffers()
    # combines them (and normalizes advantages globally) before each update.
    buffers = [TrajectoryBuffer() for _ in range(num_envs)]
    env = VecGymClient(
        num_envs,
        structured=args.structured,
        opponent="random" if selfplay else args.opponent,
        selfplay=selfplay,
    )

    def _flatten(o):
        # The MLP consumes flat vectors: (N, 12, 65) -> (N, 780) in structured
        # mode; flat mode is already (N, 100).
        return o.reshape(num_envs, -1) if args.structured else o

    def _flatten_seat(state):
        state["obs"] = _flatten(state["obs"])
        return state

    if selfplay:
        p1_state, p2_state = env.reset_all_dual()
        _flatten_seat(p1_state)
        _flatten_seat(p2_state)
        # Open p1 transitions, one per env: a transition closes at p1's NEXT
        # decision point (or terminal), accumulating rewards from any
        # opponent-only steps (e.g. p2 force-switches) in between.
        pending: list = [None] * num_envs
        selfplay_rng = np.random.default_rng()
        obs_size = p1_state["obs"].shape[1]
    else:
        obs_batch, masks = env.reset_all()
        obs_batch = _flatten(obs_batch)
        obs_size = obs_batch.shape[1]

    # obs_size is derived from the actual observation rather than hardcoded,
    # so this loop works for both the M1 flat vector (100) and the M2
    # structured flatten (780) without duplicating the PPOAgent class.
    agent = PPOAgent(obs_size=obs_size, device=args.device)
    # Self-play opponents come from this run's own checkpoints unless a pool
    # is given explicitly.
    pool_dir = Path(args.selfplay_pool) if args.selfplay_pool else checkpoint_dir

    print(
        f"Starting PPO training: {total_budget} steps | "
        f"obs_mode={'structured' if args.structured else 'flat'} (obs_size={obs_size}) | "
        f"rollout={rollout_steps} steps | "
        f"num_envs={num_envs} | device={agent.device} | "
        f"opponent={args.opponent}"
        + (f" (pool: {pool_dir})" if selfplay else "")
        + " | "
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

            if selfplay:
                # One frozen opponent per rollout, shared by all envs so its
                # inference stays batched. Re-sampled every rollout.
                opponent_agent = _sample_opponent(pool_dir, agent, args.device, selfplay_rng)

                while steps_this_rollout < rollout_steps and total_steps < total_budget:
                    # p1 (the learner): a new decision point closes the
                    # previous pending transition, whose reward has accumulated
                    # any opponent-only steps since.
                    a1: list = [None] * num_envs
                    p1_idx = np.flatnonzero(p1_state["needs"])
                    if len(p1_idx):
                        acts, lps, vals = agent.act_batch(
                            p1_state["obs"][p1_idx], p1_state["mask"][p1_idx]
                        )
                        for k, i in enumerate(p1_idx):
                            if pending[i] is not None:
                                _push_pending(buffers[i], pending[i], done=False)
                            pending[i] = {
                                "obs": p1_state["obs"][i].copy(),
                                "action": int(acts[k]),
                                "log_prob": float(lps[k]),
                                "value": float(vals[k]),
                                "mask": p1_state["mask"][i].copy(),
                                "reward": 0.0,
                            }
                            a1[i] = int(acts[k])
                        total_steps += len(p1_idx)
                        steps_this_rollout += len(p1_idx)

                    # p2 (frozen opponent)
                    a2: list = [None] * num_envs
                    p2_idx = np.flatnonzero(p2_state["needs"])
                    if len(p2_idx):
                        opp_acts, _, _ = opponent_agent.act_batch(
                            p2_state["obs"][p2_idx], p2_state["mask"][p2_idx]
                        )
                        for k, i in enumerate(p2_idx):
                            a2[i] = int(opp_acts[k])

                    p1_state, p2_state, rewards, dones, infos = env.step_dual(a1, a2)
                    _flatten_seat(p1_state)
                    _flatten_seat(p2_state)

                    for i in range(num_envs):
                        if "error" in infos[i]:
                            print(
                                f"  [step {total_steps}] env {i} error ({infos[i]['error']}) — reset",
                                flush=True,
                            )
                            pending[i] = None
                            continue
                        if pending[i] is not None:
                            pending[i]["reward"] += float(rewards[i])
                        if dones[i]:
                            if pending[i] is not None:
                                _push_pending(buffers[i], pending[i], done=True)
                                pending[i] = None
                            rollout_episodes += 1
                            if infos[i].get("winner") == "Gym":
                                rollout_wins += 1

                    # Checkpoint on step boundary
                    if total_steps - last_checkpoint_step >= checkpoint_every:
                        ckpt_path = checkpoint_dir / f"ppo_step_{total_steps}.pt"
                        agent.save(str(ckpt_path))
                        last_checkpoint_step = total_steps
                        print(f"[checkpoint] Saved {ckpt_path}")

                # Close any still-open transitions so the buffers are complete;
                # their bootstrap comes from the current obs below.
                for i in range(num_envs):
                    if pending[i] is not None:
                        _push_pending(buffers[i], pending[i], done=False)
                        pending[i] = None
                bootstrap_obs, bootstrap_masks = p1_state["obs"], p1_state["mask"]

            else:
                while steps_this_rollout < rollout_steps and total_steps < total_budget:
                    actions, log_probs, values = agent.act_batch(obs_batch, masks)
                    # The masks in force when the actions were chosen — stored per
                    # step so PPO updates re-mask with real legality (M3.2).
                    step_masks = masks
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
                        ckpt_path = checkpoint_dir / f"ppo_step_{total_steps}.pt"
                        agent.save(str(ckpt_path))
                        last_checkpoint_step = total_steps
                        print(f"[checkpoint] Saved {ckpt_path}")

                bootstrap_obs, bootstrap_masks = obs_batch, masks

            # ----------------------------------------------------------------
            # Compute advantages and update
            # ----------------------------------------------------------------
            if all(len(b) == 0 for b in buffers):
                break

            # Bootstrap each env from the value of its current obs. Where an
            # env's last stored transition was terminal, GAE's not_done factor
            # zeroes the bootstrap, so passing the fresh-episode value is safe.
            _, _, last_values = agent.act_batch(bootstrap_obs, bootstrap_masks)
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

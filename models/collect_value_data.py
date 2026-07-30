"""
collect_value_data.py — MCTS value-target collection (M8 Phase 2).

Plays games with tuned MCTS driving the searching seat, recording, at every
decision point the searcher actually made, the state it saw plus everything
needed to build an AlphaZero-style value target for it:

  obs         the observation, already sliced to the checkpoint's obs_size
  visits      root visit counts of the search that chose the action (9,)
  root_value  the search's own root Q (mean backup return), NaN when no
              search ran at that decision (forced/locked choices)
  reward      shaped gym reward accumulated from this decision to the next,
              in the SEARCHING SEAT's perspective (p2 rewards are negated)
  outcome     +1 / -1 / 0 for the searching seat's final game result

Targets themselves are computed at training time (`value_finetune.py`), so a
dataset can be re-used for outcome / discounted-MC / root-Q targets without
recollecting.

The opponent is chosen with `--opponent`:

  checkpoint    (default) dual-seat self-play against a frozen policy
                checkpoint, seats alternating per game so the dataset is
                seat-balanced (p2 rewards negated into the searcher's frame).
  random        single-seat against the bridge's scripted bot. The searcher is
  damagefirst   always p1, so rewards need no sign flip. `damagefirst` matches
                the distribution the Criterion C A/B is actually judged on —
                the M8 Phase 2 dataset was collected against a frozen policy
                but evaluated against DamageFirst, which is one of the two
                standing explanations for why its calibration gain did not
                transfer to play strength.

Shards are written as compressed .npz every `--shard-games` games, so a
killed run keeps everything it had already flushed; re-running the same
command resumes from the games already on disk.

Example
-------
    python models/collect_value_data.py \
        --checkpoint models/ppo/checkpoints/v3/ppo_step_5000002_final.pt \
        --obs-v3 --games 1000 --workers 4 \
        --out-dir data/value_targets/m8_v3

    # against the bot the A/B is judged on
    python models/collect_value_data.py --opponent damagefirst \
        --checkpoint models/ppo/checkpoints/v3/ppo_step_5000002_final.pt \
        --obs-v3 --games 2000 --workers 10 \
        --out-dir data/value_targets/m8_v3_df
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppo"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcts"))

from gym_client import GymClient, slice_structured_obs  # noqa: E402


def _shard_paths(out_dir: str, worker: int) -> list:
    return sorted(glob.glob(os.path.join(out_dir, f"w{worker:02d}-shard-*.npz")))


def _games_on_disk(out_dir: str, worker: int) -> int:
    """Number of complete games this worker has already flushed."""
    total = 0
    for path in _shard_paths(out_dir, worker):
        try:
            with np.load(path) as data:
                total += int(data["n_games"])
        except Exception as err:  # corrupt/partial shard from a hard kill
            print(f"[w{worker}] ignoring unreadable shard {path}: {err}", flush=True)
    return total


class _ShardWriter:
    """Accumulates per-decision rows and flushes them as .npz shards."""

    FIELDS = ("obs", "visits", "root_value", "reward", "outcome", "seat",
              "step_idx", "game_id")

    def __init__(self, out_dir: str, worker: int, shard_games: int, start_index: int):
        self.out_dir = out_dir
        self.worker = worker
        self.shard_games = shard_games
        self.index = start_index
        self.rows = {f: [] for f in self.FIELDS}
        self.games = 0

    def add_game(self, rows: dict) -> None:
        for f in self.FIELDS:
            self.rows[f].extend(rows[f])
        self.games += 1
        if self.games >= self.shard_games:
            self.flush()

    def flush(self) -> None:
        if self.games == 0:
            return
        path = os.path.join(self.out_dir, f"w{self.worker:02d}-shard-{self.index:04d}.npz")
        np.savez_compressed(
            path,
            obs=np.asarray(self.rows["obs"], dtype=np.float32),
            visits=np.asarray(self.rows["visits"], dtype=np.int32),
            root_value=np.asarray(self.rows["root_value"], dtype=np.float32),
            reward=np.asarray(self.rows["reward"], dtype=np.float32),
            outcome=np.asarray(self.rows["outcome"], dtype=np.float32),
            seat=np.asarray(self.rows["seat"], dtype=np.int8),
            step_idx=np.asarray(self.rows["step_idx"], dtype=np.int32),
            game_id=np.asarray(self.rows["game_id"], dtype=np.int32),
            n_games=np.int32(self.games),
        )
        print(f"[w{self.worker}] wrote {path} ({len(self.rows['obs'])} decisions, "
              f"{self.games} games)", flush=True)
        self.index += 1
        self.rows = {f: [] for f in self.FIELDS}
        self.games = 0


def _play_game(env, mcts, opponent_agent, seat: str, game_id: int) -> dict:
    """One self-play game; returns the searching seat's per-decision rows."""
    opp_seat = "p2" if seat == "p1" else "p1"
    reward_sign = 1.0 if seat == "p1" else -1.0
    opp_obs_size = opponent_agent._hparams["obs_size"]

    rows = {f: [] for f in _ShardWriter.FIELDS}
    searched = 0

    def opp_action(state):
        obs = slice_structured_obs(state["obs"].reshape(1, -1), opp_obs_size)
        return int(opponent_agent.act_batch(obs, state["mask"][None, :])[0][0])

    states = dict(zip(("p1", "p2"), env.reset_dual()))
    done = False
    info = {}
    step_idx = 0
    while not done:
        actions = {"p1": None, "p2": None}
        if states[seat]["needs"]:
            obs_flat = states[seat]["obs"].reshape(-1)
            actions[seat] = mcts.act(env, obs_flat, states[seat]["mask"])
            rows["obs"].append(slice_structured_obs(
                obs_flat[None, :], mcts.agent._hparams["obs_size"])[0])
            rows["visits"].append(np.asarray(mcts.last_root_visits, dtype=np.int32))
            rows["root_value"].append(mcts.last_root_value)
            rows["reward"].append(0.0)
            rows["seat"].append(0 if seat == "p1" else 1)
            rows["step_idx"].append(step_idx)
            rows["game_id"].append(game_id)
            step_idx += 1
            if mcts.last_search_sims > 0:
                searched += 1
        if states[opp_seat]["needs"]:
            actions[opp_seat] = opp_action(states[opp_seat])

        p1_state, p2_state, reward, done, info = env.step_dual(actions["p1"], actions["p2"])
        states = {"p1": p1_state, "p2": p2_state}
        # Rewards earned between our decisions (including from opponent-only
        # force-switch steps) belong to our most recent decision — the same
        # pending-transition accounting the PPO trainer uses.
        if rows["reward"]:
            rows["reward"][-1] += reward_sign * reward

    winner_name = "Gym" if seat == "p1" else "Opponent"
    winner = info.get("winner")
    if winner == winner_name:
        outcome = 1.0
    elif winner is None:
        outcome = 0.0
    else:
        outcome = -1.0
    rows["outcome"] = [outcome] * len(rows["obs"])
    rows["_searched"] = searched
    rows["_outcome"] = outcome
    return rows


def _play_game_scripted(env, mcts, game_id: int) -> dict:
    """One game vs a scripted bot opponent; returns the searcher's rows.

    Single-seat mode: the bridge drives the opponent internally, so the
    searcher is always p1 and every reward already arrives in its perspective
    (no seat alternation, no sign flip). This is deliberately the same
    configuration `evaluate.py --model mcts --opponent damagefirst` measures,
    which is the point of collecting this way — the dual-seat path trains
    against a frozen policy but the Criterion C A/B is judged against a bot.
    """
    rows = {f: [] for f in _ShardWriter.FIELDS}
    searched = 0

    obs, mask = env.reset()
    obs = obs.reshape(-1)
    done = False
    info = {}
    step_idx = 0
    while not done:
        rows["obs"].append(slice_structured_obs(
            obs[None, :], mcts.agent._hparams["obs_size"])[0])
        action = mcts.act(env, obs, mask)
        rows["visits"].append(np.asarray(mcts.last_root_visits, dtype=np.int32))
        rows["root_value"].append(mcts.last_root_value)
        rows["reward"].append(0.0)
        rows["seat"].append(0)
        rows["step_idx"].append(step_idx)
        rows["game_id"].append(game_id)
        step_idx += 1
        if mcts.last_search_sims > 0:
            searched += 1

        obs, reward, done, info, mask = env.step(action)
        obs = obs.reshape(-1)
        # Same pending-transition accounting as the dual-seat path: reward
        # earned on the way to the next decision belongs to this one.
        rows["reward"][-1] += reward

    winner = info.get("winner")
    outcome = 1.0 if winner == "Gym" else 0.0 if winner is None else -1.0
    rows["outcome"] = [outcome] * len(rows["obs"])
    rows["_searched"] = searched
    rows["_outcome"] = outcome
    return rows


def _worker(worker_id: int, n_games: int, args: argparse.Namespace) -> None:
    import torch

    torch.set_num_threads(1)
    from mcts_agent import MCTSAgent
    from ppo_agent import PPOAgent

    already = _games_on_disk(args.out_dir, worker_id) if args.resume else 0
    if already >= n_games:
        print(f"[w{worker_id}] {already}/{n_games} games already on disk — nothing to do",
              flush=True)
        return
    start_shard = len(_shard_paths(args.out_dir, worker_id))

    agent = PPOAgent.load(args.checkpoint, device=args.device)
    agent.eval()
    scripted = args.opponent != "checkpoint"
    if scripted:
        opponent_agent = None
    elif args.opponent_checkpoint:
        opponent_agent = PPOAgent.load(args.opponent_checkpoint, device=args.device)
        opponent_agent.eval()
    else:
        opponent_agent = agent

    seed = args.seed + 1000 * worker_id
    searchers = {
        s: MCTSAgent(
            agent,
            n_sims=args.sims,
            n_determinizations=args.determinizations,
            c_puct=args.c_puct,
            seed=seed + (0 if s == "p1" else 1),
            seat=s,
            opp_sampler=args.opp_sampler,
            has_opp_head=agent.has_opp_head,
        )
        for s in ("p1", "p2")
    }

    if scripted:
        # Single-seat: the bridge plays the scripted bot, searcher is always p1.
        env = GymClient(
            structured=True, opponent=args.opponent,
            obs_v2=args.obs_v2, obs_v3=args.obs_v3, obs_v3_extended=args.obs_v3_extended,
        )
    else:
        env = GymClient(
            structured=True, selfplay=True,
            obs_v2=args.obs_v2, obs_v3=args.obs_v3, obs_v3_extended=args.obs_v3_extended,
        )
    writer = _ShardWriter(args.out_dir, worker_id, args.shard_games, start_shard)
    wins = 0
    t0 = time.perf_counter()
    try:
        for g in range(already, n_games):
            # Alternate seats so the dataset is seat-balanced. Scripted
            # opponents are single-seat only, so the searcher is always p1.
            seat = "p1" if (scripted or g % 2 == 0) else "p2"
            # game_id is namespaced by worker so shards from parallel workers
            # merge without collisions.
            game_id = worker_id * 1_000_000 + g
            if scripted:
                rows = _play_game_scripted(env, searchers["p1"], game_id)
            else:
                rows = _play_game(env, searchers[seat], opponent_agent, seat, game_id)
            wins += rows["_outcome"] > 0
            writer.add_game(rows)
            played = g - already + 1
            if played % max(1, args.log_every) == 0:
                rate = played / (time.perf_counter() - t0)
                print(f"[w{worker_id}] game {g + 1}/{n_games} | seat {seat} | "
                      f"decisions {len(rows['obs'])} (searched {rows['_searched']}) | "
                      f"searcher win rate {wins / played:.2f} | {rate * 3600:.0f} games/h",
                      flush=True)
        writer.flush()
    finally:
        writer.flush()
        env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Collect MCTS self-play value targets (M8 Phase 2)",
    )
    parser.add_argument("--checkpoint", required=True,
                        help="PPO checkpoint the search runs over (the searching seat)")
    parser.add_argument("--opponent", choices=["checkpoint", "random", "damagefirst"],
                        default="checkpoint",
                        help="who the searcher plays. 'checkpoint' (default) is "
                             "dual-seat self-play vs --opponent-checkpoint with "
                             "alternating seats; 'random'/'damagefirst' use the "
                             "single-seat bridge bot, searcher always p1 — matches "
                             "the distribution the Criterion C A/B is judged on")
    parser.add_argument("--opponent-checkpoint", default=None,
                        help="frozen raw-policy opponent (default: the same checkpoint). "
                             "Only used with --opponent checkpoint")
    parser.add_argument("--games", type=int, default=1000, help="total games to collect")
    parser.add_argument("--out-dir", required=True, help="directory for .npz shards")
    parser.add_argument("--shard-games", type=int, default=25,
                        help="flush a shard every N games (kill-safety granularity)")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel collector processes (each hosts its own bridge)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="ignore shards already in --out-dir instead of resuming")
    parser.add_argument("--sims", type=int, default=100, help="MCTS simulation budget")
    parser.add_argument("--determinizations", type=int, default=1)
    parser.add_argument("--c-puct", type=float, default=0.5)
    parser.add_argument("--opp-sampler", choices=["policy", "head"], default="policy")
    parser.add_argument("--obs-v2", action="store_true")
    parser.add_argument("--obs-v3", action="store_true")
    parser.add_argument("--obs-v3-extended", action="store_true")
    parser.add_argument("--device", default="cpu",
                        help="torch device for the nets (cpu is fastest at batch 1)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    if sum([args.obs_v2, args.obs_v3, args.obs_v3_extended]) > 1:
        parser.error("--obs-v2, --obs-v3 and --obs-v3-extended are mutually exclusive")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.opponent != "checkpoint" and args.opponent_checkpoint:
        parser.error("--opponent-checkpoint only applies to --opponent checkpoint")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "meta.json"), "w") as fh:
        json.dump({
            "checkpoint": args.checkpoint,
            "opponent": args.opponent,
            "opponent_checkpoint": (None if args.opponent != "checkpoint"
                                    else args.opponent_checkpoint or args.checkpoint),
            "seats": "p1 only" if args.opponent != "checkpoint" else "alternating",
            "games": args.games,
            "sims": args.sims,
            "determinizations": args.determinizations,
            "c_puct": args.c_puct,
            "opp_sampler": args.opp_sampler,
            "obs_mode": ("v3-extended" if args.obs_v3_extended else
                         "v3" if args.obs_v3 else "v2" if args.obs_v2 else "v1"),
            "seed": args.seed,
        }, fh, indent=2)

    # Games are split evenly; each worker owns its own shard namespace, so
    # resuming is per-worker and needs no cross-process coordination.
    per_worker = [args.games // args.workers] * args.workers
    for i in range(args.games % args.workers):
        per_worker[i] += 1

    if args.workers == 1:
        _worker(0, per_worker[0], args)
        return

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_worker, args=(i, per_worker[i], args), daemon=False)
        for i in range(args.workers)
    ]
    for p in procs:
        p.start()
    failed = 0
    for p in procs:
        p.join()
        failed += p.exitcode != 0
    if failed:
        raise SystemExit(f"{failed}/{args.workers} collector workers exited non-zero")


if __name__ == "__main__":
    main()

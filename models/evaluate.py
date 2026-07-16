"""
evaluate.py — Standalone evaluation script for trained Pokemon Showdown RL agents.

Runs N battles against RandomPlayerAI (default) or DamageFirstAI
(--opponent damagefirst) and reports win rate to stdout. Battles run across
--num-envs parallel envs (default 8).

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
from typing import NamedTuple

import numpy as np

# ---- path setup so imports resolve regardless of cwd ----------------------
_MODELS_DIR = Path(__file__).parent
sys.path.insert(0, str(_MODELS_DIR))

from gym_client import GymClient, slice_structured_obs  # noqa: E402 (after sys.path patch)
from vec_gym_client import VecGymClient  # noqa: E402


class BattleResult(NamedTuple):
    """Result of a _run_battles* call. opp_correct/opp_total are 0 for the
    h2h/mcts variants, which don't track opp-head prediction accuracy."""
    wins: int
    total: int
    opp_correct: int = 0
    opp_total: int = 0


def _opp_head_argmax(agent, obs_batch: np.ndarray) -> np.ndarray:
    """Top-1 opponent-action predictions from the opp head for a batch of obs.

    obs_batch is (N, d) in the ENV's obs size; it is sliced down to the agent's
    obs size when the agent is a narrower (e.g. v1) checkpoint, mirroring how
    act_batch is fed. Returns an int64 array of shape (N,).
    """
    import torch

    obs = np.asarray(obs_batch, dtype=np.float32)
    obs = slice_structured_obs(obs, agent._hparams["obs_size"])
    obs_t = torch.as_tensor(obs, dtype=torch.float32).to(agent.device)
    with torch.no_grad():
        features = agent.trunk(obs_t)
        opp_logits = agent.opp_head(features)
    return opp_logits.argmax(dim=-1).cpu().numpy()


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
    opponent: str = "random", obs_v2: bool = False, track_opp: bool = False,
) -> BattleResult:
    """Run n_battles episodes across num_envs parallel envs; return
    (wins, total, opp_correct, opp_total).

    Each env gets a fixed battle quota (n_battles split as evenly as possible)
    so the count is exact and unbiased by battle length. Envs that hit their
    quota keep being stepped (VecGymClient auto-resets) but their further
    episodes aren't counted; the leftover work is bounded by one battle/env.

    Agents with act_batch() (PPO/transformer) get batched inference; the
    q_learning/dqn agents fall back to per-env act() calls — the parallel
    simulation is the speedup either way.
    """
    num_envs = max(1, min(num_envs, n_battles))
    env = VecGymClient(num_envs, structured=structured, opponent=opponent, obs_v2=obs_v2)
    quotas = [n_battles // num_envs] * num_envs
    for i in range(n_battles % num_envs):
        quotas[i] += 1

    completed = [0] * num_envs
    wins = 0
    done_total = 0
    opp_correct = 0
    opp_total = 0
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
            # Opp-head prediction is on the pre-step obs the agent just acted on;
            # the ground-truth label arrives in the resulting step's info.
            opp_preds = _opp_head_argmax(agent, obs) if track_opp else None
            obs, _rewards, dones, infos, masks = env.step(actions)
            obs = _prep(obs)
            for i in range(num_envs):
                if "error" in infos[i]:
                    print(f"  [env {i}] error ({infos[i]['error']}) — battle not counted", flush=True)
                    continue
                if track_opp:
                    label = infos[i].get("opp_action", -1)
                    if label >= 0:
                        opp_total += 1
                        if int(opp_preds[i]) == int(label):
                            opp_correct += 1
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

    return BattleResult(wins, n_battles, opp_correct, opp_total)


def _run_battles_h2h(
    agent, opponent_agent, n_battles: int, num_envs: int, structured: bool,
    obs_v2: bool = False,
) -> BattleResult:
    """Head-to-head (M3.3): agent (seat p1) vs a second checkpoint (seat p2).

    Runs the bridge in dual-seat self-play mode; both sides act through
    act_batch() on their own seat's revealed-info observation. Wins are
    counted from p1's perspective. Battle quotas per env keep the count
    exact, same as _run_battles_vec.

    Cross-schema (M3.4): with obs_v2=True the env produces (12, 77) schema-v2
    observations; each seat's view is sliced down per token to its own
    checkpoint's obs size, so a v2 agent can play a v1 agent directly.
    """
    num_envs = max(1, min(num_envs, n_battles))
    env = VecGymClient(num_envs, structured=structured, selfplay=True, obs_v2=obs_v2)
    quotas = [n_battles // num_envs] * num_envs
    for i in range(n_battles % num_envs):
        quotas[i] += 1

    completed = [0] * num_envs
    wins = 0
    done_total = 0
    log_every = min(50, max(1, n_battles // 10))

    def _prep(state):
        if structured:
            state["obs"] = state["obs"].reshape(num_envs, -1)
        return state

    def _seat_actions(seat_agent, state):
        actions = [None] * num_envs
        idx = np.flatnonzero(state["needs"])
        if len(idx):
            obs = state["obs"][idx]
            if structured:
                obs = slice_structured_obs(obs, seat_agent._hparams["obs_size"])
            acts = seat_agent.act_batch(obs, state["mask"][idx])[0]
            for k, i in enumerate(idx):
                actions[i] = int(acts[k])
        return actions

    try:
        p1_state, p2_state = env.reset_all_dual()
        _prep(p1_state)
        _prep(p2_state)
        while any(completed[i] < quotas[i] for i in range(num_envs)):
            a1 = _seat_actions(agent, p1_state)
            a2 = _seat_actions(opponent_agent, p2_state)
            p1_state, p2_state, _rewards, dones, infos = env.step_dual(a1, a2)
            _prep(p1_state)
            _prep(p2_state)
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

    return BattleResult(wins, n_battles)


def _run_battles_mcts(
    mcts, n_battles: int, opponent: str = "random", obs_v2: bool = False,
) -> BattleResult:
    """MCTS evaluation (M4): one env, search runs per decision via sim_* commands.

    Single-env by design — every MCTS decision issues its own sequence of
    sim_clone/sim_fork/sim_step commands against the env hosting the live
    battle, so battles are played one at a time. Reports decision latency.
    """
    import time

    env = GymClient(structured=True, opponent=opponent, obs_v2=obs_v2)
    wins = 0
    latencies = []
    log_every = min(50, max(1, n_battles // 10))

    try:
        for i in range(1, n_battles + 1):
            obs, valid_mask = env.reset()
            obs = obs.reshape(-1)
            done = False
            while not done:
                t0 = time.perf_counter()
                action = mcts.act(env, obs, valid_mask)
                if mcts.last_search_sims > 0:
                    latencies.append(time.perf_counter() - t0)
                obs, _reward, done, info, valid_mask = env.step(action)
                obs = obs.reshape(-1)
            if done and info.get("winner") == "Gym":
                wins += 1

            if i % log_every == 0 or i == n_battles:
                lat = np.array(latencies) * 1000 if latencies else np.zeros(1)
                print(
                    f"Battle {i}/{n_battles} | running win rate: {wins / i:.2f} ({wins}/{i}) "
                    f"| search latency ms: mean {lat.mean():.0f} / p95 {np.percentile(lat, 95):.0f} "
                    f"/ max {lat.max():.0f}",
                    flush=True,
                )
    finally:
        env.close()

    return BattleResult(wins, n_battles)


def _run_battles_mcts_h2h(
    mcts, opponent_agent, n_battles: int, obs_v2: bool = False,
) -> BattleResult:
    """Head-to-head (M4): MCTS vs a raw PPO checkpoint via dual-seat self-play.

    Single-env loop; the search drives the seat given by mcts.seat ('p1' or
    'p2' — run both orientations for a seat-balanced comparison), the raw
    checkpoint's policy drives the other. Wins are counted for the MCTS side.
    """
    import time

    env = GymClient(structured=True, selfplay=True, obs_v2=obs_v2)
    wins = 0
    latencies = []
    log_every = min(50, max(1, n_battles // 10))
    mcts_winner_name = "Gym" if mcts.seat == "p1" else "Opponent"

    def _opp_action(seat):
        obs = seat["obs"].reshape(1, -1)
        obs = slice_structured_obs(obs, opponent_agent._hparams["obs_size"])
        return int(opponent_agent.act_batch(obs, seat["mask"][None, :])[0][0])

    def _mcts_action(seat):
        t0 = time.perf_counter()
        action = mcts.act(env, seat["obs"].reshape(-1), seat["mask"])
        if mcts.last_search_sims > 0:
            latencies.append(time.perf_counter() - t0)
        return action

    try:
        for i in range(1, n_battles + 1):
            p1_state, p2_state = env.reset_dual()
            done = False
            while not done:
                if mcts.seat == "p1":
                    a1 = _mcts_action(p1_state) if p1_state["needs"] else None
                    a2 = _opp_action(p2_state) if p2_state["needs"] else None
                else:
                    a1 = _opp_action(p1_state) if p1_state["needs"] else None
                    a2 = _mcts_action(p2_state) if p2_state["needs"] else None
                p1_state, p2_state, _reward, done, info = env.step_dual(a1, a2)
            if info.get("winner") == mcts_winner_name:
                wins += 1

            if i % log_every == 0 or i == n_battles:
                lat = np.array(latencies) * 1000 if latencies else np.zeros(1)
                print(
                    f"Battle {i}/{n_battles} | running MCTS win rate: {wins / i:.2f} ({wins}/{i}) "
                    f"| search latency ms: mean {lat.mean():.0f} / p95 {np.percentile(lat, 95):.0f} "
                    f"/ max {lat.max():.0f}",
                    flush=True,
                )
    finally:
        env.close()

    return BattleResult(wins, n_battles)


def _run_battles(
    agent, n_battles: int, structured: bool = False, flatten: bool = True,
    opponent: str = "random", obs_v2: bool = False, track_opp: bool = False,
) -> BattleResult:
    """Run n_battles greedy episodes; return (wins, total, opp_correct, opp_total).

    structured=True makes GymClient return the (12, 65) M2 observation
    instead of the legacy flat 100-dim vector (or (12, 77) with obs_v2=True).
    flatten=True additionally reshapes that to (780,)/(924,), matching a PPO
    checkpoint trained with train.py --structured/--obs-v2. The transformer
    consumes the (12, 65) observation directly, so its caller passes
    structured=True, flatten=False.
    """
    env = GymClient(structured=structured, opponent=opponent, obs_v2=obs_v2)
    wins = 0
    opp_correct = 0
    opp_total = 0
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
                # Opp-head prediction on the pre-step obs; label from the step's info.
                opp_pred = int(_opp_head_argmax(agent, obs[None, :])[0]) if track_opp else None
                obs, _reward, done, info, valid_mask = env.step(action)
                if track_opp:
                    label = info.get("opp_action", -1)
                    if label >= 0:
                        opp_total += 1
                        if opp_pred == int(label):
                            opp_correct += 1
                if structured and flatten:
                    obs = obs.reshape(-1)
            if done and info.get("winner") == "Gym":
                wins += 1

            if i % log_every == 0 or i == n_battles:
                print(f"Battle {i}/{n_battles} | running win rate: {wins / i:.2f} ({wins}/{i})", flush=True)
    finally:
        env.close()

    return BattleResult(wins, n_battles, opp_correct, opp_total)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Pokemon Showdown RL agent vs RandomPlayerAI."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["q_learning", "dqn", "ppo", "transformer", "mcts"],
        help="Model architecture to evaluate. 'mcts' (M4) wraps a structured "
             "PPO checkpoint in determinized UCT search at inference time.",
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
        "--obs-v2",
        action="store_true",
        help=(
            "M3.4: evaluate a PPO checkpoint trained with train.py --obs-v2 "
            "(the (12,77)->924 schema-v2 observation). Implies --structured. "
            "With --vs-checkpoint, the opponent checkpoint may still be v1 — "
            "its view is sliced down per token."
        ),
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
    parser.add_argument(
        "--vs-checkpoint",
        default=None,
        help=(
            "Head-to-head (M3.3): path to a second PPO checkpoint to play as the "
            "opponent (seat p2) via dual-seat self-play. Both checkpoints must use "
            "the same obs mode (--structured applies to both). --model ppo or mcts."
        ),
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=100,
        help="MCTS (M4): total simulation budget per move decision (default: 100).",
    )
    parser.add_argument(
        "--determinizations",
        type=int,
        default=1,
        help="MCTS (M4): opponent-team samples per decision; --sims is split "
             "evenly across them (default: 1, per the post-M4 knob sweep — "
             "one deep tree beats a shallow ensemble at fixed --sims).",
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=0.5,
        help="MCTS (M4): PUCT exploration constant (default: 0.5, per the "
             "post-M4 knob sweep).",
    )
    parser.add_argument(
        "--no-determinize",
        action="store_true",
        help="MCTS (M4): search on the true hidden opponent team instead of "
             "sampled ones (omniscient upper-bound diagnostic only).",
    )
    parser.add_argument(
        "--mcts-seat",
        choices=["p1", "p2"],
        default="p1",
        help="MCTS (M4, --vs-checkpoint only): which seat the search drives. "
             "Run both orientations for a seat-balanced head-to-head.",
    )
    parser.add_argument(
        "--opp-sampler",
        choices=["policy", "head"],
        default="policy",
        help="MCTS (M5, --model mcts): in-search opponent model. 'policy' "
             "(default) samples the opponent's own reveal-tracked policy; "
             "'head' samples the auxiliary opp head on the searcher's own obs "
             "(masked to legal opponent actions). 'head' auto-falls-back to "
             "'policy' when the checkpoint has no trained opponent head.",
    )
    args = parser.parse_args()

    if args.obs_v2:
        if args.model not in ("ppo", "mcts"):
            parser.error("--obs-v2 is only meaningful with --model ppo/mcts")
        args.structured = True
    if args.structured and args.model not in ("ppo", "mcts"):
        parser.error(
            "--structured is only meaningful with --model ppo "
            "(transformer always uses structured, unflattened observations; no flag needed)"
        )
    if args.vs_checkpoint:
        if args.model not in ("ppo", "mcts"):
            parser.error("--vs-checkpoint only supports --model ppo/mcts")
        if args.opponent != "random":
            parser.error("--vs-checkpoint and --opponent are mutually exclusive")

    if args.model == "mcts":
        # The search wraps a structured PPO checkpoint (v1 780 or --obs-v2 924).
        base_agent = _load_agent("ppo", args.checkpoint, device=args.device)
        sys.path.insert(0, str(_MODELS_DIR / "mcts"))
        from mcts_agent import MCTSAgent

        if args.mcts_seat == "p2" and not args.vs_checkpoint:
            parser.error("--mcts-seat p2 requires --vs-checkpoint (dual-seat mode)")
        mcts = MCTSAgent(
            base_agent,
            n_sims=args.sims,
            n_determinizations=args.determinizations,
            c_puct=args.c_puct,
            determinize=not args.no_determinize,
            seed=0,
            seat=args.mcts_seat,
            opp_sampler=args.opp_sampler,
            has_opp_head=base_agent.has_opp_head,
        )
        if args.num_envs > 1:
            print("[mcts] search is single-env by design; ignoring --num-envs", flush=True)
        if args.vs_checkpoint:
            opponent_agent = _load_agent("ppo", args.vs_checkpoint, device=args.device)
            result = _run_battles_mcts_h2h(
                mcts, opponent_agent, args.battles, obs_v2=args.obs_v2,
            )
            opponent_name = args.vs_checkpoint
        else:
            result = _run_battles_mcts(
                mcts, args.battles, opponent=args.opponent, obs_v2=args.obs_v2,
            )
            opponent_name = "DamageFirstAI" if args.opponent == "damagefirst" else "RandomPlayerAI"
        wins, total = result.wins, result.total
        win_rate = wins / total if total > 0 else 0.0
        print(f"Model: mcts (sims={args.sims}, det={args.determinizations}, "
              f"c_puct={args.c_puct}, determinize={not args.no_determinize}, "
              f"seat={args.mcts_seat}, opp_sampler={mcts.opp_sampler})")
        print(f"Base checkpoint: {args.checkpoint}")
        print(f"Battles: {total}")
        print(f"MCTS win rate vs {opponent_name}: {win_rate:.2f} ({wins}/{total})")
        return

    agent = _load_agent(args.model, args.checkpoint, device=args.device)
    if args.model == "transformer":
        structured, flatten = True, False
    else:
        structured, flatten = args.structured, True

    # M5: for a headed PPO checkpoint, also report opp-prediction top-1 accuracy
    # (opp head argmax vs the ground-truth opp_action label over unmasked steps).
    # Only the single-opponent paths carry the label in `info`; h2h is skipped.
    track_opp = args.model == "ppo" and agent.has_opp_head

    if args.vs_checkpoint:
        opponent_agent = _load_agent("ppo", args.vs_checkpoint, device=args.device)
        result = _run_battles_h2h(
            agent, opponent_agent, args.battles, args.num_envs, structured=structured,
            obs_v2=args.obs_v2,
        )
        opponent_name = args.vs_checkpoint
    elif args.num_envs > 1:
        result = _run_battles_vec(
            agent, args.battles, args.num_envs, structured=structured, flatten=flatten,
            opponent=args.opponent, obs_v2=args.obs_v2, track_opp=track_opp,
        )
    else:
        result = _run_battles(
            agent, args.battles, structured=structured, flatten=flatten,
            opponent=args.opponent, obs_v2=args.obs_v2, track_opp=track_opp,
        )
    wins, total, opp_correct, opp_total = result
    win_rate = wins / total if total > 0 else 0.0

    if not args.vs_checkpoint:
        opponent_name = "DamageFirstAI" if args.opponent == "damagefirst" else "RandomPlayerAI"
    print(f"Model: {args.model} | Checkpoint: {args.checkpoint}")
    print(f"Battles: {total}")
    print(f"Win rate vs {opponent_name}: {win_rate:.2f} ({wins}/{total})")
    if track_opp:
        if opp_total > 0:
            print(
                f"Opp-prediction top-1 accuracy: {opp_correct / opp_total:.3f} "
                f"({opp_correct}/{opp_total} unmasked steps)"
            )
        else:
            print("Opp-prediction top-1 accuracy: n/a (no unmasked opp_action labels in info)")


if __name__ == "__main__":
    main()

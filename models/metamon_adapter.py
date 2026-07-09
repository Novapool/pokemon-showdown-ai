"""
metamon_adapter.py — Convert Metamon parsed-replay trajectories into the
M2 structured observation format.

Metamon (https://github.com/UT-Austin-RPL/metamon) publishes human Showdown
replays parsed into per-timestep UniversalState dicts plus action indices
(https://huggingface.co/datasets/jakegrigsby/metamon-parsed-replays).
This adapter replays those trajectories as (obs, action, done) tuples where
obs is the (12, 65) token array defined by the M2 spec in MILESTONES.md.

Observation layout — 12 tokens:
    [0]     own active
    [1-5]   own bench (available switches, in Metamon's consistent order)
    [6]     opponent active
    [7-11]  opponent bench (unknown-alive first, then fainted)

Per-token layout — 65 dims:
    [0]     HP ratio (fainted -> 0, unknown -> 1.0)
    [1]     level / 100
    [2:17]  type 1 one-hot (15 gen1 types)
    [17:32] type 2 one-hot (duplicates type 1 for monotype)
    [32:38] status one-hot (brn, frz, par, psn, slp, tox)
    [38]    active flag
    [39]    unknown flag
    [40]    fainted flag
    [41:65] 4 moves x (base_power/250, accuracy, PP ratio, type_idx/20,
            category/2, disabled)

Action alignment — Metamon's MinimalActionSpace is Discrete(9):
    0-3: the active Pokemon's moves in consistent_move_order (alphabetical)
    4-8: available switches in consistent_pokemon_order (alphabetical)
The adapter writes move features and bench tokens in those same orders, so
"action k = move in slot k / Pokemon in bench token k-3" holds exactly as it
does in PokemonGymEnv (where slots follow request order instead). Actions
stored as -1 (player choice not reconstructable from the replay) are skipped.

Sorting uses Metamon's normalization ("lowercase with special characters
removed") reimplemented locally so the adapter runs without metamon installed.
"""

import json
import os
import re
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    import lz4.frame
except ImportError as e:
    raise ImportError(
        "The 'lz4' package is required to read Metamon parsed replays. "
        "Install it with: pip install lz4"
    ) from e

N_TOKENS = 12
TOKEN_DIM = 65
N_ACTIONS = 9

GEN1_TYPES = [
    "normal", "fire", "water", "electric", "grass", "ice", "fighting",
    "poison", "ground", "flying", "psychic", "bug", "rock", "ghost", "dragon",
]
TYPE_INDEX = {t: i for i, t in enumerate(GEN1_TYPES)}

# Same order as fillStatusBitmask() in sim/tools/feature-extractor.ts
STATUS_ORDER = ["brn", "frz", "par", "psn", "slp", "tox"]
STATUS_INDEX = {s: i for i, s in enumerate(STATUS_ORDER)}

# Feature offsets within a token
HP = 0
LEVEL = 1
TYPE1 = 2
TYPE2 = 17
STATUS = 32
ACTIVE_FLAG = 38
UNKNOWN_FLAG = 39
FAINTED_FLAG = 40
MOVES = 41
MOVE_DIM = 6

_NORMALIZE_RE = re.compile(r"[^a-z0-9]")


def normalize_name(name: str) -> str:
    """Metamon's name normalization: lowercase, special characters removed."""
    return _NORMALIZE_RE.sub("", name.lower())


def _clamp01(x: float) -> float:
    return min(1.0, max(0.0, x))


def _as_fraction(x: float) -> float:
    """Normalize a value that may be stored as a percentage (0-100)."""
    if x > 1.001:
        x = x / 100.0
    return _clamp01(x)


def _unknown_token() -> np.ndarray:
    t = np.zeros(TOKEN_DIM, dtype=np.float32)
    t[HP] = 1.0  # unrevealed => assumed full HP (M2 spec)
    t[UNKNOWN_FLAG] = 1.0
    return t


def _fainted_token() -> np.ndarray:
    t = np.zeros(TOKEN_DIM, dtype=np.float32)
    t[FAINTED_FLAG] = 1.0
    return t


def _default_cache_dir() -> Path:
    env = os.environ.get("METAMON_CACHE_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "data" / "metamon_cache"


class MetamonDataAdapter:
    """Streams Metamon parsed-replay trajectories as (obs, action, done).

    Args:
        battle_format: Showdown format the replays were played in, e.g. "gen1ou".
        cache_dir:     Where the dataset was downloaded. Defaults to
                       $METAMON_CACHE_DIR, then ./data/metamon_cache.
    """

    def __init__(self, battle_format: str = "gen1ou", cache_dir: str | None = None):
        self.battle_format = battle_format
        self.cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
        # Filenames start "{format}-{battle id}_..."; the dash prevents
        # "gen1ou-" from matching longer format names.
        self.files = sorted(self.cache_dir.rglob(f"{battle_format}-*.json.lz4"))
        if not self.files:
            raise FileNotFoundError(
                f"No {battle_format} parsed replays found under {self.cache_dir}. "
                "Download the dataset first: bash scripts/download_metamon.sh"
            )

    # ------------------------------------------------------------------
    # Single-timestep conversion
    # ------------------------------------------------------------------

    def pokemon_token(self, p: dict, active: bool) -> np.ndarray:
        """Convert one UniversalPokemon dict into a 65-dim token."""
        t = np.zeros(TOKEN_DIM, dtype=np.float32)
        t[HP] = _as_fraction(float(p.get("hp_pct", 0.0)))
        t[LEVEL] = float(p.get("lvl", 100)) / 100.0

        # types: space-separated, e.g. "ice psychic"; monotype duplicates
        # type 1 into type 2, matching the flat extractor's convention
        types = [s.lower() for s in str(p.get("types", "")).split()]
        if types and types[0] in TYPE_INDEX:
            t[TYPE1 + TYPE_INDEX[types[0]]] = 1.0
            type2 = types[1] if len(types) > 1 else types[0]
            if type2 in TYPE_INDEX:
                t[TYPE2 + TYPE_INDEX[type2]] = 1.0

        status = str(p.get("status", "")).lower()  # "nostatus" -> all zeros
        if status in STATUS_INDEX:
            t[STATUS + STATUS_INDEX[status]] = 1.0
        if t[HP] <= 0.0:
            t[FAINTED_FLAG] = 1.0
            t[HP] = 0.0

        t[ACTIVE_FLAG] = 1.0 if active else 0.0

        moves = sorted(
            p.get("moves") or [],
            key=lambda m: normalize_name(str(m.get("name", ""))),
        )
        for i, m in enumerate(moves[:4]):
            o = MOVES + i * MOVE_DIM
            t[o + 0] = _clamp01(float(m.get("base_power", 0)) / 250.0)
            t[o + 1] = _as_fraction(float(m.get("accuracy", 1.0)))
            max_pp = float(m.get("max_pp", 0) or 0)
            t[o + 2] = _clamp01(float(m.get("current_pp", 0)) / max_pp) if max_pp else 1.0
            t[o + 3] = TYPE_INDEX.get(str(m.get("move_type", "")).lower(), 0) / 20.0
            category = str(m.get("category", "")).lower()
            t[o + 4] = {"physical": 0.0, "special": 1.0}.get(category, 2.0) / 2.0
            t[o + 5] = 0.0  # 'disabled' is not reconstructable from replays
        return t

    def state_to_obs(self, state: dict) -> np.ndarray:
        """Convert one UniversalState dict into the (12, 65) observation."""
        obs = np.zeros((N_TOKENS, TOKEN_DIM), dtype=np.float32)

        obs[0] = self.pokemon_token(state["player_active_pokemon"], active=True)

        # Own bench: available switches in Metamon's consistent (alphabetical)
        # order so bench token j corresponds to action 4+j. Fainted teammates
        # are absent from available_switches, so remaining slots read as
        # fainted. (Known noise source: a live-but-trapped teammate, e.g.
        # during gen1 Wrap, also drops out of available_switches.)
        bench = sorted(
            state.get("available_switches") or [],
            key=lambda p: normalize_name(str(p.get("name", ""))),
        )
        for j in range(5):
            if j < len(bench):
                obs[1 + j] = self.pokemon_token(bench[j], active=False)
            else:
                obs[1 + j] = _fainted_token()

        opp_active = state["opponent_active_pokemon"]
        obs[6] = self.pokemon_token(opp_active, active=True)

        # Opponent bench: replays never reveal bench details, so alive bench
        # slots use the unknown-token convention and the rest read as fainted.
        opp_active_alive = float(opp_active.get("hp_pct", 0.0)) > 0.0
        alive_bench = int(state.get("opponents_remaining", 6)) - int(opp_active_alive)
        alive_bench = min(5, max(0, alive_bench))
        for j in range(5):
            obs[7 + j] = _unknown_token() if j < alive_bench else _fainted_token()

        return obs

    # ------------------------------------------------------------------
    # Trajectory iteration
    # ------------------------------------------------------------------

    def load_battle(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load one replay file -> (obs (T, 12, 65), actions (T,)).

        Timesteps whose action could not be reconstructed (stored as -1)
        are dropped.
        """
        with lz4.frame.open(path, "rb") as f:
            data = json.loads(f.read().decode("utf-8"))
        states = data["states"]
        actions = data["actions"]

        obs_list, act_list = [], []
        for state, action in zip(states, actions):
            action = int(action)
            if not 0 <= action < N_ACTIONS:
                continue
            obs_list.append(self.state_to_obs(state))
            act_list.append(action)

        if not obs_list:
            return (
                np.zeros((0, N_TOKENS, TOKEN_DIM), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )
        return np.stack(obs_list), np.array(act_list, dtype=np.int64)

    def samples(
        self, shuffle_files: bool = False, seed: int | None = None
    ) -> Iterator[tuple[np.ndarray, int, bool]]:
        """Yield (obs, action, done) over every timestep of every battle.

        done is True on the final usable timestep of each battle.
        """
        files = list(self.files)
        if shuffle_files:
            rng = np.random.default_rng(seed)
            rng.shuffle(files)
        for path in files:
            obs_arr, act_arr = self.load_battle(path)
            n = len(act_arr)
            for i in range(n):
                yield obs_arr[i], int(act_arr[i]), i == n - 1

    def __iter__(self) -> Iterator[tuple[np.ndarray, int, bool]]:
        return self.samples()

    def __len__(self) -> int:
        """Number of replay files (not timesteps)."""
        return len(self.files)


if __name__ == "__main__":
    # Smoke test: stream a few battles and report shapes / action histogram.
    adapter = MetamonDataAdapter()
    print(f"{adapter.battle_format}: {len(adapter)} replay files under {adapter.cache_dir}")
    action_counts = np.zeros(N_ACTIONS, dtype=np.int64)
    steps = 0
    battles = 0
    for obs, action, done in adapter:
        assert obs.shape == (N_TOKENS, TOKEN_DIM), obs.shape
        assert np.isfinite(obs).all()
        action_counts[action] += 1
        steps += 1
        battles += int(done)
        if battles >= 20:
            break
    print(f"checked {steps} timesteps across {battles} battles")
    print(f"action histogram: {action_counts.tolist()}")

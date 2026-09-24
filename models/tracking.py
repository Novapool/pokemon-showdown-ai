"""
tracking.py — MLflow run tracking for the training and eval entry points (M13).

Every tracked run records the full argparse namespace, the git commit (SHA,
branch, dirty flag), the host, the torch version, the seed, and — when there
is one — the checkpoint path plus its sha256, so a number in the ledger can be
traced back to the exact code and weights that produced it.

The store is one SQLite file at the repo root (`mlflow.db`, gitignored) unless
MLFLOW_TRACKING_URI says otherwise. Home-box runs reach the Mac's store over a
reverse SSH tunnel — see docs/MULTI-MACHINE.md → MLflow.

Tracking is a no-op when mlflow is not installed or `--no-track` is passed, so
CI and anyone without mlflow can run every script unchanged.

Usage:
    tracking.add_args(parser)           # adds --seed / --no-track
    args = parser.parse_args()
    seed = tracking.seed_everything(args.seed)
    with tracking.run("eval", args, checkpoint=args.checkpoint) as t:
        ...
        t.log_metrics({"win_rate": 0.7}, step=0)
"""

import hashlib
import os
import random
import socket
import subprocess
from contextlib import contextmanager
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = "pokemon-showdown"


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{REPO_ROOT / 'mlflow.db'}"


def find_runs(kind: str | None = None) -> list:
    """All runs in the store (optionally one `kind`), oldest first."""
    import mlflow
    mlflow.set_tracking_uri(tracking_uri())
    return mlflow.search_runs(experiment_names=[EXPERIMENT],
                              filter_string=f"tags.kind = '{kind}'" if kind else "",
                              output_format="list",
                              order_by=["attributes.start_time ASC"])


def add_args(parser) -> None:
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed for python/numpy/torch (default: drawn at random, and "
             "always logged so the run can be repeated). The Node simulator's "
             "battle RNG is NOT covered — battles themselves stay unseeded.",
    )
    parser.add_argument(
        "--no-track", action="store_true",
        help="don't log this run to MLflow (tracking is also skipped silently "
             "when mlflow isn't installed)",
    )


def seed_everything(seed: int | None) -> int:
    """Seed python, numpy's legacy global RNG, and torch. Returns the seed
    actually used, drawing one when `seed` is None so it can be logged."""
    if seed is None:
        seed = random.SystemRandom().randrange(2**31)
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    return seed


def _git(*cmd: str) -> str:
    try:
        return subprocess.run(["git", *cmd], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_info() -> dict:
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # Untracked files don't change what ran, so only tracked edits count.
        "git_dirty": str(bool(_git("status", "--porcelain", "--untracked-files=no"))),
    }


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _Run:
    """Handle yielded by run(); every method is a no-op when not tracking.

    A logging failure (e.g. the home box's SSH tunnel to the Mac's store
    dropping) warns once and disables tracking for the rest of the run — a
    tracking hiccup must never kill a multi-hour training job.
    """

    def __init__(self, mlflow=None):
        self._mlflow = mlflow
        self.run_id = mlflow.active_run().info.run_id if mlflow else None

    def _call(self, fn, *args, **kwargs) -> None:
        if not self._mlflow:
            return
        try:
            fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 — any store failure, by design
            print(f"[tracking] MLflow logging failed, disabled for this run: {e}", flush=True)
            self._mlflow = None

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        clean = {k: float(v) for k, v in metrics.items()
                 if v is not None and np.isfinite(v)}
        self._call(lambda: self._mlflow.log_metrics(clean, step=step))

    def set_tags(self, tags: dict) -> None:
        self._call(lambda: self._mlflow.set_tags({k: str(v) for k, v in tags.items()}))


@contextmanager
def run(kind: str, args, checkpoint=None, name: str | None = None):
    """Open a tracked run of `kind` (train | bc | eval | ab).

    `checkpoint` is the input checkpoint for evals, or the output checkpoint a
    trainer will write (logged by path; hash it later with log_checkpoint).
    """
    try:
        import mlflow
    except ImportError:
        mlflow = None
    if mlflow is None or getattr(args, "no_track", False):
        yield _Run()
        return

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=name):
        import torch
        mlflow.log_params({k: str(v) for k, v in vars(args).items()})
        tags = {"kind": kind, "host": socket.gethostname(),
                "torch": torch.__version__, **git_info()}
        handle = _Run(mlflow)
        handle.set_tags(tags)
        if checkpoint is not None:
            log_checkpoint(handle, checkpoint)
        yield handle


def log_checkpoint(handle: _Run, path) -> None:
    """Record a checkpoint's path and, if it exists yet, its sha256."""
    path = Path(path)
    tags = {"checkpoint_path": str(path.resolve().relative_to(REPO_ROOT))
            if path.resolve().is_relative_to(REPO_ROOT) else str(path)}
    if path.is_file():
        tags["checkpoint_sha256"] = sha256(path)
    handle.set_tags(tags)

#!/usr/bin/env bash
# Install metamon (UT-Austin-RPL) and download the parsed-replays dataset.
#
# Metamon is cloned into vendor/metamon (gitignored) and installed editable,
# because its pip package expects git submodules that a plain
# `pip install git+...` would not fetch.
#
# The dataset lands in $METAMON_CACHE_DIR (default: ./data/metamon_cache).
# models/metamon_adapter.py reads from the same default location.
#
# Usage:
#   bash scripts/download_metamon.sh            # gen1ou only (default)
#   FORMATS="gen1ou gen2ou" bash scripts/download_metamon.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
FORMATS="${FORMATS:-gen1ou}"

export METAMON_CACHE_DIR="${METAMON_CACHE_DIR:-$REPO_ROOT/data/metamon_cache}"
mkdir -p "$METAMON_CACHE_DIR"
echo "METAMON_CACHE_DIR=$METAMON_CACHE_DIR"

if ! "$PYTHON" -c "import metamon" >/dev/null 2>&1; then
    echo "metamon not installed — cloning and installing..."
    mkdir -p "$REPO_ROOT/vendor"
    if [ ! -d "$REPO_ROOT/vendor/metamon" ]; then
        git clone --recursive https://github.com/UT-Austin-RPL/metamon.git \
            "$REPO_ROOT/vendor/metamon"
    fi
    "$PYTHON" -m pip install -e "$REPO_ROOT/vendor/metamon"
fi

# lz4 is needed by models/metamon_adapter.py to read the .json.lz4 replay files
"$PYTHON" -m pip install --quiet lz4

echo "Downloading parsed-replays for: $FORMATS"
# shellcheck disable=SC2086
"$PYTHON" -m metamon.data.download parsed-replays --formats $FORMATS

echo "Verifying gen1ou trajectories..."
"$PYTHON" - <<'PY'
import os
from pathlib import Path

cache = Path(os.environ["METAMON_CACHE_DIR"])
# Replay files are named "{format}-{battle id}_..._{WIN|LOSS}.json.lz4";
# the trailing dash keeps e.g. "gen1ou-" from matching other formats.
files = sorted(cache.rglob("gen1ou-*.json.lz4"))
print(f"gen1ou trajectories available: {len(files)}")
if not files:
    raise SystemExit(
        "ERROR: no gen1ou trajectories found under "
        f"{cache} — download may have failed."
    )
PY

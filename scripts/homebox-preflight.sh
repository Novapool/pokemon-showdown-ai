#!/usr/bin/env bash
#
# Preflight for the home machine. Run this FIRST, every time a session starts
# using the home box — before any training/eval/collection command.
#
# It answers the only question that matters at the start of a remote session:
# "am I about to run the right code, with the right toolchain, on this box?"
#
# From the Mac (note `bash -lc` — nvm only loads in a login shell):
#   ssh homebox 'bash -lc "cd ~/Projects/pokemon-showdown-ai && scripts/homebox-preflight.sh"'
#
# Flags:
#   --no-pull   report git state but don't fast-forward
#   --build     force ./build even if dist/ looks current
#
# Exit codes: 0 = ready to run. Non-zero = do NOT start the job.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DO_PULL=1
FORCE_BUILD=0
for arg in "$@"; do
	case "$arg" in
		--no-pull) DO_PULL=0 ;;
		--build) FORCE_BUILD=1 ;;
		*) echo "unknown flag: $arg" >&2; exit 2 ;;
	esac
done

fail() { echo "FAIL  $*" >&2; exit 1; }
ok()   { echo "ok    $*"; }
warn() { echo "warn  $*"; }

echo "=== preflight: $REPO ==="

# --- 1. toolchain: nvm-provided node, in-repo venv -------------------------
# A non-interactive `ssh homebox '...'` gets system node 18, which ./build
# rejects outright. Source nvm ourselves so the script is correct either way.
if [ -s "$HOME/.nvm/nvm.sh" ]; then
	# shellcheck disable=SC1091
	. "$HOME/.nvm/nvm.sh" >/dev/null 2>&1
fi

command -v node >/dev/null || fail "no node on PATH"
NODE_V="$(node -v)"
NODE_MAJOR="${NODE_V#v}"; NODE_MAJOR="${NODE_MAJOR%%.*}"
[ "$NODE_MAJOR" -ge 22 ] || fail "node $NODE_V — ./build requires >=22. Source nvm (use \`ssh homebox 'bash -lc \"...\"'\`)."
ok "node $NODE_V"

PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || fail "no .venv/bin/python — the system python3 has no torch. Create the venv first."
PY_INFO="$("$PY" - <<'EOF' 2>&1
import sys
try:
    import torch
    print(f"python {sys.version.split()[0]} / torch {torch.__version__} / cuda {torch.cuda.is_available()}", end="")
    if torch.cuda.is_available():
        print(f" ({torch.cuda.get_device_name(0)})", end="")
    print()
except ImportError:
    print(f"python {sys.version.split()[0]} / torch MISSING")
    sys.exit(1)
EOF
)" || fail "$PY_INFO"
ok "$PY_INFO"
case "$PY_INFO" in *"cuda False"*) warn "CUDA not visible — training will run on CPU" ;; esac

# --- 2. git: same commit as the machine you're driving from ----------------
DIRTY="$(git status --porcelain)"
if [ -n "$DIRTY" ]; then
	warn "working tree is DIRTY — not pulling. Resolve by hand:"
	git status --short | sed 's/^/      /'
	DO_PULL=0
fi

git fetch origin --quiet || warn "git fetch failed (offline?) — commit check is against stale refs"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
BEHIND="$(git rev-list --count "HEAD..origin/$BRANCH" 2>/dev/null || echo 0)"
AHEAD="$(git rev-list --count "origin/$BRANCH..HEAD" 2>/dev/null || echo 0)"

if [ "$BEHIND" -gt 0 ] && [ "$DO_PULL" -eq 1 ]; then
	echo "      $BEHIND commit(s) behind origin/$BRANCH — fast-forwarding"
	git merge --ff-only "origin/$BRANCH" || fail "fast-forward failed — diverged history, resolve by hand"
	BEHIND=0
fi
[ "$BEHIND" -eq 0 ] || fail "$BEHIND commit(s) behind origin/$BRANCH and not pulled — you would run stale code AND stale weights"
[ "$AHEAD" -eq 0 ] || warn "$AHEAD local commit(s) not pushed — results here may be invisible to the Mac"
ok "git $BRANCH @ $(git rev-parse --short HEAD) — $(git log -1 --format=%s | cut -c1-60)"

# --- 3. dist/: built JS is what actually runs -----------------------------
NEEDS_BUILD=$FORCE_BUILD
if [ ! -d dist/sim ]; then
	NEEDS_BUILD=1
	echo "      dist/sim missing"
elif [ -n "$(find sim data tools config -name '*.ts' -newer dist/sim/index.js -print -quit 2>/dev/null)" ]; then
	NEEDS_BUILD=1
	echo "      TypeScript source is newer than dist/ — rebuilding"
fi
if [ "$NEEDS_BUILD" -eq 1 ]; then
	./build || fail "./build failed"
	ok "dist/ rebuilt"
else
	ok "dist/ current"
fi

# --- 4. tier-2 data: never syncs, so say plainly what's here --------------
echo "--- local-only data (tier 2, never git) ---"
for d in data/replays data/replay_trajs data/value_targets data/metamon_cache vendor; do
	if [ -e "$d" ]; then
		printf '      %-24s %s\n' "$d" "$(du -sh "$d" 2>/dev/null | cut -f1)"
	else
		printf '      %-24s absent (regenerate or rsync if the job needs it)\n' "$d"
	fi
done

echo "=== READY — node $NODE_V, $(git rev-parse --short HEAD) ==="

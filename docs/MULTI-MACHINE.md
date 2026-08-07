# Multi-Machine Setup (Mac ↔ home machine)

> **Status: LIVE as of 2026-07-29.** Tailscale is installed on both machines,
> SSH from the Mac to the home machine (via the `homebox` alias) is verified
> working. Home machine is WSL2 (Windows Subsystem for Linux), reached through
> a Windows port-proxy — see [WSL2 note](#wsl2-note-on-the-home-machine) below.

## Why this exists

Work happens on two machines: a MacBook that goes everywhere, and a home
machine with real compute. The recurring failure mode is starting work on one
machine and discovering the checkpoint or dataset lives on the other — e.g. the
M8 Phase 1A checkpoint (`checkpoints/opp/ppo_step_2000003_final.pt`) exists
only on the home machine and has never been on the Mac.

The fix has two halves:

1. **Final checkpoints are committed to git** (since `344def9ef`), so the
   inputs to any eval / MCTS run / fine-tune are on every machine automatically.
2. **The home machine is reached over SSH from the Mac**, so heavy jobs run
   there without copying anything. Bulk data never syncs — it's regenerated or
   stays put.

## The machines

| | **Mac** (portable) | **Home machine** (compute) |
|---|---|---|
| Name | `MacBook-Pro-90` / "MacBook Pro (3)" | `Home-PC` (WSL2 hostname) |
| Tailscale name/IP | `100.74.212.114` | `100.97.203.71` |
| Hardware | Apple M4, 10 physical cores, 16 GB, arm64 | Intel i7-13700K, 24 threads, 16 GB (WSL2 alloc), RTX 3080 |
| OS | macOS 26.5.2 | Windows + WSL2 (kernel 6.6.87.2-microsoft-standard-WSL2) |
| Repo path | `/Users/laithassaf/Documents/Programs/Archived/pokemon-showdown` | `/home/laith/Projects/pokemon-showdown-ai` |
| Python | 3.13.12 via in-repo `.venv/` | 3.12.3 — **only the in-repo `.venv/` has torch**; system `python3` does not |
| Node | v25.8.2 | v22.20.0 **via nvm only**; system node is v18.19.1 and `./build` rejects it |
| SSH user | — | `laith` |
| Role | Editing, planning, short runs, MCTS self-play collection | Long PPO training runs (5M steps), big sweeps — has the GPU |

**Fill the TODO column in at setup time.** An accurate inventory here is the
whole point of the file — it's what stops every new chat from re-deriving it.

### What's on the Mac right now (2026-07-29)

Verified present: `data/replays` (474M), `data/replay_trajs` (1.7G),
`data/metamon_cache` (2.1G), `vendor/` (1.0G), `node_modules`, built `dist/`,
`config/showdown_login.txt` (mode 600), and PPO checkpoint dirs `bcft`, `opp`,
`selfplay`, `structured`, `v2`, `v3`, `v3-extended`.

So the Mac is currently the better-stocked machine for data. The home machine's
advantage is throughput, not contents.

## What lives where, and how it moves

Three tiers. Getting a file into the right tier is the whole discipline.

| Tier | What | How it crosses machines |
|---|---|---|
| **1. Git** | Source, docs, **`*_final.pt` checkpoints** (~2 MB each, 24 MB total) | `git push` / `git pull`. Automatic. |
| **2. Never syncs** | `data/replays`, `data/replay_trajs`, `data/metamon_cache`, `data/value_targets`, `vendor/`, intermediate sweep checkpoints | Regenerate on the machine that needs it, or `rsync` on demand. 4+ GB — never git. |
| **3. Secrets** | `config/showdown_login.txt` | Copy by hand once, `chmod 600`. Gitignored forever. |

The `.gitignore` rule that implements tier 1:

```gitignore
models/**/checkpoints/**/*.pt
!models/**/checkpoints/**/*_final.pt
```

If a run produces something another machine will need, **make it a `_final.pt`
or commit it deliberately** — don't leave it as an intermediate and hope.

## Setup (one-time) — done, kept for reference

1. **Tailscale on both machines**, signed into the same account. On the home
   machine, Tailscale is installed **on Windows**, not inside WSL2 — WSL2
   shares Windows' network stack via NAT, so an inbound connection needs to
   hit Windows first. `tailscale ip -4` on Windows gave the stable IP
   `100.97.203.71`.
2. **SSH server inside WSL2** (not on Windows): `sudo apt install -y
   openssh-server && sudo systemctl enable --now ssh`.
3. **Windows port-proxy**, because WSL2's internal IP (`hostname -I` inside
   WSL2, currently `172.20.218.172`) changes across reboots and Tailscale only
   sees the Windows host. Elevated PowerShell:
   ```powershell
   $wslIp = (wsl hostname -I).Trim().Split(" ")[0]
   netsh interface portproxy delete v4tov4 listenport=22 listenaddress=0.0.0.0 2>$null
   netsh interface portproxy add v4tov4 listenport=22 listenaddress=0.0.0.0 connectport=22 connectaddress=$wslIp
   New-NetFirewallRule -DisplayName "WSL2 SSH" -Direction Inbound -Protocol TCP -LocalPort 22 -Action Allow
   ```
   See [WSL2 note](#wsl2-note-on-the-home-machine) — **this must be re-run
   after every Windows reboot** until it's turned into a startup task.
4. **Copy the Mac's key over:** `ssh-copy-id laith@100.97.203.71` (key already
   existed at `~/.ssh/id_ed25519.pub` on the Mac).
5. **SSH alias** in `~/.ssh/config` on the Mac:
   ```
   Host homebox
       HostName 100.97.203.71
       User laith
       ServerAliveInterval 60
   ```
6. **Verified:** `ssh homebox 'hostname && nproc && python3 -V'` →
   `Home-PC` / `24` / `Python 3.12.3`, no password prompt.
7. Repo already present at `/home/laith/Projects/pokemon-showdown-ai` on the
   home machine (this is the same checkout — the setup was done sitting at the
   home machine, then verified from the Mac).

## WSL2 note (on the home machine)

The home machine's SSH server runs **inside WSL2**, but Tailscale runs on
**Windows** (the host). These are different network namespaces, so a
Windows-side port-proxy forwards port 22 to whatever WSL2's current internal
IP is. That internal IP is not stable across Windows reboots — **if `ssh
homebox` stops connecting, the fix is almost always re-running the port-proxy
PowerShell block above from an elevated prompt**, not a Tailscale or SSH key
problem. Turning this into a Windows Scheduled Task (run at startup, as
Administrator) is a known TODO to stop this from being a manual step.

## Start of every home-machine session: run the preflight

**Before any training / eval / collection command on the home box, run this
once. No exceptions.**

```bash
ssh homebox 'bash -lc "cd ~/Projects/pokemon-showdown-ai && scripts/homebox-preflight.sh"'
```

Exit 0 means it is safe to launch the job. **Non-zero means do not launch it** —
the script's job is to fail loudly instead of letting a run proceed on stale
code, stale weights, or the wrong toolchain. It:

1. **Sources nvm** and asserts node ≥ 22 (see the gotcha below).
2. **Asserts the in-repo `.venv` has torch**, and reports whether CUDA is
   visible — a silent CPU fallback turns a 2-hour run into a 20-hour one.
3. **Fast-forwards to `origin/master`** and refuses to continue if still behind.
   Since final checkpoints are committed, a stale checkout is a *wrong weights*
   bug, not just old code. A dirty tree is reported and the pull is skipped
   rather than forced.
4. **Rebuilds `dist/`** if any `.ts` under `sim/ data/ tools/ config/` is newer
   than `dist/sim/index.js`. `dist/` is what Node actually runs; a `git pull`
   that touches TypeScript leaves it stale.
5. **Lists which tier-2 data dirs exist on the box**, since none of them sync.

Flags: `--no-pull` (report git state, don't move HEAD), `--build` (force a
rebuild).

Push from the Mac *before* running it, or the fast-forward has nothing to fetch:
`git push origin master` → preflight → launch.

### The two toolchain gotchas this exists to prevent

- **Remote commands get node v18, not v22 — and `bash -lc` does NOT fix it.**
  ⚠️ **Corrected 2026-08-07; this doc previously claimed `bash -lc` was the
  fix.** Measured: `ssh homebox 'bash -lc "node -v"'` returns **v18.19.1**. nvm
  initializes from `~/.bashrc:119`, and `~/.bashrc` has Ubuntu's stock
  non-interactive guard at the top (`case $- in *i*) ;; *) return;;`). `bash -lc`
  is a **login but non-interactive** shell, so that guard fires and nvm never
  loads. **Only interactive shells get node 22.** The preflight passes solely
  because it sources nvm itself (`homebox-preflight.sh:41-44`) — and its own
  failure message still repeats the bad advice.

  **The reliable fix is the absolute path:**
  `/home/laith/.nvm/versions/node/v22.20.0/bin/node`, or prepend
  `/home/laith/.nvm/versions/node/v22.20.0/bin` to `PATH`. `./build` hard-exits
  with "We require Node.js version 22 or later", but the failures that cost real
  time are the quiet ones — node 18 has **no global `WebSocket`**, so
  `ladder-bot.js` dies at `_connect` with `ReferenceError: WebSocket is not
  defined` *after* logging in and printing a healthy banner.
- **`python3` on the home box has no torch.** Use `.venv/bin/python`
  explicitly in every remote command — a bare `python3 models/ppo/train.py`
  dies on `import torch`. **`ladder-bot.js` cannot take this advice**: it spawns
  `infer_server.py` via a hardcoded `python3` (`ladder-bot.js:143`), so the venv
  must be on `PATH` before node starts. Nothing in the bot's own flags can fix
  it.

**Both bite hardest under `tmux`**, which spawns a non-interactive shell and
inherits neither. Launch long jobs through a wrapper script that exports `PATH`
explicitly — `scripts/run-m12-ladder.sh` is the worked example.

### Home box data state (updated 2026-08-06)

| Dir | Home box | Notes |
|---|---|---|
| `data/replays/gen1ou` | ✅ **98,985 logs, 399 MB** | **home box is AUTHORITATIVE** — extended there 2026-07-31 |
| `data/replays/gen1randombattle` | ✅ **21,581 logs, 88 MB** | **home box is AUTHORITATIVE** — extended there 2026-07-31 |
| `data/value_targets/m8_v3_df` | ✅ 4.7 MB | collected there |
| `data/replay_trajs/v3` | ✅ **963 MB** | regenerated there 2026-08-06 for M12 Phase 2 |
| `data/metamon_cache` | ❌ absent | |
| `vendor/` | ❌ absent | |
| `config/showdown_login.txt` | ✅ mode 600 | copied 2026-08-06 for the M12 ladder run; `config/` is gitignored, so credentials never sync — scp them per machine |

⚠️ **Both replay corpora are authoritative on the home box.** They were grown
there by backfill runs on 2026-07-31 and the Mac copies are now behind. **Rsync
back before editing either on the Mac**, or the corpora fork:

```bash
rsync -a homebox:Projects/pokemon-showdown-ai/data/replays/ data/replays/
```

Consequences for job placement:

- Jobs needing only a **committed checkpoint** (PPO training, bot evals, MCTS
  self-play collection) run on the home box immediately.
- **Replay scraping/backfill** should run there — that is where the corpora live
  now, and it is the machine that stays up.
- **BC pretrain now runs there** — `data/replay_trajs/v3` was regenerated on the
  home box 2026-08-06 (both formats, ~25 min) rather than rsync'd, per the
  never-rsync rule below. Coverage reproduced the documented figures exactly
  (gen1ou 86.4%, randbats 90.7%), which is the check that the regeneration was
  faithful. Rebuild it with:
  ```bash
  node models/replay_adapter_cli.js --format gen1ou --obs-v3 --out-dir data/replay_trajs/v3
  node models/replay_adapter_cli.js --format gen1randombattle --obs-v3 --out-dir data/replay_trajs/v3
  ```
  ⚠️ Shards are **battle-sized** (50 gen1ou / 11 randbats at the default
  `--shard-size 2000`), not the 99/21 the M7-era archive records. `--val-shards`
  therefore holds out ~2× the data it used to, so held-out accuracies are not
  comparable to pre-2026-08-06 numbers at fine precision.
- `data/value_targets/*` dirs are ~5 MB each and rsync in seconds:
  `rsync -avz data/value_targets/ homebox:~/Projects/pokemon-showdown-ai/data/value_targets/`

## Daily use

The model is: **one Claude session, on the Mac.** Claude reaches the home
machine through SSH in its Bash tool. There is no second Claude session and
nothing to keep in sync conversationally.

Long jobs go in `tmux` on the home machine so they survive a closed laptop or a
dropped connection:

Note the shape of these: the real repo path `~/Projects/pokemon-showdown-ai`,
and `.venv/bin/python` — not `python3`. ⚠️ **`bash -lc` gets you the repo's
login environment but NOT node 22** (see the gotchas above); anything invoking
node inside tmux needs the absolute nvm path or a `PATH`-exporting wrapper.

```bash
# launch a long training run, detached
ssh homebox 'bash -lc "cd ~/Projects/pokemon-showdown-ai && tmux new -d -s train \
  \".venv/bin/python models/ppo/train.py --obs-v3 --steps 5000000 \
    --checkpoint-dir checkpoints/v3 2>&1 | tee train.log\""'

# check on it later, from anywhere
ssh homebox 'tmux capture-pane -pt train | tail -30'
ssh homebox 'tail -20 ~/Projects/pokemon-showdown-ai/train.log'

# list what's running
ssh homebox 'tmux ls'

# attach interactively (from your own terminal, not Claude's)
ssh -t homebox 'tmux attach -t train'
```

Pull a result back when a run finishes:

```bash
# small: just commit it on the home box and pull
ssh homebox 'bash -lc "cd ~/Projects/pokemon-showdown-ai && git add models/**/*_final.pt \
  && git commit -m \"...\" && git push"'
git pull

# large (a dataset you genuinely need locally):
rsync -avz --progress homebox:~/Projects/pokemon-showdown-ai/data/value_targets/ data/value_targets/
```

### Choosing a machine

- **Mac** — editing, tests, `./build`, evals, MCTS self-play collection. The M4
  handles 6-worker collection at ~35 min per 2000 games.
- **Home box** — 5M-step PPO runs, 20-checkpoint sweeps, anything measured in
  hours.

### Rules of thumb

- **Run `scripts/homebox-preflight.sh` before every remote job** (see above). It
  exists so "always `git pull` on both ends" is enforced rather than remembered
  — committed checkpoints mean a stale checkout is now a *wrong weights* bug,
  not just old code.
- **Call `.venv/bin/python`, never `python3`**, and for anything that runs node,
  use the **absolute nvm path** (`/home/laith/.nvm/versions/node/v22.20.0/bin/node`)
  or export it onto `PATH`. `bash -lc` alone does **not** provide node 22 —
  measured 2026-08-07, see the gotchas section.
- Pass `--checkpoint-dir` explicitly on training runs. `train.py`'s routing
  checks `elif args.opp_coef != 0.0` before the obs-schema branches, so runs
  land in `checkpoints/opp/` unexpectedly (this is how the M8 Phase 1A
  checkpoint got misfiled). See `IN-PROGRESS.md` → "Carried gotcha".
- Never `rsync` `data/metamon_cache` or `data/replay_trajs`. Re-download or
  regenerate; it's faster than 3.8 GB over a coffee-shop uplink.

## When to update this file

Update it — don't create a new doc — whenever:

- **Setup completes** (flip the status banner, fill every TODO). This is the
  first and most important update.
- **A machine changes**: new hardware, OS reinstall, repo moved, Python/Node
  version bump, hostname or Tailscale name change.
- **The tier table changes**: a new bulk data directory, a new gitignore
  whitelist rule, a new secret.
- **The connection method changes**: different VPN, different SSH alias, a
  jump host, added `mosh`.
- **A cross-machine gotcha bites you.** If you lost time to it once, it goes in
  "Rules of thumb" so it costs nothing the next time.

Keep the machine inventory factual and dated — a wrong inventory is worse than
no inventory, because it gets trusted.

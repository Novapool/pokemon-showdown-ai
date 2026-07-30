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
| Tailscale name/IP | _TODO: fill in_ | `100.97.203.71` |
| Hardware | Apple M4, 10 physical cores, 16 GB, arm64 | Intel i7-13700K, 24 threads, 16 GB (WSL2 alloc), RTX 3080 |
| OS | macOS 26.5.2 | Windows + WSL2 (kernel 6.6.87.2-microsoft-standard-WSL2) |
| Repo path | `/Users/laithassaf/Documents/Programs/Archived/pokemon-showdown` | `/home/laith/Projects/pokemon-showdown-ai` |
| Python | 3.13.12 via in-repo `.venv/` | 3.12.3 |
| Node | v25.8.2 | v22.20.0 |
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

## Daily use

The model is: **one Claude session, on the Mac.** Claude reaches the home
machine through SSH in its Bash tool. There is no second Claude session and
nothing to keep in sync conversationally.

Long jobs go in `tmux` on the home machine so they survive a closed laptop or a
dropped connection:

```bash
# launch a long training run, detached
ssh homebox 'cd ~/pokemon-showdown && tmux new -d -s train \
  "python models/ppo/train.py --obs-v3 --steps 5000000 --checkpoint-dir checkpoints/v3 2>&1 | tee train.log"'

# check on it later, from anywhere
ssh homebox 'tmux capture-pane -pt train | tail -30'
ssh homebox 'tail -20 ~/pokemon-showdown/train.log'

# list what's running
ssh homebox 'tmux ls'

# attach interactively (from your own terminal, not Claude's)
ssh -t homebox 'tmux attach -t train'
```

Pull a result back when a run finishes:

```bash
# small: just commit it on the home box and pull
ssh homebox 'cd ~/pokemon-showdown && git add models/**/*_final.pt && git commit -m "..." && git push'
git pull

# large (a dataset you genuinely need locally):
rsync -avz --progress homebox:~/pokemon-showdown/data/value_targets/ data/value_targets/
```

### Choosing a machine

- **Mac** — editing, tests, `./build`, evals, MCTS self-play collection. The M4
  handles 6-worker collection at ~35 min per 2000 games.
- **Home box** — 5M-step PPO runs, 20-checkpoint sweeps, anything measured in
  hours.

### Rules of thumb

- Always `git pull` on both ends before starting work. Committed checkpoints
  mean a stale checkout is now a *wrong weights* bug, not just old code.
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

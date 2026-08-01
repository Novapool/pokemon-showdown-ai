# Code Review Findings — 2026-08-01

**Read this before proposing any new training experiment, and before quoting
any pre-2026-08-01 number.** Three independent read-only reviews of the
training path, the observation pipeline, and the evaluation machinery. Several
findings **weaken or invalidate existing conclusions**, including two that are
currently recorded as closed hypotheses and one that affects the shipping
agent's headline number.

Every claim marked ✅ VERIFIED below was re-checked directly against the code or
by running it, not taken from a reviewer's report.

---

## The one-paragraph version

The agent cannot see what it needs to play Gen 1 well. Its observation encodes
a move in six numbers — base power, accuracy, type, category, PP, disabled —
with **no move identity**, and carries **no species and no stats at all**. So
`Swords Dance` and `Recover` are byte-identical inputs, and no damage estimate
is formable. This is invisible against `Random` and `DamageFirst`, which never
switch and mostly spam attacks, and expensive against humans — which is the
exact shape of the unexplained 93%-vs-bots / 30.5%-vs-humans gap. Separately,
the reward penalises long games **only when winning**, which is a standing
gradient against exactly the patient, positional play the project has spent
three milestones trying to teach.

**A hypothesis this project previously held is now measured to be wrong:** the
1032→128 first layer was suspected of being an 8:1 information bottleneck. It
is not. A live observation carries ~25 distinct values.

---

## 1. The observation is information-poor ✅ VERIFIED

Measured on the live gym (`structured-v3-extended`, 1044 dims, ~260 decision
points): **128.9 non-zero dims (12.3%)**, **25.3 distinct values** per
observation, and **667 of 1044 dims are non-zero in <1% of decisions**.

### 1a. Move identity is absent, and the encoding aliases decisive moves

A move slot is `T_MOVE_DIM = 6` dims (`sim/tools/feature-extractor.ts:455`):
base power/250, accuracy, type, category, PP ratio, disabled. There is no move
ID. Verified against `Dex.mod('gen1')`:

```
ALIASED   Rest / Reflect / Amnesia / Agility               -> [0, 1, "Psychic", "Status"]
ALIASED   Recover / Softboiled / Substitute / Swords Dance -> [0, 1, "Normal", "Status"]
ALIASED   Seismic Toss / Counter                           -> [0.004, 1, "Fighting", "Physical"]
```

**The agent cannot distinguish a sweep-setup move from a recovery move.**
Further consequences, all verified:

| Move | Encodes as | Reality |
|---|---|---|
| Wrap / Fire Spin / Clamp | 15 BP, 85% Normal physical | Among the strongest tactics in Gen 1 |
| Horn Drill / Fissure | 0 BP, 30% Normal physical | OHKO moves; look like the worst moves in the game |
| Slash | 70 BP Normal physical | `critRatio` unused; Slash is dominant *because* of it |

v3's status flags are aggregated **per token, not per slot** — the agent knows
"this Pokémon has a sleep move" but not which button causes it.

### 1b. No species identity and no stats

`feature-extractor.ts:446-455` never reads stats. The live request already
carries `stats: {atk, def, spa, spd, spe}` per own Pokémon. HP is stored as a
**ratio**, so absolute bulk is invisible too. **No damage estimate is
formable** — which is what every switch / stay / sacrifice decision is.

### 1c. Partial trapping and Hyper Beam recharge are invisible

`maybeLocked` is never read; the tracker ignores `|cant|…partiallytrapped`.
`Dex.mod('gen1').moves.get('recharge').exists === false` ✅ VERIFIED — so a
recharge turn encodes as a nonsense 0-BP **physical Normal** move. `DamageFirst`
spams Hyper Beam and the agent structurally cannot learn to punish it.

### 1d. Why this explains the gap

`Random` and `DamageFirst` never voluntarily switch and mostly attack. Against
them, base power and type genuinely suffice → 93%. Against humans, move
identity, stats, trapping and recharge all matter → 30.5%. **This is the first
hypothesis in the project that predicts the *shape* of the gap** rather than
proposing another knob.

It also gives M8 Phase 1A a second explanation: adding a speed-ratio feature to
an observation lacking species, stats and move identity is a rounding error.

---

## 2. The reward penalises the skill we are trying to teach ✅ VERIFIED

`sim/tools/pokemon-gym.ts:706,711` — terminal reward applies `−0.001 × turns`,
then clips to `[-1, 1]`. **A loss already saturates at −1, so the clip absorbs
the penalty entirely; a win does not.**

> Long games are punished only when you win. Stalling while losing is free.

Switching costs tempo and lengthens games. Also `+0.0001` for statusing the
opponent with no symmetric penalty for statuses received. Neither constant is
reachable from any flag. **Fix is ~3 lines and is a direct test of the
switching hypothesis.**

**Found independently by two reviewers**, which is why it is stated this
strongly.

`battle-sim.ts` (the MCTS forward model) applies **neither** the penalty nor the
clip, so `value_finetune --target mc` (env rewards) and `--target root` (sim
root Q) sit on different reward scales while the docs present them as
interchangeable — a live candidate explanation for the **M8 Phase 2 null**.

---

## 3. Every search-vs-no-search comparison is confounded ✅ VERIFIED

`models/mcts/mcts_agent.py:280-289` returns **argmax over visit counts**
(deterministic). `PPOAgent.act`/`act_batch` (`models/ppo/ppo_agent.py:149,182`)
call **`dist.sample()`** (stochastic). No greedy/temperature flag exists in the
eval path.

So "with search vs without search" also varied the decision rule. Measured on
M7 with **no search in either arm**, only sampling swapped for masked argmax:

| arm | vs Random | vs DamageFirst |
|---|---|---|
| sampled (production path) | 73.5% | 60.0% |
| greedy argmax | **81.2%** | 58.0% |
| difference | **+7.8pp [+2.7, +12.5]** | −2.0pp [−8.8, +4.8] |

**Argmax alone buys roughly the entire reported search gain.** The effect is
opponent-dependent, so it is not a constant offset that cancels.

**Affected:** `MILESTONES.md:713`'s claim that the M4 head-to-head is "the
cleanest causal measure — same network, same battles, only the search differs"
is **false**. M4's "+8.6pp vs Random", the 60.2% seat-balanced h2h, the shipped
**"93.0% with MCTS vs 69.7% raw"**, and every ladder `--mcts` on/off comparison
inherit this.

**How it survived nine milestones:** `models/evaluate.py:365`'s docstring reads
*"Run n_battles greedy episodes."* It is not greedy for PPO. ✅ VERIFIED

### 3a. The ladder bot has always played stochastically

`models/infer_server.py:135,156,171` all call `agent.act()`. ✅ VERIFIED
**The 30.5% ladder result was achieved while sampling from the policy rather
than playing its best move.** Greedy decoding on the ladder has never been
tested. It is free to test — inference only, no training.

Caveat before assuming it is free points: the bot-eval effect was
opponent-dependent, and determinism against adapting humans is a real
trade-off. But at 30.5% we are not losing to opponents who are reading us.

---

## 4. Training-loop defects

### 4a. `--opponent-mix` destroys ~62% of episodes

`models/ppo/train.py:365-383` — `_switch_family` resets **all** envs whenever
the sampled opponent family changes, and family is resampled every rollout
(`:478`). With the standard `selfplay=0.5,damagefirst=0.3,random=0.2` that is a
~62% switch rate, against a reward that is ~99% terminal. Truncation is
**length-biased**, preferentially destroying long battles — where switching
decides Gen 1 games.

Affects every mixed-opponent run since M3.4: **M7, m9seed, 2c, 2d**. It does
not invalidate comparisons *among* them (all share it) but may be a handbrake
on all of them.

### 4b. Learning rate was unsettable ✅ VERIFIED, FIXED 2026-08-01

There was no `--lr`. Every PPO run M2–M9 trained at the 3e-4 default. A naive
fix would not have worked: `PPOAgent.load` restores the checkpoint's optimizer
`param_groups` including `lr`, silently reverting a constructor value on every
warm-start. Now added, applied **after** the optimizer load.

Note both BC checkpoints store **zero optimizer state entries** ✅ VERIFIED, so
PPO genuinely starts from a fresh Adam at 3e-4 in every arm — historical
comparisons are internally consistent on this axis.

### 4c. Network width was unsettable — FIXED 2026-08-01

Hard-coded `128` in five places, so "is the model big enough?" was not a
runnable experiment for nine milestones. Now `--hidden-size`.

### 4d. Smaller

- `--opp-coef` silently ignored on `--resume` while the banner prints the
  requested value (`train.py:391`).
- Self-play pool silently falls back to a copy of the current policy when seeds
  fail to rsync (`:273-287`) — no warning.
- **No seeding anywhere** in the trainer, so no A/B can use common random
  numbers.
- Only in-run metrics are a mixture win-rate that does not record which
  opponent it is against, plus a single summed loss. **No entropy, KL, clip
  fraction, or value loss is ever logged.**
- No run manifest (device / argv / SHA) — the enabling condition for the
  `mps`/`cuda` confound that cost a 70-minute run.
- `models/collect_value_data.py` is clean and is the template the trainer
  should follow: it writes `meta.json`, has `--seed`, exposes every knob.

---

## 5. Evaluation machinery

### 5a. Verified sound — the statistics layer is not the problem

Algebra checked directly rather than via docstrings: Wilson, Newcombe method 10
(correctly role-swapped for `p2 − p1`), two-proportion `required_n`, `chi2_sf`
(against closed forms df=1..5), and the Pearson overdispersion statistic are
**all correct**.

### 5b. Battles are unseeded — and this is good news

`gym_bridge.js:154` passes no seed → `new PRNG(null)`. Arms are therefore
**genuinely independent**, so the Wilson/Newcombe CIs are **valid, not too
narrow**. A previously-suspected correlation defect is **not present**.

Cost: both teams redraw every battle, so each arm pays full team-draw variance.
Common random numbers + a paired test would be a large **free power gain**.
Implementation trap: `PokemonGymEnv` reuses one `_seed` on every reset, so a
naive run-level seed would make all N battles identical.

### 5c. Ties count as losses everywhere

`sim/battle.ts:1521` emits `|tie` with no `|win|`; `parseProgressLines` does not
match it, so `info.winner` becomes the `'__ended__'` sentinel. Ladder rate is
3/507 rated (0.6%); the offline rate is unmeasured.

### 5d. The "~3pp seat bias" appears to be an artifact

`models/CLAUDE.md` cites it as fact. The only direct measurement
(`MILESTONES.md:566`, 45%/51% at n=500) has a **CI including 0**, and M4's
60.4%/60.0% shows no effect. **An unmeasured constant has been serving as an
error bar.** Separately, `_run_battles_h2h` hard-seats p1 with no `--seat` flag,
so seat balancing is a convention a caller can silently violate.

### 5e. Other

- `ladder_analysis.py:294-303` prints the opponent-Elo confound warning but
  never gates or adjusts on it; `own_rating` is loaded and never used.
  Matchmaking attenuates real differences toward zero on the M9 Phase 3
  endpoint.
- `evaluate.py:71,79` set `epsilon=0` for q_learning/DQN while PPO samples — so
  the **M2 architecture comparison** ran its arms under different decision
  rules.
- Errored battles are excluded without being tallied.
- Ladder-bot's prefix-matching room cleanup (`:650-656`) can silently drop a
  live battle.

### 5f. Harness reproducibility is unestablished

M7-raw vs Random has now read **69.7%** (n=2,000, 2026-07-31), **72.0%** and
**73.5%** (n=400 each, 2026-08-01). Pooled 2026-08-01 vs the M9 reading is
**+3.8pp [+0.1, +7.4]** — same checkpoint, same harness. That is the same scale
as the project's ±3pp gates. **Day-to-day harness reproducibility has never
been measured and should be.**

---

## 6. BC ↔ gym distribution mismatch

`sim/tools/replay-adapter.ts`:
- **No v3-extended path**, so the M8 87-dim schema has no BC data — meaning the
  speed-ratio arm could not have been BC warm-started while its control lineage
  was. **A second confound on M8 Phase 1A**, on top of §1d.
- Orders slots **alphabetically** vs the request order used live, relying on a
  "slot-content invariant" that §1a makes barely learnable.
- Drops all `|cant|` states.
- **Silently discards decisions by a seat KO'd before it acted** — the
  mispredictions — and does not count them in `skipped`.

Action space and legality masking were checked against Gen 1 edge cases
(forced switch, Wrap, recharge, sleep/freeze, Struggle, Disable) and are
**correct**. One unguarded hang: `GymPlayer` does not override `receiveError`,
which throws in a detached loop.

---

## What this changes

**Weakened or invalidated:**

| Conclusion | Status now |
|---|---|
| "Search is worth +10.2pp" (M4) | **Confounded** — argmax alone is ~+7.8pp vs Random |
| "93.0% with MCTS vs 69.7% raw" | **Confounded** — same cause |
| "Richer observations are not the constraint" (M8 P1A) | **Doubly confounded** — width, and no BC warm-start for the extended schema |
| "Value-head targeting fails" (M8 P2) | **New candidate cause** — `mc` and `root` targets are on different reward scales |
| "The model is too small" | **Measured wrong** — obs carries ~25 distinct values; width 512 bought +2.8pp BC val acc |
| "~3pp seat bias" | Unsupported by its own measurement |

**Still standing:** the M9 statistical protocol, the Wilson/Newcombe
implementations, PPO run-to-run reproducibility (<1pp), the data-exhaustion
finding, and the 2a/2c format-alignment result.

## Recommended order

1. **Reward asymmetry fix** (~3 lines) — cheapest test of the switching hypothesis.
2. **Observation fix** — move identity, species/stats, trapping, recharge. The
   real candidate; re-opens a hypothesis currently in the dead-ends table.
3. **Greedy decoding** — free to test, no training, may be free ladder points.
4. **Instrumentation** — run manifest, seeding, entropy/KL logging. Cheap, and
   prevents the next confound.

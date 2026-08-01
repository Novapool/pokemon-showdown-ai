# Review C — measurement & evaluation machinery

Scope: `models/evaluate.py`, `scripts/bot_eval_ab.py`, `scripts/ladder_analysis.py`,
`tools/ladder-bot/ladder-bot.js`, `models/infer_server.py`, `models/mcts/mcts_agent.py`.
Supporting files read for verification only: `sim/tools/pokemon-gym.ts`,
`models/gym_bridge.js`, `models/ppo/ppo_agent.py`, `sim/battle.ts`, `sim/prng.ts`.

**Headline:** the statistics layer is correct — I checked the Wilson, Newcombe,
sample-size and chi-square algebra line by line and it all matches the published
formulas (F9). The defect is one level up, in *what gets played*: **the MCTS arm
of every search A/B selects actions greedily while the raw-policy arm samples
from the policy.** I measured that confound on the shipping checkpoint today:
argmax alone is worth **+7.8pp vs Random [+2.7, +12.5]**, with no search at all.

---

## F1 — CRITICAL: "search vs no search" is confounded with "argmax vs sampling"

**What is measured wrong.** MCTS returns the highest-visit action —
deterministic argmax (`mcts_agent.py:280-289`, and the no-search fallback at
`:274` is also `np.argmax`). The raw-policy arm it is compared against calls
`PPOAgent.act`/`act_batch`, which are **stochastic**: `dist.sample()` at
`models/ppo/ppo_agent.py:149` and `:182`. Every consumer inherits this —
`evaluate.py:143` and `:393` (bot evals), `infer_server.py:135` and `:171`
(ladder). There is no `--greedy`/temperature flag anywhere in the eval path.

So when the project reports "MCTS beats the same checkpoint by +Xpp", X contains
two effects: lookahead, and the removal of policy sampling noise. Nothing
separates them.

**Measured magnitude (this review, read-only probe on the shipping M7 checkpoint
`models/ppo/checkpoints/v3/ppo_step_5000002_final.pt`, obs-v3, CPU, no search in
either arm — only `act_batch` swapped for a masked argmax):**

| arm | vs Random | vs DamageFirst |
|---|---|---|
| sampled (production path) | 73.5% (588/800) | 60.0% (240/400) |
| greedy argmax | **81.2% (325/400)** | 58.0% (232/400) |
| difference (Newcombe 95%) | **+7.8pp [+2.7, +12.5]** | −2.0pp [−8.8, +4.8] |

(The sampled-vs-Random arm pools two runs: `models/evaluate.py` itself at n=400
→ 288/400, and my probe at n=400 → 300/400, so the probe is validated against the
production CLI.)

**Which conclusions this could distort.**
- `MILESTONES.md:704-705` — the M4 pre-registered head-to-head "MCTS vs raw,
  60.4% p1 / 60.0% p2, +10.2pp seat-balanced", described in `MILESTONES.md:713`
  as "the cleanest causal measure — same network, same battles, only the search
  differs". It is not: the decision rule also differs. A greedy raw policy is
  worth ~+8pp vs Random on its own; the residual attributable to search may be
  much smaller than +10pp, possibly near zero.
- `MILESTONES.md:702` — M4 criterion "vs RandomPlayerAI 66.0% (MCTS) vs 57.4%
  (raw), +8.6pp". That delta is the same size as the argmax effect I measured.
- `IN-PROGRESS.md:8-16` — the shipping claim "**93.0% vs Random / 84.2% vs
  DamageFirst** (with tuned MCTS on top)" against "69.7% raw". Part of that 23pp
  gap is decision-rule, not search.
- M6/M9 ladder runs comparing `--mcts` on vs off (`ladder-bot.js:288-305`) —
  same confound, since the non-MCTS branch samples and the MCTS branch argmaxes.
- Note the DamageFirst column shows **no** greedy gain, which is itself
  informative: the confound is opponent-dependent, so it cannot be waved away as
  a constant offset.

**Fix.** Add `--greedy` to `models/evaluate.py` and `models/infer_server.py` (a
masked-logit argmax path on `PPOAgent`), make it the default for evaluation, and
re-run the MCTS-vs-raw head-to-head with **both arms greedy**. That is the actual
causal question. Effort: ~1h of code; the re-runs are the real cost (n≥2,000/arm
per `docs/EVALUATION-METHODOLOGY.md`). **Do this before the width A/B**, because
if the width arms are ever compared with search on one side, the same confound
reappears.

---

## F2 — HIGH: seat balancing is an unenforced convention, and the "~3pp seat bias" it defends against was never established

**What is measured wrong.** `models/CLAUDE.md` ("Head-to-head (M3.3)") states
"Seat bias exists (~3pp) — run both orientations and combine". In code:

- `evaluate.py:178-250` (`_run_battles_h2h`) hard-seats `--checkpoint` at p1 and
  `--vs-checkpoint` at p2. There is **no `--seat` flag** for the PPO path;
  `--mcts-seat` exists but `evaluate.py:575` restricts it to `--model mcts`
  with `--vs-checkpoint`.
- To get the other orientation the caller must swap the two `--checkpoint`
  arguments, after which `wins` is reported for the *other* agent. Combining
  therefore requires hand-computing `n − wins` for the flipped run.
  `bot_eval_ab.py` takes raw `wins/n` strings (`:35-45`) and cannot detect the
  mistake.
- Single-opponent evals are **always p1**: `pokemon-gym.ts:593-594` names p1
  "Gym" (the agent) and p2 "Opponent", and `evaluate.py:164/239/283` count
  `winner == "Gym"`. Every absolute number in the project — 93.0%, 84.2%, 69.7%,
  the M9 Phase 2a BC A/B, the 2c regression — is a p1-only measurement.

**Second half of this finding: the 3pp constant appears to be an artifact.**
The only direct seat measurement I can locate is `MILESTONES.md:566` — "45% as
p1 (227/500), 51% as p2 (253/500) → 48.0% combined". At n=500 per orientation the
Newcombe CI on that 6pp gap is roughly ±6pp and includes 0. The M4 measurement
(`MILESTONES.md:704-705`) is 60.4% p1 vs 60.0% p2 — no seat effect at all. So a
number that has never been shown to be non-zero is being carried in docs and used
informally as an error bar. Either it is real (in which case p1-only absolute
numbers are biased and the source asymmetry in the harness needs finding), or it
is not (in which case the seat-balancing tax on h2h runs is unnecessary).

**Which conclusions this could distort.** Any h2h that was run in one
orientation only; and, if the bias is real, every absolute win rate the project
has ever reported. `MILESTONES.md:510` ("52.4% over 1000 seat-balanced battles")
and `:566` are the ones explicitly flagged as marginal — they sit inside the size
of the alleged bias.

**Fix.** (a) Add `--seat {p1,p2,both}` to `evaluate.py`'s h2h path, with `both`
running the split and reporting the correctly combined `wins/n`. (b) Settle the
seat question once: run the *same* checkpoint against itself, n≥4,000, both
orientations — under H0 that is exactly 50%, so any deviation is pure harness
asymmetry and is measurable to ±1.5pp. Effort: ~2h code + one overnight eval.

---

## F3 — HIGH: battles are unseeded and unpaired; the n=2,000/arm A/B is leaving a large power gain on the table (but the CIs are *valid*)

**What I verified.** `models/gym_bridge.js:154` constructs
`new PokemonGymEnv({ obsMode, opponent })` — **no `seed`**. So
`pokemon-gym.ts:536` leaves `_seed` undefined, `:591` does `new PRNG(null)`, and
`sim/prng.ts:46/160/210-218` draws a fresh crypto-random seed. The three derived
seeds (battle spec, p1 team, p2 team) are redrawn on every `reset`, and
`RandomPlayerAI`/`DamageFirstAI` also get `seed: undefined` (`:557-564`).

**The good news, stated explicitly because the review brief flagged the
opposite risk:** arms are therefore *independent*, each battle is an i.i.d.
Bernoulli draw from the joint distribution over (team draw × damage rolls ×
opponent), and the independent-sample Wilson/Newcombe intervals in
`bot_eval_ab.py` and `ladder_analysis.py` are **correct, not too narrow**. There
is no shared-seed inflation defect here.

**The cost.** `gen1randombattle` redraws both teams every battle, so team-draw
variance is a large share of each arm's total variance, and each arm pays it
separately. Common random numbers — identical seed sequence in both arms, so both
arms see the *same 2,000 team pairings* and the same opponent RNG stream until
the policies actually diverge — plus a paired (McNemar / paired-difference)
analysis would cut the variance of the difference substantially for the same
wall-clock. For a project whose gates are ±3pp and whose evals cost hours, that
is the single largest available efficiency win.

**Trap to avoid when implementing.** `PokemonGymEnv` stores **one** `_seed` and
reuses it on *every* `reset()` (`:536`, `:591`). Passing a run-level seed would
make all N battles in a run byte-identical — a catastrophic silent failure that
would look like a suspiciously tight win rate. The seed must be supplied
**per reset** (`{"cmd":"reset","seed":<int>}`), derived as `base + battle_index`.

**Fix.** Per-reset seed in `gym_bridge.js`'s `reset` handler, `--seed-base` in
`evaluate.py` (assigning deterministic per-battle seeds across the env quotas in
`_run_battles_vec`), and a `--paired` mode in `bot_eval_ab.py` taking per-battle
outcome vectors instead of counts. Effort: ~4-6h. It also buys reproducibility,
which the harness currently has none of.

---

## F4 — MEDIUM: ties are silently counted as losses in both harnesses

**Offline.** `sim/battle.ts:1502-1521`: a draw calls `win()` with no side, which
emits `|tie` and **never** `|win|`. `pokemon-gym.ts:1240-1267`
(`parseProgressLines`) only matches `|win|Gym` and `|win|Opponent`, so `done`
stays false; the battle then terminates through the `_waitForWin()` fallback,
which resolves the sentinel `'__ended__'` (`:1178`, consumed at `:692-701` and
`:850-858`). `info.winner` becomes `'__ended__'`, and every counter in
`evaluate.py` (`:164`, `:239`, `:283`, `:343`) tests `== "Gym"` — so a tie is a
loss. Tie frequency in offline eval is **unmeasured and unlogged**.

**Ladder.** `ladder-bot.js:243-245` correctly records `result='tie'`;
`ladder_analysis.py:162` keeps the row in `n` while `tally()` (`:191-194`) counts
only `'win'`. Same outcome: tie = loss. Observed rate in
`data/replays/self_ladder/ladder_results.csv`: **3 ties / 507 rated games
(0.6%)**.

**Distortion.** Symmetric across arms, so differences move by ~0.1pp — this does
not overturn anything. It does depress every absolute rate slightly, and the
offline rate is unknown, which is the part that should not stay unknown.

**Fix.** Recognize `|tie` in `parseProgressLines` and set an explicit
`winner: 'tie'`; report tie counts alongside wins in `evaluate.py` and
`ladder_analysis.py`; decide the convention (exclude from n, or score 0.5) and
write it into `docs/EVALUATION-METHODOLOGY.md`. Effort: ~1h.

---

## F5 — MEDIUM: the ladder's matchmaking confound is diagnosed but never adjusted for

`ladder_analysis.py:294-303` prints per-arm mean opponent Elo and warns "a large
gap here means the arms did not face the same pool and the difference is
confounded" — but there is **no threshold, no gate, and no adjustment**, and
`own_rating` (captured at `ladder-bot.js:219-223`, written at `:357`) is loaded
into `Battle` (`ladder_analysis.py:142`) and then never used.

Rating-based matchmaking is self-correcting: the arm that wins climbs and meets
stronger opponents, which **attenuates the measured difference toward zero**.
That is conservative in direction but unquantified in size, and it applies
directly to the M9 Phase 3 primary endpoint (`ladder_analysis.py:256-279`), whose
pre-registered bar is ±10pp — a bar that attenuation makes harder to clear.

**Fix.** Report the arm difference stratified into opp_rating bands, and/or fit a
logistic regression of win on arm + opp_rating; add a hard warning when the
per-arm mean opp Elo gap exceeds ~25 points. Effort: ~3h.

---

## F6 — MEDIUM: cross-model comparisons used different decision rules

`evaluate.py:71` and `:79` explicitly set `agent.epsilon = 0.0  # fully greedy`
for `q_learning` and `dqn`. PPO and transformer get no such treatment and sample
(F1). So the M2 model-selection comparison behind `docs/MODEL-COMPARISON.md`
pitted **greedy** tabular/DQN agents against a **stochastic** PPO. By F1's
measurement that is worth up to ~8pp vs Random in PPO's disfavour — i.e. the
architecture comparison that chose the project's model was run with the arms
under different decision rules.

Corroborating evidence that this was never noticed: `evaluate.py:365` documents
`_run_battles` as running "n_battles **greedy** episodes", which is false for the
PPO/transformer path it is mostly used for.

**Fix.** Same `--greedy` flag as F1, applied uniformly; add a caveat line to
`docs/MODEL-COMPARISON.md`. Effort: folded into F1.

---

## F7 — LOW: harness errors are excluded from the sample without being counted

`evaluate.py:152-154` and `:233-235` skip an env-step when `infos[i]` carries an
`"error"` — "battle not counted". The per-env quota logic means the *denominator*
stays exact (the loop keeps running until every quota is met), so this is **not**
an off-by-one. But errored battles are a systematically-excluded subpopulation
(they correlate with unusual battle states), no count of them is reported at the
end, and the serial path `_run_battles` (`:360-414`) has no error handling at
all. Fix: tally and print skipped battles; fail loudly above a threshold. ~30min.

## F8 — LOW: ladder-bot room cleanup uses prefix matching and can drop a live battle

`ladder-bot.js:650-656` retires "twin" rooms with
`id.startsWith(otherId) || otherId.startsWith(id)`. `battle-gen1randombattle-1`
is a prefix of `battle-gen1randombattle-12`, so an unrelated concurrent battle
would be deleted, `/leave`-ed and added to `finishedRooms` — never recorded in
the CSV, i.e. a silently dropped game. Latent today because the bot ladders one
battle at a time (`searchNext()` guards on `this.searching`), but it is a
silent-data-loss path in the file that produces the ladder numbers. Fix: require
the alias to differ only by a trailing `-<password>` segment. ~15min.

---

## F9 — VERIFIED CORRECT: the statistics themselves

I checked the algebra rather than the docstrings.

- **Wilson** (`ladder_analysis.py:44-52`): centre `(p + z²/2n)/(1 + z²/n)`,
  half-width `z·sqrt(p(1−p)/n + z²/4n²)/(1 + z²/n)`. Matches the standard score
  interval exactly. ✅
- **Newcombe method 10** (`:55-66`) for `p2 − p1`:
  `lo = (p2−p1) − sqrt((p2−l2)² + (u1−p1)²)`,
  `hi = (p2−p1) + sqrt((u2−p2)² + (p1−l1)²)`. This is the published form with the
  1↔2 roles correctly swapped for the `p2 − p1` orientation. ✅ (The stray
  `- 0 +` on line 65 is cosmetic.)
- **required_n** (`:69-80`): the standard two-proportion formula
  `[z_α√(2p̄q̄) + z_β√(p₁q₁+p₂q₂)]² / δ²`. ✅
- **chi2_sf** (`:103-123`): verified against closed forms for df = 1, 2, 3, 4, 5.
  Even branch gives `e^{-x/2}Σ(x/2)^i/i!`; odd branch gives
  `erfc(√(x/2)) + √(2x/π)e^{-x/2}(1 + x/3 + …)`. ✅
- **overdispersion** (`:83-100`): Pearson statistic with pooled `p`, `df = k−1`.
  ✅
- `bot_eval_ab.py` reuses all of the above unchanged. ✅

One nit: `ladder_analysis.py:284` and `report_power` hardcode `p1=0.30` in the
power tables regardless of the arms' observed rates. Correct for the ladder
(~30%), badly wrong if anyone points this script at bot-eval-scale rates (~70-93%).
`bot_eval_ab.py` does the right thing with `--baseline-p`. Add an assert or read
the observed pooled rate. ~15min.

---

## Side observation worth logging (not a code defect)

The shipping M7 checkpoint, sampled, vs Random has now been measured three times:
**69.7% (n=2,000, M9 protocol, 2026-07-31)**, **72.0% (n=400, `evaluate.py`,
today)**, **75.0% (n=400, my probe, today)**. Pooling today's two runs gives
73.5% (588/800); against the M9 reading that is **+3.8pp [+0.1, +7.4]** — a
nominally significant gap between two measurements of the same checkpoint on the
same harness against the same opponent, at exactly the ±3pp scale of the
project's gates. It is marginal (p≈0.05) and could be ordinary run-to-run drift
plus a lucky draw, but the project has never established the harness's
day-to-day reproducibility for absolute rates. F3's per-battle seeding would
settle it definitively. Recommend re-measuring M7-raw at n=2,000 once more before
any absolute number is used as a control.

---

## Recommended order

1. **F1** — add `--greedy`, re-run the MCTS-vs-raw head-to-head with both arms
   greedy. This is the finding that may materially weaken an existing shipped
   conclusion, and it is cheap to fix.
2. **F3** — per-reset seeding + paired analysis, *before* the n=2,000/arm width
   A/B is launched, so that experiment gets the power for free.
3. **F2** — `--seat both`, plus the self-vs-self run that settles whether the
   "3pp seat bias" exists at all.
4. F4, F5, F6, then F7/F8/F9-nit.

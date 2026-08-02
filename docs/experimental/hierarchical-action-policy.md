# Experiment: Hierarchical Action Policy + Move Embeddings

**Status:** 🧪 Design only — no code written, nothing in the training pipeline touched.
**Branch:** `experiment/hierarchical-action-policy`
**Written:** 2026-08-02

A parallel exploration, not a replacement for the M11 line. Read
`docs/WHERE-WE-ARE.md` first for what the main lineage is doing and why.

---

## Part 1 — Audit: how rich is our move encoding today?

### What is actually encoded

Per Pokémon token (schema v3, 86 dims/token, 12 tokens → 1032-dim observation),
each of the 4 move slots gets **6 dims** at `T_MOVES = 41` + `slot * 6`
(`sim/tools/feature-extractor.ts:625-653`):

| Dim | Feature | Encoding |
|---|---|---|
| +0 | base power | `min(1, bp/250)` |
| +1 | accuracy | `acc/100`, `1.0` if never-miss |
| +2 | PP | `pp/maxpp`, defaults `1.0` when unobservable |
| +3 | type | **`typeIndex/20` — an ordinal scalar, 0.00…0.70** |
| +4 | category | `catIndex/2` ∈ {0, 0.5, 1} |
| +5 | disabled | 0/1 |

v3 adds 9 more dims per token (`fillV3MoveDims`, `:521`), of which only the
first four are per-move-slot:

| Dim | Feature | Granularity |
|---|---|---|
| 77–80 | type effectiveness of move slot *i* vs opponent active | **per slot** ✅ |
| 81 | "some move on this token causes recharge" | **OR'd over all 4 moves** |
| 82 | "some move self-KOs" | OR'd over all 4 |
| 83 | "some move has priority" | OR'd over all 4 |
| 84 | first inflicted-status id found, `/6` | first hit wins, not per-slot |
| 85 | Sleep Clause (global) | broadcast to all tokens |

### Three concrete defects this audit found

**1. Move identity is absent, and the v3 flags do not recover it.** This is
already the headline of `docs/CODE-REVIEW-FINDINGS.md §1`, but the aggregation
detail matters for this design: dims 81–84 are OR'd **across the token's four
moves**, so they say "this Pokémon has *a* priority move," never "*slot 2* is
the priority move." The policy head chooses a slot. It is being asked to pick
slot 2 using a fact that isn't attached to slot 2. For attacking-direction type
effectiveness (77–80) the per-slot alignment is right; for everything else it
is not.

**2. Move type is an ordinal scalar; species type is a one-hot.** Compare:
species types get 15-dim one-hots at `T_TYPE1=2` / `T_TYPE2=17`
(`:615-616`), but a move's type is a single number `typeIdx/20` (`:650`). Fed
into `nn.Linear`, one scalar cannot express 15 unordered categories — the
network can only learn a monotone response to "type number." Electric (0.15)
and Grass (0.20) are forced to be near-neighbours because of an arbitrary
dictionary order. This is ~56 extra dims/token to fix (4 slots × 14) and is
independent of everything else in this doc.

**3. Nothing about the move generalizes across slots.** The policy head is
`Linear(hidden, 9)` (`models/ppo/ppo_agent.py:79`) — output unit 2 means "the
move in slot 2," not "Thunderbolt." Gen 1 randbats shuffles move order, so
everything the network learns about Thunderbolt in slot 2 has to be re-learned
in slots 0, 1 and 3. **This is the defect I'd rank second after move identity,
and it is not on the M11 v4 list.**

### What a dedicated move-embedding space would add

Gen 1 has **168 moves** and **151 species** — a tiny vocabulary. An
`nn.Embedding(169, d)` at d=16 is 2,704 parameters against a current model of
151,187, and the BC corpus is ~3.9M gen1ou + randbats records. Table capacity
is a non-issue.

What it buys over the 6 hand dims, concretely:

- **Semantics that have no hand-feature at all.** Recover (heal 50%), Amnesia
  (+2 special), Rest (heal + sleep), Substitute, Counter, Explosion, Wrap-family
  trapping, the OHKO moves. Today Recover and Swords Dance are byte-identical;
  Horn Drill and Fissure encode as *the two worst moves in the game* (0 BP,
  30% accuracy).
- **Learned similarity rather than declared similarity.** This is the part of
  the food/cuisine-embedding analogy that carries over: nobody has to decide
  that Blizzard ≈ Ice Beam ≈ Thunderbolt-but-Ice. With ~10⁴–10⁵ observations of
  each common move, that geometry is learned from human play.
- **Slot invariance**, once the embedding is used to *score* actions rather
  than just to describe them (see Arm A below).

What it does **not** buy, and the honest caveats:

- **Frequency skew.** Gen 1 usage is extremely concentrated. The ~40 moves that
  matter will get good vectors; the tail (Bide, Mirror Move, Kinesis…) will sit
  near their init. Fix: keep the 6 hand dims and concatenate — rare moves fall
  back on features, common moves get identity. Do not replace, augment.
- **It is not a substitute for stats.** No damage estimate is formable without
  species base stats, and an embedding of the *move* cannot supply the
  *defender's* Def. That stays an M11 v4 item.
- **It requires an observation-schema change** (a move-id dim per slot), which
  invalidates every existing checkpoint — exactly the M11 cost. See Sequencing.

---

## Part 2 — Design: the hierarchical orchestrator

### First, a correction to the premise

**The transformer is retired.** The production model since the M3.2 decision is
the MLP `PPOAgent` — `Linear(1032, H) → ReLU → Linear(H, H) → ReLU` with three
linear heads (`models/ppo/ppo_agent.py:71-86`). There is no attention in the
shipping path. And even in the retired `TransformerPolicy`, attention ran over
**12 Pokémon tokens**, never over moves — moves have never been tokens in this
project. So "what we get for free from the transformer's attention" is,
currently, nothing. Any cross-move reasoning would be new.

Second: **Gen 1 has no entry hazards.** No Spikes, no Stealth Rock, no Toxic
Spikes. A "hazard" category is empty in this format; the plausible category set
is smaller than it looks.

### The degeneracy problem dissolves if the levels share an objective

The stated open problem — "a move-selector trained to maximize immediate reward
degenerates into max base power" — is real, but it is **created by the proposed
training signal, not by the hierarchy.** A factorization

    π(a | s) = π_hi(c | s) · π_lo(a | c, s)

is a *reparameterization* of a categorical distribution over 9 actions. If both
factors are trained with the **same** PPO advantage on the **same** return, the
composed policy is exactly as long-horizon as the flat one — the log-prob of the
taken action is `log π_hi(c) + log π_lo(a|c)`, one ratio, one clip, one
advantage. Nothing is myopic. Myopia only appears if you deliberately hand the
low level a shorter-horizon reward. **So don't.** Train end-to-end with the
shared advantage; the "open problem" is then a non-problem, and the real
question becomes whether the discrete bottleneck helps at all (see Risks).

If you nonetheless want a separately trainable selector (for reuse or
interpretability), here are the anti-myopia options, ranked:

1. **Distill from MCTS root visit counts.** We already collect these:
   `models/collect_value_data.py` writes root visit distributions and root Q per
   decision, ~600 games/h/worker. Visit counts are long-horizon *by
   construction* — they are the output of a depth-limited search with the value
   head at leaves. Training the low level to match the visit distribution
   *within* a category gives a non-greedy target with zero new machinery. This
   is the cheapest correct answer and it reuses an existing asset.
2. **Condition on the trunk embedding, not on V(s).** ⚠️ Feeding the
   move-selector a **scalar value estimate cannot work as stated**: V(s) is
   constant across the actions being ranked, so it adds the same number to every
   logit and cannot change an argmax. It can only act as a gate (via a nonlinear
   mixing layer) — "when V(s) is low, shift the whole distribution toward
   healing/stalling." That is a weak, low-bandwidth signal. What actually
   carries the long-horizon state is the **trunk hidden vector h(s)**, and the
   selector should take that. If you want a per-action long-horizon signal, the
   object you need is Q(s,a) or an advantage, not V(s).
3. **Advantage-weighted supervised learning** on human replays: weight the CE
   loss on the human's chosen move by the trajectory's outcome. Keeps it
   supervised while making the target outcome-aware.

### What is genuinely novel here vs. already available

| Idea | Verdict |
|---|---|
| Category-then-move factorization | **Not novel and not free capacity.** A 9-way softmax can already represent any hierarchical policy exactly. Its value would be inductive bias / exploration structure, not expressiveness. Unproven. |
| Category label as an auxiliary loss | Mildly useful, near-free. Predicting "is this an attack or a status move" as an aux head is an M5-style auxiliary task (we already do this for opponent actions with `opp_head`). Cheap to test. |
| Learned move-identity embeddings | **Genuinely new information**, but it belongs to the M11 v4 schema, not to the hierarchy. Do not let the hierarchy take credit for it. |
| **Action-conditioned (pointer) scoring head** | **The strongest genuinely-novel item in this design.** Replace `Linear(H, 9)` with `logit_i = ⟨W·h(s), e(move_i)⟩` over the *legal actions present in this state*. Learning transfers across slots and across Pokémon. Needs no hierarchy. |
| Cross-move attention | New (moves have never been tokens). Plausible for "coverage" reasoning — which of my 4 moves is redundant given the other 3. Speculative; costs a token-layout redesign. |

### Proposed architecture (arms are separable — that is the point)

```
                     obs (12 × 86)  ──►  trunk  ──►  h(s)  [H dims]
                                                      │
   move ids (4)  ──► E_move (168×16) ──► e_i  ───────►│
   move hand-feats (6+4) ────────────► ─────────────► │
                                                      ▼
   ARM A   logit_i = ⟨W_q h(s), [e_i ; f_i]⟩        (pointer head, 9 logits,
           switch actions score against species embeddings   masked as today)

   ARM B   c ~ π_hi(c | h(s))         categories: {damage, status/setup, switch}
           a ~ π_lo(a | h(s), c)      = ARM A's head, restricted to c's members
           trained end-to-end, single shared PPO advantage

   ARM C   π_lo distilled from MCTS root visits (collect_value_data.py output),
           π_hi trained on the marginalized visit mass per category
```

**Arm A is the recommended first build.** It captures move identity, slot
invariance and the embedding geometry, and it is a head swap plus one obs dim
per move slot — no hierarchy, no new reward, no new training loop.

Category set for Arm B, given Gen 1: `{damage, status/setup, switch}` — three,
not five or six. "Hazard" is empty. Body Slam (damage + 30% paralysis) already
shows the labels are leaky, which is itself an argument for letting the aux
head *predict* categories rather than *route through* them.

### Risks specific to Arm B

- **Exploration collapse.** π_hi commits to a category before any member is
  evaluated; if it collapses early to `damage`, the switch branch gets no
  gradient. This project has spent three milestones trying to teach switching
  (M7–M9) and `Random`/`DamageFirst` never switch, so the training signal
  already leans this way. A hierarchy could make it worse. Mitigation: entropy
  bonus on π_hi specifically, and monitor per-category entropy (which we
  currently log for *nothing* — see the M11 instrumentation-debt item).
- **Credit assignment.** A bad outcome after a good category / bad move
  penalizes both factors. Flat softmax has the same issue in a less structured
  form; not obviously worse, but not obviously better either.
- **It is a bet on inductive bias with no expressiveness gain.** That is the
  honest one-line summary of Arm B.

---

## Part 3 — Minimal test plan (before any real training compute)

Three stages, each gating the next. Total ≈ 1 day of wall-clock, mostly on the
home box, none of it touching `models/ppo/train.py`.

### Stage 0 — Non-greedy decision probe (≈2 h, no training at all)

Build the diagnostic *before* the architecture, so there's a baseline to beat
and a direct measurement of the degeneracy the design is meant to fix.

Filter the existing BC shards (`data/replay_trajs/`) for positions where a
rated human chose a **0-BP status move or a switch while a ≥90-BP damaging move
was legal**. That is a corpus-derived "the greedy answer was wrong" set, free to
construct. Then score the **current M7 v3 checkpoint** on it, greedy decoding.

- Output: current top-1 agreement on the non-greedy set vs on the full set.
- Value regardless of this experiment: if the gap is small, "the agent plays
  greedily" is not actually our problem and this whole design loses its
  motivation. **Run this first even if you build nothing else.**

### Stage 1 — BC ablation (≈3 h on the home box, existing corpus)

Same corpus, same optimizer, same steps; only the head/features change. Four
arms:

| Arm | Change |
|---|---|
| baseline | current MLP, v3 obs (known: **55.5%** randbats val @ H=512) |
| +onehot | move type as 15-dim one-hot instead of `typeIdx/20` (defect 2) |
| +embed | move-id embedding concatenated to the hand features |
| +pointer | Arm A: embedding **and** action-conditioned scoring head |

Report top-1 / top-3 held-out accuracy on the full set **and** on Stage 0's
non-greedy subset. `bc_pretrain_mlp.py` already prints val accuracy; the
existing `55.5%` at width 512 is the number to beat.

**Gate: `+pointer` must beat baseline by ≥2pp top-1 on the non-greedy subset.**
Below that, stop — the architecture isn't extracting anything the flat head
couldn't.

⚠️ **A BC win is a screen, not a decision.** M9 Phase 2c is the standing
counterexample: a BC checkpoint that imitated +5.6pp better finished **−8.3pp**
after the same 5M-step PPO recipe. Passing Stage 1 buys the right to spend RL
compute; it does not predict RL outcome.

### Stage 2 — Short RL arm (≈2 h/arm, only if Stage 1 gates)

2M steps, not 5M — this is a signal check, not a decision run. Two arms
(baseline v3 recipe vs `+pointer`), same seed, `--opponent-mix` as M7.
Evaluate raw-policy greedy at **n=2,000/opponent**, gate **≥+3pp with CI
excluding 0**, per `docs/EVALUATION-METHODOLOGY.md`. Pre-register before
looking. Only *then* is Arm B (the actual hierarchy) worth building, and it
should be A/B'd against Arm A — not against the v3 baseline, or the embedding
win will be misattributed to the hierarchy.

---

## Sequencing note (important)

Arm A needs a **move-id dim in the observation**, which is exactly M11 Phase 1's
first bullet. Building this experiment before M11 v4 means writing the same
schema plumbing (`feature-extractor.ts` + `replay-adapter.ts` + LUT + BC
re-conversion) twice, and re-running BC twice. **Recommendation: let M11 Phase 1
land the move-id and species dims, then branch this experiment off v4.**
Stage 0 is the exception — it needs nothing new and can run today.

## Open questions

- Should switch actions score against a **species** embedding in the same
  pointer head (symmetric, elegant) or keep the 5 fixed switch logits? Symmetric
  is nicer but couples this to the species-id dim as well.
- Is per-slot move data even reachable for the *opponent's* bench, where moves
  are only partially revealed? The embedding of an unrevealed move is a
  different problem from the embedding of a known one — probably needs a
  dedicated `<unknown>` token in the table.

# Model Comparison — Results Ledger

Running record of every evaluated model/run in this project. Evaluation is
`models/evaluate.py` (sampled policy) unless noted; opponents are
`RandomPlayerAI` ("Random"; never voluntarily switches) and `DamageFirstAI`
("DamageFirst"; always picks the highest-base-power move, M3.3).

## Headline results

| Model / run | Steps | vs Random | vs DamageFirst | Status |
|---|---|---|---|---|
| Q-Learning (tabular, flat obs) | — | never fully trained | — | Archived in M1 — tabular limitation assumed confirmed by design |
| DQN (flat obs) | — | never fully trained | — | Regression baseline only |
| **MLP-PPO, structured obs (M2 baseline)** | 2.6M | **51% (254/500)** | **51% (101/200)** | ✅ The baseline to beat (`models/ppo/checkpoints/structured/ppo_step_2600000_final.pt`) |
| Transformer PPO, from scratch (M3) | 2.6M | 32% (158/500) | — | ❌ Negative result |
| Transformer PPO, BC warm-started, run 1 (M3) | 2.6M–7.6M | 41% @ 2.6M; peak 46% @ 2.5M; collapsed to 0–13% by 3.6–5.6M | — | ❌ Violent collapse (unconstrained PPO updates) |
| Transformer PPO, warm-started, run 2 (+KL early-stop, +LR anneal) (M3) | 5.0M | peak 45% @ 500k, decayed to 27% @ 5.0M | — | ❌ Collapse fixed, decay remained |
| Transformer PPO, warm-started, M3.2 fixes (value warmup + BC KL-anchor + real masks) | 5.0M | holds 44–55% through 3.5M (no collapse); best confirmed **53% (263/500)** @ 500k; decays to 25% by 5M as the anchor anneals away | **39% (77/200)** @ 500k | ❌ Parity vs Random at best, behind vs DamageFirst → **transformer retired (M3.2 decision); M4+ proceeds on MLP-PPO** |
| MLP-PPO self-play (M3.3 best) | 4.75M | **57% (287/500)** | 46% (91/200) | 🟨 First stable run; peer of M2 (52.4% h2h over 1000); no DamageFirst transfer (`models/ppo/checkpoints/selfplay/ppo_step_4750059.pt`) |
| MLP-PPO schema-v2 + opponent mix (M3.4 best) | 2.25M | 54% (272/500) | 46% (92/200) | ❌ M2/M3.3 peer (48.0% h2h vs M3.3 best); obs richness + opponent mix ruled out as bottleneck |
| MCTS over M3.3 best (M4) — original defaults (100 sims, det=4, c_puct=1.5) | inference-time | 66% (330/500) | 56% (113/200) | ✅ Positive result. Beats its own base checkpoint 60.2% (602/1000) seat-balanced h2h; 84–88ms/move. First clear improvement since M2 |
| MCTS over M3.3 best — tuned knobs (post-M4 sweep: det=1, c_puct=0.5) | inference-time | 81.2% (406/500) | 67.2% (336/500) | ✅ Tuned operating point (one deep tree, trusted prior) |
| MCTS over M3.4 v2 best (post-M4 A/B) | inference-time | 82.6% (413/500) | 70.2% (351/500) | 🟨 Statistical tie with v1 under search (raw v2 trailed raw v1 — richer obs matter more with lookahead) |
| MLP-PPO + opp-prediction aux head, raw (M5 final) | 5.0M | 57% (285/500) | 41.5% (83/200) | 🟨 Raw peer of M2/M3.3/M3.4 (fifth 5M-class run in the 51–57% band); head accuracy 35.8% vs DamageFirst |
| MCTS over M5 final (policy sampler, tuned knobs) | inference-time | 86.0% (430/500) | 72.6% (363/500) | ✅ Best of the bot-trained lineage; M6 ladder-A/B control. Head *sampler* retired: 70.4% DF / 85.8% R — parity with the policy sampler (M5 thesis negative). Checkpoint: `models/ppo/checkpoints/opp/ppo_step_5000001_final.pt` |
| MLP BC on human replays, raw (M5.5 run 2) | BC only | 22% | 14.5% | ❌ ~50% human-imitation val acc doesn't transfer to raw bot play; under tuned MCTS: 56.2% R / 45.0% DF — search helps hugely but BC-only is far below the bars |
| MLP-PPO BC-warm-started anchored fine-tune, raw (M5.5 final) | BC + 5.0M | 54.6% (273/500) | 42.0% (84/200) | 🟨 Raw peer of the 51–57% band; first BC→RL transfer that improves on BC (raw 22% → 55%) instead of eroding. Sweep 35–58%, no collapse |
| MCTS over M5.5 fine-tuned final (policy sampler, tuned knobs) | inference-time | 90.6% (453/500) | 79.2% (396/500) | ✅ Prior best agent (+4.6pp R / +6.6pp DF over the M5 best, outside noise at n=500); seat-balanced h2h vs the raw M5 checkpoint 78.4% (392/500). Human prior + anchored RL + search compounds. Checkpoint: `models/ppo/checkpoints/bcft/ppo_step_5000000_final.pt` |
| MLP-PPO obs-v3 (type-eff + move-effect flags + Sleep Clause), raw (M7 final) | 5.0M | 70.0% (350/500) | 52.0% (104/200) | 🟨 Best raw-policy numbers in project history (prior best raw: 57% R M3.3). Checkpoint: `models/ppo/checkpoints/v3/ppo_step_5000002_final.pt` |
| MCTS over M8 Phase 2 value-head fine-tune (`--target outcome`) | inference-time value-head-only FT | — | **80.0% (160/200)** vs base **82.5% (165/200)** | ❌ **Negative result (M8 Phase 2).** Value head retrained on AlphaZero outcome targets from 2000 MCTS self-play games (66,459 decisions); trunk/policy/opp heads bit-identical, so the delta isolates leaf evaluation. **−2.5pp against a ≥+3pp gate → Criterion C failed, Phase 3 skipped.** Notable: the fine-tune *worked* in-distribution (val MSE 0.7414 → 0.5907 vs a constant-predictor baseline of 0.7396, i.e. **R² 0.00 → 0.20**; the PPO-trained head scored *below* a constant) and none of it transferred to play strength. Delta is inside noise (SE ~3.9pp at n=200) — not established as harmful, but fails a point gate. Log: `models/mcts/results/m8_phase2_valft_criterionC.log`; checkpoint `models/ppo/checkpoints/v3_valft/ppo_v3_valft_outcome.pt` |
| **MCTS over M7 obs-v3 final (policy sampler, tuned knobs)** | inference-time | **93.0% (465/500)** | **84.2% (421/500)** | ✅ **Current best agent by bot eval** (+2.4pp R / +5.0pp DF over the M5.5 best). **Ladder read (Criterion C, 100 games): Elo 1034.6 / GXE 28.2%, vs M6's 1017/23.9%** — directionally up (+9pp raw win rate) but lands in the pre-registered 25–34% inconclusive band, not a confirmed ladder win. Eval logs: `models/mcts/results/m7_v3_mcts_{random,damagefirst}.log`; ladder log: `data/replays/self_ladder/m7_ladder_run.log` |

## Reference points

- Uniform-random-over-valid-actions policy: 0–3% vs Random (untrained
  policies lose almost every game because ~5/9 actions are switches and
  RandomPlayerAI never stops attacking).
- BC-pretrained transformer (`models/checkpoints/bc_pretrain_gen1ou.pt`,
  50.5% top-1 accuracy on human gen1ou): ~45% vs Random at PPO step ~0
  (inferred from early warm-started checkpoints).

## Interpretation notes

- 150-battle evals carry roughly ±8pp noise; 500-battle evals ±4.5pp. Treat
  single-checkpoint differences under that as ties.
- **Powering an A/B (added after M8 Phase 2).** The noise figures above are for
  a *single* win rate. Comparing two arms is worse: the SE on the difference is
  ~√2× a single arm's, so at n=200/arm near p≈0.8 it is **~3.9pp**, and at
  n=500/arm **~2.5pp**. M8 Phase 2's pre-registered "≥+3pp" gate at n=200 was
  therefore underpowered — a true +3pp effect would have been detected only
  about a third of the time. **Any future A/B meant to resolve ±3pp needs
  ~500–800 battles per arm.** Also note the asymmetry this creates: a gate can
  fail on a point estimate (as Phase 2 did at −2.5pp) while the delta is still
  statistically indistinguishable from zero. Record both readings.
- **In-distribution metric gains need not transfer to play strength (M8
  Phase 2).** The value-head fine-tune moved val MSE from no-better-than-a-
  constant to R² ≈ 0.20 and still lost 2.5pp of win rate. Plausible mechanism:
  MCTS selection depends on *relative* leaf values between siblings, so a large
  share of an MSE gain that comes from learning the base rate is a constant
  offset that buys nothing. Judge value-targeting work on head-to-head play,
  not on MSE or sign agreement.
- The M3.2 run confirmed the M3 conclusion with the untrained-value-head
  problem treated: the fixes eliminated the collapse/decay mechanism (the
  policy holds at its BC plateau for 3.5M steps, and decay resumes exactly as
  the KL-anchor coefficient anneals to zero), but the transformer's ceiling
  is the BC policy itself — PPO never improves on it. Full trail in
  `MILESTONES.md` → M3.2, sweep data in
  `models/transformer/checkpoints/m32/train.log`.
- vs-DamageFirst numbers only exist from M3.3 onward (the opponent was built
  then). Backfill for older checkpoints if a comparison is ever needed.
- The MCTS rows are not new networks — they are PPO checkpoints with
  determinized UCT search at inference time (`evaluate.py --model mcts`).
  The tuned operating point (sims=100, det=1, c_puct=0.5, ~57–85ms/move) is
  the default since the post-M4 sweep; eval logs in `models/mcts/results/`
  (M5 sampler A/B: `m5_ab_*.log`; M5.5 batteries: `m55_bc_mcts_*.log`,
  `m55_bcft_mcts_*.log`, h2h `m55_bcft_h2h_*.log`).
- The M5 aux head shaped the trunk (best search-amplified numbers to date)
  but its *sampler* did not beat the policy sampler inside search — see
  `MILESTONES.md` → M5 → Reading.

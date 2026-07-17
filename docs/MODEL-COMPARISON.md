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
| **MCTS over M5.5 fine-tuned final (policy sampler, tuned knobs)** | inference-time | **90.6% (453/500)** | **79.2% (396/500)** | ✅ **Current best agent** (+4.6pp R / +6.6pp DF over the M5 best, outside noise at n=500); seat-balanced h2h vs the raw M5 checkpoint 78.4% (392/500). Human prior + anchored RL + search compounds. Checkpoint: `models/ppo/checkpoints/bcft/ppo_step_5000000_final.pt` |

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

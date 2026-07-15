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
| **MCTS over M3.3 best (M4)** — 100 sims, 4 determinizations | inference-time | **66% (330/500)** | **56% (113/200)** | ✅ **Current best agent.** Beats its own base checkpoint 60.2% (602/1000) seat-balanced h2h; 84–88ms/move. First clear improvement since M2 |

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
- The M4 MCTS row is not a new network — it is the M3.3 checkpoint with
  determinized UCT search at inference time (`evaluate.py --model mcts`).
  Search knobs (sims/c_puct/determinizations) are untuned defaults; eval
  logs in `models/mcts/results/`.

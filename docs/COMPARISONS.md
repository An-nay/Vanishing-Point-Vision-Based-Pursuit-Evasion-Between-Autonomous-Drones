# MAPPO comparison, ablations and opponent-pool stability

All runs: `city` obstacle world, original unshaped rewards for evaluation, 60 paired held-out starts per matchup,
held-out PPO Evader (seed 2002, never used in training). Training seeds 2001-2003 unless stated. Training budgets match
the main recipe: Seeker pretraining 400k steps, then joint training 600k steps. **n = 3 seeds is small**: seed-to-seed
standard deviations here are 5-40 points, so only large differences (roughly more than 20 points) are clearly real.
Treat the rest as suggestive.

Columns: **Seeker vs PPO** = Evader visible % when the joint-trained Seeker faces the held-out PPO Evader (higher = stronger
Seeker); **Evader vs held-out** = Evader visible % when the joint-trained Evader faces two held-out learned Seekers from
other training runs (lower = stronger Evader); **Joint game** = Evader visible % between the two co-trained agents.
Entries are mean +/- standard deviation across seeds. Reproduce with `ablation_report.py`.

## 1. Key ablations (leave one component out of the final recipe)

| Variant | What is removed | Seeker vs PPO | Evader vs held-out | Joint game | Wall/obstacle contact steps (Evader / Seeker) |
|---|---|---:|---:|---:|---:|
| A0 final recipe | nothing | 57.9 +/- 18.8 | 25.4 +/- 9.0 | 17.4 +/- 11.4 | 222 / 167 |
| A1 | Seeker observation history | 33.0 +/- 16.8 | 24.7 +/- 9.0 | 35.7 +/- 15.8 | 214 / 131 |
| A2 | per-step wall/obstacle penalty | 50.6 +/- 22.8 | 18.5 +/- 3.6 | 12.3 +/- 5.5 | **564 / 408** |
| A3 | opponent mixing (pure joint training) | 24.9 +/- 6.5 | 34.4 +/- 14.7 | 20.3 +/- 13.2 | 150 / 150 |
| A4 | alternating learner blocks | 47.7 +/- 19.7 | 23.5 +/- 5.0 | 24.6 +/- 11.5 | 203 / 187 |
| A5 | Seeker pretraining (cold start) | 28.4 +/- 15.1 | 37.2 +/- 9.6 | 13.2 +/- 3.4 | 130 / 136 |

Reading it:
- **Clearly matter (large effects):** opponent mixing (Seeker 58 -> 25, Evader 25 -> 34), Seeker pretraining (Seeker 58 -> 28,
  Evader 25 -> 37) and Seeker observation history (Seeker 58 -> 33). These agree with the single-seed development runs.
- **Wall penalty:** removing it does NOT hurt win rates (the Evader even looks stronger), but wall/obstacle contact more than
  doubles (564 / 408 steps per 600-step episode): both drones park against walls. So the penalty's job is to stop degenerate
  wall-hugging, not to improve the visibility score.
- **Alternating blocks:** differences (Seeker 58 vs 48) are inside the noise; alternation is at most a small help.
- Held-out Seekers here are the Phase B Seekers of seeds 2004 and 2005 (for A1 the other no-history Seekers), so the
  A0 row differs slightly from the 5-seed headline table in `STAGE2_RESULTS.md`.

## 2. MAPPO comparison

MAPPO (`mappo.py`, `train_mappo.py`): the same environment, observations, wall penalty, Seeker history, opponent mixing and
alternating blocks as the MADDPG recipe, but on-policy clipped-PPO updates (per-agent Gaussian actor, centralised value critic
per agent, GAE, time-limit bootstrapping, loss masked on steps played by a frozen opponent). Same budgets (400k + 600k steps).

| | Seeker vs PPO | Evader vs held-out | Joint game | Contact steps (Evader / Seeker) |
|---|---:|---:|---:|---:|
| MADDPG (final recipe, seeds 2001-2003) | 57.9 +/- 18.8 | 25.4 +/- 9.0 | 17.4 +/- 11.4 | 222 / 167 |
| MAPPO (same seeds) | 36.5 +/- 41.1 | 30.5 +/- 11.0 | 22.4 +/- 21.2 | **36 / 33** |

Per seed (2001 / 2002 / 2003), Evader visible % to the Seeker:

| | MADDPG | MAPPO |
|---|---|---|
| Phase B Seeker vs held-out PPO | 96 / 18 / 68 | 8 / 91 / 12 |
| Joint-trained Seeker vs held-out PPO | 81 / 35 / 58 | 4 / 94 / 11 |
| Joint game | 33 / 7 / 12 | 4 / 52 / 12 |

- MAPPO's Seekers **overfit the one PPO Evader they trained against**: they see it 96 % / 95 % / 72 % of the time, but against a
  different PPO Evader only seed 2002 holds up (94 %) and the other two collapse to 6-8 %. Stochastic and deterministic action
  selection give the same numbers, so this is genuine over-specialisation, not an evaluation artefact. MADDPG's off-policy
  replay of mixed opponents generalised better in this setup.
- MAPPO's drones almost never touch walls or obstacles (about 35 steps versus 170-220). Its stochastic Gaussian policy and
  entropy bonus avoid the wall-parking that MADDPG's deterministic actors fall into.
- Overall, on this task MAPPO was not better than MADDPG: lower and far more variable Seeker skill, slightly weaker Evader,
  much cleaner movement. With 3 seeds this is a comparison of tendencies, not a verdict.

## 3. Opponent pool and stability

The pool (`maddpg_config_pool.json`, `OpponentProvider`): every 30k steps a snapshot of both learners is stored (20 kept),
and 15 % of episodes play the learning Evader against a sampled past Seeker, 10 % the learning Seeker against a sampled past
Evader. It replaces the single frozen Seeker of the main recipe. Critic targets use the snapshot that actually played each
transition. 5 seeds, same Phase B starting Seekers and budget as the main recipe. (Evaluation here uses the 5-seed pairing of the headline
table, where each run meets the Phase B Seeker of the next seed; the ablation table in section 1 uses the Phase B Seekers of seeds 2004 and 2005,
so the pool's "Evader vs held-out" is 23.3 here and 32.3 in section 1.)

| | No pool (main recipe) | With pool |
|---|---:|---:|
| Training oscillation: std of the smoothed Evader visible % over the second half (points) | 14.7 | 10.8 |
| Peak-to-peak of that curve (points) | 54.7 | 46.0 |
| Joint-game visible % at evaluation: mean (std across 5 seeds) | 20.2 (9.9) | 16.9 (5.4) |
| Joint Seeker vs held-out PPO: mean (std across seeds) | 52.8 (26.4) | 49.8 (20.3) |
| Joint Evader vs held-out Seekers: mean (std across seeds) | 19.6 (13.9) | 23.3 (11.1) |
| Cross-play 5x5 (every Seeker vs every Evader), co-trained pairs | 16.6 | 14.7 |
| Cross-play, strangers (off-diagonal) mean | 25.2 | 25.4 |
| Cross-play gap strangers - own pair | 8.6 | 10.7 |
| Cross-play worst-case partner for the Evader (mean of column maxima) | 33.4 | 38.2 |

Summary: the pool gives **smoother training and about half the seed-to-seed spread in the joint game** (std 5.4 vs 9.9), but it
does **not** improve average strength and does **not** improve cross-play robustness (strangers 25.4 vs 25.2). With 5 seeds none of
these differences is statistically solid. Cross-play matrices: `crossplay_stage2.py` output in `runs/cross_nopool` and
`runs/cross_pool` (local, git-ignored).

## 4. Physical drones

Not done; written up as future work with the sim-to-real gaps and a staged plan in [FUTURE_WORK.md](FUTURE_WORK.md).

# MAPPO comparison, fix6 journey and opponent-pool stability

All runs: `city` obstacle world, original unshaped rewards for evaluation, 3 training seeds (2001, 2002,
2003) unless stated. Current submission recipe = **fix6** (`configs/maddpg_config_fix6.json` +
`configs/maddpg_config_fix6_phaseB.json`). The earlier `maddpg_config_final.json` recipe is deprecated;
its numbers are preserved in git history only. **n = 3 seeds is small**: seed-to-seed standard deviations
here are 14 to 42 points, so only large differences (roughly more than 25 points) are clearly real.
Treat the rest as suggestive.

Columns: **Seeker vs PPO** = Evader visible % when the joint-trained Seeker faces the held-out PPO Evader
(higher = stronger Seeker). **Joint game** = Evader visible % between the two co-trained agents
(approximately 50% = balanced equilibrium). Entries are mean +/- standard deviation across seeds.
Reproduce with `scripts/multiseed_report.py`.

## 1. Baseline to fix6 journey

The baseline recipe (symmetric reward 2.0 for both agents, 21-D Seeker observation, symmetric actor LRs
3e-4, 600k Phase C steps, no proximity bonus, no last-known Evader velocity) collapsed into a wall-hugging
Evader-dominant equilibrium. Each fix is additive on top of the previous one.

| Variant | Core change                                                                          | Joint-game visible % |
|---------|--------------------------------------------------------------------------------------|---------------------:|
| baseline | symmetric wall 2.0, 21-D Seeker obs, symmetric LRs 3e-4                            |                 18.7 |
| fix3    | asymmetric wall (Evader 3.5, Seeker 2.0) + last-known Evader velocity + Seeker proximity bonus 0.2 |           75.7 +/- 9.3 |
| fix4    | fix3 minus the briefly-tried seeker_visible_bonus / evader_hidden_bonus magnitude asymmetry |      79.2 +/- 8.9 |
| fix5    | fix4 compressed to a 400k Phase C quick-feedback run                                 |          tested only |
| **fix6** | fix4 + asymmetric actor LRs (Evader 2e-3, Seeker 3e-4, 6x faster) + 1.5M Phase C steps |  **46.3 +/- 14.3** |

Reading it:
- fix3 and fix4 both tilted the game Seeker-ward (75 - 79 % visible = Seeker dominates). The dense
  Seeker proximity bonus and the last-known Evader velocity were strong enough to flip who wins.
- fix6 kept all those shaping changes and rebalanced the game through asymmetric learning rates. Making
  the Evader six times faster at updating lets it catch up with the strengthened Seeker. The resulting
  46.3 % is the submission's headline: a balanced two-sided equilibrium.
- A balanced equilibrium is harder to produce than a one-sided win in either direction. Reward shaping
  and observation changes are the levers that got us there, not the choice of MADDPG itself.

## 2. fix6 multi-seed results

From `runs/fix6_multiseed.md` (`scripts/multiseed_report.py --prefix runs/fix6_train --seeds 2001 2002 2003`):

| Experiment              | Visible % (mean +/- std) | Longest hidden s | Evader boundary steps | Reacq time s     |
|-------------------------|-------------------------:|-----------------:|----------------------:|-----------------:|
| Phase B Seeker vs PPO   |             56.9 +/- 34.2 |      16.5 +/- 16.6 |            111 +/- 65 |      1.45 +/- 0.91 |
| Joint Seeker vs PPO     |             25.8 +/- 13.8 |       36.5 +/- 6.0 |           266 +/- 116 |      5.40 +/- 4.20 |
| **Joint self-play**     |           **46.3 +/- 14.3** |      25.4 +/- 10.3 |          154 +/- 134 |      2.54 +/- 1.90 |

- Phase B Seeker vs PPO is high-variance across seeds (std 34.2); seed 2002's Phase B Seeker is weak.
- The joint Seeker is worse than the Phase B Seeker against a frozen PPO opponent (25.8 vs 56.9). That is
  the expected cost of removing opponent mixing from Phase C: the Seeker's policy drifts toward beating
  the current Evader, not toward beating PPO.
- Evader boundary steps dropped from 420 at baseline to 154 at fix6: the asymmetric wall penalty killed
  the wall-hugging reward exploit.

## 3. MAPPO comparison

MAPPO (`vp/mappo.py`, `scripts/train_mappo.py`) is implemented as the on-policy ablation: the same
environment, observations, boundary penalty and alternating blocks as the MADDPG recipe, but on-policy
clipped-PPO updates (per-agent Gaussian actor, centralised value critic per agent, GAE, time-limit
bootstrapping, loss masked on steps played by a frozen opponent).

The MAPPO comparison in this repo was last run against the pre-fix3 `maddpg_config_final` recipe. The
qualitative finding still holds and is worth stating directly:

- **MAPPO Seekers overfit the single training PPO Evader** they trained against. They see that specific
  Evader 72 to 96 % of the time, but against a different-seed PPO Evader two of three seeds collapse to
  6 - 8 % visible. MADDPG's off-policy replay generalised better across unseen PPO opponents.
- **MAPPO drones almost never touch walls or obstacles** (about 35 contact steps versus 170 - 220 for
  MADDPG). Its stochastic Gaussian policy and entropy bonus avoid the wall-parking that MADDPG's
  deterministic actors can fall into. Under the fix6 asymmetric wall penalty this gap would narrow.
- Overall MAPPO was not stronger than MADDPG on this task. Lower and far more variable Seeker skill,
  slightly weaker Evader, much cleaner movement. We did not re-run MAPPO under the fix6 recipe because
  the overfitting finding (not the raw numbers) was the headline MAPPO result.

A proper fix6-recipe MAPPO re-run is listed in [FUTURE_WORK.md](FUTURE_WORK.md).

## 4. Opponent pool and stability (deprecated recipe)

The pool (`configs/maddpg_config_pool.json`, `OpponentProvider`) was tested as a stability improvement
over the pre-fix3 recipe. Every 30k steps a snapshot of both learners is stored (20 kept), and 15 % of
episodes play the learning Evader against a sampled past Seeker, 10 % the learning Seeker against a
sampled past Evader. It replaced the single frozen Seeker of the pre-fix3 recipe.

Under the pre-fix3 recipe the pool gave smoother training and about half the seed-to-seed spread in the
joint game (std 5.4 vs 9.9). It did **not** improve average strength and did **not** improve cross-play
robustness. Under fix6 the pool is unused: Phase C is pure self-play with all `mix_*` set to 0.
Re-evaluating the pool under fix6 is listed as future work.

## 5. Physical drones

Not done. Written up as future work with the sim-to-real gaps and a staged plan in
[FUTURE_WORK.md](FUTURE_WORK.md).

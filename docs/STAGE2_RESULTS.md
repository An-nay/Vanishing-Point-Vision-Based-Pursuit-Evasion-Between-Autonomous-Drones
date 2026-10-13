# Stage 2 results (obstacle world, final recipe)

**Stage 2 contains no scripted controller.** Both agents are neural policies trained model-free with
actor-critic policy-gradient updates (MADDPG, deterministic policy gradient). The only fixed opponent used during
training is a frozen PPO Evader (PPO is also a model-free policy-gradient method) and a frozen *learned* Seeker.
There is no value iteration, policy iteration, planning, search or learned/assumed environment model anywhere.
The scripted Seeker remains Stage 1 only (`--stage1-reference` / `--baseline-only` in the evaluator).

All numbers: `city` obstacle world (6 solid boxes, line of sight blocked), ORIGINAL unshaped rewards, 60 paired
held-out starts per matchup (seeds 10001/11001/12001 x 20), 95 % bootstrap CIs. Training used PPO seed 2001 as the
frozen Evader opponent; evaluation uses PPO seed 2002, so no checkpoint is scored against a policy it trained against.

## Headline (5 training seeds)

"Visible %" = share of the 60 s episode the Evader is in the Seeker's view. Lower is better for the Evader.

| Matchup | Seed 2001 | 2002 | 2003 | 2004 | 2005 | Mean |
|---|---:|---:|---:|---:|---:|---:|
| Phase B Seeker (trained alone vs PPO) vs held-out PPO Evader | 96.4 | 17.9 | 67.6 | 87.1 | 51.7 | **64.1** |
| Joint-trained Seeker vs held-out PPO Evader | 80.8 | 34.7 | 58.1 | 11.8 | 78.7 | **52.8** |
| Joint MADDPG Seeker vs joint MADDPG Evader | 33.3 | 7.3 | 11.6 | 20.2 | 28.8 | **20.2** |
| Joint MADDPG Evader vs a held-out learned Seeker (another seed's Phase B Seeker) | 5.2 | 4.9 | 19.5 | 41.7 | 26.7 | **19.6** |
| PPO Evader vs those same held-out Seekers (same starts) | 17.9 | 67.6 | 87.1 | 51.7 | 96.4 | 64.1 |

Reading it:
- A learned Seeker finds a PPO Evader much of the time (64 % / 53 % on average), but results vary a lot by seed
  (12 % to 96 %); report the spread, not just the mean.
- The jointly trained Evader is seen far less than the PPO Evader by the same held-out learned Seekers in all 5
  pairings (19.6 % vs 64.1 % on average). Caveat: those Seekers were trained against PPO-type Evaders, which favours
  them against PPO, so this comparison is tilted against PPO.
- In the joint game neither side dominates consistently (7 % to 33 % visible).
- Per-seed tables with CIs: `runs/final2_eval_s{2001..2005}/comparison.md` (local, git-ignored).

## Proposal metrics (5 seeds x 60 starts = 300 episodes per row)

Computed by `scripts/evaluate_stage2.py` (see `comparison.md` in each evaluation folder). "Hidden-start" evaluation points the
Seeker's camera away from the Evader at step 0 (`--hidden-start`), so time-to-first-detection measures real search time.

| Matchup | Evader visible % (visible start) | Evader visible % (hidden start) | First detection, hidden start: mean / median s (episodes detected) |
|---|---:|---:|---|
| Phase B Seeker vs PPO Evader | 64.1 | 60.0 | 4.2 / 2.0 s (290 of 300) |
| Joint-trained Seeker vs PPO Evader | 52.8 | 47.1 | 6.8 / 2.1 s (280 of 300) |
| Joint Seeker vs joint Evader | 20.2 | 18.4 | 6.1 / 2.2 s (265 of 300) |
| Held-out learned Seeker vs joint Evader | 19.6 | 14.6 | 6.2 / 2.0 s (254 of 300) |

- Time to first detection is 0 s by definition when the episode starts visible. The mean is over detected episodes only;
  episodes where the Seeker never finds the Evader within 60 s are counted separately (the "of 300" column), not averaged in.
  Against the joint Evader the Seeker fails to find it at all in 12-15 % of hidden-start episodes.
- Boundary and collision violations (mean per 60 s episode, visible start):

| Matchup | Evader wall/obstacle contacts | Seeker wall/obstacle contacts | Evader steps in contact | Seeker steps in contact | Drone-drone collision steps |
|---|---:|---:|---:|---:|---:|
| Phase B Seeker vs PPO Evader | 10.3 | 8.4 | 170 | 68 | 2.4 |
| Joint Seeker vs PPO Evader | 13.0 | 8.5 | 196 | 125 | 1.8 |
| Joint Seeker vs joint Evader | 13.5 | 14.8 | 262 | 155 | 0.2 |
| Held-out learned Seeker vs joint Evader | 13.9 | 19.1 | 293 | 127 | 0.2 |

  "Contacts" count new contact streaks with a wall or obstacle; "steps in contact" counts every step spent touching.
  Drone-drone collisions (centres closer than two drone radii, 0.7 m) are measured only; they are not penalised in training.
  They are rare (under 2.5 steps per episode).
- Reward convergence: `docs/figures/final2_joint/learning_curves.png` (MADDPG, 5 seeds) and
  `docs/figures/ppo_city_convergence.png` (PPO plain vs curriculum, 3 seeds). Competing learners oscillate rather than
  converge monotonically (the PPO curriculum curve starts high because its early stages use an easier Seeker; it is only comparable to plain PPO from the final stage, after about 375k steps, where both sit near 410-430); individual seeds swing between roughly 5 % and 80 % visible time during joint training, and the seed mean ends around 25 %.

See also [COMPARISONS.md](COMPARISONS.md) (MAPPO, ablations, opponent pool) and [FUTURE_WORK.md](FUTURE_WORK.md) (physical drones).

## Recipe (`configs/maddpg_config_final.json`, `configs/maddpg_config_final_phaseB.json`)

1. Phase B: train the Seeker alone against a frozen PPO Evader (400k steps).
2. Phase C: warm-start from that Seeker, 600k steps of joint training with per-episode opponent mixing:
   50 % frozen PPO Evader, 25 % frozen learned Seeker (the Phase B Seeker), 25 % self-play, alternating learner
   blocks of 20k steps.
3. Seeker observation includes 4 past observations spaced 5 steps apart (`seeker_history`).
4. Per-step wall/obstacle penalty 2.0 and action regulariser 0.05 (training only; evaluation is unshaped).

## What mattered (seed 2001, 200k steps, evaluated vs held-out PPO)

| Change | Effect |
|---|---|
| Per-step wall penalty >= the +1 hidden reward | Removed Seeker corner-parking; 0.5 was not enough |
| More updates per step, longer training, lower LR, action regulariser alone, warm start alone | No reliable improvement |
| Seeker observation history (4 x 5 steps) | Dedicated Seeker vs PPO 45 % -> 83 % mean over 3 seeds. Biggest single gain |
| Opponent mixing + alternating learner blocks | Made joint training keep the Seeker's skill instead of collapsing |
| Larger wall penalty (3, 5) | Worse Seeker |

Earlier experiments also tried mixing in the scripted Seeker. That was removed for the final Stage 2 recipe because
the assignment requires policy-gradient, model-free learning (the scripted controller is a hand-written rule policy);
the results above were re-run without it.

Implementation notes found on the way: a frozen opponent's next action may enter the OTHER agent's critic target but
never its own critic (doing so made the Evader worse).

## Caveats

- 5 training seeds; seed 2002's Phase B Seeker is weak (18 %) and seed 2004's joint Seeker is weak vs PPO (12 %).
- The Evader sees the true Seeker state while the Seeker only has a history of detections; the Evader (2.2 m/s) is
  faster than the Seeker (1.8 m/s).
- Wall and obstacle contacts remain common for the learned Seeker in some runs; the penalty is a training aid.
- Evaluation starts are visible by default; `--hidden-start` tests the other case.

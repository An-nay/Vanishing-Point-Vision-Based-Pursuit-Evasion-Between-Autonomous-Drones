# Stage 2 results (obstacle world, fix6 recipe)

**Stage 2 contains no scripted controller.** Both agents are neural policies trained model-free with
actor-critic policy-gradient updates (MADDPG, deterministic policy gradient). There is no value iteration,
policy iteration, planning, search or learned/assumed environment model anywhere. The scripted Seeker
remains Stage 1 only (`--stage1-reference` / `--baseline-only` in the evaluator). The frozen PPO Evader
is used only during Phase B pre-training; Phase C is pure two-sided self-play with no frozen opponents
mixed in.

All numbers: `city` obstacle world (6 solid boxes, line of sight blocked), ORIGINAL unshaped rewards,
3 independent training seeds (2001, 2002, 2003), evaluated per-seed via
`scripts/multiseed_report.py --prefix runs/fix6_train --seeds 2001 2002 2003`.
Mean +/- stddev reported across seeds.

## Headline (fix6, 3 training seeds)

"Visible %" = share of the 60 s episode the Evader is in the Seeker's view. Higher = better for the Seeker.

| Matchup                                   | Visible % (mean +/- std) | Longest hidden s | Evader boundary steps | Reacq time s     |
|-------------------------------------------|-------------------------:|-----------------:|----------------------:|-----------------:|
| Phase B Seeker vs frozen PPO Evader       |            56.9 +/- 34.2 |      16.5 +/- 16.6 |            111 +/- 65 |      1.45 +/- 0.91 |
| Joint Seeker vs frozen PPO Evader         |            25.8 +/- 13.8 |       36.5 +/- 6.0 |           266 +/- 116 |      5.40 +/- 4.20 |
| **Joint Seeker vs joint Evader (fix6)**   |          **46.3 +/- 14.3** |      25.4 +/- 10.3 |          154 +/- 134 |      2.54 +/- 1.90 |

Reading it:
- The headline self-play equilibrium (46.3 % +/- 14.3) is roughly balanced: neither side dominates.
  The starting point before any of the fix3-6 shaping changes was 18.7 % visible, Evader-dominant,
  with the Evader hugging a wall for 420 of 600 steps per episode.
- Phase B Seeker vs PPO Evader is high-variance across seeds (std 34.2); individual seeds swing between
  about 25 % and 90 %. Report the spread, not just the mean.
- Against the frozen PPO Evader the joint Seeker is weaker than the Phase B Seeker (25.8 vs 56.9) --
  symptom of Phase C self-play drifting the Seeker's policy toward whatever beats the current Evader,
  not toward whatever beats PPO. This is a known cost of removing opponent mixing from Phase C.
- Per-seed CSVs: `runs/fix6_train_s{2001,2002,2003}/eval/episodes.csv` and `runs/fix6_multiseed.md`.

## Recipe (`configs/maddpg_config_fix6.json`, `configs/maddpg_config_fix6_phaseB.json`)

### Phase A: Smoke test
A short validation run to confirm env + agents + replay + logging work end-to-end before spending real
compute.

### Phase B: Seeker pre-training (400,000 steps)
Only the Seeker learns. The Evader is a frozen PPO policy from Stage 1.
Actor LR 3e-4 (symmetric), critic LR 1e-3, noise 0.3 -> 0.05 over 300k steps, `include_last_known_velocity = true`.
Produces the warm-start Seeker checkpoint that Phase C initialises from.

### Phase C: Pure self-play (1,500,000 steps)
Both agents learn jointly, warm-started from Phase B.
`mix_ppo_evader = mix_frozen_seeker = mix_pool_seeker = mix_pool_evader = 0.0`. No frozen or scripted
opponents are mixed in. Alternating learner blocks of 20k steps.

**Asymmetric actor learning rates** (fix6 key choice): Seeker 3e-4, Evader 2e-3. The Evader is six times
faster at updating so it can keep pace with a now-competent Seeker after the reward-shaping and observation
changes below.

### Reward shaping (fix6)
- Detection reward: +1 per step to the hidden agent, -1 to the visible agent (unchanged from Stage 1).
- **Asymmetric boundary penalty** (fix3): Evader -3.5 per contact step, Seeker -2.0 per contact step.
  Replaces the symmetric 2.0 of the earlier recipe. Kills the wall-hugging exploit the Evader used to
  break the game at 18.7 % visible.
- **Seeker proximity bonus** (fix3): `+0.2 / (1 + distance_to_evader)` per step. Gives the Seeker a dense
  gradient during long hidden stretches instead of a flat -1 per step.
- Action regulariser 0.05 (training only; evaluation is unshaped).

### Observation shaping (fix6)
- Evader observation: 21-D unchanged (own pos+vel, relative Seeker pos+vel, Seeker camera forward,
  6 wall distances). With obstacles on, add 9 D for the 3 nearest obstacles.
- Seeker observation: 33-D base = 21 base + **3 last-known Evader velocity** (fix3,
  `include_last_known_velocity: true`) + 9 for 3 nearest obstacles. 4-frame history, stacked 5 steps apart,
  on top. The last-known velocity lets the Seeker dead-reckon through occlusions instead of arriving at the
  last-known position with no idea where the Evader went next.

### Noise and training horizon
- Phase B: noise 0.3 -> 0.05 linearly over 300,000 steps, total 400,000 steps.
- Phase C: noise 0.3 -> 0.1 linearly over 1,100,000 of 1,500,000 steps. The final 400k steps hold a floor
  of 0.1 instead of annealing to zero, so the agents keep exploring late in training.

## Baseline -> fix6 journey

| Experiment | Phase C setup                                                 | stage2_joint visible % |
|------------|---------------------------------------------------------------|------------------------:|
| baseline   | symmetric wall penalty 2.0, 21-D Seeker obs, equal LRs        |                   18.7 |
| fix3       | asymm wall 3.5/2.0, +last-known vel, +proximity bonus         |            75.7 +/- 9.3 |
| fix4       | fix3 minus seeker_visible_bonus / evader_hidden_bonus asymm   |            79.2 +/- 8.9 |
| fix5       | fix4 with 400k Phase C quick-feedback run                     |             tested only |
| **fix6**   | fix4 + asymm actor LRs (Evader 6x Seeker), 1.5M Phase C steps |       **46.3 +/- 14.3** |

fix3 and fix4 both tilted the game Seeker-ward (75-79 % visible = Seeker dominates). fix6 added the
asymmetric learning rates so the Evader could catch up, landing at a balanced 46.3 % equilibrium.
"Balanced" is the design goal: a 50 / 50 equilibrium between an informed Evader and a sensor-restricted
Seeker is harder to produce than a one-sided win in either direction.

## What mattered (ablation summary)

| Change                                             | Effect on fix6 recipe                                                    |
|----------------------------------------------------|--------------------------------------------------------------------------|
| Asymmetric wall penalty (Evader 3.5 / Seeker 2.0)  | Killed wall-hugging exploit. Evader boundary steps 420 -> 76-140.        |
| Last-known Evader velocity in Seeker obs           | Reacq time 3.92 s -> 1.13 s (fix3-5 best); longest hidden 37.2 s -> 10.4 s. |
| Seeker proximity bonus                             | Gave Seeker a dense gradient during hidden stretches; reliable search.   |
| Asymmetric actor LRs (Evader 2e-3 / Seeker 3e-4)   | Rebalanced the game from Seeker-dominant 79 % to balanced 46 %.          |
| Removing opponent mixing (pure self-play Phase C)  | Assignment-compliant, but joint Seeker vs PPO Evader regressed (25.8 %). |
| Noise floor 0.1 (not 0.05) over 1.1M of 1.5M steps | More exploration late in training; small individual effect.              |
| Phase C 1.5M steps (vs 600k)                       | More time for self-play to settle; needed alongside the above.           |
| Per-episode seed = `seed * 1_000_003 + episode`    | Fixed cross-run contamination so independent seeds are truly independent.|

## Comparison with the earlier `maddpg_config_final` recipe

The pre-fix3 `maddpg_config_final.json` recipe is retained in the repo for history. It used symmetric
wall penalty 2.0, 21-D Seeker observation (no last-known velocity, no proximity bonus), symmetric actor
LRs 3e-4, 600k Phase C steps, and **per-episode opponent mixing in Phase C** (50 % frozen PPO Evader,
25 % frozen Phase B Seeker, 25 % self-play). It is NOT the submission recipe. Its headline numbers
(52.8 / 19.6 / 64.1 / 20.2) are listed in `git log` for the pre-fix3 commits only; do not quote them for
fix6.

See also [COMPARISONS.md](COMPARISONS.md) (MAPPO, pre-fix3 ablations, opponent pool) and
[FUTURE_WORK.md](FUTURE_WORK.md) (physical drones).

## Caveats

- 3 training seeds. seed 2002's Phase B Seeker is weak; seed-to-seed variance is substantial (std 14 - 34).
- The Evader sees the true Seeker state while the Seeker only has detections and last-known
  position/velocity; the Evader (2.2 m/s) is faster than the Seeker (1.8 m/s).
- Wall and obstacle contacts remain common for the learned Seeker in some runs; the penalty is a
  training aid, not a hard constraint.
- Evaluation starts are visible by default; `--hidden-start` tests the other case for first-detection
  timing.
- "Balanced" here means approximately 50 / 50 share-of-episode visible; it does not mean the two
  policies are strategically identical, only that neither one wins the share-of-time game decisively.

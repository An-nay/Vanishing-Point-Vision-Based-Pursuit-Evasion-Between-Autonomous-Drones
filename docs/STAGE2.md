# Stage 2: simultaneous learning with MADDPG

Stage 1's PPO commands, saved PPO checkpoints, 21-value Evader observation,
scripted Seeker and existing tests remain supported. Both RL interfaces now
advance the same NumPy simulator. The random-Evader prototype is retained
unchanged for provenance.

## Files

| File | Purpose |
|---|---|
| `vanishing_point_core.py` | Arena constants, original FOV geometry, velocity scaling, scripted Seeker, joint dynamics, memory and statistics |
| `vanishing_point_render.py` | Original arena/drone/debug rendering extracted into a lazy PyBullet mixin |
| `vanishing_point_ppo.py` | Existing Gymnasium PPO wrapper, training, curriculum and evaluation CLI |
| `vanishing_point_parallel.py` | PettingZoo Parallel API with `seeker` and `evader` |
| `maddpg.py` | Local actors, centralized critics, targets, replay, optimization and checkpoints |
| `maddpg_config.json` | All Stage 2 network, optimization, exploration and environment configuration |
| `stage2_opponents.py` | Frozen PPO, random and historical MADDPG opponents |
| `train_maddpg.py` | Phases A, B and C, multi-seed runs and CSV logging |
| `evaluate_stage2.py` | Phase D, four-way paired evaluation and historical opponents |
| `plot_stage2.py` | Learning curves, visibility comparisons and 3D trajectories |
| `test_stage2_environment.py` | Joint dynamics, FOV, camera, information restrictions, rewards and API tests |
| `test_maddpg.py` | Learning, replay alignment, targets, frozen opponents, checkpoint and reproducibility tests |
| `test_stage2_evaluation.py` | Paired starting conditions, saved PPO evaluation and censored-event aggregation |

All Python modules remain flat so existing imports and commands work.

## Setup and first validation

Use the existing working environment. For CPU-only PyTorch installation, install
its official CPU wheel before the other requirements:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pytest -q
python train_maddpg.py --phase validate --episodes 2
```

Headless environment validation never imports PyBullet or opens a GUI.
PyBullet is loaded only when requesting `--render`. A rendering-enabled run
uses the same NumPy world and sleeps 0.1 seconds per step for playback.

## Shared dynamics and rewards

The arena is 20 x 20 x 10 m, with a 0.4 m boundary margin. The timestep is
0.1 s and the default episode lasts 600 steps. Seeker and Evader speed
magnitudes are capped at 1.8 and 2.2 m/s respectively. Actions are clipped
to [-1, 1]; vectors longer than one are normalized before scaling.

Visibility is the original circular 90-degree cone with a 10 m range,
computed geometrically after both positions and camera orientation update.
There is no rotor physics, obstacle occlusion or inter-drone collision reward.

- Evader: +1 hidden, -1 visible, -2 on the start of a wall-contact streak.
- Seeker: +1 visible, -1 hidden, its own -2 on the start of a wall-contact streak.

A contact streak ends when the proposed position is strictly inside all
boundaries. Leaving and touching again incurs another penalty, matching
Stage 1. Wall contacts and detection never terminate episodes. Both agents
truncate together at the time limit; final observations are returned for replay.

No extra reward shaping is enabled. Visibility components are opposite, but
wall penalties make the full rewards non-zero-sum.

### Simultaneous actions and camera updates

`step({"seeker": seeker_action, "evader": evader_action})` requires both
actions. Dimensions and finite values are checked for both before mutating
the world. Each actor receives observations from the same pre-transition
world. Camera and position changes are committed once per joint step.

The learned Seeker action is `[vx, vy, vz, yaw_rate, pitch_rate]`, with
the last two values normalized to [-1, 1]. Defaults:

```text
yaw_new   = wrap_to_pi(yaw_old + action[3] * radians(100) * 0.1)
pitch_new = clip(pitch_old + action[4] * radians(90) * 0.1, -89deg, +89deg)
forward   = [cos(pitch)*cos(yaw), cos(pitch)*sin(yaw), sin(pitch)]
```

Yaw is about the world's z axis; pitch is elevation above the xy plane.
Movement does not automatically rotate a learned Seeker's camera.

The original scripted baseline uses `step_scripted(evader_action)`. It keeps
its original instantaneous TRACK/SEARCH turns and SCAN sweep. This adapter
uses the same position dynamics, visibility and rewards; its camera behavior
is intentionally preserved for Stage 1 comparisons. The learned camera's
turn-rate restriction is a difference to account for when interpreting results.

## World state versus actor observations

The simulator holds both true positions and velocities, camera yaw/pitch,
visibility, last-detected target position, detection age, boundary-contact
flags and episode time. Rendering and evaluation may read this complete world.

Each actor accepts just its own 21-value observation. Actor inference
never accepts a global state, an opponent observation or an opponent action.

### Evader: the saved PPO-compatible layout

| Indices | Values |
|---|---|
| 0:3 | Own position centered/scaled by arena dimensions |
| 3:6 | Own velocity / 2.2 |
| 6:9 | Relative Seeker position / arena dimensions |
| 9:12 | Relative Seeker velocity / (1.8 + 2.2) |
| 12:15 | Seeker camera forward vector |
| 15:21 | Distances to six walls / arena dimensions |

Visibility remains a reward signal rather than an explicit Evader observation,
as in the saved 21-value PPO policies. This Evader observation deliberately
retains Stage 1's greater knowledge of the opponent.

### Seeker: restricted target information

| Indices | Values |
|---|---|
| 0:3 | Own normalized position |
| 3:6 | Own velocity / 1.8 |
| 6:9 | Own camera forward vector |
| 9:15 | Own distances to six walls |
| 15 | Current visibility flag |
| 16:19 | Relative currently detected target, or relative last-seen target |
| 19 | Whether any target detection has occurred |
| 20 | Steps since last detection / episode length |

When hidden, the relative target is computed from the stored last-seen
absolute position and current Seeker position. It is not updated from the
hidden target's movement. Before any detection, the target vector is zero
and the known flag is false. Current target velocity, distance and FOV angle
are absent from both Seeker observations and its step info.

### Centralized critics

Each agent has its own critic. During training only, each critic receives
`state()` (26 normalized values) and the concatenated Seeker/Evader actions
(8 values). State contains true positions and velocities, camera forward,
last-known target position, known flag, detection age, visibility, both
boundary-contact flags, episode progress and camera yaw/pitch.

This lets critics assess joint outcomes despite partial local observations.
Target actions are produced from each agent's own next observation. Critics
and global state are not needed to execute the saved actors.

## Training

All defaults are in `maddpg_config.json`: two 128-unit ReLU layers, tanh
actors, actor LR 0.0003, critic LR 0.001, batch size 128, replay 100,000,
gamma 0.99, tau 0.01, gradient norm cap 10, 2,000 warmup steps, an update
every 10 steps, Gaussian exploration decaying from 0.3 to 0.05 over
150,000 steps, and checkpoints every 25,000 steps. CPU is the default and
only required device; one PyTorch thread avoids overhead on these small MLPs.

### Phase A: validate random joint actions

```powershell
python train_maddpg.py --phase validate --episodes 2
python train_maddpg.py --phase validate --episodes 1 --render
```

This runs PettingZoo's `parallel_api_test`, then reports random rollouts.

### Phase B: learn the Seeker against a fixed opponent

Use the existing PPO model:

```powershell
python train_maddpg.py --phase seeker --opponent ppo --opponent-model models/ppo_500k.zip --output runs/phase_b --seeds 2001
```

For a quick random-opponent check:

```powershell
python train_maddpg.py --phase seeker --opponent random --output runs/phase_b_random --total-steps 10000
```

Only the Seeker actor/critic and targets update. Replay stores the actual
frozen opponent's actions. Critic targets query the same frozen policy on
batched next observations; they do not use the untrained MADDPG Evader.
A random opponent is a fixed policy distribution rather than a learned actor.

PPO is loaded through Stable-Baselines3 on CPU with deterministic actions.
Its weights are never transferred into a MADDPG actor. The checkpoint's
unused Evader actor has not been trained in Phase B.

An older Evader can also be frozen with
`--opponent maddpg --opponent-model <checkpoint.pt>`.

### Phase C: simultaneous learning

```powershell
python train_maddpg.py --phase simultaneous --output runs/phase_c --seeds 2001 2002 2003
```

Optionally warm-start from Phase B:

```powershell
python train_maddpg.py --phase simultaneous --initialize runs/phase_b/seeker_s2001/final.pt --output runs/phase_c_warm --seeds 2001
```

Warm-start requires compatible network sizes and spaces. It loads actors,
critics and targets, but starts fresh optimizers, replay and training counters.
The Phase B Evader is still an initial actor, so Phase C begins its learning.

Each seed writes a separate directory such as
`runs/phase_c/simultaneous_s2001/` containing:

- `run.json`: resolved config, phase, trained agents and opponent sources.
- `episodes.csv`: returns, detection, first loss of sight, hidden streaks,
  each agent's contact count and boundary steps, and reacquisition statistics.
- `updates.csv`: actor and critic losses for each active learner.
- `checkpoint_XXXXXXXX.pt` and `final.pt`: networks, targets, optimizers,
  architecture/config, step/update counts and exploration/torch RNG states.

Output directories must be empty to prevent overwriting a prior experiment.
A final incomplete episode is logged with `partial=True`; learning-curve
plots exclude partial episodes.

Replay records both terminations and truncations. Only true termination
masks bootstrapping. Time-limit truncation uses the final next world/observation,
never the following reset state. This treats the 600-step cutoff as an
external time limit.

Checkpoints are loadable for inference and warm-starting. Full interruption
resume is not provided: replay and in-progress world/opponent RNG state are
not included. Runs from the same config/seed are reproducible on the tested
CPU stack; exact agreement across different library versions is not promised.

## Phase D: paired comparisons and robustness

```powershell
python evaluate_stage2.py --ppo-model models/ppo_500k.zip --seeker-checkpoint runs/phase_b/seeker_s2001/final.pt --joint-checkpoint runs/phase_c/simultaneous_s2001/final.pt --seeds 10001 11001 12001 --episodes 3 --output runs/evaluation_s2001
```

This evaluates:

| Experiment | Seeker | Evader |
|---|---|---|
| Stage 1 baseline | Original scripted | PPO |
| Learned Seeker test | Phase B MADDPG | Frozen PPO |
| Stage 2 | Joint MADDPG | Joint MADDPG |
| Cross-evaluation | Original scripted | Joint MADDPG |

If `--seeker-checkpoint` is omitted, the learned-Seeker/PPO test uses the
joint checkpoint's Seeker and records that source in `evaluation.json`.
Use a Phase B checkpoint when specifically testing independent Seeker learning.

Each matchup receives identical held-out starting positions and initial camera
direction for each episode. Default evaluation seeds are separated from the
default training seed. To assess training stability, repeat this command for
each independently trained checkpoint on the same held-out seeds, using a
different output directory.

Add historical policies with `--historical-evaders <one.pt> <two.pt>` and/or
`--historical-seekers <one.pt> <two.pt>`. Their environment settings must
match. Add `--hidden-start` for a separate experiment starting with the
camera facing away from the target; initial detection memory is empty.

For a Stage 1-only reference:
```powershell
python evaluate_stage2.py --baseline-only --output runs/stage1_reference
```

Outputs:

- `starts.json`: exact paired starting conditions.
- `evaluation.json`: policy sources and environment settings.
- `episodes.csv`: per-episode statistics.
- `comparison.csv`: per-experiment means, standard deviations and valid counts.
- `comparison.md`: concise comparison table.
- `trajectories/*.csv`: all 600 steps plus the reset frame, including both
  positions, camera direction and visibility.
- `visibility_comparison.png` and `trajectories_3d.png`: standalone figures.

Visibility percentages count post-transition timesteps. First loss of sight
is the first visible-to-hidden transition, with 0.1 s timing. Default starts
are visible, so time-to-first-detection is intentionally not a headline metric.
Hidden-start episodes may have no first-loss event.

Wall contacts count new contact streaks. Boundary steps count every proposed
position touching or exceeding a boundary; actual positions are always clipped.
Reacquisition delay is the duration of a completed hidden streak. Episodes
without reacquisition have blank delay, not zero. The comparison includes
valid-event episode counts, loss/reacquisition counts and unresolved hidden
duration, so persistent failures are not silently treated as quick recovery.
The reported mean delay is a mean of per-episode completed-streak means.
In a hidden-start experiment, an initial discovery completes the initial
hidden streak even though it was not preceded by a loss-of-sight event.

Plot learning curves for one or several independent training runs:

```powershell
python plot_stage2.py --runs runs/phase_c/simultaneous_s2001 runs/phase_c/simultaneous_s2002 runs/phase_c/simultaneous_s2003 --output runs/phase_c/plots
```

This plots each seed separately; comparison standard deviations describe
evaluation episodes of the supplied checkpoint, not uncertainty across
training seeds.

## Validation and limits

Run the complete automated suite with `python -m pytest -q`.
Tests cover joint motion and speed caps, FOV geometry, bounded camera updates,
hidden-target non-leakage, independent wall rewards and real leave/re-entry,
episode truncation/reset, finite spaces/actions, the PettingZoo contract,
Stage 1 equivalence, replay alignment, target handling, frozen opponents,
both actors updating, soft targets, checkpoint reload and reproducible CPU
training. Short tests intentionally shorten the horizon; normal training
and paired evaluation retain the original 600-step limit.

The learning problem remains asymmetric: the Evader knows the true Seeker
state, while the Seeker only gets last-seen target information. MADDPG MLPs
are memoryless apart from explicit detection memory in observations. Long
joint training can be unstable in competitive settings; smoke-test success
establishes pipeline correctness, not convergence or superiority to PPO.

Implementation references:
[PettingZoo Parallel API](https://pettingzoo.farama.org/api/parallel/) and
[the MADDPG paper](https://arxiv.org/abs/1706.02275).

Executed checks and smoke-run artifacts are recorded in [STAGE2_VALIDATION.md](STAGE2_VALIDATION.md).

# Vanishing Point

A pursuit-evasion RL project: a drone **Evader** learns to stay out of a **Seeker** drone's camera view inside a bounded 3D arena. Stage 1 trains a PPO Evader against a rule-based Seeker; Stage 2 trains BOTH drones with model-free actor-critic policy-gradient learning (MADDPG) and contains no scripted controller. No value/policy iteration or planning is used anywhere.

- **Phase 1** — built the physics/geometry simulator, a scripted Seeker (TRACK / SCAN / SEARCH), and a random-Evader baseline to sanity-check the environment.
- **Phase 2** — trained the Evader with PPO, both plainly and with a 4-stage difficulty curriculum, validated across 3 seeds, and added an automated test suite.
- **Stage 2 (implemented, trained, evaluated)** — PettingZoo two-agent environment, MADDPG with local actors and centralised critics, an optional **obstacle world** (solid boxes that block the Seeker's line of sight), Seeker observation history, and opponent-mixed training. See [Stage 2 documentation](docs/STAGE2.md) and [results](docs/STAGE2_RESULTS.md).

Full write-up, tables, and figures: [`docs/Vanishing_Point_Full_Overview.pdf`](docs/Vanishing_Point_Full_Overview.pdf).

## Repo structure

```
Codes/
├── README.md, requirements.txt, pytest.ini, .gitignore
│
├── vp/                              # the library: imported by everything, never run directly (except vp.ppo)
│   ├── core.py                      #   NumPy simulator: dynamics, FOV, obstacles + occlusion, Stage 1 scripted Seeker
│   ├── ppo.py                       #   Stage 1 Gymnasium env + PPO train/eval CLI  (run: python -m vp.ppo ...)
│   ├── parallel.py                  #   Stage 2 PettingZoo two-agent env
│   ├── render.py                    #   lazy PyBullet 3D viewer (quadcopters, obstacles, view cone)
│   ├── maddpg.py                    #   MADDPG: local actors, centralised critics, replay, checkpoints
│   ├── mappo.py                     #   MAPPO comparison (on-policy, centralised critic)
│   ├── opponents.py                 #   frozen PPO / random / historical opponents, opponent pool
│   ├── policy_io.py                 #   one loader for MADDPG and MAPPO checkpoints
│   └── paths.py                     #   ROOT / CONFIGS / MODELS / RUNS: all default paths hang off these
│
├── scripts/                         # everything you run:  python -m scripts.<name>
│   ├── train_maddpg.py              #   phases: validate / seeker (vs frozen opponent) / simultaneous (mixed opponents)
│   ├── train_mappo.py               #   same recipe with MAPPO
│   ├── evaluate_stage2.py           #   paired evaluation, bootstrap CIs, unshaped rewards
│   ├── crossplay_stage2.py          #   every Seeker vs every Evader across independent runs
│   ├── replay_stage2.py             #   PyBullet 3D replay window
│   ├── view2d.py                    #   2D viewer: top-down + side view + stats sidebar -> GIF/PNG
│   ├── plot_stage2.py               #   learning curves, visibility comparisons, 3D trajectories
│   ├── plot_convergence.py          #   PPO reward convergence from TensorBoard logs
│   └── ablation_report.py, stability_report.py   # tables across variants / seeds
│
├── configs/                         # all training configs
│   ├── maddpg_config.json           #   original Stage 2 defaults
│   ├── maddpg_config_final*.json    #   final recipe (obstacles, Seeker history, opponent mixing); phaseB = Seeker pretrain
│   ├── maddpg_config_pool.json      #   opponent-pool variant
│   └── mappo_config*.json           #   MAPPO configs
│
├── tests/                           # pytest suite (`pytest -q` from the project root)
├── models/                          # trained PPO agents (.zip): open-arena and *_city (obstacle world)
├── runs/                            # training / evaluation outputs (git-ignored)
├── logs/, logs_city/                # TensorBoard logs
├── docs/                            # Stage 2 docs, results, full overview PDFs, build_report.py, figures/
└── archive/                         # Phase 1 prototype (basic_sim_random_evader.py), kept for provenance
```

A few notes on this layout:

- **Run everything from the project root.** Libraries live in the `vp/` package and runnable programs in `scripts/`, so use `python -m scripts.<name>` (not `python scripts/<name>.py`) and `python -m vp.ppo` for Stage 1. Running as a module is what lets the scripts and tests import each other.
- **Paths are anchored to the project root.** Default locations for configs, `models/` and `runs/` come from `vp/paths.py`, so they don't depend on your working directory. If you move a folder, change it there. Paths you pass on the command line (`--output`, `--model`, ...) are used as given, relative to where you run.
- `models/` and `logs/` are small (~1.2 MB and ~1.9 MB) so they're checked into git as-is, no Git LFS needed.
- `__pycache__/`, `.pytest_cache/`, `.venv/` and `.DS_Store` are excluded via `.gitignore`.

## Environment

`VanishingPointEnv` (Gymnasium-compatible, defined in `vp/ppo.py`):

| | |
|---|---|
| Arena | 20 × 20 × 10 m, bounded box |
| Timestep | 0.1 s, 600 steps/episode (60 s) |
| Evader speed | 2.2 m/s |
| Seeker speed | 1.8 m/s (full difficulty) |
| Seeker FOV / range | 90° / 10 m (full difficulty) |
| Action space | `Box(-1, 1, shape=(3,))` — normalized 3D velocity command, scaled to Evader speed |
| Observation space | `Box(-1, 1, shape=(21,))` — Evader position & velocity, relative Seeker position & velocity, Seeker forward vector, distance to all 6 arena walls |
| Reward | `+1` per step hidden, `-1` per step visible, `-2` at the start of each wall-contact streak (not every step; leaving and touching again re-fires it) |

**Scripted Seeker** cycles through three modes: `TRACK` (Evader visible), `SCAN` (spin + ±45° pitch sweep after a 4 s timeout with no sighting), `SEARCH` (move to a random waypoint if still nothing found).

**Curriculum training** (`--curriculum`) ramps Seeker speed/FOV/range linearly across 4 stages, from an easy Seeker (1.0 m/s, 60° FOV, 6 m range) up to full difficulty, giving the Evader time to learn clean navigation before facing a fast, wide-FOV opponent.

## Setup

```bash
python3.11 -m venv .venv            # or: conda create -n vp python=3.11 -y && conda activate vp
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, if `pybullet` fails to build, install it with `CFLAGS="-Dfdopen=fdopen" pip install pybullet`. Only the 3D viewers use PyBullet; headless training and evaluation don't import it.

Check the install (about 10 seconds):

```bash
pytest -q          # expect: 74 passed
```

## Run the whole project, end to end

Everything below runs from the project root with the environment active. The repo already ships trained PPO agents in `models/` (open arena and obstacle world), so you can start at step 4 and skip the PPO training. Outputs go to `runs/` (git-ignored) unless noted.

**1. Stage 1 baseline** (open arena, scripted Seeker)
```bash
python -m vp.ppo random --episodes 20                      # random-Evader baseline
python -m vp.ppo train --timesteps 500000 --seed 577448602 --model models/ppo_500k  --tensorboard-log logs/plain
python -m vp.ppo train --curriculum --timesteps 500000 --seed 577448602 --model models/ppo_curr --tensorboard-log logs/curriculum
python -m vp.ppo eval --model models/ppo_500k --episodes 20
```
Give every training run its own `--tensorboard-log` folder; runs that share one interleave their logs. To plot, point `--plain` / `--curriculum` at the `PPO_N` folder inside each log directory:
```bash
python -m scripts.plot_convergence --plain logs/plain/PPO_1 --curriculum logs/curriculum/PPO_1 --out docs/figures/convergence.png
tensorboard --logdir logs                                  # optional: browse the raw curves
```

**2. PPO opponents in the obstacle world** (Stage 2 needs one training opponent and one held-out evaluation opponent)
```bash
python -m vp.ppo train --obstacles city --timesteps 500000 --seed 2001 --model models/ppo_city_plain_s2001 --tensorboard-log logs_city/plain_2001
python -m vp.ppo train --obstacles city --timesteps 500000 --seed 2002 --model models/ppo_city_plain_s2002 --tensorboard-log logs_city/plain_2002
```

**3. Quick MADDPG smoke test** (a couple of episodes, confirms the Stage 2 stack works)
```bash
python -m scripts.train_maddpg --phase validate --episodes 2
```

**4. MADDPG: the final recipe**
```bash
# Phase B: Seeker alone against a frozen PPO Evader
python -m scripts.train_maddpg --phase seeker --config configs/maddpg_config_final_phaseB.json \
    --opponent ppo --opponent-model models/ppo_city_plain_s2001.zip --output runs/B --seeds 2001

# Phase C: both drones learn together, with opponent mixing, starting from the Phase B Seeker
python -m scripts.train_maddpg --phase simultaneous --config configs/maddpg_config_final.json \
    --initialize runs/B/seeker_s2001/final.pt --output runs/C --seeds 2001
```

**5. Evaluate** (60 paired starts, unshaped rewards, bootstrap confidence intervals; add `--hidden-start` for time-to-first-detection)
```bash
python -m scripts.evaluate_stage2 --ppo-model models/ppo_city_plain_s2002.zip \
    --seeker-checkpoint runs/B/seeker_s2001/final.pt \
    --joint-checkpoint runs/C/simultaneous_s2001/final.pt --output runs/eval
python -m scripts.plot_stage2 --runs runs/C/simultaneous_s2001 --output runs/C/plots     # learning curves
```
Read `runs/eval/comparison.md` for the results table.

**6. Watch the agents**
```bash
python -m scripts.view2d --matchup joint --obstacles city \
    --joint-checkpoint runs/C/simultaneous_s2001/final.pt --out runs/joint.gif      # 2D GIF, no PyBullet needed
python -m scripts.replay_stage2 --joint-checkpoint runs/C/simultaneous_s2001/final.pt   # PyBullet 3D window
```

**7. MAPPO comparison** (same environment and recipe, on-policy learner)
```bash
python -m scripts.train_mappo --phase seeker --config configs/mappo_config_phaseB.json \
    --opponent-model models/ppo_city_plain_s2001.zip --output runs/mappo_B --seeds 2001
python -m scripts.train_mappo --phase simultaneous --initialize runs/mappo_B/seeker_s2001/final.pt \
    --output runs/mappo_C --seeds 2001
python -m scripts.evaluate_stage2 --ppo-model models/ppo_city_plain_s2002.zip \
    --joint-checkpoint runs/mappo_C/simultaneous_s2001/final.pt --output runs/ev_MAPPO_s2001
```
Evaluation, cross-play and the viewers load MADDPG and MAPPO checkpoints interchangeably.

**8. Stability and ablation reports** (these need several independent runs, e.g. repeat step 4 with different `--seeds` / `--output`)
```bash
python -m scripts.crossplay_stage2 --checkpoints runs/C_a/simultaneous_s2001/final.pt runs/C_b/simultaneous_s2002/final.pt \
    --labels s2001 s2002 --output runs/cross
python -m scripts.stability_report --group main runs/C_*/simultaneous_s*
python -m scripts.ablation_report --prefix runs/ev_ A0_final A1_nohist       # reads runs/ev_<variant>_s<seed>/episodes.csv
```

**9. Rebuild the report PDF** (needs `pip install reportlab`; reads the figures in `docs/figures/`)
```bash
python docs/build_report.py
```

The details behind each stage are in [docs/STAGE2.md](docs/STAGE2.md); results are in [docs/STAGE2_RESULTS.md](docs/STAGE2_RESULTS.md) and [docs/COMPARISONS.md](docs/COMPARISONS.md).

## Usage

All commands below assume you're in the project root with the environment active.

**Random baseline**
```bash
python -m vp.ppo random --episodes 20
```

**Train PPO** (plain or curriculum, ~4–5 min per 500k-step run on CPU)
```bash
python -m vp.ppo train --timesteps 500000 --seed 577448602 \
    --model models/ppo_500k --tensorboard-log logs/

python -m vp.ppo train --curriculum --timesteps 500000 --seed 577448602 \
    --model models/ppo_curr --tensorboard-log logs/
```

**Evaluate a trained model**
```bash
python -m vp.ppo eval --model models/ppo_500k --episodes 20
```

**Watch an episode render** (PyBullet GUI)
```bash
python -m vp.ppo eval --model models/ppo_500k --episodes 3 --render
```

**Override difficulty** (works on `eval` and `random`)
```bash
python -m vp.ppo eval --model models/ppo_500k --episodes 20 \
    --seeker-speed 1.2 --fov 70 --range 8
```

**Plot reward convergence** (plain vs. curriculum, from TensorBoard logs)
```bash
python -m scripts.plot_convergence --plain logs/PPO_1 --curriculum logs/PPO_3 --out docs/figures/convergence.png
```

**Run the test suite**
```bash
pytest -q
```

## Results

Full tables and discussion are in [`docs/Vanishing_Point_Full_Overview.pdf`](docs/Vanishing_Point_Full_Overview.pdf). Headline numbers, evaluated over 20 episodes at full difficulty:

| Setup | Mean return | Undetected time | Wall touches (mean) |
|---|---|---|---|
| Random baseline | 173.3 | 66.0% | 9.25 |
| Plain PPO (avg of 3 seeds) | 469.9 | 90.2% | 6.08 |
| Curriculum PPO (avg of 3 seeds) | 452.8 | 88.3% | 3.13 |

PPO massively outperforms random behaviour on every metric. Between the two PPO variants there's a consistent, repeatable trade-off across all 3 seeds: plain PPO scores a little higher, while curriculum PPO roughly halves wall collisions — a plausible explanation is that full-difficulty training rewards riskier wall-hugging shortcuts while the curriculum learns safer navigation first, but no ablation was run to confirm this.

## Known limitations

- The scripted Seeker (1.8 m/s) is slower than the Evader (2.2 m/s) and doesn't predict movement, so part of the Evader's score may reflect out-running rather than pure evasive skill.
- The Seeker's search waypoints are random, so worst-case hidden stretches stay long even after tuning.
- The Evader's observation doesn't include the Seeker's current mode (TRACK/SCAN/SEARCH).
- Only 3 seeds have been tested per setup, and the head-to-head comparison is only run at full difficulty.
- Drones can overlap with no collision penalty between them.

## Roadmap

- **Stage 2 (done, see [docs/STAGE2_RESULTS.md](docs/STAGE2_RESULTS.md))**: longer multi-seed MADDPG runs in the obstacle world. MAPPO comparison, 3-seed ablations and an opponent pool are in [docs/COMPARISONS.md](docs/COMPARISONS.md). Physical-drone deployment (proposal Phase 4) is future work: [docs/FUTURE_WORK.md](docs/FUTURE_WORK.md). Next: recurrent policies, symmetric information.
- Add a minimum drone-separation penalty.
- Bias Seeker search waypoints toward the Evader's last known position instead of pure random sampling.
- Give the Evader the Seeker's current mode as an extra observation.

## License

Add a license of your choice (e.g. MIT) if you intend this to be reused — none is currently specified.

## Stage 2 usage

The shared simulator, PettingZoo environment, MADDPG implementation, phased training, evaluation, plots and tests are documented in [docs/STAGE2.md](docs/STAGE2.md). All Stage 1 commands above still work.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m scripts.train_maddpg --phase validate --episodes 2
python -m scripts.train_maddpg --phase seeker --opponent ppo --opponent-model models/ppo_500k.zip --output runs/phase_b
python -m scripts.train_maddpg --phase simultaneous --output runs/phase_c --seeds 2001 2002 2003
```

Headless training uses CPU and does not import PyBullet. Add `--render` for playback. Checkpoints and CSV outputs go to ignored `runs/` directories; existing PPO models and logs are preserved.

## Obstacle world and viewers

```bash
# Train / evaluate PPO inside the obstacle world (obstacle observations are appended, so retrain; the open-arena models stay valid without --obstacles)
python -m vp.ppo train --obstacles city --timesteps 500000 --seed 2001 --model models/ppo_city_plain_s2001
python -m vp.ppo eval  --obstacles city --model models/ppo_city_plain_s2002 --episodes 50

# Final Stage 2 recipe: Seeker pretrain (frozen PPO opponent) then mixed joint training
python -m scripts.train_maddpg --phase seeker --config configs/maddpg_config_final_phaseB.json --opponent ppo --opponent-model models/ppo_city_plain_s2001.zip --output runs/B
python -m scripts.train_maddpg --phase simultaneous --config configs/maddpg_config_final.json --initialize runs/B/seeker_s2001/final.pt --output runs/C --seeds 2001

# Evaluate (60 paired starts, unshaped rewards, bootstrap CIs) and watch
python -m scripts.evaluate_stage2 --ppo-model models/ppo_city_plain_s2002.zip --seeker-checkpoint runs/B/seeker_s2001/final.pt --joint-checkpoint runs/C/simultaneous_s2001/final.pt --output runs/eval
python -m scripts.view2d --matchup joint --obstacles city --joint-checkpoint runs/C/simultaneous_s2001/final.pt --out runs/joint.gif   # 2D top-down + side view
python -m scripts.replay_stage2 --joint-checkpoint runs/C/simultaneous_s2001/final.pt                                                                  # PyBullet 3D window
```

## Figures and metrics

- `docs/figures/ppo_city_convergence.png`: PPO reward convergence, plain vs curriculum (obstacle world, 3 seeds).
- `docs/figures/final2_joint/learning_curves.png`: MADDPG joint-training curves (5 seeds).
- Time-to-first-detection (use `--hidden-start`), drone-drone collision steps, wall/obstacle contacts and bootstrap CIs are in each evaluation's `comparison.md`; see [docs/STAGE2_RESULTS.md](docs/STAGE2_RESULTS.md).

# Vanishing Point

A pursuit-evasion RL project: a drone **Evader** learns to stay out of a **Seeker** drone's camera view inside a bounded 3D arena. Stage 1 trains a PPO Evader against a rule-based Seeker; Stage 2 trains BOTH drones with model-free actor-critic policy-gradient learning (MADDPG) and contains no scripted controller. No value/policy iteration or planning is used anywhere.

- **Phase 1** — built the physics/geometry simulator, a scripted Seeker (TRACK / SCAN / SEARCH), and a random-Evader baseline to sanity-check the environment.
- **Phase 2** — trained the Evader with PPO, both plainly and with a 4-stage difficulty curriculum, validated across 3 seeds, and added an automated test suite.
- **Stage 2 (implemented, trained, evaluated)** — PettingZoo two-agent environment, MADDPG with local actors and centralised critics, an optional **obstacle world** (solid boxes that block the Seeker's line of sight), Seeker observation history, and opponent-mixed training. See [Stage 2 documentation](docs/STAGE2.md) and [results](docs/STAGE2_RESULTS.md).

Full write-up, tables, and figures: [`docs/Vanishing_Point_Full_Overview.pdf`](docs/Vanishing_Point_Full_Overview.pdf).

## Repo structure

```
vanishing-point/
├── README.md, requirements.txt, .gitignore
│
├── vanishing_point_core.py        # shared NumPy simulator: dynamics, FOV, obstacles + occlusion, Stage 1 scripted Seeker
├── vanishing_point_render.py      # lazy PyBullet 3D viewer (quadcopters, obstacles, view cone)
├── vanishing_point_view2d.py      # polished 2D viewer: top-down + side view + stats sidebar -> GIF/PNG
├── vanishing_point_ppo.py         # Stage 1 Gymnasium env + PPO train/eval CLI
├── vanishing_point_parallel.py    # Stage 2 PettingZoo two-agent env
├── maddpg.py                      # MADDPG: local actors, centralised critics, replay, checkpoints
├── train_maddpg.py                # phases: validate / seeker (vs frozen opponent) / simultaneous (mixed opponents)
├── evaluate_stage2.py             # paired evaluation, bootstrap CIs, unshaped rewards
├── stage2_opponents.py            # frozen PPO / random / historical MADDPG opponents
├── plot_stage2.py, plot_convergence.py, replay_stage2.py
├── vanishing_point_basic_sim_random_evader.py  # Phase 1 prototype (kept for provenance)
├── maddpg_config.json             # original Stage 2 defaults
├── maddpg_config_final.json       # final recipe (obstacles, Seeker history, opponent mixing)
├── maddpg_config_pool.json        # opponent-pool variant
├── mappo.py, train_mappo.py, mappo_config*.json  # MAPPO comparison (on-policy, centralised critic)
├── policy_io.py                   # one loader for MADDPG and MAPPO checkpoints
├── crossplay_stage2.py, stability_report.py, ablation_report.py  # stability and ablation analysis
│
├── tests/                         # pytest suite (`pytest -q` from the repo root)
├── pytest.ini
├── models/                        # trained PPO agents (.zip): open-arena and *_city (obstacle world)
├── logs/, logs_city/              # TensorBoard logs
├── convergence.png
└── docs/                          # Stage 2 docs, results, full overview PDF (Phases 1-2 and Stage 2),
                                   # build_report.py (rebuilds the PDF; needs reportlab), figures/
```

A few notes on this layout:

- All Python files stay **flat at the repo root** (not in a `src/` folder) because the tests import `vanishing_point_ppo` directly (`import vanishing_point_ppo as vp`), and every command in the report's appendix assumes you're running from this same folder. Nesting them would break both without extra path plumbing.
- `models/` and `logs/` are small (~1.2 MB and ~1.9 MB) so they're checked into git as-is — no Git LFS needed. If you'd rather not version binary model checkpoints, add `models/*.zip` to `.gitignore` and note in this README that they're regenerated via the training commands below.
- `__pycache__/`, `.pytest_cache/`, and `.DS_Store` from the zip are excluded via `.gitignore` (see below) — they're local build artifacts, not source.

## Environment

`VanishingPointEnv` (Gymnasium-compatible, defined in `vanishing_point_ppo.py`):

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
conda create -n vp python=3.11 -y
conda activate vp
pip install -r requirements.txt
```

## Usage

All commands below assume you're in this repo's root with the `vp` environment active.

**Random baseline**
```bash
python vanishing_point_ppo.py random --episodes 20
```

**Train PPO** (plain or curriculum, ~4–5 min per 500k-step run on CPU)
```bash
python vanishing_point_ppo.py train --timesteps 500000 --seed 577448602 \
    --model models/ppo_500k --tensorboard-log logs/

python vanishing_point_ppo.py train --curriculum --timesteps 500000 --seed 577448602 \
    --model models/ppo_curr --tensorboard-log logs/
```

**Evaluate a trained model**
```bash
python vanishing_point_ppo.py eval --model models/ppo_500k --episodes 20
```

**Watch an episode render** (PyBullet GUI)
```bash
python vanishing_point_ppo.py eval --model models/ppo_500k --episodes 3 --render
```

**Override difficulty** (works on `eval` and `random`)
```bash
python vanishing_point_ppo.py eval --model models/ppo_500k --episodes 20 \
    --seeker-speed 1.2 --fov 70 --range 8
```

**Plot reward convergence** (plain vs. curriculum, from TensorBoard logs)
```bash
python plot_convergence.py --plain logs/PPO_1 --curriculum logs/PPO_3 --out convergence.png
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

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python train_maddpg.py --phase validate --episodes 2
python train_maddpg.py --phase seeker --opponent ppo --opponent-model models/ppo_500k.zip --output runs/phase_b
python train_maddpg.py --phase simultaneous --output runs/phase_c --seeds 2001 2002 2003
```

Headless training uses CPU and does not import PyBullet. Add `--render` for playback. Checkpoints and CSV outputs go to ignored `runs/` directories; existing PPO models and logs are preserved.

## Obstacle world and viewers

```bash
# Train / evaluate PPO inside the obstacle world (obstacle observations are appended, so retrain; the open-arena models stay valid without --obstacles)
python vanishing_point_ppo.py train --obstacles city --timesteps 500000 --seed 2001 --model models/ppo_city_plain_s2001
python vanishing_point_ppo.py eval  --obstacles city --model models/ppo_city_plain_s2002 --episodes 50

# Final Stage 2 recipe: Seeker pretrain (frozen PPO opponent) then mixed joint training
python train_maddpg.py --phase seeker --config maddpg_config_final_phaseB.json --opponent ppo --opponent-model models/ppo_city_plain_s2001.zip --output runs/B
python train_maddpg.py --phase simultaneous --config maddpg_config_final.json --initialize runs/B/seeker_s2001/final.pt --output runs/C --seeds 2001

# Evaluate (60 paired starts, unshaped rewards, bootstrap CIs) and watch
python evaluate_stage2.py --ppo-model models/ppo_city_plain_s2002.zip --seeker-checkpoint runs/B/seeker_s2001/final.pt --joint-checkpoint runs/C/simultaneous_s2001/final.pt --output runs/eval
python vanishing_point_view2d.py --matchup joint --obstacles city --joint-checkpoint runs/C/simultaneous_s2001/final.pt --out runs/joint.gif   # 2D top-down + side view
python replay_stage2.py --joint-checkpoint runs/C/simultaneous_s2001/final.pt                                                                  # PyBullet 3D window
```

## Figures and metrics

- `docs/figures/ppo_city_convergence.png`: PPO reward convergence, plain vs curriculum (obstacle world, 3 seeds).
- `docs/figures/final2_joint/learning_curves.png`: MADDPG joint-training curves (5 seeds).
- Time-to-first-detection (use `--hidden-start`), drone-drone collision steps, wall/obstacle contacts and bootstrap CIs are in each evaluation's `comparison.md`; see [docs/STAGE2_RESULTS.md](docs/STAGE2_RESULTS.md).

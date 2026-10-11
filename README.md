# Vanishing Point

A pursuit-evasion RL project: a PPO-controlled **Evader** drone learns to stay out of a scripted **Seeker** drone's camera view inside a bounded 3D arena.

- **Phase 1** — built the physics/geometry simulator, a scripted Seeker (TRACK / SCAN / SEARCH), and a random-Evader baseline to sanity-check the environment.
- **Phase 2** — trained the Evader with PPO, both plainly and with a 4-stage difficulty curriculum, validated across 3 seeds, and added an automated test suite.
- **Stage 2 (implemented)** — added a PettingZoo two-agent environment and MADDPG with local actors and centralized critics; Stage 1 remains available for comparison. See [Stage 2 documentation](docs/STAGE2.md).

Full write-up, tables, and figures: [`docs/Vanishing_Point_Full_Overview.pdf`](docs/Vanishing_Point_Full_Overview.pdf).

## Repo structure

```
vanishing-point/
├── README.md
├── requirements.txt
├── .gitignore
│
├── vanishing_point_ppo.py                     # main env + scripted Seeker + PPO train/eval CLI
├── vanishing_point_basic_sim_random_evader.py # Phase 1 prototype: sim + random Evader (kept for provenance)
├── plot_convergence.py                        # reads TensorBoard logs -> reward convergence plot
├── test_vanishing_point.py                    # pytest suite (11 tests) for the environment/CLI
│
├── models/                # trained PPO agents (Stable-Baselines3 .zip checkpoints)
│   ├── ppo_500k.zip            # plain PPO, seed 577448602 (headline model)
│   ├── ppo_curr.zip            # curriculum PPO, seed 577448602
│   ├── ppo_plain_s2001.zip     # plain PPO, seed 2001
│   ├── ppo_plain_s2002.zip     # plain PPO, seed 2002
│   ├── ppo_curr_s2001.zip      # curriculum PPO, seed 2001
│   └── ppo_curr_s2002.zip      # curriculum PPO, seed 2002
│
├── logs/                  # raw TensorBoard event files, one PPO_N folder per training run
│   └── PPO_1 ... PPO_10/
│
├── convergence.png        # plain-vs-curriculum reward convergence plot (used in the report)
│
└── docs/
    └── Vanishing_Point_Full_Overview.pdf   # full report: methodology, results, pros/cons, appendix
```

A few notes on this layout:

- All Python files stay **flat at the repo root** (not in a `src/` folder) because `test_vanishing_point.py` imports `vanishing_point_ppo` directly (`import vanishing_point_ppo as vp`), and every command in the report's appendix assumes you're running from this same folder. Nesting them would break both without extra path plumbing.
- `models/` and `logs/` are small (~1.2 MB and ~1.9 MB) so they're checked into git as-is — no Git LFS needed. If you'd rather not version binary model checkpoints, add `models/*.zip` to `.gitignore` and note in this README that they're regenerated via the training commands below.
- I didn't include `models/vanishing_point_ppo.zip` in the list above — it looks like a leftover from a run without an explicit `--model` name (it doesn't correspond to any seed/setup discussed in the report). Worth deleting or renaming before you push, so the models folder only contains the six models the report actually evaluates.
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
| Reward | `+1` per step hidden, `-1` per step visible, `-2` once on first wall contact (not per step) |

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
pytest test_vanishing_point.py -v
```

## Results

Full tables and discussion are in [`docs/Vanishing_Point_Full_Overview.pdf`](docs/Vanishing_Point_Full_Overview.pdf). Headline numbers, evaluated over 20 episodes at full difficulty:

| Setup | Mean return | Undetected time | Wall touches (mean) |
|---|---|---|---|
| Random baseline | 173.3 | 66.0% | 9.25 |
| Plain PPO (avg of 3 seeds) | 469.9 | 90.2% | 6.08 |
| Curriculum PPO (avg of 3 seeds) | 452.8 | 88.3% | 3.13 |

PPO massively outperforms random behaviour on every metric. Between the two PPO variants there's a consistent, repeatable trade-off across all 3 seeds: plain PPO scores a little higher, while curriculum PPO roughly halves wall collisions — because training the whole run at full difficulty lets the agent find and keep riskier, wall-hugging shortcuts, whereas curriculum training locks in collision-safe navigation habits early, before the Seeker ever gets fast.

## Known limitations

- The scripted Seeker (1.8 m/s) is slower than the Evader (2.2 m/s) and doesn't predict movement, so part of the Evader's score may reflect out-running rather than pure evasive skill.
- The Seeker's search waypoints are random, so worst-case hidden stretches stay long even after tuning.
- The Evader's observation doesn't include the Seeker's current mode (TRACK/SCAN/SEARCH).
- Only 3 seeds have been tested per setup, and the head-to-head comparison is only run at full difficulty.
- Drones can overlap with no collision penalty between them.

## Roadmap

- **Stage 2**: run longer multi-seed MADDPG experiments and compare checkpoints against the retained PPO/scripted baselines.
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

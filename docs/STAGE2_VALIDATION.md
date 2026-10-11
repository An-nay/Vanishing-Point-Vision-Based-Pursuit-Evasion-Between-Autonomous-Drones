# Executed Stage 2 validation

Validated on 5 October 2026 using the existing Windows project environment:
Python 3.14.3, NumPy 2.5.3, Gymnasium 1.3.0, PettingZoo 1.27.0,
PyTorch 2.14.1+cpu, Stable-Baselines3 2.9.0, PyBullet 3.2.7,
pytest 9.1.1 and Matplotlib 3.11.2.

These checks were actually executed. Short training results are pipeline
checks, not evidence of convergence or improved policies.

| Check | Result |
|---|---|
| `python -m pytest -q` | 43 tests passed in 33.87 seconds |
| Original Git HEAD Stage 1 versus refactored wrapper | All 9,000 transitions matched exactly, including observations, rewards, flags, info and episode summaries |
| Stage 1 saved PPO evaluation CLI | Completed one 600-step episode with `models/ppo_500k` |
| Stable-Baselines3 `check_env` | Passed on the refactored Gymnasium environment |
| Phase A CLI | PettingZoo Parallel API check passed; a full random 600-step rollout completed |
| Simultaneous training CLI | 1,200 steps, two full episodes, 147 joint updates, periodic and final checkpoints saved |
| Frozen PPO Seeker training CLI | 600 steps, one full episode, 72 Seeker updates; existing PPO loaded separately |
| Phase B checkpoint warm-start CLI | 64 steps, five joint updates; final checkpoint saved |
| Paired evaluation CLI | Four required matchups plus two historical-opponent matchups, each on seeds 41001 and 42001 |
| Hidden-start evaluation CLI | Four required matchups completed on seed 43001 |
| Plot generation | Learning curves, visibility comparisons and 3D trajectories generated |
| Renderer integration | Both interfaces rendered shared world positions through PyBullet DIRECT and released separate clients |
| Headless isolation | Confirmed `pybullet` absent from `sys.modules` after a headless joint transition |
| `python -m pip check` | No broken requirements found |
| Python AST parsing and `git diff --check` | Passed |

The independent Stage 1 regression used the original source retrieved with
`git show HEAD:vanishing_point_ppo.py`. It checked three seeds for each of
five settings: fixed/default, randomized/default, randomized/easy curriculum,
randomized/mid curriculum and fixed/easy curriculum. Each rollout lasted 600
steps. A redundant normalization that initially changed an FOV angle at
machine precision was corrected before the final 9,000-transition run.

The smoke configuration uses 32-unit hidden layers, batch size 32, replay
capacity 2,000, 32 warmup steps, updates every eight steps and checkpointing
every 600 steps. It retains the normal 600-step episode horizon and 0.1 s
timestep. Default longer runs use `maddpg_config.json`.

Local generated artifacts (ignored by Git):

- `runs/smoke/smoke_config.json`
- `runs/smoke/training/simultaneous_s2001/final.pt`
- `runs/smoke/training/seeker_s2001/final.pt`
- `runs/smoke/warm_start/simultaneous_s2001/final.pt`
- `runs/smoke/evaluation/comparison.md`
- `runs/smoke/evaluation/comparison.csv`
- `runs/smoke/evaluation/visibility_comparison.png`
- `runs/smoke/evaluation/trajectories_3d.png`
- `runs/smoke/plots/learning_curves.png`
- `runs/smoke/hidden_evaluation/`

Interactive GUI playback was not visually tested. The actual renderer was
tested in DIRECT mode without opening windows. Multiple independent training
seeds, long-run convergence and robustness of fully trained policies remain
experiments for the user; the scripts and documented commands support them.

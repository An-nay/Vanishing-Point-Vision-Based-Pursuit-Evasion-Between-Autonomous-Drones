"""
Plot PPO reward convergence from TensorBoard logs.

Usage (from the project root):

    python -m scripts.plot_convergence --plain logs/PPO_1 --curriculum logs/PPO_3 --out docs/figures/convergence.png
    python -m scripts.plot_convergence --plain A/PPO_1 B/PPO_1 C/PPO_1 --curriculum D/PPO_1 E/PPO_1 F/PPO_1   # several seeds: mean + min-max band

Each training run creates its own PPO_N subfolder inside the --tensorboard-log directory. Give EVERY run its own
--tensorboard-log folder: runs started at the same time that share one folder race for the PPO_N names and
interleave their logs (this corrupted logs_city/PPO_1..5).
"""

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from vp.paths import ROOT


def load_curve(log_dir, tag="rollout/ep_rew_mean"):
    ea = EventAccumulator(log_dir)
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        available = ea.Tags().get("scalars", [])
        raise SystemExit(f"Tag '{tag}' not found in {log_dir}. Available tags: {available}")
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def mean_band(runs, grid):
    """Interpolate each run onto a common step grid -> (mean, min, max)."""
    curves = np.array([np.interp(grid, steps, values) for steps, values in runs])
    return curves.mean(axis=0), curves.min(axis=0), curves.max(axis=0)


def main():
    parser = argparse.ArgumentParser(description="Plot PPO reward convergence (one or several seeds per group)")
    parser.add_argument("--plain", required=True, nargs="+", help="TensorBoard log dir(s), plain PPO")
    parser.add_argument("--curriculum", required=True, nargs="+", help="TensorBoard log dir(s), curriculum PPO")
    parser.add_argument("--out", default=str(ROOT / "docs" / "figures" / "convergence.png"), help="Output image path")
    parser.add_argument("--title", default="PPO Reward Convergence: Plain vs Curriculum")
    args = parser.parse_args()

    groups = {"Plain PPO": [load_curve(d) for d in args.plain],
              "Curriculum PPO": [load_curve(d) for d in args.curriculum]}
    last = min(steps[-1] for runs in groups.values() for steps, _ in runs)
    grid = np.linspace(0, last, 300)
    plt.figure(figsize=(9, 5.5))
    for label, runs in groups.items():
        mean, low, high = mean_band(runs, grid)
        plt.plot(grid, mean, label=f"{label} (mean of {len(runs)})", linewidth=2)
        if len(runs) > 1:
            plt.fill_between(grid, low, high, alpha=0.2)
    plt.xlabel("Training timesteps")
    plt.ylabel("Mean episode return (rollout/ep_rew_mean)")
    plt.title(args.title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()

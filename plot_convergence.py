"""
Plot PPO reward convergence from TensorBoard logs.

Usage (from the same folder as vanishing_point_ppo.py, with the 'vp'
conda environment active):

    $PY plot_convergence.py --plain logs/PPO_1 --curriculum logs/PPO_2 --out convergence.png

Find your actual log folder names with:  ls logs/
Each --curriculum/--plain training run creates its own PPO_N subfolder
inside the --tensorboard-log directory you passed at training time.
"""

import argparse
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


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


def main():
    parser = argparse.ArgumentParser(description="Plot PPO reward convergence")
    parser.add_argument("--plain", required=True, help="TensorBoard log dir for the plain PPO run")
    parser.add_argument("--curriculum", required=True, help="TensorBoard log dir for the curriculum PPO run")
    parser.add_argument("--out", default="convergence.png", help="Output image path")
    args = parser.parse_args()

    plain_steps, plain_values = load_curve(args.plain)
    curr_steps, curr_values = load_curve(args.curriculum)

    plt.figure(figsize=(9, 5.5))
    plt.plot(plain_steps, plain_values, label="Plain PPO", linewidth=2)
    plt.plot(curr_steps, curr_values, label="Curriculum PPO", linewidth=2)
    plt.xlabel("Training timesteps")
    plt.ylabel("Mean episode return (rollout/ep_rew_mean)")
    plt.title("PPO Reward Convergence: Plain vs Curriculum")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()

"""Standalone learning curves, comparison charts and 3D trajectory figures."""
import argparse
import csv
from pathlib import Path
import numpy as np


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def smooth(values, window):
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def _on_grid(xs, ys, grid):
    return np.interp(grid, xs, ys)


def plot_learning(run_directories, output, window=20, mode="selfplay", loss_window=1000):
    """Learning curves: per-seed thin lines and the seed mean (bold).

    Competitive co-training oscillates by design (each side adapts to the other), so expect swings in
    visible time rather than a monotone curve. Losses are smoothed over `loss_window` updates; the
    sawtooth in raw losses comes from alternating learner blocks.
    """
    plt = plotting()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    runs = []
    for directory in map(Path, run_directories):
        episodes = [r for r in read_rows(directory / "episodes.csv") if r["partial"] == "False"]
        if mode and episodes and "mode" in episodes[0] and any(r["mode"] == mode for r in episodes):
            episodes = [r for r in episodes if r["mode"] == mode]
        runs.append((directory.name, episodes, read_rows(directory / "updates.csv")))
    last = max(int(e[-1]["step"]) for _, e, _ in runs if e)
    grid = np.linspace(0, last, 300)
    colours = {"seeker": "tab:red", "evader": "tab:blue"}

    series = {"visible": [], "seeker": [], "evader": []}
    for name, episodes, _ in runs:
        steps = np.array([int(r["step"]) for r in episodes])
        trim = len(steps) - len(smooth(steps, window))
        xs = steps[trim:]
        for key, field in (("visible", "visible_pct"), ("seeker", "seeker_reward"), ("evader", "evader_reward")):
            ys = smooth([float(r[field]) for r in episodes], window)
            series[key].append(_on_grid(xs, ys, grid))
            if key == "visible":
                axes[0, 0].plot(xs, ys, lw=0.8, alpha=0.45, label=name.split("_")[-1])
    axes[0, 0].plot(grid, np.mean(series["visible"], axis=0), "k", lw=2.4, label="mean")
    axes[0, 0].set_title(f"Evader visible time %, joint game ({window}-episode mean)")
    axes[0, 0].legend(fontsize=7, ncol=3)
    for agent in ("seeker", "evader"):
        mean = np.mean(series[agent], axis=0)
        axes[0, 1].plot(grid, mean, color=colours[agent], lw=2.2, label=agent + " (seed mean)")
        axes[0, 1].fill_between(grid, np.min(series[agent], axis=0), np.max(series[agent], axis=0),
                                color=colours[agent], alpha=0.12)
    axes[0, 1].set_title(f"Episode returns, shaped reward ({window}-episode mean, min-max band)")
    axes[0, 1].legend(fontsize=8)
    for index, kind in enumerate(["actor", "critic"]):
        ax = axes[1, index]
        for agent in ("seeker", "evader"):
            curves = []
            for _, _, updates in runs:
                valid = [r for r in updates if r[agent + "_" + kind + "_loss"]]
                xs = np.array([int(r["step"]) for r in valid])
                ys = smooth([float(r[agent + "_" + kind + "_loss"]) for r in valid], loss_window)
                curves.append(_on_grid(xs[len(xs) - len(ys):], ys, grid))
            ax.plot(grid, np.mean(curves, axis=0), color=colours[agent], lw=2, label=agent + " (seed mean)")
            for curve in curves:
                ax.plot(grid, curve, color=colours[agent], lw=0.6, alpha=0.3)
        ax.set_title(f"{kind.capitalize()} loss (smoothed over {loss_window} updates)")
        ax.legend(fontsize=8)
        if kind == "critic":
            ax.set_yscale("log")
    for ax in axes.flat:
        ax.set_xlabel("Environment steps")
        ax.grid(alpha=.25)
    fig.suptitle("MADDPG joint training (mixed-opponent recipe), 5 seeds", fontsize=12)
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=150)
    plt.close(fig)


def plot_evaluation(directory):
    plt = plotting()
    directory = Path(directory)
    rows = read_rows(directory / "comparison.csv")
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = [row["experiment"] for row in rows]
    ax.bar(labels, [float(row["visible_pct_mean"]) for row in rows],
           yerr=[float(row["visible_pct_std"]) for row in rows], capsize=4)
    ax.set_ylabel("Visible time (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    fig.savefig(directory / "visibility_comparison.png", dpi=180)
    plt.close(fig)
    # One 3D panel per matchup for the first paired initial condition.
    episodes = read_rows(directory / "episodes.csv")
    first_seed = episodes[0]["seed"]
    selected = [row for row in episodes if row["seed"] == first_seed]
    fig = plt.figure(figsize=(6 * min(len(selected), 3), 5 * int(np.ceil(len(selected) / 3))))
    for index, row in enumerate(selected):
        trajectory = np.genfromtxt(directory / "trajectories" /
                                   f"{row['experiment']}_s{row['seed']}_e{row['episode']}.csv",
                                   delimiter=",", names=True)
        ax = fig.add_subplot(int(np.ceil(len(selected) / 3)), min(len(selected), 3),
                             index + 1, projection="3d")
        for agent, color in [("seeker", "red"), ("evader", "blue")]:
            xyz = [trajectory[agent + "_" + axis] for axis in ["x", "y", "z"]]
            ax.plot(*xyz, color=color, label=agent, linewidth=1.)
            ax.scatter(*[v[0] for v in xyz], color=color, marker="o", s=30)
        ax.set(xlim=(0, 20), ylim=(0, 20), zlim=(0, 10),
               xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
               title=row["experiment"])
        ax.set_box_aspect((2, 2, 1))
        ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "trajectories_3d.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="*")
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/plots"))
    parser.add_argument("--smooth", type=int, default=20, help="moving-average window in episodes")
    parser.add_argument("--mode", default="selfplay", help="episode mode to plot ('' = all modes)")
    args = parser.parse_args()
    if not args.runs and not args.evaluation:
        parser.error("provide --runs or --evaluation")
    if args.runs:
        plot_learning(args.runs, args.output, args.smooth, args.mode)
    if args.evaluation:
        plot_evaluation(args.evaluation)


if __name__ == "__main__":
    main()

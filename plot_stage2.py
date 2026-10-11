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


def plot_learning(run_directories, output):
    plt = plotting()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for directory in map(Path, run_directories):
        episodes = [r for r in read_rows(directory / "episodes.csv") if r["partial"] == "False"]
        updates = read_rows(directory / "updates.csv")
        label = directory.name
        if episodes:
            steps = [int(r["step"]) for r in episodes]
            for agent in ["seeker", "evader"]:
                axes[0, 0].plot(steps, [float(r[agent + "_reward"]) for r in episodes],
                                label=label + ":" + agent)
            axes[0, 1].plot(steps, [float(r["visible_pct"]) for r in episodes], label=label)
        for agent in ["seeker", "evader"]:
            for index, kind in enumerate(["actor", "critic"]):
                valid = [r for r in updates if r[agent + "_" + kind + "_loss"]]
                if valid:
                    axes[1, index].plot([int(r["step"]) for r in valid],
                                        [float(r[agent + "_" + kind + "_loss"]) for r in valid],
                                        label=label + ":" + agent, alpha=.75)
    for ax, title in zip(axes.flat, ["Episode returns", "Visible time (%)",
                                   "Actor loss", "Critic loss"]):
        ax.set_title(title)
        ax.set_xlabel("Environment steps")
        ax.grid(alpha=.25)
        if ax.get_lines():
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=180)
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
    args = parser.parse_args()
    if not args.runs and not args.evaluation:
        parser.error("provide --runs or --evaluation")
    if args.runs:
        plot_learning(args.runs, args.output)
    if args.evaluation:
        plot_evaluation(args.evaluation)


if __name__ == "__main__":
    main()

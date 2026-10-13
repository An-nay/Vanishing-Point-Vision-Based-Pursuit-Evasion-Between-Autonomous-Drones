"""Cross-play stability test: every Seeker from independent training runs against every Evader.

    python -m scripts.crossplay_stage2 --checkpoints runs/a/final.pt runs/b/final.pt ... --labels s1 s2 ... --output runs/cross

Entry (i, j) = mean % of the episode the Evader of run j is visible to the Seeker of run i on the same held-out
starts. The diagonal is each run's own co-trained pair; off-diagonals are strangers that never trained together.
A stable method keeps the off-diagonal close to the diagonal (no overfitting to one partner) and has no
catastrophic rows/columns (an exploitable Seeker or Evader).
"""
import argparse
import csv
from pathlib import Path
import numpy as np
from vp.policy_io import load_policy
from vp.core import sample_initial_state
from vp.parallel import VanishingPointParallelEnv


def play(env, seeker, evader, seed, start):
    obs, _ = env.reset(seed=seed, options={"initial_state": start})
    while env.agents:
        obs, *_ = env.step({"seeker": seeker.act("seeker", obs["seeker"]), "evader": evader.act("evader", obs["evader"])})
    return env.sim.episode_summary()["visible_pct"]


def crossplay(checkpoints, seeds, episodes):
    policies = [load_policy(path) for path in checkpoints]
    options = {**policies[0].config["environment"], "boundary_penalty_per_step": 0.0}  # unshaped scoring
    for policy in policies[1:]:
        if {**policy.config["environment"], "boundary_penalty_per_step": 0.0} != options:
            raise ValueError("all checkpoints must share environment settings")
    env = VanishingPointParallelEnv(**options)
    starts = []
    for base in seeds:
        for episode in range(episodes):
            s, e, f = sample_initial_state(np.random.default_rng(base + episode), obstacles=options.get("obstacles"))
            starts.append((base + episode, {"seeker_pos": s.tolist(), "evader_pos": e.tolist(), "seeker_forward": f.tolist()}))
    n = len(policies)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = np.mean([play(env, policies[i], policies[j], seed, start) for seed, start in starts])
            print(f"seeker {i} vs evader {j}: visible {matrix[i, j]:.1f}%", flush=True)
    return matrix


def summarize(matrix):
    n = len(matrix)
    off = matrix[~np.eye(n, dtype=bool)]
    return {"own_pair_mean": float(np.mean(np.diag(matrix))), "stranger_mean": float(off.mean()),
            "stranger_std": float(off.std()), "overfit_gap": float(off.mean() - np.mean(np.diag(matrix))),
            # exploitability proxies: how badly the worst partner treats each side
            "worst_case_evader_visible": float(np.mean([matrix[:, j].max() for j in range(n)])),
            "worst_case_seeker_visible": float(np.mean([matrix[i, :].min() for i in range(n)]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20001, 21001])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = args.labels or [f"run{i}" for i in range(len(args.checkpoints))]
    if len(labels) != len(args.checkpoints) or len(args.checkpoints) < 2:
        parser.error("need >= 2 checkpoints and one label each")
    matrix = crossplay(args.checkpoints, args.seeds, args.episodes)
    summary = summarize(matrix)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "crossplay.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seeker \\ evader"] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + [f"{v:.1f}" for v in row])
    lines = ["| Seeker \\ Evader | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
    lines += [f"| {label} | " + " | ".join(f"{v:.1f}" for v in row) + " |" for label, row in zip(labels, matrix)]
    lines += ["", "Entry = Evader visible % (lower is better for the Evader). Diagonal = co-trained pair.", ""]
    lines += [f"- {k}: {v:.1f}" for k, v in summary.items()]
    (args.output / "crossplay.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5 + 0.4 * len(labels), 4.5))
    image = ax.imshow(matrix, vmin=0, vmax=100, cmap="RdYlGn_r")
    ax.set_xticks(range(len(labels)), labels, rotation=45)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Evader from run")
    ax.set_ylabel("Seeker from run")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, label="Evader visible %")
    fig.tight_layout()
    fig.savefig(args.output / "crossplay.png", dpi=150)


if __name__ == "__main__":
    main()

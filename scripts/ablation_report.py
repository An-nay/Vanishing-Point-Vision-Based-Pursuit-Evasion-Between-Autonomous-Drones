"""Aggregate evaluate_stage2 outputs of several variants x seeds into one comparison table.

    python -m scripts.ablation_report --prefix runs/ev_ A0_final A2_nowall ...   # reads runs/ev_<variant>_s<seed>/episodes.csv

Columns (each = mean over seeds, with the standard deviation ACROSS seeds, so seed-to-seed stability is visible):
  Seeker vs PPO       Evader visible %, joint-trained Seeker vs the held-out PPO Evader (higher = stronger Seeker)
  Evader vs held-out  Evader visible %, joint-trained Evader vs held-out learned Seekers (lower = stronger Evader)
  Joint game          Evader visible % when the two co-trained agents meet (balance; far from 0/100 is healthy)
  Contact steps       Steps per episode in contact with a wall/obstacle, Evader / Seeker (joint game)
"""
import argparse
import csv
import glob
from pathlib import Path
import numpy as np
from vp.paths import RUNS


def per_seed(prefix, variant):
    runs = {}
    for directory in sorted(glob.glob(f"{prefix}{variant}_s*")):
        path = Path(directory) / "episodes.csv"
        if not path.is_file():
            continue
        rows = list(csv.DictReader(path.open()))
        seed = int(directory.rsplit("_s", 1)[1])
        by = {}
        for r in rows:
            by.setdefault(r["experiment"], []).append(r)
        mean = lambda exp, key: float(np.mean([float(r[key]) for r in by[exp]]))
        seeker_exp = "joint_seeker_ppo" if "joint_seeker_ppo" in by else "learned_seeker_ppo"
        held = [e for e in by if e.startswith("historical_seeker")]
        runs[seed] = {
            "seeker_vs_ppo": mean(seeker_exp, "visible_pct"),
            "evader_vs_heldout": float(np.mean([mean(e, "visible_pct") for e in held])),
            "joint": mean("stage2_joint", "visible_pct"),
            "ev_contact": mean("stage2_joint", "evader_boundary_steps"),
            "sk_contact": mean("stage2_joint", "seeker_boundary_steps"),
        }
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefix", default=str(RUNS) + "/ev_")
    parser.add_argument("variants", nargs="+")
    args = parser.parse_args()
    cell = lambda values: f"{np.mean(values):.1f} +/- {np.std(values):.1f}"
    print("| Variant | Seeds | Seeker vs PPO | Evader vs held-out | Joint game | Contact steps Evader / Seeker |")
    print("|---|---:|---:|---:|---:|---:|")
    for variant in args.variants:
        runs = per_seed(args.prefix, variant)
        if not runs:
            print(f"| {variant} | 0 | - | - | - | - |")
            continue
        get = lambda key: [r[key] for r in runs.values()]
        print(f"| {variant} | {len(runs)} | {cell(get('seeker_vs_ppo'))} | {cell(get('evader_vs_heldout'))} | "
              f"{cell(get('joint'))} | {np.mean(get('ev_contact')):.0f} / {np.mean(get('sk_contact')):.0f} |")


if __name__ == "__main__":
    main()

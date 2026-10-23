"""Aggregate paired-eval results across multiple seeds.

Reads runs/<prefix>_s<seed>/eval/episodes.csv for each seed, averages the per-experiment
metrics, and prints a comparison table with mean + stddev across seeds.

Usage:
    python -m scripts.multiseed_report --prefix runs/fix3_train --seeds 2001 2002 2003
"""
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True,
                        help="e.g. runs/fix3_train (reads runs/fix3_train_s<seed>/eval/episodes.csv)")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--out", default=None, help="Optional output markdown path")
    args = parser.parse_args()

    frames = []
    for seed in args.seeds:
        path = Path(f"{args.prefix}_s{seed}/eval/episodes.csv")
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        df = pd.read_csv(path)
        df["seed_group"] = seed
        frames.append(df)
    if not frames:
        raise SystemExit("no episode files found")
    all_df = pd.concat(frames, ignore_index=True)

    cols = ["visible_pct", "longest_hidden_s", "evader_boundary_steps",
            "seeker_boundary_steps", "evader_wall_contacts", "seeker_wall_contacts",
            "reacquisition_mean_s", "first_loss_time_s"]
    cols = [c for c in cols if c in all_df.columns]

    rows = []
    for exp in all_df.experiment.unique():
        sub = all_df[all_df.experiment == exp]
        per_seed = sub.groupby("seed_group")[cols].mean()
        row = {"experiment": exp, "n_seeds": per_seed.shape[0],
               "episodes_per_seed": len(sub) // per_seed.shape[0]}
        for c in cols:
            row[f"{c}_mean"] = per_seed[c].mean()
            row[f"{c}_std"] = per_seed[c].std()
        rows.append(row)
    summary = pd.DataFrame(rows)

    # Build markdown
    lines = [f"# Multi-seed summary ({len(args.seeds)} seeds: {args.seeds})", ""]
    lines.append("| Experiment | Visible % (mean +/- std) | Longest hidden s | Evader boundary steps | Reacq time s |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {r.experiment} | "
            f"{r.visible_pct_mean:.1f} +/- {r.visible_pct_std:.1f} | "
            f"{r.longest_hidden_s_mean:.1f} +/- {r.longest_hidden_s_std:.1f} | "
            f"{r.evader_boundary_steps_mean:.0f} +/- {r.evader_boundary_steps_std:.0f} | "
            f"{r.reacquisition_mean_s_mean:.2f} +/- {r.reacquisition_mean_s_std:.2f} |"
        )
    md = "\n".join(lines) + "\n"
    print(md)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

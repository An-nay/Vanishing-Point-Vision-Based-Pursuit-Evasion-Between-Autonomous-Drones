"""Training-stability statistics from episodes.csv of several runs (joint-game episodes only).

    python -m scripts.stability_report --group no_pool runs/final2_C_s*/simultaneous_s* --group pool runs/pool_C_s*/simultaneous_s*

Per run: the Evader's visible % smoothed over a window; oscillation = standard deviation and peak-to-peak of that
curve over the second half of training; final level = mean of the last 10 % of episodes. Reported per group as the
mean over runs plus the spread of the final level across runs (seed-to-seed stability).
"""
import argparse
import csv
from pathlib import Path
import numpy as np


def run_stats(directory, window=15, mode="selfplay"):
    rows = [r for r in csv.DictReader((Path(directory) / "episodes.csv").open()) if r["partial"] == "False"]
    if rows and "mode" in rows[0] and any(r["mode"] == mode for r in rows):
        rows = [r for r in rows if r["mode"] == mode]
    steps = np.array([int(r["step"]) for r in rows])
    visible = np.array([float(r["visible_pct"]) for r in rows])
    smooth = np.convolve(visible, np.ones(window) / window, mode="valid")
    xs = steps[len(steps) - len(smooth):]
    half = smooth[xs >= xs.max() / 2]
    tail = visible[int(0.9 * len(visible)):]
    return {"oscillation_std": float(half.std()), "peak_to_peak": float(half.max() - half.min()),
            "final_level": float(tail.mean()), "episodes": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", nargs="+", action="append", metavar=("NAME", "RUN"), required=True)
    parser.add_argument("--window", type=int, default=15)
    args = parser.parse_args()
    print("| Group | Runs | Oscillation (std, % pts) | Peak-to-peak (% pts) | Final level % (mean) | Final level spread across runs (std) |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, *runs in args.group:
        stats = [run_stats(r, args.window) for r in runs]
        finals = [s["final_level"] for s in stats]
        print(f"| {name} | {len(stats)} | {np.mean([s['oscillation_std'] for s in stats]):.1f} | "
              f"{np.mean([s['peak_to_peak'] for s in stats]):.1f} | {np.mean(finals):.1f} | {np.std(finals):.1f} |")


if __name__ == "__main__":
    main()

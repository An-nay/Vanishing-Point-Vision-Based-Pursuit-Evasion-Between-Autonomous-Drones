"""Phase D: paired starts, multi-seed evaluation and historical-opponent tests."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from vp.core import sample_initial_state
from vp.parallel import VanishingPointParallelEnv, AGENTS
from vp.maddpg import load_config
from vp.policy_io import load_policy
from vp.opponents import PPOEvader, CheckpointOpponent
from vp.paths import MODELS, RUNS

SCALAR_METRICS = [
    "seeker_reward", "evader_reward", "undetected_pct", "visible_pct",
    "first_loss_time_s", "longest_hidden_s", "seeker_wall_contacts",
    "evader_wall_contacts", "seeker_boundary_steps", "evader_boundary_steps",
    "loss_events", "reacquisitions", "reacquisition_mean_s", "unresolved_hidden_s",
    "first_detection_time_s", "drone_collision_steps", "drone_collision_events",
]


_SHAPING_KEYS = (
    "boundary_penalty_per_step",
    "seeker_boundary_penalty_per_step",
    "seeker_proximity_coef",
    "seeker_visible_bonus",
    "evader_hidden_bonus",
)


def same_world(first, second):
    """Environment settings match, ignoring training-only reward shaping (evaluation is unshaped)."""
    strip = lambda cfg: {k: v for k, v in cfg.items() if k not in _SHAPING_KEYS}
    return strip(first) == strip(second)


def _unshaped(env_options):
    # Scrub every shaping knob to its no-op value so the eval reward matches Stage 1.
    scrubbed = {**env_options}
    for key in _SHAPING_KEYS:
        if key == "seeker_visible_bonus" or key == "evader_hidden_bonus":
            scrubbed[key] = None  # None keeps the default +/-1 symmetric Stage 1 rule
        else:
            scrubbed[key] = 0.0
    return scrubbed


def evaluate_matchup(label, seeker, evader, seed, episode, start, env_options,
                     output, render=False):
    # Always score the ORIGINAL (unshaped) rewards so numbers match Stage 1 and across configs.
    env_options = _unshaped(env_options)
    env = VanishingPointParallelEnv(render_mode="human" if render else None, **env_options)
    trajectory = []
    try:
        observations, _ = env.reset(seed=seed, options={"initial_state": start})

        def capture():
            trajectory.append([env.sim.step_count, *env.sim.seeker_pos,
                               *env.sim.evader_pos, *env.sim.camera_forward, int(env.sim.visible)])

        capture()
        while env.agents:
            # Both policies see the same pre-transition observation dictionary.
            evader_action = evader(observations["evader"])
            if seeker is None:
                observations, _, _, _, _ = env.step_scripted(evader_action)
            else:
                seeker_action = seeker(observations["seeker"])
                observations, _, _, _, _ = env.step({"seeker": seeker_action, "evader": evader_action})
            capture()
        row = {"experiment": label, "seed": seed, "episode": episode,
               **env.sim.episode_summary()}
        path = output / "trajectories" / f"{label}_s{seed}_e{episode}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["step", "seeker_x", "seeker_y", "seeker_z",
                             "evader_x", "evader_y", "evader_z",
                             "forward_x", "forward_y", "forward_z", "visible"])
            writer.writerows(trajectory)
        return row
    finally:
        env.close()


def bootstrap_ci(values, resamples=4000, seed=0):
    """95% percentile bootstrap CI of the mean (resampling episodes)."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return (float(values.mean()), float(values.mean())) if len(values) else (None, None)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (resamples, len(values)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_differences(rows, reference="stage1_baseline", metric="visible_pct"):
    """Mean (experiment - reference) over episodes sharing the same start, with bootstrap CI."""
    base = {(r["seed"]): r[metric] for r in rows if r["experiment"] == reference}
    out = []
    for label in dict.fromkeys(r["experiment"] for r in rows):
        if label == reference:
            continue
        diffs = [r[metric] - base[r["seed"]] for r in rows if r["experiment"] == label and r["seed"] in base]
        if diffs:
            lo, hi = bootstrap_ci(diffs)
            out.append({"experiment": label, "metric": metric, "mean_diff": float(np.mean(diffs)),
                        "ci_lo": lo, "ci_hi": hi, "pairs": len(diffs)})
    return out


def comparison_rows(rows):
    result = []
    for label in dict.fromkeys(row["experiment"] for row in rows):
        selected = [row for row in rows if row["experiment"] == label]
        aggregate = {"experiment": label, "episodes": len(selected),
                     "seed_count": len({row["base_seed"] for row in selected}),
                     "sight_loss_episode_pct": 100 * sum(row["first_loss_time_s"] is not None
                                                        for row in selected) / len(selected),
                     "detected_episode_pct": 100 * sum(row["first_detection_time_s"] is not None
                                                       for row in selected) / len(selected)}
        for key in SCALAR_METRICS:
            values = np.array([row[key] for row in selected if row[key] is not None], dtype=float)
            aggregate[key + "_mean"] = float(values.mean()) if len(values) else None
            aggregate[key + "_std"] = float(values.std()) if len(values) else None
            aggregate[key + "_count"] = len(values)
        for key in ("visible_pct", "evader_reward"):
            aggregate[key + "_ci_lo"], aggregate[key + "_ci_hi"] = bootstrap_ci([row[key] for row in selected])
        result.append(aggregate)
    return result


def write_csv(path, rows):
    if not rows:
        raise ValueError("no evaluation results")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(ppo_path, joint_path, seeker_path, seeds, episodes, output,
             historical_evaders=(), historical_seekers=(), hidden_start=False, render=False,
             baseline_only=False, stage1_reference=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"evaluation directory is not empty: {output}")
    ppo = PPOEvader(ppo_path)
    # Stage 2 matchups contain only learned policies. The scripted Seeker belongs to Stage 1 and is
    # evaluated only for --baseline-only or when --stage1-reference is requested explicitly.
    matches = []
    if baseline_only:
        matches.append(("stage1_baseline", None, ppo.act))
    elif stage1_reference:
        matches.append(("stage1_reference", None, ppo.act))
    joint = None
    env_options = load_config()["environment"].copy()
    checkpoints = {}
    if not baseline_only:
        if not joint_path:
            raise ValueError("--joint-checkpoint is required unless --baseline-only is set")
        joint = load_policy(joint_path)
        expected = VanishingPointParallelEnv(**joint.config["environment"])
        expected_dims = {a: expected.observation_space(a).shape[0] for a in AGENTS}
        if joint.obs_dims != expected_dims or joint.action_dims != {"seeker": 5, "evader": 3}:
            raise ValueError("joint checkpoint is incompatible with Stage 2")
        env_options = joint.config["environment"].copy()
        if env_options["episode_steps"] != 600:
            raise ValueError("comparison requires the original 600-step horizon")
        learned_seeker = load_policy(seeker_path) if seeker_path else joint
        if not same_world(learned_seeker.config["environment"], joint.config["environment"]):
            raise ValueError("seeker and joint checkpoints must share environment settings")
        seeker_act = lambda obs: joint.act("seeker", obs)
        evader_act = lambda obs: joint.act("evader", obs)
        matches.append(("learned_seeker_ppo", lambda obs: learned_seeker.act("seeker", obs), ppo.act))
        if seeker_path and Path(seeker_path) != Path(joint_path):
            matches.append(("joint_seeker_ppo", seeker_act, ppo.act))
        matches.append(("stage2_joint", seeker_act, evader_act))
        checkpoints["joint"] = str(joint_path)
        checkpoints["seeker"] = str(seeker_path or joint_path)
        for index, checkpoint in enumerate(historical_evaders):
            prior = CheckpointOpponent(checkpoint, "evader")
            if not same_world(prior.learner.config["environment"], env_options):
                raise ValueError("historical Evader environment differs from evaluation")
            matches.append((f"historical_evader_{index}", seeker_act, prior.act))
            checkpoints[f"historical_evader_{index}"] = str(checkpoint)
        for index, checkpoint in enumerate(historical_seekers):
            prior = CheckpointOpponent(checkpoint, "seeker")
            if not same_world(prior.learner.config["environment"], env_options):
                raise ValueError("historical Seeker environment differs from evaluation")
            matches.append((f"historical_seeker_{index}", prior.act, evader_act))
            checkpoints[f"historical_seeker_{index}"] = str(checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    starts, rows = [], []
    for base_seed in seeds:
        for episode in range(episodes):
            seed = base_seed + episode
            seeker, evader, forward = sample_initial_state(
                np.random.default_rng(seed), obstacles=env_options.get("obstacles"))
            if hidden_start:
                forward = -forward
            start = {"seeker_pos": seeker.tolist(), "evader_pos": evader.tolist(),
                     "seeker_forward": forward.tolist()}
            starts.append({"seed": seed, "episode": episode, "initial_state": start})
            for label, seeker_policy, evader_policy in matches:
                row = evaluate_matchup(label, seeker_policy, evader_policy, seed, episode,
                                       start, env_options, output, render)
                row["base_seed"] = base_seed
                rows.append(row)
                print(f"{label} seed={seed} visible={row['visible_pct']:.1f}% "
                      f"returns=({row['seeker_reward']:.0f}, {row['evader_reward']:.0f})", flush=True)
    (output / "starts.json").write_text(json.dumps(starts, indent=2) + "\n", encoding="utf-8")
    metadata = {"ppo_model": str(ppo_path), "checkpoints": checkpoints, "seed_groups": seeds,
                "episodes_per_seed": episodes, "hidden_start": hidden_start,
                "environment": env_options,
                "learned_seeker_source": str(seeker_path or joint_path) if not baseline_only else None}
    (output / "evaluation.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "episodes.csv", rows)
    aggregates = comparison_rows(rows)
    write_csv(output / "comparison.csv", aggregates)
    table = ["| Experiment | Episodes | Visible % [95% CI] | Hidden % | Seeker return | Evader return |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in aggregates:
        table.append(f"| {row['experiment']} | {row['episodes']} | {row['visible_pct_mean']:.1f} "
                     f"[{row['visible_pct_ci_lo']:.1f}, {row['visible_pct_ci_hi']:.1f}] | "
                     f"{row['undetected_pct_mean']:.1f} | {row['seeker_reward_mean']:.1f} | "
                     f"{row['evader_reward_mean']:.1f} |")
    reference = "stage1_reference" if any(r["experiment"] == "stage1_reference" for r in rows) else rows[0]["experiment"]
    differences = paired_differences(rows, reference=reference)
    if differences:
        table += ["", f"Paired difference in visible % vs {reference} (same starts):", "",
                  "| Experiment | Mean diff | 95% CI | Pairs |", "|---|---:|---:|---:|"]
        table += [f"| {d['experiment']} | {d['mean_diff']:+.1f} | [{d['ci_lo']:+.1f}, {d['ci_hi']:+.1f}] | {d['pairs']} |"
                  for d in differences]
    table += ["", "Detection and violation metrics (proposal metrics):", "",
              "| Experiment | First detection s (mean, detected/episodes) | Evader wall+obstacle contacts / ep | "
              "Seeker contacts / ep | Drone-drone collision steps / ep |", "|---|---:|---:|---:|---:|"]
    for row in aggregates:
        first = row["first_detection_time_s_mean"]
        table.append(f"| {row['experiment']} | "
                     f"{'-' if first is None else format(first, '.1f')} "
                     f"({row['first_detection_time_s_count']}/{row['episodes']}) | "
                     f"{row['evader_wall_contacts_mean']:.1f} | {row['seeker_wall_contacts_mean']:.1f} | "
                     f"{row['drone_collision_steps_mean']:.1f} |")
    table += ["", "Rewards are the original unshaped Stage 1 rewards (no wall-penalty shaping).",
              "Time to first detection is 0 for episodes that start visible; use --hidden-start to measure real search time."]
    (output / "comparison.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo-model", type=Path, default=MODELS / "ppo_500k.zip")
    parser.add_argument("--joint-checkpoint", type=Path)
    parser.add_argument("--seeker-checkpoint", type=Path,
                        help="Phase B checkpoint; defaults to the joint Seeker if omitted")
    parser.add_argument("--historical-evaders", type=Path, nargs="*", default=[])
    parser.add_argument("--historical-seekers", type=Path, nargs="*", default=[])
    parser.add_argument("--seeds", type=int, nargs="+", default=[10001, 11001, 12001])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", type=Path, default=RUNS / "evaluation")
    parser.add_argument("--hidden-start", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--baseline-only", action="store_true", help="Stage 1 only: scripted Seeker vs PPO")
    parser.add_argument("--stage1-reference", action="store_true",
                        help="also include the Stage 1 scripted-Seeker-vs-PPO reference (not part of Stage 2)")
    args = parser.parse_args()
    if args.episodes <= 0 or any(seed < 0 for seed in args.seeds):
        parser.error("episodes must be positive and seeds non-negative")
    expanded_seeds = [seed + e for seed in args.seeds for e in range(args.episodes)]
    if len(set(expanded_seeds)) != len(expanded_seeds):
        parser.error("seed groups overlap; use distinct episode seeds")
    try:
        evaluate(args.ppo_model, args.joint_checkpoint, args.seeker_checkpoint, args.seeds,
                 args.episodes, args.output, args.historical_evaders, args.historical_seekers,
                 args.hidden_start, args.render, args.baseline_only, args.stage1_reference)
        from scripts.plot_stage2 import plot_evaluation
        plot_evaluation(args.output)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

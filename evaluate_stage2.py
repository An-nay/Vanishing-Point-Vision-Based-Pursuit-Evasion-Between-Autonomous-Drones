"""Phase D: paired starts, multi-seed evaluation and historical-opponent tests."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from vanishing_point_core import sample_initial_state
from vanishing_point_parallel import VanishingPointParallelEnv, AGENTS
from maddpg import MADDPG, load_config
from stage2_opponents import PPOEvader, CheckpointOpponent

SCALAR_METRICS = [
    "seeker_reward", "evader_reward", "undetected_pct", "visible_pct",
    "first_loss_time_s", "longest_hidden_s", "seeker_wall_contacts",
    "evader_wall_contacts", "seeker_boundary_steps", "evader_boundary_steps",
    "loss_events", "reacquisitions", "reacquisition_mean_s", "unresolved_hidden_s",
]


def evaluate_matchup(label, seeker, evader, seed, episode, start, env_options,
                     output, render=False):
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


def comparison_rows(rows):
    result = []
    for label in dict.fromkeys(row["experiment"] for row in rows):
        selected = [row for row in rows if row["experiment"] == label]
        aggregate = {"experiment": label, "episodes": len(selected),
                     "seed_count": len({row["seed"] for row in selected}),
                     "sight_loss_episode_pct": 100 * sum(row["first_loss_time_s"] is not None
                                                        for row in selected) / len(selected)}
        for key in SCALAR_METRICS:
            values = np.array([row[key] for row in selected if row[key] is not None], dtype=float)
            aggregate[key + "_mean"] = float(values.mean()) if len(values) else None
            aggregate[key + "_std"] = float(values.std()) if len(values) else None
            aggregate[key + "_count"] = len(values)
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
             baseline_only=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"evaluation directory is not empty: {output}")
    ppo = PPOEvader(ppo_path)
    matches = [("stage1_baseline", None, ppo.act)]
    joint = None
    env_options = load_config()["environment"].copy()
    checkpoints = {}
    if not baseline_only:
        if not joint_path:
            raise ValueError("--joint-checkpoint is required unless --baseline-only is set")
        joint = MADDPG.load(joint_path)
        if joint.obs_dims != {"seeker": 21, "evader": 21} or joint.action_dims != {"seeker": 5, "evader": 3}:
            raise ValueError("joint checkpoint is incompatible with Stage 2")
        env_options = joint.config["environment"].copy()
        if env_options["episode_steps"] != 600:
            raise ValueError("comparison requires the original 600-step horizon")
        learned_seeker = MADDPG.load(seeker_path) if seeker_path else joint
        if learned_seeker.config["environment"] != joint.config["environment"]:
            raise ValueError("seeker and joint checkpoints must share environment settings")
        seeker_act = lambda obs: joint.act("seeker", obs)
        evader_act = lambda obs: joint.act("evader", obs)
        matches += [
            ("learned_seeker_ppo", lambda obs: learned_seeker.act("seeker", obs), ppo.act),
            ("stage2_joint", seeker_act, evader_act),
            ("scripted_maddpg", None, evader_act),
        ]
        checkpoints["joint"] = str(joint_path)
        checkpoints["seeker"] = str(seeker_path or joint_path)
        for index, checkpoint in enumerate(historical_evaders):
            prior = CheckpointOpponent(checkpoint, "evader")
            if prior.learner.config["environment"] != env_options:
                raise ValueError("historical Evader environment differs from evaluation")
            matches.append((f"historical_evader_{index}", seeker_act, prior.act))
            checkpoints[f"historical_evader_{index}"] = str(checkpoint)
        for index, checkpoint in enumerate(historical_seekers):
            prior = CheckpointOpponent(checkpoint, "seeker")
            if prior.learner.config["environment"] != env_options:
                raise ValueError("historical Seeker environment differs from evaluation")
            matches.append((f"historical_seeker_{index}", prior.act, evader_act))
            checkpoints[f"historical_seeker_{index}"] = str(checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    starts, rows = [], []
    for base_seed in seeds:
        for episode in range(episodes):
            seed = base_seed + episode
            seeker, evader, forward = sample_initial_state(np.random.default_rng(seed))
            if hidden_start:
                forward = -forward
            start = {"seeker_pos": seeker.tolist(), "evader_pos": evader.tolist(),
                     "seeker_forward": forward.tolist()}
            starts.append({"seed": seed, "episode": episode, "initial_state": start})
            for label, seeker_policy, evader_policy in matches:
                row = evaluate_matchup(label, seeker_policy, evader_policy, seed, episode,
                                       start, env_options, output, render)
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
    table = ["| Experiment | Episodes | Visible % | Hidden % | Seeker return | Evader return |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in aggregates:
        table.append(f"| {row['experiment']} | {row['episodes']} | {row['visible_pct_mean']:.1f} | "
                     f"{row['undetected_pct_mean']:.1f} | {row['seeker_reward_mean']:.1f} | "
                     f"{row['evader_reward_mean']:.1f} |")
    (output / "comparison.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo-model", type=Path, default=Path("models/ppo_500k.zip"))
    parser.add_argument("--joint-checkpoint", type=Path)
    parser.add_argument("--seeker-checkpoint", type=Path,
                        help="Phase B checkpoint; defaults to the joint Seeker if omitted")
    parser.add_argument("--historical-evaders", type=Path, nargs="*", default=[])
    parser.add_argument("--historical-seekers", type=Path, nargs="*", default=[])
    parser.add_argument("--seeds", type=int, nargs="+", default=[10001, 11001, 12001])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation"))
    parser.add_argument("--hidden-start", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    if args.episodes <= 0 or any(seed < 0 for seed in args.seeds):
        parser.error("episodes must be positive and seeds non-negative")
    expanded_seeds = [seed + e for seed in args.seeds for e in range(args.episodes)]
    if len(set(expanded_seeds)) != len(expanded_seeds):
        parser.error("seed groups overlap; use distinct episode seeds")
    try:
        evaluate(args.ppo_model, args.joint_checkpoint, args.seeker_checkpoint, args.seeds,
                 args.episodes, args.output, args.historical_evaders, args.historical_seekers,
                 args.hidden_start, args.render, args.baseline_only)
        from plot_stage2 import plot_evaluation
        plot_evaluation(args.output)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

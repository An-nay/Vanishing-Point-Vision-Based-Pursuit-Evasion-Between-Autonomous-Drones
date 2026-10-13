"""Replay trained MADDPG checkpoints in the PyBullet GUI (one episode per matchup)."""
import argparse
import tempfile
from pathlib import Path
import numpy as np
from vanishing_point_core import sample_initial_state
from policy_io import load_policy
from stage2_opponents import PPOEvader
from evaluate_stage2 import evaluate_matchup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--seeker-checkpoint", type=Path,
                        help="Phase B checkpoint for the learned-Seeker-vs-PPO matchup")
    parser.add_argument("--ppo-model", type=Path, default=Path("models/ppo_500k.zip"))
    parser.add_argument("--matchups", nargs="+", default=["stage2_joint"],
                        choices=["stage1_baseline", "learned_seeker_ppo", "stage2_joint"])
    parser.add_argument("--seed", type=int, default=10001)
    args = parser.parse_args()

    joint = load_policy(args.joint_checkpoint)
    seeker_net = load_policy(args.seeker_checkpoint) if args.seeker_checkpoint else joint
    ppo = PPOEvader(args.ppo_model)
    options = joint.config["environment"].copy()
    table = {
        "stage1_baseline": (None, ppo.act),
        "learned_seeker_ppo": (lambda o: seeker_net.act("seeker", o), ppo.act),
        "stage2_joint": (lambda o: joint.act("seeker", o), lambda o: joint.act("evader", o)),
    }
    seeker, evader, forward = sample_initial_state(
        np.random.default_rng(args.seed), obstacles=options.get("obstacles"))
    start = {"seeker_pos": seeker.tolist(), "evader_pos": evader.tolist(),
             "seeker_forward": forward.tolist()}
    with tempfile.TemporaryDirectory() as scratch:
        for label in args.matchups:
            print(f"Replaying {label} (seed {args.seed}) ...", flush=True)
            seeker_policy, evader_policy = table[label]
            row = evaluate_matchup(label, seeker_policy, evader_policy, args.seed, 0,
                                   start, options, Path(scratch), render=True)
            print(f"  visible={row['visible_pct']:.1f}% returns=({row['seeker_reward']:.0f}, "
                  f"{row['evader_reward']:.0f})", flush=True)


if __name__ == "__main__":
    main()

"""Paired evaluation and censored-metric checks, including the existing PPO."""
from pathlib import Path
import copy
import csv
import json
import numpy as np
import pytest
from evaluate_stage2 import evaluate, comparison_rows, SCALAR_METRICS
from maddpg import MADDPG, load_config
from vanishing_point_parallel import AGENTS


def test_missing_events_are_not_counted_as_zero_delay():
    first = {"experiment": "x", "seed": 1, **{key: 0 for key in SCALAR_METRICS}}
    second = copy.deepcopy(first)
    first.update(first_loss_time_s=None, reacquisition_mean_s=None, visible_pct=20.)
    second.update(seed=2, first_loss_time_s=3., reacquisition_mean_s=5., visible_pct=80.)
    summary = comparison_rows([first, second])[0]
    assert summary["first_loss_time_s_mean"] == 3.
    assert summary["first_loss_time_s_count"] == 1
    assert summary["reacquisition_mean_s_mean"] == 5.
    assert summary["reacquisition_mean_s_count"] == 1
    assert summary["visible_pct_mean"] == 50.
    assert summary["visible_pct_std"] == 30.
    assert summary["sight_loss_episode_pct"] == 50.


def test_four_way_evaluation_uses_identical_initial_conditions(tmp_path):
    config = load_config()
    config["hidden_sizes"] = [16, 16]
    learner = MADDPG({a: 21 for a in AGENTS}, {"seeker": 5, "evader": 3}, 26, config)
    checkpoint = tmp_path / "joint.pt"
    learner.save(checkpoint)
    output = tmp_path / "evaluation"
    rows = evaluate(Path(__file__).with_name("models") / "ppo_500k.zip",
                    checkpoint, checkpoint, [51001], 1, output)
    assert len(rows) == 4
    assert {row["experiment"] for row in rows} == {
        "stage1_baseline", "learned_seeker_ppo", "stage2_joint", "scripted_maddpg"}
    trajectories = list((output / "trajectories").glob("*.csv"))
    assert len(trajectories) == 4
    starts = []
    for path in trajectories:
        with path.open(newline="") as stream:
            frames = list(csv.DictReader(stream))
        assert len(frames) == 601
        starts.append([frames[0][key] for key in ["seeker_x", "seeker_y", "seeker_z",
                                                "evader_x", "evader_y", "evader_z",
                                                "forward_x", "forward_y", "forward_z"]])
    assert all(start == starts[0] for start in starts)
    for row in rows:
        assert row["visible_pct"] + row["undetected_pct"] == pytest.approx(100.)
        assert np.isfinite(row["seeker_reward"]) and np.isfinite(row["evader_reward"])
    assert (output / "comparison.csv").is_file() and (output / "comparison.md").is_file()
    metadata = json.loads((output / "evaluation.json").read_text())
    assert metadata["learned_seeker_source"] == str(checkpoint)


def test_config_rejects_unknown_environment_option(tmp_path):
    config = load_config()
    config["environment"]["dt"] = .2
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="environment must contain"):
        load_config(path)

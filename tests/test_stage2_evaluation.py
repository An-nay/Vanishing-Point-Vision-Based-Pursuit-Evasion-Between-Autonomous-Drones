"""Paired evaluation and censored-metric checks, including the existing PPO."""
from pathlib import Path
import copy
import csv
import json
import numpy as np
import pytest
from scripts.evaluate_stage2 import evaluate, comparison_rows, SCALAR_METRICS, bootstrap_ci, paired_differences
from vp.maddpg import MADDPG, load_config
from vp.parallel import AGENTS


def test_missing_events_are_not_counted_as_zero_delay():
    first = {"experiment": "x", "seed": 1, "base_seed": 1, **{key: 0 for key in SCALAR_METRICS}}
    second = copy.deepcopy(first)
    first.update(first_loss_time_s=None, reacquisition_mean_s=None, visible_pct=20.)
    second.update(seed=2, base_seed=2, first_loss_time_s=3., reacquisition_mean_s=5., visible_pct=80.)
    summary = comparison_rows([first, second])[0]
    assert summary["first_loss_time_s_mean"] == 3.
    assert summary["first_loss_time_s_count"] == 1
    assert summary["reacquisition_mean_s_mean"] == 5.
    assert summary["reacquisition_mean_s_count"] == 1
    assert summary["visible_pct_mean"] == 50.
    assert summary["visible_pct_std"] == 30.
    assert summary["sight_loss_episode_pct"] == 50.


def test_evaluation_uses_identical_initial_conditions_and_only_learned_policies(tmp_path):
    config = load_config()
    config["hidden_sizes"] = [16, 16]
    learner = MADDPG({a: 21 for a in AGENTS}, {"seeker": 5, "evader": 3}, 26, config)
    checkpoint = tmp_path / "joint.pt"
    learner.save(checkpoint)
    output = tmp_path / "evaluation"
    rows = evaluate(Path(__file__).resolve().parent.parent / "models" / "ppo_500k.zip",
                    checkpoint, checkpoint, [51001], 1, output)
    assert len(rows) == 2
    assert {row["experiment"] for row in rows} == {"learned_seeker_ppo", "stage2_joint"}  # no scripted policy
    trajectories = list((output / "trajectories").glob("*.csv"))
    assert len(trajectories) == 2
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


def test_seed_count_counts_seed_groups_not_episodes():
    rows = [{"experiment": "x", "seed": 10 + e + 100 * g, "base_seed": 10 + 100 * g,
             **{key: 1.0 for key in SCALAR_METRICS}} for g in range(3) for e in range(4)]
    assert comparison_rows(rows)[0]["seed_count"] == 3 and comparison_rows(rows)[0]["episodes"] == 12


def test_bootstrap_ci_brackets_the_mean_and_paired_differences_use_same_starts():
    low, high = bootstrap_ci(np.arange(100.0))
    assert low < 49.5 < high and high - low < 15
    rows = []
    for seed in range(20):
        base = {key: 0.0 for key in SCALAR_METRICS}
        rows.append({"experiment": "stage1_baseline", "seed": seed, "base_seed": 0, **base, "visible_pct": 10.0})
        rows.append({"experiment": "other", "seed": seed, "base_seed": 0, **base, "visible_pct": 10.0 + seed % 2})
    diff = paired_differences(rows)[0]
    assert diff["pairs"] == 20 and diff["mean_diff"] == pytest.approx(0.5) and diff["ci_lo"] <= 0.5 <= diff["ci_hi"]


def test_scripted_stage1_reference_is_opt_in(tmp_path):
    config = load_config()
    config["hidden_sizes"] = [16, 16]
    learner = MADDPG({a: 21 for a in AGENTS}, {"seeker": 5, "evader": 3}, 26, config)
    checkpoint = tmp_path / "joint.pt"
    learner.save(checkpoint)
    ppo = Path(__file__).resolve().parent.parent / "models" / "ppo_500k.zip"
    rows = evaluate(ppo, checkpoint, checkpoint, [51001], 1, tmp_path / "ref", stage1_reference=True)
    assert {row["experiment"] for row in rows} == {"stage1_reference", "learned_seeker_ppo", "stage2_joint"}


def test_crossplay_matrix_and_summary(tmp_path):
    from scripts.crossplay_stage2 import crossplay, summarize
    config = load_config()
    config["hidden_sizes"] = [16, 16]
    config["environment"]["episode_steps"] = 30
    paths = []
    for seed in (1, 2, 3):
        config["seed"] = seed
        learner = MADDPG({a: 21 for a in AGENTS}, {"seeker": 5, "evader": 3}, 26, config)
        learner.save(tmp_path / f"c{seed}.pt")
        paths.append(tmp_path / f"c{seed}.pt")
    matrix = crossplay(paths, [30001], 2)
    assert matrix.shape == (3, 3) and np.all((matrix >= 0) & (matrix <= 100))
    summary = summarize(np.array([[10., 20., 30.], [20., 10., 30.], [40., 20., 10.]]))
    assert summary["own_pair_mean"] == pytest.approx(10.) and summary["stranger_mean"] == pytest.approx(26.6667, abs=1e-3)
    assert summary["overfit_gap"] == pytest.approx(16.6667, abs=1e-3)

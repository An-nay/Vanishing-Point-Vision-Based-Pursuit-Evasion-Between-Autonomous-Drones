"""MAPPO: GAE, masking, bootstrapping, persistence, training smoke runs and evaluator compatibility."""
import copy
import csv
import numpy as np
import pytest
import torch
from mappo import MAPPO, gae, load_mappo_config
from train_mappo import train
from policy_io import load_policy
from vanishing_point_parallel import VanishingPointParallelEnv, AGENTS
from evaluate_stage2 import evaluate
from pathlib import Path


def small_config():
    config = load_mappo_config()
    config.update(hidden_sizes=[16, 16], rollout_steps=64, minibatch=32, epochs=2, total_steps=192,
                  checkpoint_every=64, mix_ppo_evader=0.0, mix_frozen_seeker=0.0)
    config.pop("mix_ppo_model", None)
    config["environment"].update(episode_steps=20)
    return config


def test_gae_matches_hand_computation_and_cuts_at_done():
    rewards, values = [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]
    adv, ret = gae(rewards, values, [0.0, 0.0, 0.0], last_value=2.0, gamma=0.5, lam=1.0)
    # with these numbers the discounted return plus bootstrap is 2.0 at every step
    assert adv == pytest.approx([2.0, 2.0, 2.0])
    cut, _ = gae(rewards, values, [0.0, 1.0, 0.0], last_value=2.0, gamma=0.5, lam=1.0)
    assert cut[1] == pytest.approx(1.0) and cut[0] == pytest.approx(1.5)  # nothing leaks across the episode end


def test_masked_steps_do_not_train_the_agent():
    config = small_config()
    env = VanishingPointParallelEnv(**config["environment"])
    dims = {a: env.observation_space(a).shape[0] for a in AGENTS}
    acts = {a: env.action_space(a).shape[0] for a in AGENTS}
    learner = MAPPO(dims, acts, 26, config)
    n = 64
    rng = np.random.default_rng(0)
    data = {"obs": rng.normal(size=(n, dims["seeker"])), "state": rng.normal(size=(n, 26)),
            "act": rng.normal(size=(n, 5)), "logp": np.zeros(n), "adv": rng.normal(size=n),
            "ret": rng.normal(size=n), "valid": np.zeros(n)}
    before = [p.detach().clone() for p in learner.actors["seeker"].parameters()]
    learner.update("seeker", data, rng)
    assert all(torch.equal(a, b) for a, b in zip(before, learner.actors["seeker"].parameters()))
    data["valid"] = np.ones(n)
    learner.update("seeker", data, rng)
    assert any(not torch.equal(a, b) for a, b in zip(before, learner.actors["seeker"].parameters()))


def test_checkpoint_roundtrip_and_shared_loader(tmp_path):
    config = small_config()
    env = VanishingPointParallelEnv(**config["environment"])
    dims = {a: env.observation_space(a).shape[0] for a in AGENTS}
    acts = {a: env.action_space(a).shape[0] for a in AGENTS}
    learner = MAPPO(dims, acts, 26, config)
    learner.save(tmp_path / "m.pt")
    loaded = load_policy(tmp_path / "m.pt")
    assert isinstance(loaded, MAPPO)
    obs, _ = env.reset(seed=1)
    for agent in AGENTS:
        assert np.array_equal(learner.act(agent, obs[agent]), loaded.act(agent, obs[agent]))
        assert np.all(np.abs(loaded.act(agent, obs[agent])) <= 1)


def test_training_phases_log_opponent_modes(tmp_path):
    config = small_config()
    seeker_ckpt = train(copy.deepcopy(config), "seeker", tmp_path / "B", opponent_model="models/ppo_500k.zip")
    config.update(mix_frozen_seeker=0.5, alternate_every=64)
    train(copy.deepcopy(config), "simultaneous", tmp_path / "C", initialize=seeker_ckpt)
    with (tmp_path / "C" / "episodes.csv").open() as stream:
        assert {row["mode"] for row in csv.DictReader(stream)} <= {"selfplay", "frozen_seeker"}
    with (tmp_path / "B" / "episodes.csv").open() as stream:
        assert {row["mode"] for row in csv.DictReader(stream)} == {"ppo_evader"}


def test_evaluator_accepts_mappo_checkpoints(tmp_path):
    config = small_config()
    config["environment"]["episode_steps"] = 600  # the evaluator requires the original horizon
    config.update(total_steps=64)
    seeker_ckpt = train(copy.deepcopy(config), "seeker", tmp_path / "B", opponent_model="models/ppo_500k.zip")
    joint_ckpt = train(copy.deepcopy(config), "simultaneous", tmp_path / "C", initialize=seeker_ckpt)
    rows = evaluate(Path(__file__).resolve().parent.parent / "models" / "ppo_500k.zip", joint_ckpt, seeker_ckpt,
                    [51001], 1, tmp_path / "eval")
    assert {row["experiment"] for row in rows} == {"learned_seeker_ppo", "joint_seeker_ppo", "stage2_joint"}

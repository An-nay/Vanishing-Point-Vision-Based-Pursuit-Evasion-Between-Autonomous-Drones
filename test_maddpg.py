"""Learning, replay, checkpoint and frozen-opponent integration tests."""
import copy
import numpy as np
import pytest
import torch
from maddpg import MADDPG, ReplayBuffer, load_config, bellman_target
from train_maddpg import train, noise_at
from vanishing_point_parallel import VanishingPointParallelEnv, AGENTS


def small_config():
    config = load_config()
    config.update(hidden_sizes=[16, 16], batch_size=8, replay_capacity=64,
                  warmup_steps=8, learn_every=4, checkpoint_every=16, total_steps=32)
    config["environment"]["episode_steps"] = 8
    return config


def setup_batch(config):
    env = VanishingPointParallelEnv(**config["environment"])
    dimensions = {a: env.observation_space(a).shape[0] for a in AGENTS}
    actions = {a: env.action_space(a).shape[0] for a in AGENTS}
    learner = MADDPG(dimensions, actions, 26, config)
    replay = ReplayBuffer(config["replay_capacity"], dimensions, actions, 26, seed=11)
    obs, _ = env.reset(seed=11)
    for _ in range(16):
        if not env.agents:
            obs, _ = env.reset(seed=12)
        controls = {a: env.action_space(a).sample() for a in AGENTS}
        state = env.state().copy()
        nxt, reward, term, trunc, _ = env.step(controls)
        replay.add(obs, controls, reward, nxt, state, env.state().copy(), term, trunc)
        obs = nxt
    return learner, replay


def test_both_actors_and_critics_learn_finite_updates():
    learner, replay = setup_batch(small_config())
    before = {a: [p.detach().clone() for p in learner.actors[a].parameters()] for a in AGENTS}
    target_before = {a: [p.detach().clone() for p in learner.target_actors[a].parameters()] for a in AGENTS}
    metrics = learner.update(replay)
    assert len(metrics) == 4 and all(np.isfinite(v) for v in metrics.values())
    for agent in AGENTS:
        current = list(learner.actors[agent].parameters())
        assert any(not torch.equal(a, b) for a, b in zip(before[agent], current))
        for previous, source, target in zip(target_before[agent], current,
                                             learner.target_actors[agent].parameters()):
            torch.testing.assert_close(target, previous.lerp(source.detach(), learner.config["tau"]))


def test_frozen_opponent_used_for_targets_and_not_updated():
    learner, replay = setup_batch(small_config())
    before = copy.deepcopy(learner.actors["evader"].state_dict())
    seen = []

    def frozen(next_observations):
        seen.append(next_observations.shape)
        return np.full((len(next_observations), 3), .25, np.float32)

    metrics = learner.update(replay, ("seeker",), {"evader": frozen})
    assert seen == [(8, 21)] and set(metrics) == {"seeker_actor_loss", "seeker_critic_loss"}
    for key, value in learner.actors["evader"].state_dict().items():
        assert torch.equal(value, before[key])


def test_time_limit_bootstraps_but_termination_does_not():
    rewards = torch.tensor([[1.], [1.]])
    terminal = torch.tensor([[0.], [1.]])
    result = bellman_target(rewards, terminal, torch.tensor([[10.], [10.]]), .9)
    torch.testing.assert_close(result, torch.tensor([[10.], [1.]]))


def test_replay_ring_alignment_and_final_next_state():
    config = small_config()
    _, replay = setup_batch(config)
    replay.capacity = 3
    replay.size = replay.position = 0
    for index in range(5):
        obs = {a: np.full(21, index, np.float32) for a in AGENTS}
        controls = {"seeker": np.zeros(5), "evader": np.zeros(3)}
        nxt = {a: np.full(21, index + .5, np.float32) for a in AGENTS}
        replay.add(obs, controls, {a: index for a in AGENTS}, nxt,
                   np.full(26, index), np.full(26, index + .5),
                   {a: False for a in AGENTS}, {a: True for a in AGENTS})
    batch = replay.sample(3)
    assert set(batch["state"][:, 0].tolist()) == {2., 3., 4.}
    torch.testing.assert_close(batch["next_state"][:, 0], batch["state"][:, 0] + .5)
    torch.testing.assert_close(batch["next_observations"]["evader"][:, 0],
                               batch["rewards"][:, 1] + .5)
    assert torch.all(batch["truncated"] == 1)


def test_checkpoint_round_trip_preserves_actions_and_noise(tmp_path):
    learner, replay = setup_batch(small_config())
    learner.update(replay)
    learner.training_step = 16
    path = tmp_path / "model.pt"
    learner.save(path)
    loaded = MADDPG.load(path)
    assert loaded.update_count == learner.update_count and loaded.training_step == 16
    obs = np.zeros(21, np.float32)
    for agent in AGENTS:
        np.testing.assert_array_equal(learner.act(agent, obs), loaded.act(agent, obs))
        np.testing.assert_array_equal(learner.act(agent, obs, .1), loaded.act(agent, obs, .1))
    assert not path.with_suffix(".pt.tmp").exists()


def test_actor_rejects_nonfinite_or_invalid_local_inputs():
    learner, _ = setup_batch(small_config())
    with pytest.raises(ValueError):
        learner.act("seeker", np.zeros(20))
    with pytest.raises(ValueError):
        learner.act("evader", np.full(21, np.nan))
    with pytest.raises(ValueError):
        MADDPG.load("models/ppo_500k.zip")


def test_replay_rejects_invalid_transition_before_advancing():
    learner, replay = setup_batch(small_config())
    size, position = len(replay), replay.position
    with pytest.raises(ValueError):
        replay.add({a: np.zeros(21) for a in AGENTS},
                   {"seeker": np.zeros(5), "evader": np.full(3, np.inf)},
                   {a: 0. for a in AGENTS}, {a: np.zeros(21) for a in AGENTS},
                   np.zeros(26), np.zeros(26),
                   {a: False for a in AGENTS}, {a: False for a in AGENTS})
    assert (len(replay), replay.position) == (size, position)


def test_noise_decays_to_configured_floor():
    config = small_config()
    assert noise_at(0, config) == config["noise_start"]
    assert noise_at(config["noise_decay_steps"] * 2, config) == pytest.approx(config["noise_end"])


@pytest.mark.parametrize("phase", ["seeker", "simultaneous"])
def test_short_training_writes_checkpoints_and_logs(tmp_path, phase):
    import csv
    config = small_config()
    final = train(config, phase, tmp_path / phase)
    saved = MADDPG.load(final)
    assert saved.training_step == 32 and saved.update_count > 0
    assert (final.parent / "checkpoint_00000016.pt").is_file()
    with (final.parent / "episodes.csv").open(newline="") as stream:
        episodes = list(csv.DictReader(stream))
    assert len(episodes) == 4 and all(row["partial"] == "False" for row in episodes)
    with (final.parent / "updates.csv").open(newline="") as stream:
        updates = list(csv.DictReader(stream))
    assert len(updates) == saved.update_count
    assert all(float(row["seeker_critic_loss"]) >= 0 for row in updates)
    if phase == "seeker":
        assert all(not row["evader_actor_loss"] for row in updates)


def test_training_reproducible_on_cpu(tmp_path):
    config = small_config()
    first = MADDPG.load(train(config, "simultaneous", tmp_path / "first"))
    second = MADDPG.load(train(config, "simultaneous", tmp_path / "second"))
    for agent in AGENTS:
        for key, weights in first.actors[agent].state_dict().items():
            torch.testing.assert_close(weights, second.actors[agent].state_dict()[key], rtol=0, atol=0)

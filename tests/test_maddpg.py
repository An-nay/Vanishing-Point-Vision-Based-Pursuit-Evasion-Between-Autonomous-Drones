"""Learning, replay, checkpoint and frozen-opponent integration tests."""
import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from vp.maddpg import MADDPG, ReplayBuffer, load_config, bellman_target
from scripts.train_maddpg import train, noise_at
from vp.parallel import VanishingPointParallelEnv, AGENTS


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


def test_mixed_opponent_training_uses_a_frozen_learned_seeker(tmp_path):
    import csv
    config = small_config()
    config.update(total_steps=160, mix_frozen_seeker=0.5)
    warm = tmp_path / "warm.pt"
    learner, _ = setup_batch(config)
    learner.save(warm)
    train(config, "simultaneous", tmp_path / "mix", initialize=warm)
    with (tmp_path / "mix" / "episodes.csv").open() as stream:
        modes = {row["mode"] for row in csv.DictReader(stream)}
    assert modes == {"selfplay", "frozen_seeker"}
    with pytest.raises(ValueError, match="initialize"):
        train(config, "simultaneous", tmp_path / "nomix")


def test_mix_config_validation():
    config = small_config()
    config.update(mix_frozen_seeker=0.7, mix_ppo_evader=0.5, mix_ppo_model="x")
    import json, tempfile, os
    path = os.path.join(tempfile.mkdtemp(), "c.json")
    Path(path).write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_config(path)
    config.update(mix_frozen_seeker=0.2, mix_ppo_evader=0.2)
    config.pop("mix_ppo_model")
    Path(path).write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_config(path)


def test_flagged_frozen_next_actions_only_replace_flagged_rows():
    config = small_config()
    learner, replay = setup_batch(config)
    seen = []

    def frozen(next_obs):
        seen.append(len(next_obs))
        return np.zeros((len(next_obs), 3), np.float32)

    replay.frozen[:] = 0.0
    learner.update(replay, ("seeker",), {"evader": frozen}, flagged_only=True)
    assert seen == []                      # nothing flagged -> frozen policy never queried
    replay.frozen[:5, 1] = 1.0  # column 1 = the Evader
    learner.update(replay, ("seeker",), {"evader": frozen}, flagged_only=True)
    assert seen and all(n <= config["batch_size"] for n in seen)


def test_frozen_next_action_enters_other_critics_but_never_the_frozen_agents_own_critic():
    config = small_config()
    learner, replay = setup_batch(config)
    replay.frozen[:, 1] = 1.0
    seen = {}
    for agent in AGENTS:
        learner.target_critics[agent].register_forward_hook(
            lambda module, inputs, output, name=agent: seen.__setitem__(name, inputs[1].detach().clone()))
    constant = lambda next_obs: np.full((len(next_obs), 3), 0.123, np.float32)
    learner.update(replay, AGENTS, {"evader": constant}, flagged_only=True)
    evader_slice = slice(5, 8)  # joint action layout: seeker(5) then evader(3)
    assert torch.allclose(seen["seeker"][:, evader_slice], torch.full((config["batch_size"], 3), 0.123))
    assert not torch.allclose(seen["evader"][:, evader_slice], torch.full((config["batch_size"], 3), 0.123))


def test_frozen_seeker_action_enters_the_evaders_critic_but_never_the_seekers_own():
    config = small_config()
    learner, replay = setup_batch(config)
    replay.frozen[:, 0] = 1.0  # column 0 = the Seeker
    seen = {}
    for agent in AGENTS:
        learner.target_critics[agent].register_forward_hook(
            lambda module, inputs, output, name=agent: seen.__setitem__(name, inputs[1].detach().clone()))
    constant = lambda next_obs: np.full((len(next_obs), 5), 0.321, np.float32)
    learner.update(replay, AGENTS, {"seeker": constant}, flagged_only=True)
    ones = torch.full((config["batch_size"], 5), 0.321)
    assert torch.allclose(seen["evader"][:, :5], ones)
    assert not torch.allclose(seen["seeker"][:, :5], ones)


def test_opponent_provider_uses_each_rows_own_snapshot_and_flags_evicted_rows():
    from vp.maddpg import Actor
    from vp.opponents import OpponentProvider
    torch.manual_seed(0)
    build = lambda: Actor(21, 3, [8, 8])
    provider = OpponentProvider("evader", 3, fixed=None, capacity=2, build_actor=build)
    nets = [build() for _ in range(3)]
    ids = [provider.add_snapshot(net.state_dict()) for net in nets]  # capacity 2 evicts the first
    obs = np.random.default_rng(0).normal(size=(4, 21)).astype(np.float32)
    out = provider.batch(obs, np.array([ids[1], ids[2], ids[1], ids[0]]))
    expected = [np.clip(nets[1](torch.as_tensor(obs[0])).detach().numpy(), -1, 1),
                np.clip(nets[2](torch.as_tensor(obs[1])).detach().numpy(), -1, 1)]
    assert np.allclose(out[0], expected[0], atol=1e-6) and np.allclose(out[1], expected[1], atol=1e-6)
    assert np.all(np.isnan(out[3]))                     # evicted snapshot -> caller must fall back
    assert provider.snapshot_ids() == ids[1:]


def test_pool_training_samples_past_snapshots_and_stays_finite(tmp_path):
    import csv
    config = small_config()
    config.update(total_steps=240, mix_pool_seeker=0.4, mix_pool_evader=0.4, pool_every=40, pool_size=3,
                  warmup_steps=8)
    train(config, "simultaneous", tmp_path / "pool")
    with (tmp_path / "pool" / "episodes.csv").open() as stream:
        modes = {row["mode"] for row in csv.DictReader(stream)}
    assert {"pool_seeker", "pool_evader"} <= modes
    with (tmp_path / "pool" / "updates.csv").open() as stream:
        assert all(np.isfinite(float(r["evader_critic_loss"] or 0)) for r in csv.DictReader(stream))
    bad = dict(config, mix_pool_seeker=0.7, mix_pool_evader=0.5)
    import json, os, tempfile
    path = os.path.join(tempfile.mkdtemp(), "c.json")
    Path(path).write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        load_config(path)

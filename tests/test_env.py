import warnings

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from steerbench.config import BenchmarkConfig
from steerbench.env import SteeringEnv, encode_observation
from steerbench.machine import make_machines


@pytest.fixture
def bench() -> BenchmarkConfig:
    return BenchmarkConfig()


def make_env(bench):
    return SteeringEnv(bench.machine, bench.env, bench.randomisation)


def test_passes_check_env(bench):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*render.*")
        check_env(make_env(bench))


def rollout(env, seed, actions):
    obs, _ = env.reset(seed=seed)
    out = [obs]
    rewards = []
    for a in actions:
        obs, r, term, trunc, _ = env.step(a)
        out.append(obs)
        rewards.append(r)
        if term or trunc:
            break
    return np.array(out), np.array(rewards)


def test_reproducible_under_fixed_seed(bench):
    actions = np.random.default_rng(0).uniform(-1, 1, (15, 10)).astype(np.float32)
    o1, r1 = rollout(make_env(bench), 42, actions)
    o2, r2 = rollout(make_env(bench), 42, actions)
    np.testing.assert_array_equal(o1, o2)
    np.testing.assert_array_equal(r1, r2)
    o3, _ = rollout(make_env(bench), 43, actions)
    assert not np.array_equal(o1[0], o3[0])


def test_observation_layout_and_masking(bench):
    env = make_env(bench)
    obs, _ = env.reset(seed=1)
    n_b, n_c = env.machine.n_bpms, env.machine.n_correctors
    assert obs.shape == (2 * n_b + n_c,)
    x, mask, theta = obs[:n_b], obs[n_b : 2 * n_b], obs[2 * n_b :]
    assert np.all(x[mask == 0] == 0)
    np.testing.assert_array_equal(mask, env.obs.mask.astype(np.float32))
    np.testing.assert_array_equal(theta, 0.0)


def test_rewards_telescope_and_horizon(bench):
    env = make_env(bench)
    env.reset(seed=3)
    r0 = env.rms_initial
    total, steps, done = 0.0, 0, False
    while not done:
        _, r, term, trunc, _ = env.step(np.zeros(10, dtype=np.float32))
        total += r
        steps += 1
        done = term or trunc
    assert steps == bench.env.horizon
    assert total == pytest.approx((r0 - env.obs.rms) / r0, abs=1e-6)


def test_action_scaling_and_limit_penalty(bench):
    env = make_env(bench)
    env.reset(seed=5)
    prev = env.obs.rms
    _, r, *_ = env.step(np.ones(10, dtype=np.float32))
    np.testing.assert_allclose(env.machine.theta, bench.machine.limits.dtheta_max_mrad)
    # theta_max == dtheta_max by default, so every corrector is at its limit.
    unpenalised = (prev - env.obs.rms) / env.rms_initial
    assert r == pytest.approx(unpenalised - bench.env.limit_penalty, abs=1e-9)


def test_encode_observation_matches_env(bench):
    m = make_machines(bench.machine, 1, 9)[0]
    obs = m.reset()
    vec = encode_observation(obs, m.model_info, bench.env)
    assert vec.dtype == np.float32
    np.testing.assert_allclose(vec[: m.n_bpms], obs.x / bench.env.x_scale_mm, rtol=1e-6)


def test_rl_controller_uses_same_action_scale_as_env(bench):
    from dataclasses import replace

    from steerbench.controllers.rl import RLController

    class FixedPolicy:
        def predict(self, obs, deterministic=True):
            return np.full(10, 0.5, dtype=np.float32), None

    hp = {"action_scale_mrad": 0.02}
    env = SteeringEnv(bench.machine, replace(bench.env, action_scale_mrad=0.02))
    env.reset(seed=0)
    env.step(np.full(10, 0.5, dtype=np.float32))
    m = make_machines(bench.machine, 1, 0)[0]
    ctrl = RLController(FixedPolicy(), m.model_info, bench, "x", hp)
    step = ctrl.act(m.reset())
    np.testing.assert_allclose(step, env.machine.theta)
    np.testing.assert_allclose(step, 0.01)
    # Larger requested scales are capped at the hardware rate limit.
    big = RLController(FixedPolicy(), m.model_info, bench, "x", {"action_scale_mrad": 5.0})
    assert big.scale == bench.machine.limits.dtheta_max_mrad

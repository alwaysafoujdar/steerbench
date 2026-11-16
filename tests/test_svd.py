from dataclasses import replace

import numpy as np
import pytest

from steerbench.controllers.svd import SVDController, pinv_reg
from steerbench.evaluation import run_episode
from steerbench.machine import exact_model_config, make_machines


def test_exact_model_reaches_oracle_in_one_step(cfg):
    machines = make_machines(exact_model_config(cfg, noise=False), 30, root_seed=11)
    checked = 0
    for m in machines:
        if not m.oracle().within_limits:
            continue
        trace = run_episode(m, SVDController(m.model_info, gain=1.0), max_interactions=1)
        assert trace.interactions[-1] == 1
        assert trace.rms_true[1] == pytest.approx(trace.r_star, abs=1e-9)
        assert trace.excess[1] == pytest.approx(0.0, abs=1e-8)
        checked += 1
    assert checked >= 25


def test_masked_exact_model_still_corrects(cfg):
    mcfg = exact_model_config(cfg, noise=False)
    mcfg = replace(mcfg, measurement=replace(mcfg.measurement, masked_fraction=0.25))
    m = make_machines(mcfg, 1, root_seed=3)[0]
    trace = run_episode(m, SVDController(m.model_info), max_interactions=3)
    assert trace.excess[-1] < 0.2


def test_gain_below_one_converges_geometrically(cfg):
    m = make_machines(exact_model_config(cfg, noise=False), 1, root_seed=5)[0]
    assert m.oracle().within_limits
    ctrl = SVDController(m.model_info, gain=0.5)
    obs = m.reset()
    x_star = m.true_trajectory(m.oracle().theta)
    deviation = [m.true_trajectory() - x_star]
    for _ in range(4):
        obs = m.apply(ctrl.act(obs))
        deviation.append(m.true_trajectory() - x_star)
    # With an exact model the deviation from the oracle trajectory shrinks by (1 - g) per step.
    for k in range(1, 5):
        np.testing.assert_allclose(deviation[k], 0.5**k * deviation[0], atol=1e-10)


def test_pinv_reg_reduces_to_pinv_and_regularises(rng):
    a = rng.normal(size=(8, 4))
    np.testing.assert_allclose(pinv_reg(a, 0.0), np.linalg.pinv(a), atol=1e-12)
    lam = 0.7
    expected = np.linalg.solve(a.T @ a + lam**2 * np.eye(4), a.T)
    np.testing.assert_allclose(pinv_reg(a, lam), expected, atol=1e-12)


def test_svd_respects_limits(cfg):
    m = make_machines(cfg, 1, root_seed=2)[0]
    obs = m.reset()
    ctrl = SVDController(m.model_info)
    step = ctrl.act(obs)
    assert np.max(np.abs(step)) <= cfg.limits.dtheta_max_mrad + 1e-12
    assert np.max(np.abs(obs.theta + step)) <= cfg.limits.theta_max_mrad + 1e-12

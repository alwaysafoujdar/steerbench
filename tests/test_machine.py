from dataclasses import replace

import numpy as np
import pytest

from steerbench.config import DriftConfig, ErrorConfig, MeasurementConfig
from steerbench.machine import Machine, MachineErrors, exact_model_config, make_machines


def test_zero_errors_give_true_equal_model(cfg, rng):
    m = Machine(exact_model_config(cfg), rng)
    np.testing.assert_allclose(m.R_true, m.R_model, atol=1e-12)
    np.testing.assert_allclose(m.R_eff, m.R_model, atol=1e-12)


def test_errors_change_response(cfg, rng):
    m = Machine(cfg, rng)
    assert not np.allclose(m.R_true, m.R_model)


def test_noise_statistics(cfg, rng):
    sigma = 0.05
    mcfg = replace(exact_model_config(cfg), measurement=MeasurementConfig(bpm_noise_mm=sigma))
    m = Machine(mcfg, rng)
    m.reset()
    noise = np.array([m.measure().x - m.true_trajectory() for _ in range(4000)])
    assert noise.mean() == pytest.approx(0.0, abs=4 * sigma / np.sqrt(noise.size))
    assert noise.std() == pytest.approx(sigma, rel=0.02)
    # Independent across BPMs.
    corr = np.corrcoef(noise.T)
    assert np.abs(corr[~np.eye(len(corr), dtype=bool)]).max() < 0.1


def test_bpm_gain_scales_readings(cfg, rng):
    mcfg = replace(cfg, measurement=MeasurementConfig(bpm_noise_mm=0.0))
    m = Machine(mcfg, rng)
    obs = m.reset()
    np.testing.assert_allclose(obs.x, m.errors.bpm_gain * m.true_trajectory())


@pytest.mark.parametrize("fraction", [0.0, 0.1, 0.25, 0.4])
def test_masking(cfg, rng, fraction):
    mcfg = replace(cfg, measurement=MeasurementConfig(masked_fraction=fraction))
    m = Machine(mcfg, rng)
    masks = []
    for _ in range(50):
        obs = m.reset()
        assert (~obs.mask).sum() == round(fraction * m.n_bpms)
        assert np.all(obs.x[~obs.mask] == 0.0)
        assert np.all(obs.x[obs.mask] != 0.0)
        masks.append(obs.mask)
    if fraction > 0:  # the masked set is redrawn each episode
        assert len({mk.tobytes() for mk in masks}) > 1


def test_drift_rate(cfg, rng):
    rate = DriftConfig(
        focal_error_sigma=0.002,
        bpm_gain_sigma=0.001,
        corrector_gain_sigma=0.003,
        quad_misalignment_mm=0.005,
    )
    m = Machine(replace(cfg, drift=rate), rng)
    history = [m.errors]
    for _ in range(400):
        m.drift_step()
        history.append(m.errors)
    for field, expected in [
        ("focal", rate.focal_error_sigma),
        ("bpm_gain", rate.bpm_gain_sigma),
        ("corrector_gain", rate.corrector_gain_sigma),
        ("misalignment_mm", rate.quad_misalignment_mm),
    ]:
        steps = np.diff(np.array([getattr(e, field) for e in history]), axis=0)
        assert steps.std() == pytest.approx(expected, rel=0.05)
        assert steps.mean() == pytest.approx(0.0, abs=4 * expected / np.sqrt(steps.size))
    # The optics follow the drifted errors.
    fresh = Machine(cfg, np.random.default_rng(0), errors=m.errors)
    np.testing.assert_allclose(m.R_true, fresh.R_true)
    np.testing.assert_allclose(m.x0, fresh.x0)


def test_no_drift_by_default(cfg, rng):
    m = Machine(cfg, rng)
    before = m.errors
    m.drift_step()
    for field in ("focal", "bpm_gain", "corrector_gain", "misalignment_mm", "injection"):
        np.testing.assert_array_equal(getattr(before, field), getattr(m.errors, field))


def test_limits_are_enforced(cfg, rng):
    m = Machine(cfg, rng)
    m.reset()
    lim = cfg.limits
    m.apply(np.full(m.n_correctors, 10.0))
    assert np.allclose(m.theta, lim.dtheta_max_mrad)
    m.apply(np.full(m.n_correctors, 10.0))
    assert np.allclose(m.theta, lim.theta_max_mrad)
    assert m.n_interactions == 2


def test_oracle_is_optimal(cfg, rng):
    m = Machine(cfg, rng)
    o = m.oracle()
    assert np.all(np.abs(o.theta) <= cfg.limits.theta_max_mrad + 1e-12)
    for _ in range(200):
        trial = np.clip(o.theta + rng.normal(0, 0.01, o.theta.size), -0.2, 0.2)
        assert m.true_rms(trial) >= o.rms - 1e-12


def test_oracle_respects_tight_limits(cfg, rng):
    tight = replace(cfg, limits=replace(cfg.limits, theta_max_mrad=0.01))
    m = Machine(tight, rng)
    o = m.oracle()
    assert not o.within_limits
    assert np.all(np.abs(o.theta) <= 0.01 + 1e-12)


def test_machines_reproducible_from_seed(cfg):
    a = make_machines(cfg, 3, root_seed=7)
    b = make_machines(cfg, 3, root_seed=7)
    for ma, mb in zip(a, b, strict=True):
        np.testing.assert_array_equal(ma.x0, mb.x0)
        np.testing.assert_array_equal(ma.reset().x, mb.reset().x)
    c = make_machines(cfg, 1, root_seed=8)[0]
    assert not np.allclose(a[0].x0, c.x0)


def test_error_sampling_statistics(rng):
    cfg = ErrorConfig(focal_error_sigma=0.05, bpm_gain_sigma=0.02, corrector_gain_sigma=0.03)
    errs = [MachineErrors.sample(cfg, 20, 10, rng) for _ in range(500)]
    assert np.std([e.focal for e in errs]) == pytest.approx(0.05, rel=0.05)
    assert np.mean([e.bpm_gain for e in errs]) == pytest.approx(1.0, abs=0.003)
    assert np.std([e.corrector_gain for e in errs]) == pytest.approx(0.03, rel=0.05)

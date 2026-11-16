import numpy as np
import pandas as pd
import pytest

from steerbench.controllers.base import BaseController, ZeroController
from steerbench.controllers.svd import SVDController
from steerbench.evaluation import bootstrap_ci, episode_table, evaluate, run_episode, summarise
from steerbench.machine import make_machines


class CountingController(BaseController):
    """Records how often it is asked to act and observe."""

    name = "counting"

    def __init__(self, n):
        self.n, self.acts, self.observes, self.resets = n, 0, 0, 0

    def reset(self, obs):
        self.resets += 1

    def act(self, obs):
        self.acts += 1
        return np.zeros(self.n)

    def observe(self, obs, action, next_obs):
        self.observes += 1


def test_interaction_counting(cfg):
    m = make_machines(cfg, 1, root_seed=1)[0]
    ctrl = CountingController(m.n_correctors)
    trace = run_episode(m, ctrl, max_interactions=7)
    assert ctrl.resets == 1 and ctrl.acts == 7 and ctrl.observes == 7
    np.testing.assert_array_equal(trace.interactions, np.arange(8))
    assert m.n_interactions == 7
    # The per-episode counter restarts; the lifetime counter does not.
    run_episode(m, ctrl, max_interactions=3)
    assert m.n_interactions == 3 and m.total_interactions == 10


def test_excess_starts_at_one_and_zero_controller_stays(cfg):
    m = make_machines(cfg, 1, root_seed=1)[0]
    trace = run_episode(m, ZeroController(m.model_info), max_interactions=4)
    np.testing.assert_allclose(trace.excess, 1.0)
    assert trace.interactions_to_success(0.05) == float("inf")


def test_interactions_to_success(cfg):
    m = make_machines(cfg, 1, root_seed=1)[0]
    trace = run_episode(m, SVDController(m.model_info, gain=0.5), max_interactions=10)
    its = trace.interactions_to_success(0.05)
    assert np.isfinite(its)
    k = int(its)
    assert trace.excess[k] < 0.05 and np.all(trace.excess[:k] >= 0.05)


def test_episode_table_and_summary(cfg):
    machines = make_machines(cfg, 6, root_seed=4)
    df = pd.concat(
        [
            evaluate(machines, lambda m: SVDController(m.model_info, gain=0.5), 8, method="svd"),
            evaluate(machines, lambda m: ZeroController(m.model_info), 8, method="none"),
        ]
    )
    ep = episode_table(df, 0.05, ["method"])
    assert len(ep) == 12
    summ = summarise(ep, ["method"], n_boot=200, rng=np.random.default_rng(0)).set_index("method")
    assert summ.loc["none", "success_rate"] == 0.0
    assert summ.loc["none", "its_median"] == float("inf")
    assert summ.loc["svd", "success_rate"] == 1.0
    assert summ.loc["svd", "success_ci_low"] <= 1.0


def test_bootstrap_ci_brackets_statistic(rng):
    v = rng.normal(3.0, 1.0, 400)
    lo, hi = bootstrap_ci(v, np.mean, 500, rng)
    assert lo < v.mean() < hi
    assert hi - lo == pytest.approx(2 * 1.96 / np.sqrt(400), rel=0.25)


def test_response_measurement_cost_is_counted(cfg):
    from dataclasses import replace

    from steerbench.controllers.svd import ResponseMeasurementSVD
    from steerbench.machine import exact_model_config

    # Real model errors, no noise: the measured matrix must equal the true one.
    mcfg = replace(cfg, measurement=replace(cfg.measurement, bpm_noise_mm=0.0))
    m = make_machines(mcfg, 1, root_seed=21)[0]
    ctrl = ResponseMeasurementSVD(m.model_info, kick=0.05)
    trace = run_episode(m, ctrl, max_interactions=m.n_correctors + 3)
    n = m.n_correctors
    # Every kick is a counted interaction; correction starts at interaction n + 1.
    np.testing.assert_array_equal(trace.interactions, np.arange(n + 4))
    assert m.n_interactions == n + 3
    assert ctrl.measured is not None
    expected = (m.errors.bpm_gain[:, None] * m.R_eff) * m.model_info.structure
    np.testing.assert_allclose(ctrl.measured, expected, atol=1e-9)
    assert np.all(trace.excess[1 : n + 1] > 0.05)  # nothing corrected while measuring
    assert trace.interactions_to_success(0.05) >= n + 1
    # With the exact model the method still pays the measurement cost.
    m2 = make_machines(exact_model_config(mcfg), 1, root_seed=21)[0]
    t2 = run_episode(m2, ResponseMeasurementSVD(m2.model_info), max_interactions=n + 1)
    assert t2.interactions_to_success(0.05) == n + 1


def test_cluster_bootstrap_is_wider_for_correlated_seeds(rng):
    machine_effect = rng.normal(0, 1, 20)
    values = np.repeat(machine_effect, 5) + rng.normal(0, 0.05, 100)
    clusters = np.repeat(np.arange(20), 5)
    lo_i, hi_i = bootstrap_ci(values, np.mean, 400, np.random.default_rng(1))
    lo_c, hi_c = bootstrap_ci(values, np.mean, 400, np.random.default_rng(1), clusters=clusters)
    assert (hi_c - lo_c) > 1.5 * (hi_i - lo_i)


def test_tuning_helpers(rng):
    from steerbench.evaluation import propose_trials, sample_params, tuning_score

    space = {"a": ("uniform", 0, 1), "b": ("log", 1e-3, 1e-1), "c": ("choice", [1, 2])}
    trials = propose_trials({"a": 0.5, "b": 0.01, "c": 1}, space, 5, rng)
    assert len(trials) == 5 and trials[0] == {"a": 0.5, "b": 0.01, "c": 1}
    for t in trials[1:]:
        assert 0 <= t["a"] <= 1 and 1e-3 <= t["b"] <= 1e-1 and t["c"] in (1, 2)
    with pytest.raises(ValueError):
        sample_params({"x": ("bogus", 0, 1)}, rng)
    df = pd.DataFrame({"interaction": [0, 1, 2, 0, 1, 2], "excess": [1, 2.0, 0.0, 1, 0.5, 0.5]})
    assert tuning_score(df) == pytest.approx((1 + 0 + 0.5 + 0.5) / 4)

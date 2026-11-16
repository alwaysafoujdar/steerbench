import numpy as np
import pytest
from scipy.stats import norm

from steerbench.controllers.bayesopt import BayesOptController, expected_improvement
from steerbench.evaluation import run_episode
from steerbench.machine import make_machines


def test_expected_improvement_closed_form():
    mu, sd, best = np.array([0.5, 1.0, 2.0]), np.array([0.2, 0.5, 1e-9]), 1.0
    ei = expected_improvement(mu, sd, best)
    u = (best - mu) / sd
    expected = (best - mu) * norm.cdf(u) + sd * norm.pdf(u)
    np.testing.assert_allclose(ei, np.maximum(expected, 0), atol=1e-12)
    assert expected_improvement(np.array([0.0]), np.array([0.0]), 1.0)[0] == 0.0
    # Larger uncertainty never lowers EI at equal mean.
    assert expected_improvement(np.array([1.2]), np.array([0.5]), 1.0) > expected_improvement(
        np.array([1.2]), np.array([0.1]), 1.0
    )


def test_modes_basis_is_orthonormal(cfg, rng):
    m = make_machines(cfg, 1, 3)[0]
    bo = BayesOptController(m.model_info, rng, n_modes=4)
    np.testing.assert_allclose(bo.basis.T @ bo.basis, np.eye(4), atol=1e-12)
    assert bo.name == "BO-SVDmodes"
    assert BayesOptController(m.model_info, rng).dim == m.n_correctors


@pytest.mark.parametrize("n_modes", [None, 4])
def test_bo_respects_limits_and_improves(cfg, n_modes):
    machines = make_machines(cfg, 4, 31)
    best = []
    for i, m in enumerate(machines):
        bo = BayesOptController(m.model_info, np.random.default_rng(i), n_modes=n_modes)
        trace = run_episode(m, bo, 25)
        assert np.all(np.abs(m.theta) <= cfg.limits.theta_max_mrad + 1e-12)
        assert len(bo.y) == 26  # initial reading plus one evaluation per interaction
        best.append(trace.excess.min())
    assert np.median(best) < 0.9

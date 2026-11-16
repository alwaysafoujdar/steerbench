import numpy as np
import pytest

from steerbench.config import BenchmarkConfig
from steerbench.controllers.offline import (
    BehaviourController,
    OfflineController,
    OfflineJob,
    collect_dataset,
    load_offline,
    train_offline,
)
from steerbench.controllers.svd import SVDController
from steerbench.evaluation import run_episode
from steerbench.machine import make_machines


@pytest.fixture
def bench() -> BenchmarkConfig:
    return BenchmarkConfig()


def test_dataset_layout(bench):
    data = collect_dataset(bench, 700, noise=0.02, seed=1)
    assert len(data) == 700
    assert data.observations.shape == (700, 50) and data.actions.shape == (700, 10)
    assert np.all(np.abs(data.actions) <= 1.0)
    # Episodes end every `horizon` steps (no early terminations at this threshold).
    ends = np.flatnonzero(data.timeouts + data.terminals)
    assert ends[-1] == 699
    assert np.all(np.diff(ends[:-1]) == bench.env.horizon) or len(ends) == 1
    again = collect_dataset(bench, 700, noise=0.02, seed=1)
    np.testing.assert_array_equal(data.observations, again.observations)


def test_behaviour_is_svd_plus_noise(bench, rng):
    m = make_machines(bench.machine, 1, 2)[0]
    obs = m.reset()
    svd = SVDController(m.model_info, gain=0.5).act(obs)
    noiseless = BehaviourController(m.model_info, rng, gain=0.5, noise=0.0).act(obs)
    np.testing.assert_allclose(noiseless, svd)
    steps = np.array(
        [BehaviourController(m.model_info, rng, 0.5, 0.01).act(obs) for _ in range(500)]
    )
    assert np.std(steps - svd, axis=0).mean() == pytest.approx(0.01, rel=0.15)


def test_bc_trains_and_acts(bench, tmp_path):
    bench.offline.n_steps = 300
    job = OfflineJob(bench, "BC", {"learning_rate": 1e-3}, 1500, 0.02, 0, str(tmp_path / "bc.d3"))
    info = train_offline(job)
    assert info["gradient_steps"] == 300
    agent = load_offline(info["path"])
    m = make_machines(bench.machine, 1, 4)[0]
    trace = run_episode(m, OfflineController(agent, m.model_info, bench, "BC"), 5)
    assert trace.interactions[-1] == 5
    assert np.all(np.abs(m.theta) <= bench.machine.limits.theta_max_mrad + 1e-12)


def test_terminal_and_timeout_never_coincide(bench):
    bench.env.rms_threshold_mm = 10.0  # every step terminates, including the last one
    data = collect_dataset(bench, 30, noise=0.0, seed=2)
    assert np.all(data.terminals == 1.0)
    assert not np.any((data.terminals > 0) & (data.timeouts > 0))

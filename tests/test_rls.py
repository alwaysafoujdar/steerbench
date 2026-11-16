from dataclasses import replace

import numpy as np

from steerbench.controllers.rls import RLSController
from steerbench.evaluation import run_episode
from steerbench.machine import make_machines


def test_rls_learns_true_response(cfg, rng):
    mcfg = replace(cfg, errors=replace(cfg.errors, focal_error_sigma=0.1))
    mcfg = replace(mcfg, measurement=replace(mcfg.measurement, bpm_noise_mm=0.001))
    m = make_machines(mcfg, 1, root_seed=4)[0]
    ctrl = RLSController(m.model_info, rng, forgetting=1.0, prior_std=5.0, dither=0.01)
    target = (m.errors.bpm_gain[:, None] * m.R_eff) * m.model_info.structure
    err0 = np.linalg.norm(ctrl.R_hat - target)
    obs = m.reset()
    ctrl.reset(obs)
    for _ in range(60):  # random excitation, staying inside the limits
        a = rng.normal(0, 0.03, m.n_correctors) - 0.5 * obs.theta
        nxt = m.apply(a)
        ctrl.observe(obs, a, nxt)
        obs = nxt
    assert np.linalg.norm(ctrl.R_hat - target) < 0.05 * err0
    # Structural zeros (BPM upstream of corrector) are never touched.
    assert np.all(ctrl.R_hat[~m.model_info.structure] == 0)


def test_rls_estimate_persists_across_episodes(cfg, rng):
    m = make_machines(cfg, 1, root_seed=6)[0]
    ctrl = RLSController(m.model_info, rng)
    run_episode(m, ctrl, 10)
    before = ctrl.R_hat.copy()
    assert ctrl.n_updates > 0
    ctrl.reset(m.reset())
    np.testing.assert_array_equal(ctrl.R_hat, before)


def test_rls_skips_masked_rows(cfg, rng):
    mcfg = replace(cfg, measurement=replace(cfg.measurement, masked_fraction=0.4))
    m = make_machines(mcfg, 1, root_seed=8)[0]
    ctrl = RLSController(m.model_info, rng, dither=0.01)
    init = ctrl.R_hat.copy()
    run_episode(m, ctrl, 5)  # one episode, one mask
    masked = ~m.mask
    assert masked.sum() == 8
    np.testing.assert_array_equal(ctrl.R_hat[masked], init[masked])
    assert not np.array_equal(ctrl.R_hat[~masked], init[~masked])


def test_fine_tuning_uses_no_extra_interactions(cfg):
    from stable_baselines3 import SAC

    from steerbench.config import BenchmarkConfig
    from steerbench.controllers.rl import FineTunedRLController
    from steerbench.env import SteeringEnv

    bench = BenchmarkConfig()
    hp = {"action_scale_mrad": 0.02}
    env = SteeringEnv(bench.machine, bench.env, bench.randomisation)
    agent = SAC("MlpPolicy", env, batch_size=16, policy_kwargs={"net_arch": [16]}, seed=0)
    m = make_machines(bench.machine, 1, 5)[0]
    ctrl = FineTunedRLController(agent, m.model_info, bench, "ft", hp, gradient_steps=3)
    before = [p.detach().clone() for p in agent.policy.actor.parameters()]
    for _ in range(3):
        run_episode(m, ctrl, 8)
    assert m.total_interactions == 24  # learning happens between episodes, not on the machine
    assert agent.replay_buffer.size() == 24
    after = list(agent.policy.actor.parameters())
    assert any(
        not np.array_equal(a.detach().numpy(), b.detach().numpy())
        for a, b in zip(before, after, strict=True)
    )


def test_covariance_stays_bounded_without_excitation(cfg, rng):
    from dataclasses import replace

    mcfg = replace(
        cfg, drift=replace(cfg.drift, focal_error_sigma=0.007, quad_misalignment_mm=0.005)
    )
    m = make_machines(mcfg, 1, root_seed=1000)[0]
    ctrl = RLSController(m.model_info, rng, forgetting=0.95, dither=0.0003)
    for _ in range(150):
        run_episode(m, ctrl, 15)
        m.drift_step()
    for P, cap in zip(ctrl.P, ctrl.trace_cap, strict=True):
        assert np.trace(P) <= cap * (1 + 1e-9)
    assert np.all(np.isfinite(ctrl.R_hat))

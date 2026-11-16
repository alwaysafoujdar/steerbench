import numpy as np
import pytest

from steerbench.config import LatticeConfig
from steerbench.optics import (
    build_fodo,
    design_focal_length,
    periodic_twiss,
    propagate_twiss,
    response_matrix,
    track,
)


@pytest.mark.parametrize("mu_deg", [45.0, 60.0, 90.0, 120.0])
@pytest.mark.parametrize("length", [2.0, 5.0])
def test_phase_advance_matches_thin_lens_formula(mu_deg, length):
    cfg = LatticeConfig(n_cells=4, drift_length=length, phase_advance_deg=mu_deg)
    lat = build_fodo(cfg)
    f = design_focal_length(length, mu_deg)
    _, _, mu = periodic_twiss(lat.cell_matrix(0))
    assert np.cos(mu) == pytest.approx(1 - length**2 / (2 * f**2), abs=1e-12)
    assert np.rad2deg(mu) == pytest.approx(mu_deg, abs=1e-9)
    # The propagated phase per cell agrees with the one-cell matrix.
    tw = propagate_twiss(lat)
    assert np.diff(tw.mu[::2]) == pytest.approx(np.full(cfg.n_cells - 1, mu), abs=1e-12)


def test_beta_is_periodic_in_matched_solution():
    lat = build_fodo(LatticeConfig())
    tw = propagate_twiss(lat)
    assert tw.beta[::2] == pytest.approx(np.full(lat.n_cells, tw.beta[0]), rel=1e-12)
    assert tw.beta[1::2] == pytest.approx(np.full(lat.n_cells, tw.beta[1]), rel=1e-12)
    assert tw.alpha[::2] == pytest.approx(np.full(lat.n_cells, tw.alpha[0]), abs=1e-12)


def test_thin_lens_fodo_beta_extrema():
    # Textbook thin-lens FODO: beta_max/min = 2L (1 +/- sin(mu/2)) / sin(mu).
    cfg = LatticeConfig()
    tw = propagate_twiss(build_fodo(cfg))
    mu = np.deg2rad(cfg.phase_advance_deg)
    L = cfg.drift_length
    assert tw.beta[0] == pytest.approx(2 * L * (1 + np.sin(mu / 2)) / np.sin(mu))
    assert tw.beta[1] == pytest.approx(2 * L * (1 - np.sin(mu / 2)) / np.sin(mu))


def test_response_zero_upstream_of_corrector():
    lat = build_fodo(LatticeConfig())
    R = response_matrix(lat)
    for j, cq in enumerate(lat.corrector_quads):
        assert np.all(R[lat.bpm_quads <= cq, j] == 0.0)
        assert np.any(R[lat.bpm_quads > cq, j] != 0.0)


def test_response_matches_tracking_with_errors(rng):
    cfg = LatticeConfig()
    lat = build_fodo(cfg, focal_errors=rng.normal(0, 0.05, 2 * cfg.n_cells))
    beta0, alpha0, _ = periodic_twiss(build_fodo(cfg).cell_matrix(0))
    R = response_matrix(lat, propagate_twiss(lat, beta0, alpha0))
    for j, cq in enumerate(lat.corrector_quads):
        kicks = np.zeros(lat.n_quads)
        kicks[cq] = 1.0
        assert track(lat, 0.0, 0.0, kicks) == pytest.approx(R[:, j], abs=1e-10)

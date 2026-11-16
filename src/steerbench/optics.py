"""Thin-lens FODO lattice, Twiss propagation, trajectory tracking and response matrix.

Conventions
-----------
* The line is a sequence of ``2 * n_cells`` thin quadrupoles separated by drifts
  of length ``L``: QF, drift, QD, drift, QF, ... Quadrupole ``k`` sits at
  ``s = k * L``.
* A BPM sits at every quadrupole and a horizontal corrector at every focusing
  quadrupole (even quadrupole indices). BPMs, correctors and lenses at one
  location are co-located; a reading at a location is taken where ``x`` is
  continuous, so ordering within a location does not matter.
* Positions are in mm, angles and kicks in mrad, lengths in m, so that
  ``x [mm] = M12 [m] * theta [mrad]``.
* Twiss functions are recorded at the entrance of each thin quadrupole, i.e.
  just before its focusing kick. ``beta`` and ``mu`` are continuous across a
  thin lens; ``alpha`` is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from steerbench.config import LatticeConfig


def design_focal_length(drift_length: float, phase_advance_deg: float) -> float:
    """Focal length giving the requested phase advance per FODO cell.

    Parameters
    ----------
    drift_length : float
        Distance L between neighbouring quadrupoles, in m.
    phase_advance_deg : float
        Target phase advance per cell, in degrees (0 < mu < 180).

    Returns
    -------
    float
        Focal length f such that ``sin(mu / 2) = L / (2 f)``.
    """
    mu = np.deg2rad(phase_advance_deg)
    if not 0.0 < mu < np.pi:
        raise ValueError("phase advance must lie strictly between 0 and 180 degrees")
    return float(drift_length / (2.0 * np.sin(mu / 2.0)))


def drift_matrix(length: float) -> np.ndarray:
    """2x2 transfer matrix of a field-free drift."""
    return np.array([[1.0, length], [0.0, 1.0]])


def thin_quad_matrix(focal_length: float) -> np.ndarray:
    """2x2 transfer matrix of a thin quadrupole (positive f focuses)."""
    return np.array([[1.0, 0.0], [-1.0 / focal_length, 1.0]])


@dataclass(frozen=True)
class Lattice:
    """A thin-lens FODO transfer line.

    Attributes
    ----------
    drift_length : float
        Drift between quadrupoles, in m.
    focal_lengths : numpy.ndarray
        Signed focal length of each quadrupole, shape ``(n_quads,)``;
        positive for focusing (QF), negative for defocusing (QD).
    """

    drift_length: float
    focal_lengths: np.ndarray

    @property
    def n_quads(self) -> int:
        """Number of quadrupoles (and BPMs)."""
        return int(self.focal_lengths.size)

    @property
    def n_cells(self) -> int:
        """Number of FODO cells."""
        return self.n_quads // 2

    @property
    def s(self) -> np.ndarray:
        """Longitudinal position of each quadrupole, in m."""
        return np.arange(self.n_quads) * self.drift_length

    @property
    def bpm_quads(self) -> np.ndarray:
        """Quadrupole index of each BPM (one BPM per quadrupole)."""
        return np.arange(self.n_quads)

    @property
    def corrector_quads(self) -> np.ndarray:
        """Quadrupole index of each corrector (one per focusing quadrupole)."""
        return np.arange(0, self.n_quads, 2)

    def cell_matrix(self, cell: int = 0) -> np.ndarray:
        """One-cell transfer matrix from the entrance of QF to the next QF entrance."""
        k = 2 * cell
        m = thin_quad_matrix(self.focal_lengths[k])
        m = drift_matrix(self.drift_length) @ m
        m = thin_quad_matrix(self.focal_lengths[k + 1]) @ m
        return drift_matrix(self.drift_length) @ m


def build_fodo(cfg: LatticeConfig, focal_errors: np.ndarray | None = None) -> Lattice:
    """Build the design (or an erroneous) FODO lattice.

    Parameters
    ----------
    cfg : LatticeConfig
        Lattice geometry and target phase advance.
    focal_errors : numpy.ndarray, optional
        Relative focal-length error of each quadrupole, shape ``(2 * n_cells,)``.
        Quadrupole ``k`` gets ``f_k * (1 + focal_errors[k])``.

    Returns
    -------
    Lattice
    """
    f = design_focal_length(cfg.drift_length, cfg.phase_advance_deg)
    signs = np.tile([1.0, -1.0], cfg.n_cells)
    focal = f * signs
    if focal_errors is not None:
        focal = focal * (1.0 + np.asarray(focal_errors, dtype=float))
    return Lattice(drift_length=float(cfg.drift_length), focal_lengths=focal)


def periodic_twiss(matrix: np.ndarray) -> tuple[float, float, float]:
    """Matched (periodic) Twiss parameters of a one-period transfer matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        2x2 one-period transfer matrix.

    Returns
    -------
    beta, alpha, mu : float
        Periodic beta (m), alpha and phase advance per period (rad).

    Raises
    ------
    ValueError
        If the motion is not stable (``|trace| >= 2``).
    """
    cos_mu = 0.5 * np.trace(matrix)
    if abs(cos_mu) >= 1.0:
        raise ValueError("unstable period: |trace/2| >= 1")
    sin_mu = np.sign(matrix[0, 1]) * np.sqrt(1.0 - cos_mu**2)
    beta = matrix[0, 1] / sin_mu
    alpha = (matrix[0, 0] - matrix[1, 1]) / (2.0 * sin_mu)
    return float(beta), float(alpha), float(np.arctan2(sin_mu, cos_mu))


def transport_twiss(matrix: np.ndarray, beta: float, alpha: float) -> tuple[float, float, float]:
    """Propagate Twiss parameters through a transfer matrix.

    Returns
    -------
    beta, alpha, dmu : float
        Twiss parameters after the element and the phase advance across it.
    """
    (m11, m12), (m21, m22) = matrix
    gamma = (1.0 + alpha**2) / beta
    beta2 = m11**2 * beta - 2.0 * m11 * m12 * alpha + m12**2 * gamma
    alpha2 = -m11 * m21 * beta + (m11 * m22 + m12 * m21) * alpha - m12 * m22 * gamma
    dmu = np.arctan2(m12, m11 * beta - m12 * alpha)
    return float(beta2), float(alpha2), float(dmu)


@dataclass(frozen=True)
class Twiss:
    """Twiss functions at the entrance of every quadrupole.

    Attributes
    ----------
    s, beta, alpha, mu : numpy.ndarray
        Position (m), beta (m), alpha and cumulative phase (rad), each of shape
        ``(n_quads,)``.
    """

    s: np.ndarray
    beta: np.ndarray
    alpha: np.ndarray
    mu: np.ndarray


def propagate_twiss(
    lattice: Lattice, beta0: float | None = None, alpha0: float | None = None
) -> Twiss:
    """Propagate Twiss functions along the line.

    Parameters
    ----------
    lattice : Lattice
        Lattice to propagate through.
    beta0, alpha0 : float, optional
        Initial Twiss at the entrance of the first quadrupole. By default the
        periodic solution of the lattice's first cell is used; pass the design
        values to propagate an injected, design-matched beam through an
        erroneous line.

    Returns
    -------
    Twiss
    """
    if beta0 is None or alpha0 is None:
        beta0, alpha0, _ = periodic_twiss(lattice.cell_matrix(0))
    n = lattice.n_quads
    beta = np.empty(n)
    alpha = np.empty(n)
    mu = np.empty(n)
    b, a, phase = beta0, alpha0, 0.0
    drift = drift_matrix(lattice.drift_length)
    for k in range(n):
        beta[k], alpha[k], mu[k] = b, a, phase
        b, a, d1 = transport_twiss(thin_quad_matrix(lattice.focal_lengths[k]), b, a)
        b, a, d2 = transport_twiss(drift, b, a)
        phase += d1 + d2
    return Twiss(s=lattice.s, beta=beta, alpha=alpha, mu=mu)


def response_matrix(lattice: Lattice, twiss: Twiss | None = None) -> np.ndarray:
    """Orbit response matrix from correctors to BPMs.

    ``R[i, j] = sqrt(beta_i beta_j) sin(mu_i - mu_j)`` for BPM ``i`` strictly
    downstream of corrector ``j``, and zero otherwise.

    Parameters
    ----------
    lattice : Lattice
        Lattice defining BPM and corrector positions.
    twiss : Twiss, optional
        Twiss functions to use; by default the lattice's own periodic solution.

    Returns
    -------
    numpy.ndarray
        Response matrix of shape ``(n_bpms, n_correctors)`` in mm/mrad.
    """
    if twiss is None:
        twiss = propagate_twiss(lattice)
    bi = lattice.bpm_quads[:, None]
    cj = lattice.corrector_quads[None, :]
    amp = np.sqrt(twiss.beta[bi] * twiss.beta[cj])
    r = amp * np.sin(twiss.mu[bi] - twiss.mu[cj])
    return np.where(bi > cj, r, 0.0)


def track(
    lattice: Lattice,
    x_in: float,
    xp_in: float,
    kicks: np.ndarray | None = None,
) -> np.ndarray:
    """Track a single trajectory through the line.

    Parameters
    ----------
    lattice : Lattice
        Lattice to track through.
    x_in, xp_in : float
        Position (mm) and angle (mrad) at the entrance of the first quadrupole.
    kicks : numpy.ndarray, optional
        Additional dipole kick (mrad) applied at each quadrupole location,
        shape ``(n_quads,)``.

    Returns
    -------
    numpy.ndarray
        Position (mm) at every BPM, shape ``(n_quads,)``.
    """
    n = lattice.n_quads
    kicks = np.zeros(n) if kicks is None else np.asarray(kicks, dtype=float)
    out = np.empty(n)
    x, xp = float(x_in), float(xp_in)
    for k in range(n):
        out[k] = x
        xp = xp - x / lattice.focal_lengths[k] + kicks[k]
        x = x + lattice.drift_length * xp
    return out


def misalignment_kicks(lattice: Lattice, offsets_mm: np.ndarray) -> np.ndarray:
    """Dipole kicks (mrad) from horizontal quadrupole offsets (mm).

    A thin quadrupole displaced by ``d`` acts as ``x' -> x' - (x - d) / f``,
    i.e. the centred lens plus a kick ``d / f``.
    """
    return np.asarray(offsets_mm, dtype=float) / lattice.focal_lengths

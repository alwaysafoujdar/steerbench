"""The simulated 'true' machine: errors, noisy masked measurement, limits, drift, oracle.

A :class:`Machine` owns one random generator, from which it draws its static
errors at construction and all later noise, masks and drift. Two machines built
from equal seeds therefore behave identically when driven identically, which
gives every controller the same machines and noise (common random numbers).

Controllers never see a :class:`Machine`. They receive a :class:`ModelInfo`
(design response matrix and limits) and a stream of :class:`Observation` objects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import lsq_linear

from steerbench.config import DriftConfig, ErrorConfig, MachineConfig
from steerbench.optics import (
    Lattice,
    build_fodo,
    misalignment_kicks,
    periodic_twiss,
    propagate_twiss,
    response_matrix,
    track,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Observation:
    """What a controller sees after each interaction.

    Attributes
    ----------
    x : numpy.ndarray
        Measured BPM positions in mm, shape ``(n_bpms,)``; masked BPMs read 0.
    mask : numpy.ndarray
        Boolean availability of each BPM, shape ``(n_bpms,)``.
    theta : numpy.ndarray
        Current corrector settings in mrad, shape ``(n_correctors,)``.
    """

    x: np.ndarray
    mask: np.ndarray
    theta: np.ndarray

    @property
    def rms(self) -> float:
        """RMS of the available BPM readings, in mm."""
        return float(np.sqrt(np.mean(self.x[self.mask] ** 2))) if self.mask.any() else 0.0


@dataclass(frozen=True)
class ModelInfo:
    """Operator-side knowledge handed to model-based controllers.

    Attributes
    ----------
    R : numpy.ndarray
        Design (ideal) response matrix, shape ``(n_bpms, n_correctors)``.
    theta_max, dtheta_max : float
        Corrector limits in mrad.
    downstream : numpy.ndarray, optional
        Boolean ``(n_bpms, n_correctors)`` matrix, true where BPM ``i`` lies
        downstream of corrector ``j``. This is line geometry, known exactly to
        the operator, so methods that estimate R may use it as structure.
    bpm_noise : float
        Nominal BPM resolution in mm (a published instrument specification).
    """

    R: np.ndarray
    theta_max: float
    dtheta_max: float
    downstream: np.ndarray | None = None
    bpm_noise: float = 0.05

    @property
    def structure(self) -> np.ndarray:
        """Downstream mask, defaulting to 'all entries free'."""
        if self.downstream is None:
            return np.ones(self.R.shape, dtype=bool)
        return self.downstream

    @property
    def n_bpms(self) -> int:
        """Number of BPMs."""
        return int(self.R.shape[0])

    @property
    def n_correctors(self) -> int:
        """Number of correctors."""
        return int(self.R.shape[1])


@dataclass(frozen=True)
class MachineErrors:
    """Error state of one machine.

    Attributes
    ----------
    focal : numpy.ndarray
        Relative focal-length errors, shape ``(n_quads,)``.
    bpm_gain : numpy.ndarray
        Multiplicative BPM gains (1 is perfect), shape ``(n_bpms,)``.
    corrector_gain : numpy.ndarray
        Multiplicative corrector calibrations, shape ``(n_correctors,)``.
    misalignment_mm : numpy.ndarray
        Quadrupole offsets in mm, shape ``(n_quads,)``.
    injection : numpy.ndarray
        Injection position (mm) and angle (mrad), shape ``(2,)``.
    """

    focal: np.ndarray
    bpm_gain: np.ndarray
    corrector_gain: np.ndarray
    misalignment_mm: np.ndarray
    injection: np.ndarray

    @classmethod
    def sample(
        cls, cfg: ErrorConfig, n_quads: int, n_correctors: int, rng: np.random.Generator
    ) -> MachineErrors:
        """Draw independent Gaussian errors with the configured RMS values."""
        return cls(
            focal=rng.normal(0.0, cfg.focal_error_sigma, n_quads),
            bpm_gain=1.0 + rng.normal(0.0, cfg.bpm_gain_sigma, n_quads),
            corrector_gain=1.0 + rng.normal(0.0, cfg.corrector_gain_sigma, n_correctors),
            misalignment_mm=rng.normal(0.0, cfg.quad_misalignment_mm, n_quads),
            injection=np.array(
                [
                    rng.normal(0.0, cfg.injection_offset_mm),
                    rng.normal(0.0, cfg.injection_angle_mrad),
                ]
            ),
        )

    def drifted(self, cfg: DriftConfig, rng: np.random.Generator) -> MachineErrors:
        """Return a copy after one random-walk step with the configured rates."""
        return MachineErrors(
            focal=self.focal + rng.normal(0.0, cfg.focal_error_sigma, self.focal.size),
            bpm_gain=self.bpm_gain + rng.normal(0.0, cfg.bpm_gain_sigma, self.bpm_gain.size),
            corrector_gain=self.corrector_gain
            + rng.normal(0.0, cfg.corrector_gain_sigma, self.corrector_gain.size),
            misalignment_mm=self.misalignment_mm
            + rng.normal(0.0, cfg.quad_misalignment_mm, self.misalignment_mm.size),
            injection=self.injection
            + np.array(
                [
                    rng.normal(0.0, cfg.injection_offset_mm),
                    rng.normal(0.0, cfg.injection_angle_mrad),
                ]
            ),
        )


@dataclass(frozen=True)
class OracleResult:
    """Best achievable correction for the true machine.

    Attributes
    ----------
    theta : numpy.ndarray
        Optimal corrector settings within limits (mrad).
    rms : float
        Residual true RMS r* at ``theta`` (mm).
    within_limits : bool
        Whether the unconstrained least-squares solution already satisfied the
        absolute corrector limit.
    max_abs_unconstrained : float
        Largest absolute kick of the unconstrained solution (mrad).
    """

    theta: np.ndarray
    rms: float
    within_limits: bool
    max_abs_unconstrained: float


def rms(x: np.ndarray) -> float:
    """Root-mean-square of an array."""
    return float(np.sqrt(np.mean(np.square(x))))


def oracle_correction(R_eff: np.ndarray, x0: np.ndarray, theta_max: float) -> OracleResult:
    """Solve ``min ||x0 + R_eff theta||_2`` subject to ``|theta| <= theta_max``.

    Parameters
    ----------
    R_eff : numpy.ndarray
        True response including corrector calibration, ``(n_bpms, n_correctors)``.
    x0 : numpy.ndarray
        True uncorrected trajectory at the BPMs (mm).
    theta_max : float
        Absolute corrector limit (mrad).

    Returns
    -------
    OracleResult
    """
    theta_free, *_ = np.linalg.lstsq(R_eff, -x0, rcond=None)
    max_abs = float(np.max(np.abs(theta_free)))
    if max_abs <= theta_max:
        theta = theta_free
    else:
        theta = lsq_linear(R_eff, -x0, bounds=(-theta_max, theta_max)).x
    return OracleResult(
        theta=theta,
        rms=rms(x0 + R_eff @ theta),
        within_limits=max_abs <= theta_max,
        max_abs_unconstrained=max_abs,
    )


class Machine:
    """A simulated transfer line with errors, noise, masking and drift.

    Parameters
    ----------
    cfg : MachineConfig
        Machine configuration.
    rng : numpy.random.Generator
        Source of all randomness for this machine.
    errors : MachineErrors, optional
        Explicit error state. If omitted, errors are sampled from ``cfg.errors``.

    Notes
    -----
    Interaction accounting: every call to :meth:`apply` sets the correctors and
    acquires one BPM reading, and counts as one deployment interaction. Reading
    the machine at :meth:`reset` is free (it is the state the operator finds).
    """

    def __init__(
        self,
        cfg: MachineConfig,
        rng: np.random.Generator,
        errors: MachineErrors | None = None,
    ) -> None:
        self.cfg = cfg
        self.rng = rng
        self.design = build_fodo(cfg.lattice)
        self._design_twiss0 = periodic_twiss(self.design.cell_matrix(0))
        self.R_model = response_matrix(self.design)
        n_quads = self.design.n_quads
        n_corr = self.design.corrector_quads.size
        if errors is None:
            errors = MachineErrors.sample(cfg.errors, n_quads, n_corr, rng)
        self.theta = np.zeros(n_corr)
        self.mask = np.ones(n_quads, dtype=bool)
        self.n_interactions = 0
        self.total_interactions = 0
        self.n_episodes = 0
        self._set_errors(errors)

    # ------------------------------------------------------------------ state
    def _set_errors(self, errors: MachineErrors) -> None:
        """Install an error state and recompute the true optics and x0."""
        self.errors = errors
        self.true_lattice: Lattice = build_fodo(self.cfg.lattice, errors.focal)
        beta0, alpha0, _ = self._design_twiss0
        self.true_twiss = propagate_twiss(self.true_lattice, beta0, alpha0)
        self.R_true = response_matrix(self.true_lattice, self.true_twiss)
        self.R_eff = self.R_true * errors.corrector_gain[None, :]
        kicks = misalignment_kicks(self.true_lattice, errors.misalignment_mm)
        self.x0 = track(self.true_lattice, errors.injection[0], errors.injection[1], kicks)
        self._oracle: OracleResult | None = None

    @property
    def model_info(self) -> ModelInfo:
        """Design model and limits, as known to the operator."""
        lim = self.cfg.limits
        lat = self.design
        return ModelInfo(
            R=self.R_model.copy(),
            theta_max=lim.theta_max_mrad,
            dtheta_max=lim.dtheta_max_mrad,
            downstream=lat.bpm_quads[:, None] > lat.corrector_quads[None, :],
            bpm_noise=self.cfg.measurement.bpm_noise_mm,
        )

    @property
    def n_bpms(self) -> int:
        """Number of BPMs."""
        return int(self.R_model.shape[0])

    @property
    def n_correctors(self) -> int:
        """Number of correctors."""
        return int(self.R_model.shape[1])

    # ------------------------------------------------------------ episode API
    def reset(self, theta: np.ndarray | None = None) -> Observation:
        """Start an episode: set correctors, draw a new BPM mask, read the BPMs.

        Parameters
        ----------
        theta : numpy.ndarray, optional
            Initial corrector settings; zero by default.

        Returns
        -------
        Observation
        """
        lim = self.cfg.limits.theta_max_mrad
        self.theta = np.zeros(self.n_correctors) if theta is None else np.clip(theta, -lim, lim)
        self.mask = self._sample_mask()
        self.n_interactions = 0
        self.n_episodes += 1
        return self.measure()

    def apply(self, dtheta: np.ndarray) -> Observation:
        """Change the correctors (with hardware limits) and read the BPMs.

        Parameters
        ----------
        dtheta : numpy.ndarray
            Requested corrector change in mrad. Each component is clipped to
            ``dtheta_max`` and the resulting setting to ``theta_max``.

        Returns
        -------
        Observation
        """
        lim = self.cfg.limits
        step = np.clip(np.asarray(dtheta, dtype=float), -lim.dtheta_max_mrad, lim.dtheta_max_mrad)
        self.theta = np.clip(self.theta + step, -lim.theta_max_mrad, lim.theta_max_mrad)
        self.n_interactions += 1
        self.total_interactions += 1
        return self.measure()

    def measure(self) -> Observation:
        """Read the BPMs at the current settings without counting an interaction."""
        noise = self.rng.normal(0.0, self.cfg.measurement.bpm_noise_mm, self.n_bpms)
        x = self.errors.bpm_gain * self.true_trajectory() + noise
        x = np.where(self.mask, x, 0.0)
        return Observation(x=x, mask=self.mask.copy(), theta=self.theta.copy())

    def drift_step(self) -> None:
        """Advance the error state by one random-walk step (between episodes)."""
        self._set_errors(self.errors.drifted(self.cfg.drift, self.rng))

    def _sample_mask(self) -> np.ndarray:
        """Draw this episode's set of available BPMs."""
        n_off = int(round(self.cfg.measurement.masked_fraction * self.n_bpms))
        mask = np.ones(self.n_bpms, dtype=bool)
        if n_off > 0:
            mask[self.rng.choice(self.n_bpms, size=n_off, replace=False)] = False
        return mask

    # ------------------------------------------------- privileged evaluation
    def true_trajectory(self, theta: np.ndarray | None = None) -> np.ndarray:
        """True trajectory at the BPMs for given (default: current) settings."""
        theta = self.theta if theta is None else theta
        return self.x0 + self.R_eff @ theta

    def true_rms(self, theta: np.ndarray | None = None) -> float:
        """RMS of the true trajectory over all BPMs, in mm."""
        return rms(self.true_trajectory(theta))

    def oracle(self) -> OracleResult:
        """Best correction within limits for the current error state (cached)."""
        if self._oracle is None:
            self._oracle = oracle_correction(self.R_eff, self.x0, self.cfg.limits.theta_max_mrad)
        return self._oracle

    def normalised_excess(self, theta: np.ndarray | None = None) -> float:
        """Normalised excess RMS ``(RMS - r*) / (RMS(theta=0) - r*)``."""
        r_star = self.oracle().rms
        r0 = rms(self.x0)
        return (self.true_rms(theta) - r_star) / (r0 - r_star)


def make_machines(cfg: MachineConfig, n: int, root_seed: int) -> list[Machine]:
    """Build ``n`` independent machines from one root seed.

    Each machine gets its own generator spawned from
    ``numpy.random.SeedSequence(root_seed)``, so machine ``i`` is the same in
    every run and for every controller.
    """
    seqs = np.random.SeedSequence(root_seed).spawn(n)
    return [Machine(cfg, np.random.default_rng(s)) for s in seqs]


def exact_model_config(cfg: MachineConfig, noise: bool = False) -> MachineConfig:
    """Copy of ``cfg`` with zero model errors (and optionally zero BPM noise)."""
    errors = replace(
        cfg.errors, focal_error_sigma=0.0, bpm_gain_sigma=0.0, corrector_gain_sigma=0.0
    )
    meas = cfg.measurement if noise else replace(cfg.measurement, bpm_noise_mm=0.0)
    return replace(cfg, errors=errors, measurement=meas)

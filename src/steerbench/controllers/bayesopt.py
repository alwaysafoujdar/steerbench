"""Gaussian-process Bayesian optimisation of the measured trajectory RMS.

Each interaction evaluates the black-box objective ``f(theta) = RMS_meas`` at
one corrector setting. A scikit-learn GP (Matern 5/2 with automatic relevance
determination plus a white-noise term) models ``f``; the next setting
maximises expected improvement (implemented here) over random and local
candidates inside a search box.

Two parameterisations are offered:

* **full**: optimise all corrector settings directly (dimension
  ``n_correctors``);
* **SVD modes**: ``theta = theta_0 + V_k z`` with ``V_k`` the top-``k`` right
  singular vectors of the design response matrix, a physics-informed,
  lower-dimensional search space. With ``k < n_correctors`` the oracle may be
  out of reach, which is the price of the reduction.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from steerbench.controllers.base import BaseController
from steerbench.machine import ModelInfo, Observation


def expected_improvement(
    mu: np.ndarray, sigma: np.ndarray, best: float, xi: float = 0.0
) -> np.ndarray:
    """Expected improvement for minimisation.

    ``EI = (best - mu - xi) Phi(u) + sigma phi(u)`` with
    ``u = (best - mu - xi) / sigma``; zero where ``sigma == 0``.
    """
    sigma = np.asarray(sigma, dtype=float)
    imp = best - np.asarray(mu, dtype=float) - xi
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(sigma > 0, imp / sigma, 0.0)
    ei = imp * norm.cdf(u) + sigma * norm.pdf(u)
    return np.where(sigma > 0, np.maximum(ei, 0.0), 0.0)


class BayesOptController(BaseController):
    """GP-EI optimisation of the measured RMS over corrector settings.

    Parameters
    ----------
    model : ModelInfo
        Limits and (for the modes variant) the design response matrix.
    rng : numpy.random.Generator
        Source of candidate points.
    n_modes : int or None
        ``None`` optimises all correctors; an integer uses that many SVD modes.
    box_fraction : float
        Half-width of the search box as a fraction of ``theta_max`` (in mode
        coordinates, the box is scaled so that a unit-norm mode step equals
        ``theta_max``).
    xi : float
        Exploration margin of expected improvement, in mm.
    local_fraction : float
        Share of candidates drawn as Gaussian perturbations of the incumbent.
    n_candidates : int
        Candidates scored per acquisition.
    """

    def __init__(
        self,
        model: ModelInfo,
        rng: np.random.Generator,
        n_modes: int | None = None,
        box_fraction: float = 0.5,
        xi: float = 0.001,
        local_fraction: float = 0.5,
        n_candidates: int = 2000,
    ) -> None:
        self.model = model
        self.rng = rng
        self.box_fraction = box_fraction
        self.xi = xi
        self.local_fraction = local_fraction
        self.n_candidates = n_candidates
        n = model.n_correctors
        if n_modes is None:
            self.basis = np.eye(n)
            self.name = "BO"
        else:
            _, _, vt = np.linalg.svd(model.R, full_matrices=False)
            self.basis = vt[:n_modes].T
            self.name = "BO-SVDmodes"
        self.dim = self.basis.shape[1]

    def reset(self, obs: Observation) -> None:
        """Start a new optimisation around the initial settings."""
        self.theta0 = obs.theta.copy()
        self.half = self.box_fraction * self.model.theta_max
        self.Z: list[np.ndarray] = [np.zeros(self.dim)]
        self.y: list[float] = [obs.rms]
        self.pending: np.ndarray | None = None

    def _theta(self, z: np.ndarray) -> np.ndarray:
        """Corrector settings for search coordinates ``z``."""
        lim = self.model.theta_max
        return np.clip(self.theta0 + self.basis @ z, -lim, lim)

    def _gp(self) -> GaussianProcessRegressor:
        """GP fitted to the evaluations so far."""
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=np.full(self.dim, self.half), length_scale_bounds=(1e-3, 1e2), nu=2.5
        ) + WhiteKernel(1e-3, (1e-6, 1e0))
        gp = GaussianProcessRegressor(
            kernel,
            normalize_y=True,
            n_restarts_optimizer=0,
            random_state=int(self.rng.integers(2**31)),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(np.array(self.Z), np.array(self.y))
        return gp

    def _candidates(self) -> np.ndarray:
        """Random points in the box plus perturbations of the incumbent."""
        n_local = int(self.local_fraction * self.n_candidates)
        glob = self.rng.uniform(-self.half, self.half, (self.n_candidates - n_local, self.dim))
        best = self.Z[int(np.argmin(self.y))]
        loc = best + self.rng.normal(0.0, 0.1 * self.half, (n_local, self.dim))
        return np.clip(np.vstack([glob, loc]), -self.half, self.half)

    def act(self, obs: Observation) -> np.ndarray:
        """Move the correctors to the point of maximal expected improvement."""
        if len(self.y) < 2:  # one random probe before the first fit
            z = self.rng.uniform(-self.half, self.half, self.dim) * 0.5
        else:
            gp = self._gp()
            cand = self._candidates()
            mu, sd = gp.predict(cand, return_std=True)
            z = cand[int(np.argmax(expected_improvement(mu, sd, min(self.y), self.xi)))]
        self.pending = z
        return self._theta(z) - obs.theta

    def observe(self, obs: Observation, action: np.ndarray, next_obs: Observation) -> None:
        """Record the measured RMS at the setting actually reached."""
        reached = next_obs.theta - self.theta0
        z = self.basis.T @ reached  # project the applied setting back to search coordinates
        self.Z.append(z)
        self.y.append(next_obs.rms)

"""Adaptive SVD correction with a recursive-least-squares response estimate."""

from __future__ import annotations

import numpy as np

from steerbench.controllers.base import BaseController, limit_step
from steerbench.controllers.svd import pinv_reg
from steerbench.machine import ModelInfo, Observation


class RLSController(BaseController):
    """SVD correction whose response matrix is learned online by RLS.

    The estimate ``R_hat`` starts at the design matrix. After each interaction
    the pair ``(dtheta, dx)`` (actually applied change, measured change) updates
    every available BPM row ``i`` by exponentially weighted recursive least
    squares over the correctors upstream of that BPM::

        e   = dx_i - R_hat[i] @ dtheta
        k   = P_i dtheta / (lambda + dtheta^T P_i dtheta)
        R_hat[i] += k e
        P_i = (P_i - k dtheta^T P_i) / lambda

    A small Gaussian dither is added to each correction step to keep the
    regressor exciting. ``R_hat`` and ``P`` persist across :meth:`reset`, so the
    estimate carries over between episodes on the same machine.

    Parameters
    ----------
    model : ModelInfo
        Design model, limits and geometry.
    rng : numpy.random.Generator
        Source of dither noise.
    forgetting : float
        Forgetting factor lambda in (0, 1].
    prior_std : float
        Prior standard deviation of each response entry around the design value,
        in mm/mrad. With ``dx`` noise variance ``2 sigma_bpm^2`` this sets
        ``P_i(0) = prior_std^2 / (2 sigma_bpm^2) I``; larger values trust the
        design model less.
    dither : float
        Standard deviation of the dither, in mrad.
    gain, tikhonov : float
        SVD correction parameters.
    min_excitation : float
        Updates with ``||dtheta||`` below this (mrad) are skipped.
    max_trace_ratio : float
        Anti-wind-up bound: after each update ``trace(P_i)`` is scaled back to
        at most this multiple of its prior trace. Without it, exponential
        forgetting under weak excitation lets ``P`` grow as ``lambda^-k`` in
        unexcited directions until the estimate diverges (observed in E4 over
        thousands of updates). This is the standard bounded-covariance fix.
    """

    def __init__(
        self,
        model: ModelInfo,
        rng: np.random.Generator,
        forgetting: float = 0.98,
        prior_std: float = 1.0,
        dither: float = 0.001,
        gain: float = 1.0,
        tikhonov: float = 0.0,
        min_excitation: float = 1e-4,
        max_trace_ratio: float = 1.0,
    ) -> None:
        if not 0.0 < forgetting <= 1.0:
            raise ValueError("forgetting factor must lie in (0, 1]")
        self.model = model
        self.rng = rng
        self.forgetting = forgetting
        self.prior_std = prior_std
        self.dither = dither
        self.gain = gain
        self.tikhonov = tikhonov
        self.min_excitation = min_excitation
        self.max_trace_ratio = max_trace_ratio
        self.name = "RLS"
        self.R_hat = model.R.copy()
        self.cols = [np.flatnonzero(row) for row in model.structure]
        p0 = prior_std**2 / (2.0 * max(model.bpm_noise, 1e-6) ** 2)
        self.P = [p0 * np.eye(c.size) for c in self.cols]
        self.trace_cap = [max_trace_ratio * p0 * c.size for c in self.cols]
        self.R_hat = self.R_hat * model.structure
        self.n_updates = 0

    def act(self, obs: Observation) -> np.ndarray:
        """SVD step with the current estimate plus dither."""
        dtheta = -self.gain * pinv_reg(self.R_hat[obs.mask], self.tikhonov) @ obs.x[obs.mask]
        if self.dither > 0:
            dtheta = dtheta + self.rng.normal(0.0, self.dither, dtheta.size)
        return limit_step(dtheta, obs.theta, self.model.theta_max, self.model.dtheta_max)

    def observe(self, obs: Observation, action: np.ndarray, next_obs: Observation) -> None:
        """Recursive-least-squares update from the applied change."""
        phi_all = next_obs.theta - obs.theta
        if np.linalg.norm(phi_all) < self.min_excitation:
            return
        dx = next_obs.x - obs.x
        lam = self.forgetting
        for i in np.flatnonzero(obs.mask & next_obs.mask):
            cols = self.cols[i]
            if cols.size == 0:
                continue
            phi = phi_all[cols]
            P = self.P[i]
            p_phi = P @ phi
            k = p_phi / (lam + phi @ p_phi)
            err = dx[i] - self.R_hat[i, cols] @ phi
            self.R_hat[i, cols] += k * err
            P = (P - np.outer(k, p_phi)) / lam
            tr = np.trace(P)
            if tr > self.trace_cap[i]:
                P = P * (self.trace_cap[i] / tr)
            self.P[i] = P
        self.n_updates += 1

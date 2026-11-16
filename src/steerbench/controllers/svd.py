"""SVD trajectory correction with a fixed response model (the operators' standard)."""

from __future__ import annotations

import numpy as np

from steerbench.controllers.base import BaseController, limit_step
from steerbench.machine import ModelInfo, Observation


def pinv_reg(a: np.ndarray, tikhonov: float = 0.0, rcond: float = 1e-10) -> np.ndarray:
    """Tikhonov-regularised pseudo-inverse via the SVD.

    Each singular value ``s`` is inverted as ``s / (s**2 + tikhonov**2)``;
    singular values below ``rcond * s_max`` are discarded.

    Parameters
    ----------
    a : numpy.ndarray
        Matrix of shape ``(m, n)``.
    tikhonov : float
        Regularisation strength, in the units of ``a``.
    rcond : float
        Relative cut-off for small singular values.

    Returns
    -------
    numpy.ndarray
        Matrix of shape ``(n, m)``.
    """
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    keep = s > rcond * (s[0] if s.size else 0.0)
    s_inv = np.zeros_like(s)
    s_inv[keep] = s[keep] / (s[keep] ** 2 + tikhonov**2)
    return (vt.T * s_inv) @ u.T


class SVDController(BaseController):
    """Iterative SVD correction: ``dtheta = -g pinv_reg(R[mask]) x[mask]``.

    Parameters
    ----------
    model : ModelInfo
        Design response matrix and limits.
    gain : float
        Feedback gain ``g`` in (0, 1].
    tikhonov : float
        Tikhonov regularisation strength (mm/mrad).
    response : numpy.ndarray, optional
        Response matrix to use instead of ``model.R`` (e.g. a measured one).
    name : str
        Label used in results.
    """

    def __init__(
        self,
        model: ModelInfo,
        gain: float = 1.0,
        tikhonov: float = 0.0,
        response: np.ndarray | None = None,
        name: str = "SVD",
    ) -> None:
        if not 0.0 < gain <= 1.0:
            raise ValueError("gain must lie in (0, 1]")
        self.model = model
        self.gain = gain
        self.tikhonov = tikhonov
        self.response = model.R if response is None else response
        self.name = name
        self._cache: tuple[bytes, np.ndarray] | None = None

    def set_response(self, response: np.ndarray) -> None:
        """Replace the response matrix used for correction."""
        self.response = np.asarray(response, dtype=float)
        self._cache = None

    def _inverse(self, mask: np.ndarray) -> np.ndarray:
        """Regularised inverse of the masked response, cached per mask."""
        key = mask.tobytes()
        if self._cache is None or self._cache[0] != key:
            self._cache = (key, pinv_reg(self.response[mask], self.tikhonov))
        return self._cache[1]

    def act(self, obs: Observation) -> np.ndarray:
        """Return the regularised least-squares correction step."""
        dtheta = -self.gain * self._inverse(obs.mask) @ obs.x[obs.mask]
        return limit_step(dtheta, obs.theta, self.model.theta_max, self.model.dtheta_max)


class ResponseMeasurementSVD(BaseController):
    """Measure the response matrix by kicking each corrector, then correct with SVD.

    The first ``n_correctors`` interactions each kick one corrector by
    ``kick`` mrad (while returning the previous one to its reference value), so
    that column ``j`` is ``(x_after - x_reference) / dtheta_j``. These
    interactions are charged like any other. Afterwards the controller behaves
    like :class:`SVDController` with the measured matrix. Entries for BPMs
    upstream of a corrector are set to zero (known geometry); rows of
    unavailable BPMs keep the design values (they are never used while masked).

    Parameters
    ----------
    model : ModelInfo
        Design model and limits.
    kick : float
        Measurement kick amplitude in mrad.
    gain, tikhonov : float
        SVD correction parameters.
    """

    def __init__(
        self, model: ModelInfo, kick: float = 0.1, gain: float = 1.0, tikhonov: float = 0.0
    ) -> None:
        self.model = model
        self.kick = min(kick, model.dtheta_max / 2.0, model.theta_max)
        self.svd = SVDController(model, gain=gain, tikhonov=tikhonov)
        self.name = "RM-SVD"
        self.measured: np.ndarray | None = None

    def reset(self, obs: Observation) -> None:
        """Start a fresh response measurement from the current state."""
        self.ref_x = obs.x.copy()
        self.ref_theta = obs.theta.copy()
        self.column = 0
        self.R_hat = self.model.R.copy()
        self.measured = None

    @property
    def measuring(self) -> bool:
        """Whether the controller is still measuring columns."""
        return self.column < self.model.n_correctors

    def act(self, obs: Observation) -> np.ndarray:
        """Kick the next corrector, or correct once all columns are measured."""
        if not self.measuring:
            return self.svd.act(obs)
        j = self.column
        target = self.ref_theta.copy()
        sign = 1.0 if self.ref_theta[j] <= 0 else -1.0  # stay inside the limits
        target[j] += sign * self.kick
        return target - obs.theta

    def observe(self, obs: Observation, action: np.ndarray, next_obs: Observation) -> None:
        """Turn the reading after a kick into a response column."""
        if not self.measuring:
            return
        j = self.column
        dtheta = next_obs.theta[j] - self.ref_theta[j]
        avail = next_obs.mask & obs.mask
        col = (next_obs.x - self.ref_x) / dtheta
        self.R_hat[avail, j] = col[avail]
        self.R_hat[:, j] *= self.model.structure[:, j]
        self.column += 1
        if not self.measuring:
            self.measured = self.R_hat.copy()
            self.svd.set_response(self.R_hat)

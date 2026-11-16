"""Common controller interface and shared helpers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from steerbench.machine import ModelInfo, Observation


@runtime_checkable
class Controller(Protocol):
    """Interface every correction strategy implements.

    The episode runner calls :meth:`reset` once with the initial observation,
    then alternates :meth:`act` (returns a corrector change ``dtheta`` in mrad),
    one machine interaction, and :meth:`observe`.
    """

    name: str

    def reset(self, obs: Observation) -> None:
        """Prepare for a new episode on the current machine."""
        ...

    def act(self, obs: Observation) -> np.ndarray:
        """Return the requested corrector change in mrad."""
        ...

    def observe(self, obs: Observation, action: np.ndarray, next_obs: Observation) -> None:
        """Optional online update after an interaction."""
        ...


class BaseController:
    """Convenience base class with no-op ``reset`` and ``observe``."""

    name: str = "base"

    def reset(self, obs: Observation) -> None:
        """Do nothing by default."""

    def act(self, obs: Observation) -> np.ndarray:  # pragma: no cover - abstract
        """Return the requested corrector change in mrad."""
        raise NotImplementedError

    def observe(self, obs: Observation, action: np.ndarray, next_obs: Observation) -> None:
        """Do nothing by default."""


def limit_step(
    dtheta: np.ndarray, theta: np.ndarray, theta_max: float, dtheta_max: float
) -> np.ndarray:
    """Make a requested step feasible while keeping its direction where possible.

    The target setting is clipped to ``[-theta_max, theta_max]``; the resulting
    step is then scaled down uniformly if any component exceeds ``dtheta_max``.
    Uniform scaling (instead of the per-component clipping done by the hardware)
    preserves the direction of a model-based correction.

    Parameters
    ----------
    dtheta : numpy.ndarray
        Requested change (mrad).
    theta : numpy.ndarray
        Current settings (mrad).
    theta_max, dtheta_max : float
        Limits (mrad).

    Returns
    -------
    numpy.ndarray
        Feasible change (mrad).
    """
    target = np.clip(theta + dtheta, -theta_max, theta_max)
    step = target - theta
    peak = float(np.max(np.abs(step))) if step.size else 0.0
    if peak > dtheta_max:
        step = step * (dtheta_max / peak)
    return step


class ZeroController(BaseController):
    """Does nothing; a floor for comparisons and a test fixture."""

    name = "none"

    def __init__(self, model: ModelInfo) -> None:
        self.n = model.n_correctors

    def act(self, obs: Observation) -> np.ndarray:
        """Return a zero step."""
        return np.zeros(self.n)

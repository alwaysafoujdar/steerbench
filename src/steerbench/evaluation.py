"""Episode runner, interaction accounting and summary metrics.

Every interaction a controller makes with the machine goes through
:meth:`steerbench.machine.Machine.apply`, which increments the machine's
interaction counter. The runner never counts on the controller's behalf, so a
controller that spends interactions on measuring the response (or on
exploration) is charged for them automatically.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from steerbench.controllers.base import Controller
from steerbench.machine import Machine

logger = logging.getLogger(__name__)


@dataclass
class EpisodeTrace:
    """Record of one episode.

    Attributes
    ----------
    interactions : numpy.ndarray
        Deployment-interaction count after each step (starts at 0).
    excess : numpy.ndarray
        Normalised excess RMS of the true trajectory after each step.
    rms_true, rms_meas : numpy.ndarray
        True RMS over all BPMs and measured RMS over available BPMs (mm).
    r_star : float
        Oracle residual RMS (mm).
    rms_initial : float
        True RMS at the start of the episode (mm).
    """

    interactions: np.ndarray
    excess: np.ndarray
    rms_true: np.ndarray
    rms_meas: np.ndarray
    r_star: float
    rms_initial: float

    def interactions_to_success(self, threshold: float) -> float:
        """First interaction count at which the excess drops below ``threshold``.

        Returns ``numpy.inf`` if the episode never succeeds.
        """
        hit = np.flatnonzero(self.excess < threshold)
        return float(self.interactions[hit[0]]) if hit.size else float("inf")

    @property
    def final_excess(self) -> float:
        """Normalised excess RMS at the end of the episode."""
        return float(self.excess[-1])

    def to_frame(self, **labels: object) -> pd.DataFrame:
        """Long-format table of the trace with constant label columns."""
        df = pd.DataFrame(
            {
                "interaction": self.interactions,
                "excess": self.excess,
                "rms_true": self.rms_true,
                "rms_meas": self.rms_meas,
            }
        )
        for key, value in labels.items():
            df[key] = value
        return df


def run_episode(
    machine: Machine,
    controller: Controller,
    max_interactions: int,
    reset_controller: bool = True,
) -> EpisodeTrace:
    """Run one correction episode.

    Parameters
    ----------
    machine : Machine
        The machine to correct. It is reset at the start of the episode.
    controller : Controller
        Correction strategy.
    max_interactions : int
        Interaction budget; the episode stops once the machine has counted this
        many interactions.
    reset_controller : bool
        Call ``controller.reset`` with the first observation.

    Returns
    -------
    EpisodeTrace
    """
    obs = machine.reset()
    if reset_controller:
        controller.reset(obs)
    r_star = machine.oracle().rms
    r0 = machine.true_rms()
    denom = max(r0 - r_star, 1e-12)
    counts, rms_t, rms_m = [0], [r0], [obs.rms]
    while machine.n_interactions < max_interactions:
        action = np.asarray(controller.act(obs), dtype=float)
        next_obs = machine.apply(action)
        controller.observe(obs, action, next_obs)
        obs = next_obs
        counts.append(machine.n_interactions)
        rms_t.append(machine.true_rms())
        rms_m.append(obs.rms)
    rms_true = np.asarray(rms_t)
    return EpisodeTrace(
        interactions=np.asarray(counts),
        excess=(rms_true - r_star) / denom,
        rms_true=rms_true,
        rms_meas=np.asarray(rms_m),
        r_star=r_star,
        rms_initial=r0,
    )


def trajectory_snapshots(
    machine: Machine, controller: Controller, n_interactions: int
) -> tuple[np.ndarray, np.ndarray]:
    """True trajectory before and after an episode (for visualisation).

    Returns
    -------
    before, after : numpy.ndarray
        True positions at every BPM (mm) at the start and after
        ``n_interactions`` interactions.
    """
    obs = machine.reset()
    before = machine.true_trajectory().copy()
    controller.reset(obs)
    for _ in range(n_interactions):
        action = np.asarray(controller.act(obs), dtype=float)
        nxt = machine.apply(action)
        controller.observe(obs, action, nxt)
        obs = nxt
    return before, machine.true_trajectory().copy()


def evaluate(
    machines: Sequence[Machine],
    make_controller: Callable[[Machine], Controller],
    max_interactions: int,
    **labels: object,
) -> pd.DataFrame:
    """Run one episode per machine with a fresh controller for each.

    Parameters
    ----------
    machines : sequence of Machine
        Evaluation machines (each is reset by the runner).
    make_controller : callable
        Builds a controller from a machine. It should use only
        ``machine.model_info``; the machine is passed so that the controller
        can be sized correctly.
    max_interactions : int
        Interaction budget per episode.
    **labels
        Constant columns added to the output (e.g. ``method``, ``seed``).

    Returns
    -------
    pandas.DataFrame
        Long-format traces with a ``machine`` column.
    """
    frames = []
    for i, machine in enumerate(machines):
        trace = run_episode(machine, make_controller(machine), max_interactions)
        frames.append(trace.to_frame(machine=i, r_star=trace.r_star, **labels))
    return pd.concat(frames, ignore_index=True)


def episode_table(traces: pd.DataFrame, threshold: float, keys: Sequence[str]) -> pd.DataFrame:
    """Collapse long traces to one row per episode.

    Parameters
    ----------
    traces : pandas.DataFrame
        Output of :func:`evaluate` (possibly concatenated).
    threshold : float
        Success threshold on the normalised excess RMS.
    keys : sequence of str
        Columns identifying an episode besides ``machine`` (e.g. method, seed).

    Returns
    -------
    pandas.DataFrame
        Columns ``keys + ['machine', 'final_excess', 'success',
        'interactions_to_success']``.
    """
    group = [*keys, "machine"]

    def per_episode(df: pd.DataFrame) -> pd.Series:
        df = df.sort_values("interaction")
        hit = df.loc[df["excess"] < threshold, "interaction"]
        its = float(hit.iloc[0]) if len(hit) else float("inf")
        return pd.Series(
            {
                "final_excess": float(df["excess"].iloc[-1]),
                "success": bool(np.isfinite(its)),
                "interactions_to_success": its,
            }
        )

    rows = [
        dict(zip(group, k, strict=True)) | per_episode(g).to_dict()
        for k, g in traces.groupby(group, sort=True)
    ]
    return pd.DataFrame(rows)


def bootstrap_ci(
    values: np.ndarray,
    stat: Callable[[np.ndarray], float],
    n_boot: int,
    rng: np.random.Generator,
    level: float = 0.95,
    clusters: np.ndarray | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of a statistic.

    Parameters
    ----------
    values : numpy.ndarray
        One value per episode.
    stat : callable
        Statistic, e.g. ``numpy.median``.
    n_boot : int
        Number of resamples.
    rng : numpy.random.Generator
        Source of resampling randomness.
    level : float
        Confidence level.
    clusters : numpy.ndarray, optional
        Cluster label of each value (e.g. the machine). If given, whole clusters
        are resampled, so repeated seeds on one machine are not treated as
        independent evidence.

    Returns
    -------
    (low, high) : tuple of float
    """
    values = np.asarray(values, dtype=float)
    if clusters is None:
        idx = rng.integers(0, values.size, size=(n_boot, values.size))
        boots = np.array([stat(values[row]) for row in idx])
    else:
        labels = np.asarray(clusters)
        groups = [np.flatnonzero(labels == c) for c in np.unique(labels)]
        picks = rng.integers(0, len(groups), size=(n_boot, len(groups)))
        boots = np.array([stat(values[np.concatenate([groups[k] for k in row])]) for row in picks])
    alpha = (1.0 - level) / 2.0
    return _quantile(boots, alpha), _quantile(boots, 1.0 - alpha)


def _quantile(values: np.ndarray, q: float) -> float:
    """Empirical (inverted-CDF) quantile; well defined when values contain inf."""
    return float(np.quantile(values, q, method="inverted_cdf"))


def summarise(
    episodes: pd.DataFrame,
    keys: Sequence[str],
    n_boot: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Median, IQR and bootstrap CIs of per-episode metrics.

    Failed episodes enter ``interactions_to_success`` as infinity, so its median
    is infinite when fewer than half the episodes succeed. Confidence intervals
    resample machines (clusters), keeping all seeds of a machine together.

    Parameters
    ----------
    episodes : pandas.DataFrame
        Output of :func:`episode_table`.
    keys : sequence of str
        Grouping columns (e.g. ``['method']``).
    n_boot : int
        Bootstrap resamples.
    rng : numpy.random.Generator
        Resampling randomness.

    Returns
    -------
    pandas.DataFrame
        One row per group.
    """
    rows = []
    for k, g in episodes.groupby(list(keys), sort=False):
        k = k if isinstance(k, tuple) else (k,)
        fe = g["final_excess"].to_numpy()
        its = g["interactions_to_success"].to_numpy()
        succ = g["success"].to_numpy(dtype=float)
        row = dict(zip(keys, k, strict=True))
        row |= {
            "n_episodes": len(g),
            "success_rate": succ.mean(),
            "final_excess_median": np.median(fe),
            "final_excess_q25": np.quantile(fe, 0.25),
            "final_excess_q75": np.quantile(fe, 0.75),
            "its_median": _quantile(its, 0.5),
            "its_q25": _quantile(its, 0.25),
            "its_q75": _quantile(its, 0.75),
        }
        cl = g["machine"].to_numpy()
        lo, hi = bootstrap_ci(succ, np.mean, n_boot, rng, clusters=cl)
        row |= {"success_ci_low": lo, "success_ci_high": hi}
        lo, hi = bootstrap_ci(fe, np.median, n_boot, rng, clusters=cl)
        row |= {"final_excess_ci_low": lo, "final_excess_ci_high": hi}
        lo, hi = bootstrap_ci(its, lambda v: _quantile(v, 0.5), n_boot, rng, clusters=cl)
        row |= {"its_ci_low": lo, "its_ci_high": hi}
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- tuning
#: A search-space entry: ``("uniform", lo, hi)``, ``("log", lo, hi)`` or
#: ``("choice", [options])``.
SearchSpace = dict[str, tuple[Any, ...]]


def sample_params(space: SearchSpace, rng: np.random.Generator) -> dict[str, Any]:
    """Draw one configuration from a search space."""
    out: dict[str, Any] = {}
    for key, spec in space.items():
        kind = spec[0]
        if kind == "uniform":
            out[key] = float(rng.uniform(spec[1], spec[2]))
        elif kind == "log":
            out[key] = float(np.exp(rng.uniform(np.log(spec[1]), np.log(spec[2]))))
        elif kind == "choice":
            options = list(spec[1])
            out[key] = options[int(rng.integers(len(options)))]
        else:
            raise ValueError(f"unknown search-space kind {kind!r}")
    return out


def propose_trials(
    default: dict[str, Any], space: SearchSpace, n_trials: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    """Trial 0 is the default configuration; the rest are random draws."""
    return [dict(default)] + [default | sample_params(space, rng) for _ in range(n_trials - 1)]


def tuning_score(traces: pd.DataFrame) -> float:
    """Tuning objective (lower is better): mean area under ``min(e, 1)``.

    The area is taken over interactions ``1 .. budget`` and averaged over
    episodes, so it rewards both speed and final quality, and charges any
    interactions spent on measurement or exploration.
    """
    after = traces[traces["interaction"] > 0]
    return float(after["excess"].clip(upper=1.0).mean())

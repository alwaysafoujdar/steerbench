"""Benchmark experiments E0-E5.

Each ``run_eX`` function takes a :class:`BenchmarkConfig` and returns a mapping
from result name to table; figures are produced separately from those tables so
that the notebook can rebuild them from cached CSVs without rerunning anything.
"""

from __future__ import annotations

import json
import logging
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from steerbench.config import BenchmarkConfig, MachineConfig
from steerbench.controllers.base import Controller
from steerbench.controllers.bayesopt import BayesOptController
from steerbench.controllers.offline import (
    BehaviourController,
    OfflineController,
    OfflineJob,
    load_offline,
    train_many_offline,
)
from steerbench.controllers.rl import (
    RLController,
    TrainJob,
    default_hyperparams,
    load_agent,
    train_many,
)
from steerbench.controllers.rls import RLSController
from steerbench.controllers.svd import ResponseMeasurementSVD, SVDController
from steerbench.evaluation import (
    SearchSpace,
    episode_table,
    evaluate,
    propose_trials,
    run_episode,
    summarise,
    tuning_score,
)
from steerbench.machine import Machine, ModelInfo, exact_model_config, make_machines

logger = logging.getLogger(__name__)

#: Focal-length error levels of the E2 sweep.
E2_FOCAL_SIGMAS: tuple[float, ...] = (0.0, 0.01, 0.03, 0.05, 0.10)
#: Extension beyond the specified sweep, to locate where classical correction breaks.
#: RL agents were trained on at most 10%, so these levels are out of their training range.
E2X_FOCAL_SIGMAS: tuple[float, ...] = (0.15, 0.20, 0.30)
#: Unavailable-BPM fractions of the E3 sweep.
E3_MASKED_FRACTIONS: tuple[float, ...] = (0.0, 0.10, 0.25, 0.40)


def with_focal_sigma(cfg: MachineConfig, sigma: float) -> MachineConfig:
    """Copy of a machine config with a different focal-length error level."""
    return replace(cfg, errors=replace(cfg.errors, focal_error_sigma=sigma))


def oracle_limit_check(cfg: BenchmarkConfig, sigmas: tuple[float, ...]) -> pd.DataFrame:
    """Per-machine oracle statistics for several focal-error levels.

    Returns
    -------
    pandas.DataFrame
        One row per sampled machine with ``max_abs_theta``, ``within_limits``,
        ``r_star`` and ``rms0``.
    """
    rows = []
    for sigma in sigmas:
        mcfg = with_focal_sigma(cfg.machine, sigma)
        for i, m in enumerate(make_machines(mcfg, cfg.eval.n_oracle_check, cfg.eval.oracle_seed)):
            o = m.oracle()
            rows.append(
                {
                    "focal_error_sigma": sigma,
                    "machine": i,
                    "max_abs_theta": o.max_abs_unconstrained,
                    "within_limits": o.within_limits,
                    "r_star": o.rms,
                    "rms0": m.true_rms(),
                }
            )
    return pd.DataFrame(rows)


def run_e0(cfg: BenchmarkConfig) -> dict[str, pd.DataFrame]:
    """E0 sanity checks.

    1. Fraction of machines whose unconstrained oracle respects ``theta_max``,
       at every E2 model-error level.
    2. SVD with the exact model, no noise and ``g = 1`` on the test machines:
       the residual after one interaction should equal the oracle residual.

    Returns
    -------
    dict of pandas.DataFrame
        ``e0_oracle_samples``, ``e0_oracle_limits`` and ``e0_svd_exact``.
    """
    samples = oracle_limit_check(cfg, E2_FOCAL_SIGMAS)
    limits = (
        samples.groupby("focal_error_sigma")
        .agg(
            n_machines=("machine", "size"),
            fraction_within_limits=("within_limits", "mean"),
            max_abs_theta_p95=("max_abs_theta", lambda v: np.quantile(v, 0.95)),
            r_star_median=("r_star", "median"),
            rms0_median=("rms0", "median"),
        )
        .reset_index()
    )
    limits["theta_max"] = cfg.machine.limits.theta_max_mrad
    for _, r in limits.iterrows():
        logger.info(
            "E0 oracle within limits at focal error %.0f%%: %.3f",
            100 * r.focal_error_sigma,
            r.fraction_within_limits,
        )

    exact = exact_model_config(cfg.machine, noise=False)
    rows = []
    for i, m in enumerate(make_machines(exact, cfg.eval.n_test_machines, cfg.eval.test_seed)):
        trace = run_episode(m, SVDController(m.model_info, gain=1.0), max_interactions=3)
        rows.append(
            {
                "machine": i,
                "oracle_within_limits": m.oracle().within_limits,
                "r_star": trace.r_star,
                "rms_initial": trace.rms_initial,
                "rms_after_1": trace.rms_true[1],
                "rms_after_3": trace.rms_true[3],
                "excess_after_1": trace.excess[1],
                "abs_gap_after_1": abs(trace.rms_true[1] - trace.r_star),
            }
        )
    svd_exact = pd.DataFrame(rows)
    ok = svd_exact[svd_exact.oracle_within_limits]
    logger.info(
        "E0 SVD exact model: %d/%d machines within limits; max |RMS_1 - r*| there = %.2e mm",
        len(ok),
        len(svd_exact),
        ok.abs_gap_after_1.max(),
    )
    return {"e0_oracle_samples": samples, "e0_oracle_limits": limits, "e0_svd_exact": svd_exact}


# ------------------------------------------------------------------ methods
Factory = Callable[[ModelInfo, np.random.Generator, dict[str, Any]], Controller]


@dataclass
class Method:
    """A non-learned correction method with its tuning space.

    Attributes
    ----------
    name : str
        Label in results.
    factory : callable
        ``factory(model_info, rng, hyperparams) -> Controller``.
    default : dict
        Default hyperparameters (tuning trial 0).
    space : SearchSpace
        Random-search space.
    """

    name: str
    factory: Factory
    default: dict[str, Any]
    space: SearchSpace = field(default_factory=dict)


_BO_SPACE: SearchSpace = {
    "box_fraction": ("uniform", 0.2, 1.0),
    "xi": ("log", 1e-4, 1e-2),
    "local_fraction": ("uniform", 0.2, 0.9),
}

CLASSICAL: dict[str, Method] = {
    "SVD": Method(
        "SVD",
        lambda m, rng, hp: SVDController(m, **hp),
        {"gain": 1.0, "tikhonov": 0.0},
        {"gain": ("uniform", 0.2, 1.0), "tikhonov": ("log", 0.1, 10.0)},
    ),
    "RM-SVD": Method(
        "RM-SVD",
        lambda m, rng, hp: ResponseMeasurementSVD(m, **hp),
        {"kick": 0.1, "gain": 1.0, "tikhonov": 0.0},
        {
            "kick": ("uniform", 0.02, 0.1),
            "gain": ("uniform", 0.2, 1.0),
            "tikhonov": ("log", 0.1, 10.0),
        },
    ),
    "RLS": Method(
        "RLS",
        lambda m, rng, hp: RLSController(m, rng, **hp),
        {"forgetting": 0.98, "prior_std": 1.0, "dither": 0.001, "gain": 1.0},
        {
            "forgetting": ("uniform", 0.9, 1.0),
            "prior_std": ("log", 0.1, 10.0),
            "dither": ("log", 1e-4, 1e-2),
            "gain": ("uniform", 0.2, 1.0),
        },
    ),
    "BO": Method(
        "BO",
        lambda m, rng, hp: BayesOptController(m, rng, n_modes=None, **hp),
        {"box_fraction": 0.5, "xi": 0.001, "local_fraction": 0.5},
        _BO_SPACE,
    ),
    "BO-SVDmodes": Method(
        "BO-SVDmodes",
        lambda m, rng, hp: BayesOptController(m, rng, **hp),
        {"n_modes": 4, "box_fraction": 0.5, "xi": 0.001, "local_fraction": 0.5},
        _BO_SPACE | {"n_modes": ("choice", [2, 3, 4, 6, 8])},
    ),
}

RL_SPACE: SearchSpace = {
    "learning_rate": ("log", 1e-4, 1e-3),
    "gamma": ("choice", [0.9, 0.95, 0.99]),
    "batch_size": ("choice", [128, 256]),
    "tau": ("choice", [0.005, 0.01, 0.02]),
    "action_scale_mrad": ("choice", [0.01, 0.02, 0.05, 0.2]),
}

OFFLINE_SPACES: dict[str, SearchSpace] = {
    "BC": {"learning_rate": ("log", 1e-4, 3e-3)},
    "TD3+BC": {"learning_rate": ("log", 1e-4, 1e-3), "alpha": ("log", 0.05, 5.0)},
}

#: Privileged reference: SVD using the true effective response (not a real method).
REFERENCE = "SVD (true R, reference)"
BEHAVIOUR = "Behaviour"


def offline_label(algo: str, size: int) -> str:
    """Result label of an offline agent, e.g. ``TD3+BC (50k)``."""
    return f"{algo} ({size // 1000}k)"


def method_rng(cfg: BenchmarkConfig, name: str) -> np.random.Generator:
    """Tuning generator of one method, independent of which others are tuned."""
    return np.random.default_rng([cfg.tuning.seed, zlib.crc32(name.encode())])


def controller_rng(machine_index: int, seed: int = 0) -> np.random.Generator:
    """Generator for a controller's own randomness on one evaluation machine."""
    return np.random.default_rng([9001, seed, machine_index])


# ------------------------------------------------------------- evaluators
def evaluate_factory(
    name: str,
    make: Callable[[Machine, int], Controller],
    machines: list[Machine],
    budget: int,
    seed: int = 0,
    **labels: object,
) -> pd.DataFrame:
    """Evaluate ``make(machine, machine_index)`` on every machine."""
    index = {id(m): i for i, m in enumerate(machines)}
    return evaluate(
        machines, lambda m: make(m, index[id(m)]), budget, method=name, seed=seed, **labels
    )


def evaluate_classical(
    method: Method, hp: dict[str, Any], machines: list[Machine], budget: int, **labels: object
) -> pd.DataFrame:
    """Evaluate a non-learned method with fixed hyperparameters."""
    return evaluate_factory(
        method.name,
        lambda m, i: method.factory(m.model_info, controller_rng(i), hp),
        machines,
        budget,
        **labels,
    )


def evaluate_rl(
    cfg: BenchmarkConfig,
    algo: str,
    agents: list[tuple[int, Any]],
    hp: dict[str, Any],
    machines: list[Machine],
    budget: int,
    **labels: object,
) -> pd.DataFrame:
    """Evaluate trained online-RL agents (one block per training seed)."""
    frames = [
        evaluate_factory(
            algo,
            lambda m, i, a=agent: RLController(a, m.model_info, cfg, algo, hp),
            machines,
            budget,
            seed=seed,
            **labels,
        )
        for seed, agent in agents
    ]
    return pd.concat(frames, ignore_index=True)


def evaluate_offline(
    cfg: BenchmarkConfig,
    label: str,
    agents: list[tuple[int, Any]],
    machines: list[Machine],
    budget: int,
    **labels: object,
) -> pd.DataFrame:
    """Evaluate offline-trained agents (one block per training seed)."""
    frames = [
        evaluate_factory(
            label,
            lambda m, i, a=agent: OfflineController(a, m.model_info, cfg, label),
            machines,
            budget,
            seed=seed,
            **labels,
        )
        for seed, agent in agents
    ]
    return pd.concat(frames, ignore_index=True)


def behaviour_factory(cfg: BenchmarkConfig, noise: float) -> Callable[[Machine, int], Controller]:
    """The offline logging policy as an evaluable controller."""
    return lambda m, i: BehaviourController(
        m.model_info, controller_rng(i, 7), cfg.offline.behaviour_gain, noise
    )


def reference_factory(svd_hp: dict[str, Any]) -> Callable[[Machine, int], Controller]:
    """SVD with the machine's true (measured-space) response: a privileged reference."""

    def make(m: Machine, i: int) -> Controller:
        info = replace(m.model_info, R=m.errors.bpm_gain[:, None] * m.R_eff)
        return SVDController(info, name=REFERENCE, **svd_hp)

    return make


# ------------------------------------------------------------- paths / io
def models_dir(cfg: BenchmarkConfig) -> Path:
    """Directory holding trained agents for this run."""
    return Path(cfg.results_dir) / "models"


def hyperparams_path(cfg: BenchmarkConfig) -> Path:
    """JSON file with the tuned hyperparameters of every method."""
    return Path(cfg.results_dir) / "hyperparams.json"


def load_hyperparams(cfg: BenchmarkConfig) -> dict[str, dict[str, Any]]:
    """Tuned hyperparameters, written by the TUNE steps."""
    path = hyperparams_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run the TUNE steps first")
    return json.loads(path.read_text())


def _store_best(cfg: BenchmarkConfig, trials: pd.DataFrame) -> None:
    """Merge the best trial of each tuned method into ``hyperparams.json``."""
    path = hyperparams_path(cfg)
    chosen = json.loads(path.read_text()) if path.exists() else {}
    best = trials.loc[trials.groupby("method")["score"].idxmin()]
    chosen |= {r.method: json.loads(r.params) for r in best.itertuples()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chosen, indent=2, sort_keys=True))


def rl_agent_path(cfg: BenchmarkConfig, algo: str, seed: int) -> Path:
    """File of a final online-RL agent."""
    return models_dir(cfg) / f"{algo}_seed{seed}.zip"


def offline_agent_path(cfg: BenchmarkConfig, algo: str, size: int, noise: float, seed: int) -> Path:
    """File of an offline agent trained on one dataset cell."""
    slug = algo.replace("+", "")
    return models_dir(cfg) / "offline" / f"{slug}_n{size}_noise{noise:g}_seed{seed}.d3"


def _load_all(paths: list[Path], loader: Callable[[Path], Any], step: str) -> list[tuple[int, Any]]:
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing agents {missing}: run the {step} step first")
    return [(s, loader(p)) for s, p in enumerate(paths)]


def load_trained_agents(cfg: BenchmarkConfig, algo: str) -> list[tuple[int, Any]]:
    """Online-RL agents written by the TRAIN step, as ``(seed, agent)`` pairs."""
    paths = [rl_agent_path(cfg, algo, s) for s in range(cfg.rl.n_seeds)]
    return _load_all(paths, lambda p: load_agent(algo, p), "TRAIN")


def load_offline_agents(
    cfg: BenchmarkConfig, algo: str, size: int, noise: float, n_seeds: int
) -> list[tuple[int, Any]]:
    """Offline agents of one dataset cell, as ``(seed, agent)`` pairs."""
    paths = [offline_agent_path(cfg, algo, size, noise, s) for s in range(n_seeds)]
    return _load_all(paths, load_offline, "TRAIN_OFFLINE")


# ------------------------------------------------------------------ TUNE
def _validation(cfg: BenchmarkConfig) -> list[Machine]:
    return make_machines(cfg.machine, cfg.eval.n_val_machines, cfg.eval.val_seed)


def run_tune(cfg: BenchmarkConfig) -> dict[str, pd.DataFrame]:
    """Equal-budget random search for every non-learned method on validation machines.

    Every method in the benchmark (here and in :func:`run_tune_rl` and
    :func:`run_tune_offline`) gets ``cfg.tuning.n_trials`` configurations; trial 0
    is its default. Each is scored by :func:`~steerbench.evaluation.tuning_score`
    on the same validation machines at nominal conditions. Test machines are
    never used. The best configuration is merged into ``hyperparams.json``.
    """
    budget, val = cfg.eval.max_interactions, _validation(cfg)
    rows: list[dict[str, Any]] = []
    for method in CLASSICAL.values():
        trials = propose_trials(
            method.default, method.space, cfg.tuning.n_trials, method_rng(cfg, method.name)
        )
        for t, hp in enumerate(trials):
            score = tuning_score(evaluate_classical(method, hp, val, budget))
            rows.append(
                {"method": method.name, "trial": t, "params": json.dumps(hp), "score": score}
            )
            logger.info("TUNE %s trial %d score %.4f %s", method.name, t, score, hp)
    out = pd.DataFrame(rows)
    _store_best(cfg, out)
    return {"tuning_classical": out}


def run_tune_rl(cfg: BenchmarkConfig) -> dict[str, pd.DataFrame]:
    """Equal-budget search for SAC and TD3 (one training seed per trial)."""
    budget, val = cfg.eval.max_interactions, _validation(cfg)
    jobs = []
    for algo in cfg.rl.algorithms:
        trials = propose_trials(
            default_hyperparams(cfg), RL_SPACE, cfg.tuning.n_trials, method_rng(cfg, algo)
        )
        for t, hp in enumerate(trials):
            path = models_dir(cfg) / "tuning" / f"{algo}_trial{t}.zip"
            jobs.append(TrainJob(cfg, algo, hp, 100 + t, str(path), {"trial": t, "hp": hp}))
    logger.info("TUNE_RL training %d trial agents", len(jobs))
    rows = []
    for info in train_many(jobs, cfg.rl.n_jobs):
        agent = load_agent(info["algo"], info["path"])
        score = tuning_score(evaluate_rl(cfg, info["algo"], [(0, agent)], info["hp"], val, budget))
        rows.append(
            {
                "method": info["algo"],
                "trial": info["trial"],
                "params": json.dumps(info["hp"]),
                "score": score,
                "training_steps": info["training_steps"],
                "wall_time_s": info["wall_time_s"],
            }
        )
        logger.info(
            "TUNE_RL %s trial %d score %.4f %s", info["algo"], info["trial"], score, info["hp"]
        )
    out = pd.DataFrame(rows)
    _store_best(cfg, out)
    return {"tuning_rl": out}


def _offline_default(cfg: BenchmarkConfig, algo: str) -> dict[str, Any]:
    hp: dict[str, Any] = {"learning_rate": cfg.offline.learning_rate}
    if algo == "TD3+BC":
        hp["alpha"] = cfg.offline.alpha
    return hp


def run_tune_offline(cfg: BenchmarkConfig) -> dict[str, pd.DataFrame]:
    """Equal-budget search for BC and TD3+BC on the largest main dataset."""
    budget, val = cfg.eval.max_interactions, _validation(cfg)
    size, noise = max(cfg.offline.dataset_sizes), cfg.offline.behaviour_noise_mrad
    jobs = []
    for algo in cfg.offline.algorithms:
        trials = propose_trials(
            _offline_default(cfg, algo),
            OFFLINE_SPACES[algo],
            cfg.tuning.n_trials,
            method_rng(cfg, algo),
        )
        for t, hp in enumerate(trials):
            path = models_dir(cfg) / "tuning" / f"{algo.replace('+', '')}_trial{t}.d3"
            jobs.append(
                OfflineJob(cfg, algo, hp, size, noise, 100 + t, str(path), {"trial": t, "hp": hp})
            )
    logger.info("TUNE_OFFLINE training %d trial agents", len(jobs))
    rows = []
    for info in train_many_offline(jobs, cfg.rl.n_jobs):
        agent = load_offline(info["path"])
        score = tuning_score(evaluate_offline(cfg, info["algo"], [(0, agent)], val, budget))
        rows.append(
            {
                "method": info["algo"],
                "trial": info["trial"],
                "params": json.dumps(info["hp"]),
                "score": score,
            }
        )
        logger.info(
            "TUNE_OFFLINE %s trial %d score %.4f %s", info["algo"], info["trial"], score, info["hp"]
        )
    out = pd.DataFrame(rows)
    _store_best(cfg, out)
    return {"tuning_offline": out}


# ----------------------------------------------------------------- TRAIN
def run_train(cfg: BenchmarkConfig) -> dict[str, pd.DataFrame]:
    """Train the final online-RL agents (``n_seeds`` per algorithm) with tuned settings.

    Returns
    -------
    dict of pandas.DataFrame
        ``rl_training``: simulated training steps and wall time per agent.
    """
    hps = load_hyperparams(cfg)
    jobs = [
        TrainJob(cfg, algo, hps[algo], s, str(rl_agent_path(cfg, algo, s)))
        for algo in cfg.rl.algorithms
        for s in range(cfg.rl.n_seeds)
    ]
    logger.info("TRAIN %d agents (%d steps each)", len(jobs), cfg.rl.total_steps)
    out = pd.DataFrame(train_many(jobs, cfg.rl.n_jobs)).drop(columns=["path"])
    out["tuning_steps"] = cfg.tuning.n_trials * cfg.rl.total_steps
    return {"rl_training": out}


def _offline_jobs(
    cfg: BenchmarkConfig, cells: list[tuple[int, float]], n_seeds: int
) -> list[OfflineJob]:
    hps = load_hyperparams(cfg)
    return [
        OfflineJob(
            cfg, algo, hps[algo], size, noise, s, str(offline_agent_path(cfg, algo, size, noise, s))
        )
        for algo in cfg.offline.algorithms
        for size, noise in cells
        for s in range(n_seeds)
    ]


def run_train_offline(cfg: BenchmarkConfig) -> dict[str, pd.DataFrame]:
    """Train the final offline agents on the main datasets (every size, main noise)."""
    noise = cfg.offline.behaviour_noise_mrad
    jobs = _offline_jobs(cfg, [(n, noise) for n in cfg.offline.dataset_sizes], cfg.offline.n_seeds)
    logger.info("TRAIN_OFFLINE %d agents (%d gradient steps each)", len(jobs), cfg.offline.n_steps)
    out = pd.DataFrame(train_many_offline(jobs, cfg.rl.n_jobs)).drop(columns=["path"])
    return {"offline_training": out}


# ------------------------------------------------------ test-set sweeps
@dataclass
class Learned:
    """Trained agents needed by the test-set sweeps, loaded once."""

    rl: dict[str, list[tuple[int, Any]]]
    offline: dict[str, list[tuple[int, Any]]]


def load_learned(cfg: BenchmarkConfig, offline_sizes: list[int] | None = None) -> Learned:
    """Load the final online and offline agents (offline: given dataset sizes)."""
    sizes = cfg.offline.dataset_sizes if offline_sizes is None else offline_sizes
    noise = cfg.offline.behaviour_noise_mrad
    offline = {
        offline_label(algo, n): load_offline_agents(cfg, algo, n, noise, cfg.offline.n_seeds)
        for algo in cfg.offline.algorithms
        for n in sizes
    }
    rl = {algo: load_trained_agents(cfg, algo) for algo in cfg.rl.algorithms}
    return Learned(rl=rl, offline=offline)


def evaluate_all(
    cfg: BenchmarkConfig, mcfg: MachineConfig, learned: Learned, **labels: object
) -> pd.DataFrame:
    """Every method, the behaviour policy and the privileged reference on the test set.

    Every method gets fresh copies of the same test machines, so machines and
    measurement noise are identical across methods (common random numbers).
    """
    hps = load_hyperparams(cfg)
    budget = cfg.eval.max_interactions

    def machines() -> list[Machine]:
        return make_machines(mcfg, cfg.eval.n_test_machines, cfg.eval.test_seed)

    frames = [
        evaluate_classical(m, hps[m.name], machines(), budget, **labels) for m in CLASSICAL.values()
    ]
    frames.append(
        evaluate_factory(REFERENCE, reference_factory(hps["SVD"]), machines(), budget, **labels)
    )
    frames.append(
        evaluate_factory(
            BEHAVIOUR,
            behaviour_factory(cfg, cfg.offline.behaviour_noise_mrad),
            machines(),
            budget,
            **labels,
        )
    )
    for algo, agents in learned.rl.items():
        frames.append(evaluate_rl(cfg, algo, agents, hps[algo], machines(), budget, **labels))
    for label, agents in learned.offline.items():
        frames.append(evaluate_offline(cfg, label, agents, machines(), budget, **labels))
    return pd.concat(frames, ignore_index=True)


def curve_summary(traces: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Median and quartiles of the excess at each interaction."""
    g = traces.groupby([*keys, "interaction"])["excess"]
    return pd.DataFrame(
        {"median": g.median(), "q25": g.quantile(0.25), "q75": g.quantile(0.75)}
    ).reset_index()


def _summaries(
    cfg: BenchmarkConfig, traces: pd.DataFrame, keys: list[str], prefix: str
) -> dict[str, pd.DataFrame]:
    """Episode table, summary and median curves for one experiment."""
    episodes = episode_table(traces, cfg.eval.success_threshold, [*keys, "seed"])
    rng = np.random.default_rng(cfg.eval.test_seed + 1)
    summary = summarise(episodes, keys, cfg.eval.n_bootstrap, rng)
    return {
        f"{prefix}_episodes": episodes,
        f"{prefix}_summary": summary,
        f"{prefix}_curves": curve_summary(traces, keys),
    }


def run_e1(cfg: BenchmarkConfig) -> dict[str, pd.DataFrame]:
    """E1 nominal conditions: every method on the held-out test machines."""
    traces = evaluate_all(cfg, cfg.machine, load_learned(cfg))
    out = _summaries(cfg, traces, ["method"], "e1")
    out["e1_traces"] = traces
    for r in out["e1_summary"].itertuples():
        logger.info(
            "E1 %-26s success %.2f  median its %s  median final e %.4f",
            r.method,
            r.success_rate,
            r.its_median,
            r.final_excess_median,
        )
    return out


def _sweep(
    cfg: BenchmarkConfig,
    column: str,
    levels: tuple[float, ...],
    configure: Callable[[MachineConfig, float], MachineConfig],
    prefix: str,
) -> dict[str, pd.DataFrame]:
    """Evaluate every method at each level of one non-ideality."""
    learned = load_learned(cfg, offline_sizes=[max(cfg.offline.dataset_sizes)])
    frames = []
    for level in levels:
        frames.append(evaluate_all(cfg, configure(cfg.machine, level), learned, **{column: level}))
        logger.info("%s %s = %g done", prefix.upper(), column, level)
    return _summaries(cfg, pd.concat(frames, ignore_index=True), [column, "method"], prefix)


def with_masked_fraction(cfg: MachineConfig, fraction: float) -> MachineConfig:
    """Copy of a machine config with a different unavailable-BPM fraction."""
    return replace(cfg, measurement=replace(cfg.measurement, masked_fraction=fraction))


def run_e2(cfg: BenchmarkConfig) -> dict[str, pd.DataFrame]:
    """E2 model-error sweep over the focal-length error level."""
    return _sweep(cfg, "focal_error_sigma", E2_FOCAL_SIGMAS, with_focal_sigma, "e2")


def run_e2x(cfg: BenchmarkConfig) -> dict[str, pd.DataFrame]:
    """E2 extension to larger focal errors, to locate where SVD stops winning."""
    return _sweep(cfg, "focal_error_sigma", E2X_FOCAL_SIGMAS, with_focal_sigma, "e2x")


def run_e3(cfg: BenchmarkConfig) -> dict[str, pd.DataFrame]:
    """E3 missing-BPM sweep at nominal model error."""
    return _sweep(cfg, "masked_fraction", E3_MASKED_FRACTIONS, with_masked_fraction, "e3")


# --------------------------------------------------------------------- E4
def run_e4(cfg: BenchmarkConfig) -> dict[str, pd.DataFrame]:
    """E4 drift: consecutive episodes on drifting machines.

    Each of ``cfg.drift.n_machines`` test machines random-walks its errors
    between episodes. Methods: static SVD, RLS whose estimate persists across
    episodes, frozen SAC and SAC fine-tuned online on its own deployment data.
    Every episode starts from zero corrector settings.
    """
    from steerbench.controllers.rl import FineTunedRLController  # local: optional SB3 internals

    dc = cfg.drift
    hps = load_hyperparams(cfg)
    algo = "SAC" if "SAC" in cfg.rl.algorithms else cfg.rl.algorithms[0]
    mcfg = replace(cfg.machine, drift=dc.rates)
    rows: list[dict[str, Any]] = []

    def run(name: str, seed: int, make: Callable[[Machine, int], Controller]) -> None:
        for i, m in enumerate(make_machines(mcfg, dc.n_machines, cfg.eval.test_seed)):
            ctrl = make(m, i)  # one controller per machine, kept across all episodes
            for ep in range(dc.n_episodes):
                tr = run_episode(m, ctrl, dc.budget)
                its = tr.interactions_to_success(cfg.eval.success_threshold)
                rows.append(
                    {
                        "method": name,
                        "seed": seed,
                        "machine": i,
                        "episode": ep,
                        "final_excess": tr.final_excess,
                        "interactions_to_success": its,
                        "success": bool(np.isfinite(its)),
                        "focal_error_rms": float(np.sqrt(np.mean(m.errors.focal**2))),
                    }
                )
                m.drift_step()
        logger.info("E4 %s seed %d done", name, seed)

    run(
        "SVD", 0, lambda m, i: CLASSICAL["SVD"].factory(m.model_info, controller_rng(i), hps["SVD"])
    )
    run(
        "RLS", 0, lambda m, i: CLASSICAL["RLS"].factory(m.model_info, controller_rng(i), hps["RLS"])
    )
    for seed, path in [(s, rl_agent_path(cfg, algo, s)) for s in range(cfg.rl.n_seeds)]:
        frozen = load_agent(algo, path)
        run(
            f"{algo} (frozen)",
            seed,
            lambda m, i, a=frozen: RLController(a, m.model_info, cfg, algo, hps[algo]),
        )
        run(
            f"{algo} (fine-tuned)",
            seed,
            lambda m, i, p=path: FineTunedRLController(
                load_agent(algo, p),
                m.model_info,
                cfg,
                f"{algo} (fine-tuned)",
                hps[algo],
                dc.finetune_gradient_steps,
            ),
        )
    return {"e4_episodes": pd.DataFrame(rows)}


# --------------------------------------------------------------------- E5
def run_e5(cfg: BenchmarkConfig) -> dict[str, pd.DataFrame]:
    """E5 offline-data ablation: dataset size x behaviour-noise level.

    Agents for the main cells (trained by TRAIN_OFFLINE) are reused; the other
    cells are trained here with ``cfg.offline.e5_seeds`` seeds.
    """
    oc = cfg.offline
    main = {(n, oc.behaviour_noise_mrad) for n in oc.dataset_sizes}
    cells = [(n, z) for n in oc.e5_sizes for z in oc.e5_noises]
    todo = [c for c in cells if c not in main or oc.e5_seeds > oc.n_seeds]
    jobs = _offline_jobs(cfg, todo, oc.e5_seeds)
    logger.info("E5 training %d offline agents", len(jobs))
    train_many_offline(jobs, cfg.rl.n_jobs)
    budget = cfg.eval.max_interactions
    frames = []
    for size, noise in cells:
        labels = {"dataset_size": size, "behaviour_noise": noise}
        machines = make_machines(cfg.machine, cfg.eval.n_test_machines, cfg.eval.test_seed)
        frames.append(
            evaluate_factory(BEHAVIOUR, behaviour_factory(cfg, noise), machines, budget, **labels)
        )
        for algo in oc.algorithms:
            agents = load_offline_agents(cfg, algo, size, noise, oc.e5_seeds)
            machines = make_machines(cfg.machine, cfg.eval.n_test_machines, cfg.eval.test_seed)
            frames.append(evaluate_offline(cfg, algo, agents, machines, budget, **labels))
        logger.info("E5 cell n=%d noise=%g done", size, noise)
    traces = pd.concat(frames, ignore_index=True)
    return _summaries(cfg, traces, ["dataset_size", "behaviour_noise", "method"], "e5")


EXPERIMENTS: dict[str, Callable[[BenchmarkConfig], dict[str, pd.DataFrame]]] = {
    "E0": run_e0,
    "TUNE": run_tune,
    "TUNE_RL": run_tune_rl,
    "TUNE_OFFLINE": run_tune_offline,
    "TRAIN": run_train,
    "TRAIN_OFFLINE": run_train_offline,
    "E1": run_e1,
    "E2": run_e2,
    "E2X": run_e2x,
    "E3": run_e3,
    "E4": run_e4,
    "E5": run_e5,
}

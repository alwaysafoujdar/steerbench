"""Offline RL from logged data of an imperfect controller (d3rlpy v2).

The behaviour policy is the operators' SVD controller with the (mismatched)
design model and a conservative gain, plus Gaussian exploration noise on every
corrector step. It is run on domain-randomised training machines, through the
same environment and reward as online RL. From the logged transitions we train

* **BC**: behaviour cloning (supervised regression onto the logged actions);
* **TD3+BC**: TD3 with a behaviour-cloning regulariser (Fujimoto & Gu, 2021).

Both are compared with the behaviour policy itself. Actions are normalised by
``OfflineConfig.action_scale_mrad`` (the hardware step limit by default, so the
logged SVD steps are represented without clipping).

Note: d3rlpy seeds torch's global generator. Training therefore runs in worker
processes (see :func:`train_many_offline`) so no global state leaks into the
benchmark process.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from steerbench.config import BenchmarkConfig, EnvConfig
from steerbench.controllers.base import BaseController, limit_step
from steerbench.controllers.svd import SVDController
from steerbench.env import SteeringEnv, action_scale, encode_observation
from steerbench.machine import ModelInfo, Observation

logger = logging.getLogger(__name__)

OFFLINE_ALGORITHMS = ("BC", "TD3+BC")


def offline_env_config(cfg: BenchmarkConfig) -> EnvConfig:
    """Environment settings used for logging and for offline policies."""
    return replace(cfg.env, action_scale_mrad=cfg.offline.action_scale_mrad)


class BehaviourController(BaseController):
    """SVD with the design model plus Gaussian exploration noise (logging policy).

    Parameters
    ----------
    model : ModelInfo
        Design model and limits.
    rng : numpy.random.Generator
        Source of exploration noise.
    gain : float
        SVD gain.
    noise : float
        Standard deviation of the per-corrector exploration noise, in mrad.
    """

    def __init__(
        self, model: ModelInfo, rng: np.random.Generator, gain: float, noise: float
    ) -> None:
        self.svd = SVDController(model, gain=gain)
        self.model = model
        self.rng = rng
        self.noise = noise
        self.name = "Behaviour"

    def act(self, obs: Observation) -> np.ndarray:
        """SVD step plus exploration noise, within limits."""
        step = self.svd.act(obs) + self.rng.normal(0.0, self.noise, self.model.n_correctors)
        return limit_step(step, obs.theta, self.model.theta_max, self.model.dtheta_max)


@dataclass
class OfflineData:
    """Logged transitions in the layout d3rlpy expects."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminals: np.ndarray
    timeouts: np.ndarray

    def __len__(self) -> int:
        return int(self.actions.shape[0])


def collect_dataset(
    cfg: BenchmarkConfig, n_transitions: int, noise: float, seed: int
) -> OfflineData:
    """Log ``n_transitions`` steps of the behaviour policy on training machines.

    Parameters
    ----------
    cfg : BenchmarkConfig
        Machine, randomisation and offline settings.
    n_transitions : int
        Dataset size (whole episodes; the last one is cut at the size limit and
        marked as a timeout).
    noise : float
        Behaviour exploration noise in mrad.
    seed : int
        Seed for machines and noise (offset from ``eval.train_seed``).

    Returns
    -------
    OfflineData
    """
    env_cfg = offline_env_config(cfg)
    env = SteeringEnv(cfg.machine, env_cfg, cfg.randomisation)
    scale = action_scale(env_cfg, env.model)
    rng = np.random.default_rng([cfg.eval.train_seed, 77, seed])
    policy = BehaviourController(env.model, rng, cfg.offline.behaviour_gain, noise)
    obs_l, act_l, rew_l, term_l, tout_l = [], [], [], [], []
    vec, _ = env.reset(seed=int(rng.integers(2**31)))
    while len(act_l) < n_transitions:
        a = np.clip(policy.act(env.obs) / scale, -1.0, 1.0).astype(np.float32)
        nxt, r, term, trunc, _ = env.step(a)
        obs_l.append(vec)
        act_l.append(a)
        rew_l.append(r)
        term_l.append(float(term))
        # A terminal takes precedence over a time-out (d3rlpy forbids both at once).
        tout_l.append(float((trunc or len(act_l) == n_transitions) and not term))
        vec = nxt
        if term or trunc:
            vec, _ = env.reset()
    return OfflineData(
        observations=np.asarray(obs_l, dtype=np.float32),
        actions=np.asarray(act_l, dtype=np.float32),
        rewards=np.asarray(rew_l, dtype=np.float32),
        terminals=np.asarray(term_l, dtype=np.float32),
        timeouts=np.asarray(tout_l, dtype=np.float32),
    )


def _quiet_d3rlpy() -> None:
    """Route d3rlpy's structlog output through a warning-level filter."""
    import structlog  # installed with d3rlpy

    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))


def _make_algo(algo: str, hp: dict[str, Any], gamma: float) -> Any:
    """Create a d3rlpy algorithm on the CPU."""
    import d3rlpy

    if algo == "BC":
        conf = d3rlpy.algos.BCConfig(learning_rate=float(hp["learning_rate"]), batch_size=256)
    elif algo == "TD3+BC":
        conf = d3rlpy.algos.TD3PlusBCConfig(
            actor_learning_rate=float(hp["learning_rate"]),
            critic_learning_rate=float(hp["learning_rate"]),
            alpha=float(hp["alpha"]),
            batch_size=256,
            gamma=gamma,
        )
    else:
        raise ValueError(f"unknown offline algorithm {algo!r}")
    return conf.create(device="cpu:0")


@dataclass
class OfflineJob:
    """One offline training run (picklable)."""

    cfg: BenchmarkConfig
    algo: str
    hp: dict[str, Any]
    dataset_size: int
    noise: float
    seed: int
    path: str
    meta: dict[str, Any] = field(default_factory=dict)


def train_offline(job: OfflineJob) -> dict[str, Any]:
    """Log a dataset, train one offline agent on it and save the agent.

    The dataset depends on ``(dataset_size, noise, seed)`` only, so every
    algorithm trained with the same seed sees the same data.
    """
    import d3rlpy
    import torch

    torch.set_num_threads(1)
    _quiet_d3rlpy()
    d3rlpy.seed(job.seed)  # process-local: runs in a worker process
    data = collect_dataset(job.cfg, job.dataset_size, job.noise, job.seed)
    dataset = d3rlpy.dataset.MDPDataset(
        data.observations, data.actions, data.rewards, data.terminals, data.timeouts
    )
    agent = _make_algo(job.algo, job.hp, job.cfg.offline.gamma)
    t0 = time.perf_counter()
    agent.fit(
        dataset,
        n_steps=job.cfg.offline.n_steps,
        n_steps_per_epoch=job.cfg.offline.n_steps,
        show_progress=False,
        logger_adapter=d3rlpy.logging.NoopAdapterFactory(),
    )
    Path(job.path).parent.mkdir(parents=True, exist_ok=True)
    agent.save(job.path)
    return job.meta | {
        "algo": job.algo,
        "seed": job.seed,
        "dataset_size": job.dataset_size,
        "noise": job.noise,
        "path": job.path,
        "gradient_steps": job.cfg.offline.n_steps,
        "wall_time_s": time.perf_counter() - t0,
        "mean_logged_return": float(
            data.rewards.sum() / max(1.0, data.timeouts.sum() + data.terminals.sum())
        ),
    }


def train_many_offline(jobs: list[OfflineJob], n_jobs: int) -> list[dict[str, Any]]:
    """Run offline jobs in worker processes (always, to isolate global seeding)."""
    if not jobs:
        return []
    with ProcessPoolExecutor(
        max(1, min(n_jobs, len(jobs))), mp_context=get_context("spawn")
    ) as pool:
        return list(pool.map(train_offline, jobs))


def load_offline(path: str | Path) -> Any:
    """Load a saved d3rlpy agent on the CPU."""
    import d3rlpy

    _quiet_d3rlpy()
    return d3rlpy.load_learnable(str(path), device="cpu:0")


class OfflineController(BaseController):
    """Deterministic policy of an offline-trained agent.

    Parameters
    ----------
    agent : d3rlpy algorithm
        Trained BC or TD3+BC agent.
    model : ModelInfo
        Limits.
    cfg : BenchmarkConfig
        Supplies the observation encoding and action scale.
    name : str
        Label used in results.
    """

    def __init__(self, agent: Any, model: ModelInfo, cfg: BenchmarkConfig, name: str) -> None:
        self.agent = agent
        self.model = model
        self.env_cfg = offline_env_config(cfg)
        self.scale = action_scale(self.env_cfg, model)
        self.name = name

    def act(self, obs: Observation) -> np.ndarray:
        """Map the observation through the policy and scale to mrad."""
        vec = encode_observation(obs, self.model, self.env_cfg)[None, :]
        action = np.asarray(self.agent.predict(vec))[0]
        return np.clip(action, -1.0, 1.0).astype(float) * self.scale

"""Online RL (SAC, TD3 from stable-baselines3) trained with domain randomisation.

Training happens entirely in simulation on machines drawn from the training
population; the number of simulated environment steps is recorded separately
from the deployment interactions counted at evaluation time.
"""

from __future__ import annotations

import contextlib
import io
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from steerbench.config import BenchmarkConfig
from steerbench.controllers.base import BaseController
from steerbench.env import SteeringEnv, action_scale, encode_observation
from steerbench.machine import Machine, ModelInfo, Observation

logger = logging.getLogger(__name__)

ALGORITHMS = ("SAC", "TD3")


def _algo_class(algo: str) -> Any:
    """Return the stable-baselines3 class for an algorithm name."""
    from stable_baselines3 import SAC, TD3

    return {"SAC": SAC, "TD3": TD3}[algo]


def default_hyperparams(cfg: BenchmarkConfig) -> dict[str, Any]:
    """Hyperparameters taken directly from the config (tuning trial 0)."""
    rl = cfg.rl
    return {
        "learning_rate": rl.learning_rate,
        "gamma": rl.gamma,
        "batch_size": rl.batch_size,
        "tau": rl.tau,
        "action_scale_mrad": rl.action_scale_mrad,
    }


def env_config_for(cfg: BenchmarkConfig, hp: dict[str, Any]) -> Any:
    """Environment settings with the agent's own action scale."""
    return replace(cfg.env, action_scale_mrad=float(hp["action_scale_mrad"]))


def make_agent(
    algo: str, env: SteeringEnv, cfg: BenchmarkConfig, hp: dict[str, Any], seed: int
) -> Any:
    """Construct an untrained SAC or TD3 agent.

    Parameters
    ----------
    algo : str
        ``"SAC"`` or ``"TD3"``.
    env : SteeringEnv
        Training environment.
    cfg : BenchmarkConfig
        Supplies replay-buffer and network settings.
    hp : dict
        ``learning_rate``, ``gamma``, ``batch_size`` and ``tau`` (other keys,
        such as the action scale, are used by the environment, not the agent).
    seed : int
        Seed for the agent's own randomness (network init, exploration).
    """
    from stable_baselines3.common.noise import NormalActionNoise

    kwargs: dict[str, Any] = {
        "learning_rate": float(hp["learning_rate"]),
        "gamma": float(hp["gamma"]),
        "batch_size": int(hp["batch_size"]),
        "tau": float(hp["tau"]),
        "buffer_size": cfg.rl.buffer_size,
        "learning_starts": cfg.rl.learning_starts,
        "policy_kwargs": {"net_arch": list(cfg.rl.net_arch)},
        "seed": seed,
        "device": "cpu",
        "verbose": 0,
    }
    if algo == "TD3":
        n = env.action_space.shape[0]
        sigma = cfg.rl.td3_action_noise * np.ones(n)
        kwargs["action_noise"] = NormalActionNoise(np.zeros(n), sigma)
    return _algo_class(algo)("MlpPolicy", env, **kwargs)


@dataclass
class TrainJob:
    """One RL training run (picklable, for process pools)."""

    cfg: BenchmarkConfig
    algo: str
    hp: dict[str, Any]
    seed: int
    path: str
    meta: dict[str, Any] = field(default_factory=dict)


def train_agent(job: TrainJob) -> dict[str, Any]:
    """Train one agent with domain randomisation and save it.

    Returns
    -------
    dict
        ``job.meta`` plus ``algo``, ``seed``, ``path``, ``training_steps`` and
        ``wall_time_s``.
    """
    import torch

    torch.set_num_threads(1)
    cfg = job.cfg
    env = SteeringEnv(cfg.machine, env_config_for(cfg, job.hp), cfg.randomisation)
    # SB3 seeds the env (and so the training-machine stream) from the agent seed;
    # offsetting by train_seed keeps training machines apart from val/test ones.
    agent = make_agent(job.algo, env, cfg, job.hp, cfg.eval.train_seed + job.seed)
    t0 = time.perf_counter()
    agent.learn(total_timesteps=cfg.rl.total_steps)
    path = Path(job.path if job.path.endswith(".zip") else job.path + ".zip")
    path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(path))  # explicit suffix: SB3 would treat e.g. "d0.05" as an extension
    return job.meta | {
        "algo": job.algo,
        "seed": job.seed,
        "path": str(path),
        "training_steps": int(agent.num_timesteps),
        "wall_time_s": time.perf_counter() - t0,
    }


def train_many(jobs: list[TrainJob], n_jobs: int) -> list[dict[str, Any]]:
    """Run training jobs, in parallel processes when ``n_jobs > 1``."""
    if n_jobs <= 1 or len(jobs) <= 1:
        return [train_agent(j) for j in jobs]
    with ProcessPoolExecutor(min(n_jobs, len(jobs)), mp_context=get_context("spawn")) as pool:
        return list(pool.map(train_agent, jobs))


def _import_legacy_gym_quietly() -> None:
    """Import the legacy ``gym`` package (a d3rlpy dependency) without its stderr banner.

    stable-baselines3 imports ``gym`` while loading a model if it is installed; the
    package prints an unconditional deprecation notice on import. The benchmark
    itself only uses gymnasium.
    """
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            import gym  # noqa: F401
        except ImportError:
            pass


def load_agent(algo: str, path: str | Path) -> Any:
    """Load a saved agent onto the CPU."""
    _import_legacy_gym_quietly()
    return _algo_class(algo).load(str(path), device="cpu")


class RLController(BaseController):
    """Deterministic policy of a trained agent behind the common interface.

    Parameters
    ----------
    agent : stable-baselines3 model
        Trained SAC or TD3 agent.
    model : ModelInfo
        Limits (the policy does not use the response matrix).
    cfg : BenchmarkConfig
        Supplies the observation encoding.
    name : str
        Label used in results.
    hp : dict
        The agent's hyperparameters; supplies its action scale.
    """

    def __init__(
        self, agent: Any, model: ModelInfo, cfg: BenchmarkConfig, name: str, hp: dict[str, Any]
    ) -> None:
        self.agent = agent
        self.model = model
        self.env_cfg = env_config_for(cfg, hp)
        self.scale = action_scale(self.env_cfg, model)
        self.name = name

    def act(self, obs: Observation) -> np.ndarray:
        """Map the observation through the policy and scale to mrad."""
        vec = encode_observation(obs, self.model, self.env_cfg)
        action, _ = self.agent.predict(vec, deterministic=True)
        return np.clip(action, -1.0, 1.0).astype(float) * self.scale


def fine_tune_env(cfg: BenchmarkConfig, hp: dict[str, Any], machine: Machine) -> SteeringEnv:
    """Environment bound to one machine, for online fine-tuning (E4)."""
    return SteeringEnv(cfg.machine, env_config_for(cfg, hp), randomisation=None, machine=machine)


class FineTunedRLController(RLController):
    """An RL policy that keeps learning from its own deployment data.

    Each deployment transition (observation, applied action, environment
    reward, next observation) is stored in the agent's replay buffer. Before
    every new episode the agent takes ``gradient_steps`` off-policy updates.
    No extra machine interactions are spent: exploration comes only from what
    the deterministic policy does anyway (and the measurement noise).

    Parameters
    ----------
    agent, model, cfg, name, hp
        As for :class:`RLController`. The agent is modified in place.
    gradient_steps : int
        Updates between episodes.
    """

    def __init__(
        self,
        agent: Any,
        model: ModelInfo,
        cfg: BenchmarkConfig,
        name: str,
        hp: dict[str, Any],
        gradient_steps: int,
    ) -> None:
        from stable_baselines3.common.logger import configure

        super().__init__(agent, model, cfg, name, hp)
        agent.set_logger(configure(None, []))
        self.gradient_steps = gradient_steps
        self.n_added = 0
        self.rms_initial = 1.0

    def reset(self, obs: Observation) -> None:
        """Update on everything collected so far, then start the episode."""
        if self.gradient_steps > 0 and self.n_added >= self.agent.batch_size:
            self.agent.train(gradient_steps=self.gradient_steps, batch_size=self.agent.batch_size)
        self.rms_initial = max(obs.rms, 1e-9)

    def observe(self, obs: Observation, action: np.ndarray, next_obs: Observation) -> None:
        """Store the transition with the environment's reward."""
        at_limit = np.mean(np.abs(next_obs.theta) >= self.model.theta_max * (1 - 1e-9))
        reward = (obs.rms - next_obs.rms) / self.rms_initial - self.env_cfg.limit_penalty * at_limit
        a = np.clip(action / self.scale, -1.0, 1.0).astype(np.float32)
        self.agent.replay_buffer.add(
            encode_observation(obs, self.model, self.env_cfg)[None],
            encode_observation(next_obs, self.model, self.env_cfg)[None],
            a[None],
            np.array([reward], dtype=np.float32),
            np.array([False]),
            [{}],
        )
        self.n_added += 1

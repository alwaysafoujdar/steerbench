"""Gymnasium environment for trajectory correction.

Observation (float32, length ``2 * n_bpms + n_correctors``)::

    [x_meas / x_scale (masked readings zeroed, clipped), mask, theta / theta_max]

Action: normalised corrector change in ``[-1, 1]^n_correctors``, multiplied by
``action_scale`` = ``min(env_cfg.action_scale_mrad, dtheta_max)`` (the hardware
``dtheta_max`` when unset).

Reward: ``(RMS_meas_prev - RMS_meas_new) / RMS_meas_initial`` minus
``limit_penalty`` times the fraction of correctors sitting at ``theta_max``.
Summed over an episode the first term telescopes to the relative reduction of
the measured RMS.

By default each :meth:`SteeringEnv.reset` samples a fresh machine with
domain-randomised focal-error level and BPM-mask fraction; passing a fixed
``machine`` instead keeps training on one machine (used for online fine-tuning).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from steerbench.config import EnvConfig, MachineConfig, RandomisationConfig
from steerbench.machine import Machine, ModelInfo, Observation


def encode_observation(obs: Observation, model: ModelInfo, env_cfg: EnvConfig) -> np.ndarray:
    """Flatten an :class:`Observation` into the RL observation vector."""
    x = np.clip(obs.x / env_cfg.x_scale_mm, -env_cfg.obs_clip, env_cfg.obs_clip)
    return np.concatenate([x, obs.mask.astype(float), obs.theta / model.theta_max]).astype(
        np.float32
    )


def action_scale(env_cfg: EnvConfig, model: ModelInfo) -> float:
    """Corrector change (mrad) for a normalised action of 1."""
    if env_cfg.action_scale_mrad is None:
        return model.dtheta_max
    return min(env_cfg.action_scale_mrad, model.dtheta_max)


def randomised_machine_config(
    base: MachineConfig, rand: RandomisationConfig, rng: np.random.Generator
) -> MachineConfig:
    """Draw a machine configuration from the randomisation ranges."""
    sigma = rng.uniform(*rand.focal_error_sigma)
    frac = rng.uniform(*rand.masked_fraction)
    return replace(
        base,
        errors=replace(base.errors, focal_error_sigma=float(sigma)),
        measurement=replace(base.measurement, masked_fraction=float(frac)),
    )


class SteeringEnv(gym.Env):
    """Trajectory-correction environment over one or many simulated machines.

    Parameters
    ----------
    machine_cfg : MachineConfig
        Base machine configuration.
    env_cfg : EnvConfig
        Horizon, reward and observation settings.
    randomisation : RandomisationConfig, optional
        Domain-randomisation ranges; ``None`` keeps ``machine_cfg`` as is.
    machine : Machine, optional
        Fixed machine to train on. When given, ``reset`` reuses it (calling
        :meth:`Machine.reset`) instead of sampling a new one.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        machine_cfg: MachineConfig,
        env_cfg: EnvConfig,
        randomisation: RandomisationConfig | None = None,
        machine: Machine | None = None,
    ) -> None:
        super().__init__()
        self.machine_cfg = machine_cfg
        self.env_cfg = env_cfg
        self.randomisation = randomisation
        self.fixed_machine = machine
        probe = machine or Machine(machine_cfg, np.random.default_rng(0))
        self.model = probe.model_info
        n_b, n_c = probe.n_bpms, probe.n_correctors
        clip = env_cfg.obs_clip
        low = np.concatenate([np.full(n_b, -clip), np.zeros(n_b), np.full(n_c, -1.0)])
        high = np.concatenate([np.full(n_b, clip), np.ones(n_b), np.full(n_c, 1.0)])
        self.observation_space = spaces.Box(low.astype(np.float32), high.astype(np.float32))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_c,), dtype=np.float32)
        self.machine: Machine | None = machine
        self.action_scale = action_scale(env_cfg, self.model)

    def _new_machine(self) -> Machine:
        """Sample a training machine from this env's generator."""
        cfg = self.machine_cfg
        if self.randomisation is not None:
            cfg = randomised_machine_config(cfg, self.randomisation, self.np_random)
        child = np.random.default_rng(int(self.np_random.integers(2**63)))
        return Machine(cfg, child)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start an episode on a new (or the fixed) machine."""
        super().reset(seed=seed)
        if self.fixed_machine is None:
            self.machine = self._new_machine()
        assert self.machine is not None
        self.obs = self.machine.reset()
        self.rms_initial = max(self.obs.rms, 1e-9)
        self.steps = 0
        return encode_observation(self.obs, self.model, self.env_cfg), self._info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply a normalised corrector change and read the BPMs."""
        assert self.machine is not None
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        prev = self.obs.rms
        self.obs = self.machine.apply(a * self.action_scale)
        self.steps += 1
        at_limit = np.mean(np.abs(self.obs.theta) >= self.model.theta_max * (1 - 1e-9))
        reward = (prev - self.obs.rms) / self.rms_initial - self.env_cfg.limit_penalty * at_limit
        terminated = bool(self.obs.rms < self.env_cfg.rms_threshold_mm)
        truncated = bool(self.steps >= self.env_cfg.horizon)
        obs = encode_observation(self.obs, self.model, self.env_cfg)
        return obs, float(reward), terminated, truncated, self._info()

    def _info(self) -> dict[str, Any]:
        """Diagnostics; the true RMS is for logging only, never for the agent."""
        assert self.machine is not None
        return {"rms_meas": self.obs.rms, "rms_true": self.machine.true_rms()}

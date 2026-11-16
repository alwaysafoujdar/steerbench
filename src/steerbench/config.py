"""Dataclass configuration and YAML loading.

Every tunable number in the benchmark lives in one of these dataclasses. A YAML
file mirrors the nesting of :class:`BenchmarkConfig`; keys that are omitted keep
their defaults and unknown keys raise an error, so typos cannot silently pass.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


@dataclass
class LatticeConfig:
    """Geometry of the periodic thin-lens FODO transfer line.

    Attributes
    ----------
    n_cells : int
        Number of FODO cells. Each cell holds one focusing and one defocusing
        quadrupole, a BPM at each quadrupole and a corrector at the focusing one.
    drift_length : float
        Drift length L between neighbouring quadrupoles, in metres.
    phase_advance_deg : float
        Design betatron phase advance per cell, in degrees. The focal length is
        derived from ``sin(mu / 2) = L / (2 f)``.
    """

    n_cells: int = 10
    drift_length: float = 5.0
    phase_advance_deg: float = 90.0


@dataclass
class ErrorConfig:
    """Static machine errors, sampled once per machine.

    Attributes
    ----------
    focal_error_sigma : float
        Relative RMS error of each quadrupole focal length (0.03 means 3%).
    bpm_gain_sigma : float
        Relative RMS error of each BPM gain around 1.
    corrector_gain_sigma : float
        Relative RMS error of each corrector calibration around 1.
    quad_misalignment_mm : float
        RMS horizontal quadrupole offset in mm; each offset ``d`` gives a dipole
        kick ``d / f``.
    injection_offset_mm : float
        RMS injection position error in mm.
    injection_angle_mrad : float
        RMS injection angle error in mrad.
    """

    focal_error_sigma: float = 0.03
    bpm_gain_sigma: float = 0.01
    corrector_gain_sigma: float = 0.01
    quad_misalignment_mm: float = 0.1
    injection_offset_mm: float = 0.5
    injection_angle_mrad: float = 0.05


@dataclass
class MeasurementConfig:
    """BPM read-out.

    Attributes
    ----------
    bpm_noise_mm : float
        Standard deviation of additive Gaussian BPM noise, in mm.
    masked_fraction : float
        Fraction of BPMs unavailable in each episode (0 to 1). The masked set is
        redrawn at every episode reset.
    """

    bpm_noise_mm: float = 0.05
    masked_fraction: float = 0.0


@dataclass
class LimitsConfig:
    """Corrector hardware limits.

    Attributes
    ----------
    theta_max_mrad : float
        Maximum absolute corrector kick.
    dtheta_max_mrad : float
        Maximum change of any corrector in one interaction.
    """

    theta_max_mrad: float = 0.2
    dtheta_max_mrad: float = 0.2


@dataclass
class DriftConfig:
    """Random walk of the error parameters between episodes on one machine.

    Each attribute is the standard deviation of the per-episode increment of the
    corresponding error parameter. The walk is applied by
    :meth:`steerbench.machine.Machine.drift_step`.
    """

    focal_error_sigma: float = 0.0
    bpm_gain_sigma: float = 0.0
    corrector_gain_sigma: float = 0.0
    quad_misalignment_mm: float = 0.0
    injection_offset_mm: float = 0.0
    injection_angle_mrad: float = 0.0


@dataclass
class MachineConfig:
    """Everything needed to sample a simulated machine."""

    lattice: LatticeConfig = field(default_factory=LatticeConfig)
    errors: ErrorConfig = field(default_factory=ErrorConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)


@dataclass
class EvalConfig:
    """Evaluation protocol.

    Attributes
    ----------
    n_test_machines : int
        Size of the held-out test set.
    n_val_machines : int
        Size of the separate validation set used for tuning.
    test_seed, val_seed, train_seed, oracle_seed : int
        Root seeds of disjoint machine populations (test, validation,
        training and the E0 oracle-limit check).
    max_interactions : int
        Deployment-interaction budget per episode.
    success_threshold : float
        An episode succeeds once the normalised excess RMS drops below this.
    n_bootstrap : int
        Bootstrap resamples for confidence intervals.
    n_oracle_check : int
        Number of machines sampled to report the fraction whose oracle lies
        inside the corrector limits.
    """

    n_test_machines: int = 50
    n_val_machines: int = 20
    test_seed: int = 1000
    val_seed: int = 2000
    train_seed: int = 3000
    oracle_seed: int = 4000
    max_interactions: int = 30
    success_threshold: float = 0.05
    n_bootstrap: int = 1000
    n_oracle_check: int = 1000


@dataclass
class EnvConfig:
    """Gymnasium environment settings.

    Attributes
    ----------
    horizon : int
        Episode length T in interactions.
    rms_threshold_mm : float
        Episode terminates when the measured RMS drops below this.
    limit_penalty : float
        Reward penalty per step times the fraction of correctors at their limit.
    action_scale_mrad : float or None
        Corrector change for a normalised action of 1; ``None`` uses the
        hardware ``dtheta_max`` (values above it are capped there).
    x_scale_mm : float
        BPM readings are divided by this before entering the observation.
    obs_clip : float
        Scaled readings are clipped to ``[-obs_clip, obs_clip]``.
    """

    horizon: int = 15
    rms_threshold_mm: float = 0.05
    limit_penalty: float = 0.01
    action_scale_mrad: float | None = None
    x_scale_mm: float = 1.0
    obs_clip: float = 10.0


@dataclass
class RandomisationConfig:
    """Domain randomisation for RL training (uniform ranges, inclusive).

    Attributes
    ----------
    focal_error_sigma : list of float
        Range of the focal-error level drawn per training machine.
    masked_fraction : list of float
        Range of the unavailable-BPM fraction drawn per training episode.
    """

    focal_error_sigma: list[float] = field(default_factory=lambda: [0.0, 0.10])
    masked_fraction: list[float] = field(default_factory=lambda: [0.0, 0.4])


@dataclass
class RLConfig:
    """Online RL (stable-baselines3) settings.

    Attributes
    ----------
    algorithms : list of str
        Subset of ``["SAC", "TD3"]``.
    total_steps : int
        Simulated environment steps per training run.
    n_seeds : int
        Training seeds per algorithm for the final evaluation.
    learning_rate, gamma, batch_size, tau : float or int
        Default hyperparameters (the tuner searches around these).
    buffer_size, learning_starts : int
        Replay-buffer settings.
    net_arch : list of int
        Hidden layer sizes of actor and critic.
    action_scale_mrad : float
        Corrector change (mrad) that a normalised action of 1 maps to. It is
        an agent design choice, capped at the hardware ``dtheta_max``; the
        pre-study showed that mapping [-1, 1] onto the full hardware step
        leaves the required precision at ~1% of the action range, where SAC and
        TD3 fail to learn. It is tuned like any other hyperparameter.
    td3_action_noise : float
        Std of TD3's Gaussian exploration noise (normalised action units).
    n_jobs : int
        Parallel training processes.
    """

    algorithms: list[str] = field(default_factory=lambda: ["SAC", "TD3"])
    total_steps: int = 50_000
    n_seeds: int = 5
    learning_rate: float = 3e-4
    gamma: float = 0.95
    batch_size: int = 256
    tau: float = 0.005
    buffer_size: int = 100_000
    learning_starts: int = 1000
    net_arch: list[int] = field(default_factory=lambda: [256, 256])
    action_scale_mrad: float = 0.02
    td3_action_noise: float = 0.1
    n_jobs: int = 8


@dataclass
class OfflineConfig:
    """Offline-RL settings.

    Attributes
    ----------
    algorithms : list of str
        Subset of ``["BC", "TD3+BC"]``.
    behaviour_gain : float
        SVD gain of the logging policy.
    behaviour_noise_mrad : float
        Main behaviour exploration noise (std per corrector step, mrad).
    dataset_sizes : list of int
        Dataset sizes compared in E1-E3 (transitions).
    n_steps : int
        Gradient steps per training run.
    n_seeds : int
        Seeds per algorithm and dataset size.
    gamma : float
        Discount for TD3+BC.
    action_scale_mrad : float or None
        Normalisation of logged actions; ``None`` uses ``dtheta_max``.
    learning_rate, alpha : float
        Defaults (tuning trial 0); ``alpha`` is TD3+BC's BC weight.
    e5_sizes : list of int
        Dataset sizes of the E5 ablation.
    e5_noises : list of float
        Behaviour-noise levels (mrad) of the E5 ablation.
    e5_seeds : int
        Seeds per E5 cell.
    """

    algorithms: list[str] = field(default_factory=lambda: ["BC", "TD3+BC"])
    behaviour_gain: float = 0.5
    behaviour_noise_mrad: float = 0.02
    dataset_sizes: list[int] = field(default_factory=lambda: [10_000, 50_000])
    n_steps: int = 20_000
    n_seeds: int = 5
    gamma: float = 0.95
    action_scale_mrad: float | None = None
    learning_rate: float = 3e-4
    alpha: float = 2.5
    e5_sizes: list[int] = field(default_factory=lambda: [10_000, 50_000])
    e5_noises: list[float] = field(default_factory=lambda: [0.005, 0.02, 0.05])
    e5_seeds: int = 3


@dataclass
class DriftExperimentConfig:
    """E4 settings: consecutive episodes on drifting machines.

    Attributes
    ----------
    n_episodes : int
        Consecutive episodes per machine.
    n_machines : int
        Independent drifting machines (taken from the test population).
    budget : int
        Interactions per episode.
    rates : DriftConfig
        Per-episode random-walk standard deviations.
    finetune_gradient_steps : int
        Gradient steps of online RL fine-tuning after each episode.
    """

    n_episodes: int = 200
    n_machines: int = 5
    budget: int = 15
    rates: DriftConfig = field(
        default_factory=lambda: DriftConfig(
            focal_error_sigma=0.007,
            bpm_gain_sigma=0.001,
            corrector_gain_sigma=0.001,
            quad_misalignment_mm=0.005,
            injection_offset_mm=0.02,
            injection_angle_mrad=0.002,
        )
    )
    finetune_gradient_steps: int = 50


@dataclass
class TuningConfig:
    """Equal-budget random search on validation machines.

    Attributes
    ----------
    n_trials : int
        Hyperparameter configurations evaluated per method (the same for every
        method; trial 0 is always the method's default configuration).
    seed : int
        Seed of the search.
    """

    n_trials: int = 6
    seed: int = 5000


@dataclass
class BenchmarkConfig:
    """Top-level configuration for one benchmark run.

    Attributes
    ----------
    name : str
        Label written into result files.
    experiments : list of str
        Experiments to run, e.g. ``["E0", "E1"]``.
    results_dir, figures_dir : str
        Output directories, relative to the repository root.
    """

    name: str = "default"
    experiments: list[str] = field(default_factory=lambda: ["E0"])
    results_dir: str = "results"
    figures_dir: str = "figures"
    machine: MachineConfig = field(default_factory=MachineConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    randomisation: RandomisationConfig = field(default_factory=RandomisationConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    offline: OfflineConfig = field(default_factory=OfflineConfig)
    drift: DriftExperimentConfig = field(default_factory=DriftExperimentConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)


def from_dict(cls: type[T], data: dict[str, Any] | None) -> T:
    """Build a (possibly nested) dataclass from a plain dictionary.

    Parameters
    ----------
    cls : type
        Target dataclass type.
    data : dict or None
        Values to override. Missing keys keep their defaults.

    Returns
    -------
    instance of ``cls``

    Raises
    ------
    KeyError
        If ``data`` contains a key that is not a field of ``cls``.
    """
    data = dict(data or {})
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        kwargs[key] = _coerce(hints[key], value)
    return cls(**kwargs)


def _coerce(hint: Any, value: Any) -> Any:
    """Convert a YAML value to the annotated field type where needed."""
    if dataclasses.is_dataclass(hint) and isinstance(value, dict):
        return from_dict(hint, value)
    if isinstance(hint, types.UnionType) or typing.get_origin(hint) is typing.Union:
        return value
    if hint is float and isinstance(value, int):
        return float(value)
    return value


def to_dict(cfg: Any) -> dict[str, Any]:
    """Return a plain-dictionary copy of a dataclass configuration."""
    return dataclasses.asdict(cfg)


def load_config(path: str | Path) -> BenchmarkConfig:
    """Load a :class:`BenchmarkConfig` from a YAML file.

    Parameters
    ----------
    path : str or Path
        YAML file whose nesting mirrors :class:`BenchmarkConfig`.

    Returns
    -------
    BenchmarkConfig
    """
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    return from_dict(BenchmarkConfig, data)

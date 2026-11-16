# steerbench

Sample-efficient trajectory correction on a simulated particle-accelerator transfer line: a benchmark of classical orbit correction, Bayesian optimisation, online RL and offline RL.

Author: Aman K. Foujdar (Ashoka University). Built as supporting work for an application to CERN's Efficient Particle Accelerators (EPA) project, which applies AI and automation to accelerator operation. Deadline for a presentable version: 15 November 2026.

## 1. Research question

When the operator's optics model is imperfect, some beam-position monitors (BPMs) are unavailable, and the machine drifts over time, which correction strategy reaches target trajectory quality in the fewest interactions with the machine?

Every interaction with a real accelerator costs beam time. The central metric is therefore the number of deployment interactions on the target machine, reported separately from any simulation training cost.

Hypotheses to test (not to assume):

* H1. SVD correction with an exact optics model is near-optimal in the ideal case.
* H2. SVD with a mismatched model degrades as model error grows.
* H3. Adaptive response estimation (recursive least squares) recovers much of that loss.
* H4. RL trained in simulation with domain randomisation is competitive in deployment interactions, but at a large simulation-training cost.
* H5. Offline RL trained on logged data from an imperfect controller improves on that controller only when data coverage is adequate.

Report whatever the results show. If the classical baseline wins, say so plainly and show where it stops winning. Honest negative results are part of the deliverable.

## 2. Physics model (horizontal plane only)

* Lattice: a periodic thin-lens FODO transfer line, default 10 cells, drift length L between quadrupoles, focal length chosen from a target phase advance per cell (default 90 degrees, using sin(mu/2) = L / (2f)).
* Twiss parameters (beta, alpha, phase mu) propagated through 2x2 transfer matrices (drift and thin-lens quadrupole), starting from the periodic (matched) solution.
* Correctors: one horizontal thin-kick corrector per cell at the focusing quadrupole (default 10). BPMs: one at every quadrupole (default 20).
* Response matrix (thin kicks, no dispersion or coupling): R[i, j] = sqrt(beta_i * beta_j) * sin(mu_i - mu_j) if BPM i is downstream of corrector j, otherwise 0.
* Units: positions in mm, kicks in mrad.
* Initial trajectory error x0: injection offset and angle, plus dipole kicks from random quadrupole misalignments propagated through the true optics.
* Measurement: x_meas = bpm_gain * (x0 + R_true @ (corrector_gain * theta)) + noise, noise ~ N(0, sigma_bpm^2), default sigma_bpm = 0.05 mm.
* Corrector limits: |theta_j| <= theta_max; per-step change |dtheta_j| <= dtheta_max. Choose defaults so that the oracle correction lies within limits for at least 95% of sampled machines, and report that fraction.

Non-idealities, all configurable:

1. Model mismatch: relative quadrupole focal-length errors (sigma 0 to 10%), BPM gain errors, corrector gain errors. Controllers that use a model receive the ideal design R_model; the simulator uses R_true.
2. Partial observability: a random fraction of BPMs (0 to 40%) unavailable per episode. Observations carry a mask; masked readings are set to 0.
3. Drift: slow random walk of the error parameters across episodes on the same machine.

Oracle: theta* = argmin ||x0 + R_true theta||_2 subject to limits (simulator privilege, used only for evaluation). Oracle residual RMS r*.

Evaluation uses the true trajectory. Agents see only noisy, masked measurements.

* Normalised excess RMS: e = (RMS_true - r*) / (RMS_true_initial - r*), from 1 towards 0.
* Success: e < 0.05.

## 3. Methods

Common interface in `controllers/base.py`:

```python
class Controller(Protocol):
    name: str
    def reset(self, obs: Observation) -> None: ...
    def act(self, obs: Observation) -> np.ndarray: ...        # returns dtheta
    def observe(self, obs: Observation, action: np.ndarray,
                next_obs: Observation) -> None: ...            # optional online update
```

1. `SVDController`: dtheta = -g * pinv_reg(R_model[mask]) @ x_meas[mask], with Tikhonov regularisation and gain g in (0, 1]. This is the operators' standard method.
2. `ResponseMeasurementSVD`: spends interactions measuring R by kicking each corrector (counted honestly), then corrects with SVD.
3. `RLSController`: starts from R_model and updates the response estimate by recursive least squares on (dtheta, dx) pairs, with a forgetting factor and small dither for excitation. The estimate persists across episodes, which matters under drift.
4. `BayesOptController`: Gaussian-process BO (scikit-learn GP, Matern 5/2, expected improvement implemented in-package) on measured RMS. Two variants: full corrector space, and the top-k SVD modes of R_model (physics-informed, lower-dimensional).
5. Online RL: SAC and TD3 (stable-baselines3) on a Gymnasium environment, trained across randomised machines and evaluated on held-out machines. Report training environment steps separately.
6. Offline RL: logged dataset from a behaviour policy (SVD with mismatched model plus Gaussian exploration noise), dataset sizes 10k and 50k transitions. Train TD3+BC or IQL (d3rlpy v2) and compare with behaviour cloning and with the behaviour policy itself. If d3rlpy causes friction, implement TD3+BC directly in PyTorch.

Gymnasium environment:

* Observation: [x_meas (masked readings zeroed), mask, theta / theta_max].
* Action: normalised dtheta in [-1, 1], scaled by dtheta_max.
* Reward: (RMS_meas_prev - RMS_meas_new) / RMS_meas_initial, with a small penalty for hitting corrector limits.
* Termination: measured RMS below a threshold, or horizon T (default 15 steps).
* Must pass `gymnasium.utils.env_checker.check_env`.

## 4. Experiments

Held-out test set: 50 machines. Learned methods: 5 training seeds. Report the median with interquartile range, plus bootstrap confidence intervals. Tune every method's hyperparameters with the same budget on separate validation machines, never on the test machines.

* E0 Sanity: oracle; SVD with exact model, no noise and g = 1 converges in one step.
* E1 Nominal: 3% model error, 50 um noise, no masking, no drift. All methods. Figure: normalised excess RMS against deployment interactions.
* E2 Model-error sweep: focal-length error sigma in {0, 1, 3, 5, 10}%. Figure: success rate and interactions-to-success against model error.
* E3 Missing-BPM sweep: {0, 10, 25, 40}% unavailable.
* E4 Drift: 200 consecutive episodes on one drifting machine; static SVD, RLS, frozen RL, and RL with online fine-tuning.
* E5 Offline-data ablation: dataset size and behaviour-noise level.

Headline figure for the README: E2 (where each method holds up as the model degrades).

## 5. Repository layout

```
steerbench/
  CLAUDE.md
  README.md
  pyproject.toml
  configs/            quick.yaml, full.yaml
  src/steerbench/
    __init__.py
    config.py         dataclass configs, YAML loading
    optics.py         FODO lattice, Twiss propagation, response matrix
    machine.py        true machine: errors, noise, masking, drift
    env.py            Gymnasium environment
    controllers/      base.py, svd.py, rls.py, bayesopt.py, rl.py, offline.py
    evaluation.py     episode runner, interaction accounting, metrics
    experiments.py    E0-E5 sweeps
    plotting.py
  scripts/
    run_benchmark.py  --config configs/full.yaml; writes results/ and figures/
    make_readme_table.py
  notebooks/
    steerbench_walkthrough.ipynb
  tests/
  results/            CSV or parquet outputs
  figures/
```

## 6. Notebook

One notebook, `notebooks/steerbench_walkthrough.ipynb`, that imports the package and reads cached results. It must not retrain models and must run top to bottom in under two minutes. No function longer than about ten lines may be defined in the notebook. Commit it executed, with outputs, so figures render on GitHub.

Sections:

1. Problem and motivation (brief; cite the references below).
2. Lattice and response matrix: beta functions, phase advance, heatmap of R.
3. One episode visualised: trajectory before and after correction for SVD, RLS and SAC.
4. Benchmark results E1 to E5, loaded from `results/`.
5. Findings, limitations and next steps.

## 7. README

* Headline figure first, then a three-sentence summary of findings taken from actual results.
* How to reproduce: one install command and one benchmark command.
* Results table generated by `scripts/make_readme_table.py`, never typed by hand.
* Limitations: linear optics, horizontal plane only, no dispersion or coupling, synthetic errors, no real CERN data, simulation-only RL training cost.
* References:
  * V. Kain et al., "Sample-efficient reinforcement learning for CERN accelerator control", Phys. Rev. Accel. Beams 23, 124801 (2020).
  * V. Kain, "Accelerators on autopilot", CERN Courier (May 2025).

## 8. Commands

```
pip install -e ".[dev]"
pytest
python scripts/run_benchmark.py --config configs/quick.yaml   # under 10 min, laptop CPU
python scripts/run_benchmark.py --config configs/full.yaml    # under 2 h, laptop CPU
ruff check . && ruff format .
```

Local environment: a Python 3.11 virtualenv lives in `.venv/` (created with `uv venv --python 3.11 .venv`). Activate it with `source .venv/bin/activate` before running the commands above.

## 9. Required tests

* `test_optics.py`: FODO phase advance per cell matches cos(mu) = 1 - L^2 / (2 f^2); R is zero for BPMs upstream of a corrector; beta is periodic in the matched solution.
* `test_machine.py`: noise statistics; masking; drift changes parameters at the configured rate; zero errors give R_true == R_model.
* `test_svd.py`: exact model, no noise, g = 1 reaches the oracle residual in one step.
* `test_env.py`: passes `check_env`; reproducible under a fixed seed.
* `test_evaluation.py`: interaction counting is correct, including response-measurement cost.

## 10. Coding standards

* Python 3.11, type hints throughout, NumPy-style docstrings, ruff for linting and formatting.
* Randomness only through an explicit `np.random.Generator` passed in; no global seeding.
* Configuration through dataclasses loaded from YAML; no hidden module-level state.
* Logging via the `logging` module, not print statements.
* Dependencies: numpy, scipy, matplotlib, pandas, pyyaml, gymnasium, stable-baselines3, torch, scikit-learn, d3rlpy, pytest, ruff, jupyter. Ask before adding anything else.
* Check the installed versions' APIs (gymnasium, stable-baselines3, d3rlpy v2) rather than assuming them.

## 11. Working rules

* Work one phase at a time. At the end of each phase, run `pytest` and the quick benchmark, then summarise what was built, what was verified and any open issues.
* Never invent or hand-type numbers in the README or notebook. Every figure and number must come from a run.
* Never tune the benchmark to favour any method. Equal tuning budgets, validation machines only.
* Keep everything runnable on a laptop CPU.

## 12. Phases

* Phase 1 (core): config, optics, machine, oracle, SVD controller, evaluation runner, tests, lattice plots, E0.
* Phase 2: RLS, response-measurement SVD, Gymnasium env, SAC and TD3 with domain randomisation, E1 and E2.
* Phase 3: offline RL, Bayesian optimisation, E3 and E4 (E5 if time permits).
* Phase 4: notebook, README, headline figure, cleanup.

Cut line: Phases 1, 2 and 4 alone form a complete, presentable project. If time runs short, list the remaining work under "In progress" in the README.

## 13. Implementation notes (kept up to date)

* Benchmark steps (config `experiments`): `E0, TUNE, TUNE_RL, TUNE_OFFLINE, TRAIN, TRAIN_OFFLINE, E1, E2, E2X, E3, E4, E5`. Any subset can be rerun with `--only`; tuning steps merge into `results/hyperparams.json`, each method has its own tuning seed, so retuning one family leaves the others unchanged.
* README numbers live only inside `<!-- BEGIN:x -->` blocks written by `scripts/make_readme_table.py` via `steerbench.report`; `report.check_claims` raises if new results contradict the prose, so update the wording rather than the check.
* The notebook is generated from cached results; execute it with `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=steerbench notebooks/steerbench_walkthrough.ipynb` (the `steerbench` kernel is registered inside `.venv` with `python -m ipykernel install --sys-prefix --name steerbench`; the user-level `python3` kernel points at miniconda).
* Decisions to preserve: the RL action scale is a tuned agent hyperparameter (full-range actions made SAC/TD3 fail); RLS uses a bounded covariance (unbounded forgetting diverged in E4); E2X extends E2 beyond 10% to find where SVD stops winning.
* d3rlpy seeds torch globally, so offline training always runs in worker processes.

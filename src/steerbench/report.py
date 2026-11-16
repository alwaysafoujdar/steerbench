"""Findings and tables generated from cached results (for the README and notebook).

Nothing here types a number by hand: every value is read from ``results/``.
The qualitative wording of :func:`findings` is guarded by :func:`check_claims`,
which raises if a rerun changes the conclusions the sentences assert, so stale
prose cannot silently survive a new benchmark run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from steerbench.config import BenchmarkConfig

#: Methods shown in the headline figure and the README tables (in this order).
HEADLINE_METHODS: tuple[str, ...] = (
    "SVD",
    "RLS",
    "RM-SVD",
    "BO-SVDmodes",
    "SAC",
    "BC (50k)",
    "SVD (true R, reference)",
)
TABLE_METHODS: tuple[str, ...] = (
    "SVD",
    "RLS",
    "RM-SVD",
    "BO",
    "BO-SVDmodes",
    "SAC",
    "TD3",
    "Behaviour",
    "BC (10k)",
    "BC (50k)",
    "TD3+BC (10k)",
    "TD3+BC (50k)",
    "SVD (true R, reference)",
)
LEARNED = ("SAC", "TD3", "BC (50k)", "TD3+BC (50k)")


def repo_root(start: Path | None = None) -> Path:
    """Directory containing ``pyproject.toml`` (searching upwards)."""
    here = (start or Path.cwd()).resolve()
    for p in (here, *here.parents):
        if (p / "pyproject.toml").exists():
            return p
    raise FileNotFoundError("repository root (pyproject.toml) not found")


@dataclass
class Results:
    """Cached result tables of one benchmark run."""

    e1: pd.DataFrame
    e2: pd.DataFrame
    e3: pd.DataFrame
    e4: pd.DataFrame
    e5: pd.DataFrame
    rl_training: pd.DataFrame
    offline_training: pd.DataFrame
    budget: int

    @classmethod
    def load(cls, results_dir: str | Path, budget: int) -> Results:
        """Read the summary CSVs (E2 and its extension are concatenated)."""
        d = Path(results_dir)
        e2 = pd.concat([pd.read_csv(d / "e2_summary.csv"), pd.read_csv(d / "e2x_summary.csv")])
        return cls(
            e1=pd.read_csv(d / "e1_summary.csv").set_index("method"),
            e2=e2,
            e3=pd.read_csv(d / "e3_summary.csv"),
            e4=pd.read_csv(d / "e4_episodes.csv"),
            e5=pd.read_csv(d / "e5_summary.csv"),
            rl_training=pd.read_csv(d / "rl_training.csv"),
            offline_training=pd.read_csv(d / "offline_training.csv"),
            budget=budget,
        )

    # ------------------------------------------------------------ lookups
    def sweep(self, table: str, column: str, level: float, method: str, field: str) -> float:
        """One value of a sweep summary."""
        df = getattr(self, table)
        row = df[(np.isclose(df[column], level)) & (df["method"] == method)]
        if len(row) != 1:
            raise KeyError(f"{table}: no unique row for {method} at {column}={level}")
        return float(row[field].iloc[0])

    def e4_block(self, method: str, first: bool, field: str = "interactions_to_success") -> float:
        """Median of a per-episode E4 field over the first or last quarter of episodes."""
        g = self.e4[self.e4["method"] == method]
        n = g["episode"].max() + 1
        part = g[g["episode"] < n // 4] if first else g[g["episode"] >= n - n // 4]
        if field == "success":
            return float(part[field].mean())
        return float(np.quantile(part[field], 0.5, method="inverted_cdf"))


def pct(x: float) -> str:
    """Format a fraction as a whole percentage."""
    return f"{100 * x:.0f}%"


def its(x: float, budget: int) -> str:
    """Format an interactions-to-success value (infinite means not reached)."""
    return f">{budget}" if not np.isfinite(x) else f"{x:.0f}"


def check_claims(r: Results) -> None:
    """Assert the qualitative statements made by :func:`findings`.

    Raises
    ------
    AssertionError
        If a conclusion no longer holds; then the wording must be revisited.
    """
    e1 = r.e1
    best = e1.drop(index="SVD (true R, reference)")["success_rate"].max()
    assert e1.loc["SVD", "success_rate"] >= best, "SVD is no longer the most reliable at E1"
    ref = e1.loc["SVD (true R, reference)"]
    assert abs(e1.loc["SVD", "its_median"] - ref["its_median"]) <= 1, "SVD no longer matches ref"
    assert e1.loc["SVD", "success_rate"] >= ref["success_rate"] - 0.02
    svd_e2 = [
        r.sweep("e2", "focal_error_sigma", s, "SVD", "success_rate")
        for s in (0.0, 0.01, 0.03, 0.05, 0.10)
    ]
    assert min(svd_e2) >= 0.95, "SVD degrades within the specified 0-10% sweep"
    for m in ("RLS", "RM-SVD"):
        assert r.sweep("e2", "focal_error_sigma", 0.30, m, "success_rate") > r.sweep(
            "e2", "focal_error_sigma", 0.30, "SVD", "success_rate"
        ), f"{m} no longer beats SVD at 30% focal error"
    for m in ("SAC", "TD3", "BO", "BO-SVDmodes"):
        assert e1.loc[m, "success_rate"] < e1.loc["SVD", "success_rate"], f"{m} matches SVD"
    for table, column in (("e2", "focal_error_sigma"), ("e3", "masked_fraction")):
        df = getattr(r, table)
        for level in df[column].unique():
            svd = r.sweep(table, column, level, "SVD", "success_rate")
            for m in ("SAC", "TD3"):
                assert r.sweep(table, column, level, m, "success_rate") < svd, (
                    f"{m} is not less reliable than SVD at {column}={level}"
                )
    assert r.sweep("e2", "focal_error_sigma", 0.30, "SVD", "success_rate") < 0.9
    its_0, its_10 = (r.sweep("e2", "focal_error_sigma", x, "SVD", "its_median") for x in (0, 0.1))
    assert its_0 <= its_10 <= its_0 + 2, "SVD interactions no longer grow 'slightly' to 10%"
    ft, fr = (
        r.e4_block("SAC (fine-tuned)", False, "success"),
        r.e4_block("SAC (frozen)", False, "success"),
    )
    assert fr <= ft < r.e4_block("SVD", False, "success"), "fine-tuning claim no longer holds"
    for m in ("BC (50k)", "TD3+BC (50k)", "BC (10k)", "TD3+BC (10k)"):
        assert e1.loc[m, "success_rate"] > e1.loc["Behaviour", "success_rate"], (
            f"{m} no longer improves on the behaviour policy"
        )
    assert e1.loc["BC (50k)", "success_rate"] > e1.loc["TD3+BC (50k)", "success_rate"]
    assert e1.loc["BC (50k)", "success_rate"] > e1.loc["BC (10k)", "success_rate"]
    m40 = {m: r.sweep("e3", "masked_fraction", 0.40, m, "success_rate") for m in ("SVD", *LEARNED)}
    assert all(m40["SVD"] > m40[m] for m in LEARNED), "a learned method beats SVD at 40% masking"
    assert r.e4_block("RLS", first=False, field="success") >= r.e4_block(
        "SVD", first=False, field="success"
    )
    assert r.e4_block("RLS", False) < r.e4_block("SVD", False), "RLS no longer faster under drift"


def findings(r: Results) -> list[str]:
    """Three-sentence summary of the benchmark, with every number from ``r``."""
    check_claims(r)
    e1, b = r.e1, r.budget

    def f2(method: str, level: float, field: str = "success_rate") -> float:
        return r.sweep("e2", "focal_error_sigma", level, method, field)

    def m3(method: str, level: float) -> float:
        return r.sweep("e3", "masked_fraction", level, method, "success_rate")

    def ok(method: str) -> str:
        return pct(e1.loc[method, "success_rate"])

    rl_steps = int(r.rl_training["training_steps"].median())
    tune_steps = int(r.rl_training["tuning_steps"].median())
    svd_min_e2 = min(f2("SVD", x) for x in (0.0, 0.01, 0.03, 0.05, 0.10))
    e2_clause = (
        "it succeeded on every machine at every level of"
        if svd_min_e2 >= 1.0
        else f"its success rate never fell below {pct(svd_min_e2)} across"
    )
    bo_best = max(e1.loc["BO", "success_rate"], e1.loc["BO-SVDmodes", "success_rate"])
    best_learned = max(m3(m, 0.4) for m in LEARNED)
    n_ep = r.e4["episode"].max() + 1
    one = (
        f"With only the design optics model, iterative SVD correction reached the target "
        f"(normalised excess RMS below 0.05) on {ok('SVD')} of held-out machines in a median of "
        f"{its(e1.loc['SVD', 'its_median'], b)} interactions at nominal conditions, matching a "
        f"privileged true-model reference, and {e2_clause} the specified 0-10% "
        f"focal-error sweep; it only lost ground beyond that range "
        f"(at 30%: {pct(f2('SVD', 0.3))} success, median {its(f2('SVD', 0.3, 'its_median'), b)} "
        f"interactions), where adaptive RLS ({pct(f2('RLS', 0.3))}) and response measurement "
        f"({pct(f2('RM-SVD', 0.3))}, but always {its(f2('RM-SVD', 0.3, 'its_median'), b)} "
        f"interactions) held up."
    )
    two = (
        f"Online RL trained with domain randomisation ({rl_steps:,} simulated steps per agent "
        f"plus {tune_steps:,} for tuning) reached the target on only {ok('SAC')} (SAC) and "
        f"{ok('TD3')} (TD3) of machines, Bayesian optimisation on {pct(bo_best)} within {b} "
        f"interactions, and offline learning from a noisy SVD log beat its behaviour policy "
        f"({ok('Behaviour')}) mainly by imitation: behaviour cloning reached {ok('BC (50k)')} "
        f"with 50k transitions but {ok('BC (10k)')} with 10k, and TD3+BC {ok('TD3+BC (50k)')}."
    )
    three = (
        f"Missing BPMs hurt every method most: with 40% unavailable, SVD succeeded on "
        f"{pct(m3('SVD', 0.4))} of machines and no learned method on more than "
        f"{pct(best_learned)}; under {n_ep} episodes of drift, RLS kept a median of "
        f"{its(r.e4_block('RLS', False), b)} interactions while static SVD slowed from "
        f"{its(r.e4_block('SVD', True), b)} to {its(r.e4_block('SVD', False), b)}."
    )
    return [one, two, three]


def _cost(r: Results, method: str) -> str:
    """Simulation cost column of the main table."""
    if method in ("SAC", "TD3"):
        g = r.rl_training[r.rl_training["algo"] == method]
        return (
            f"{int(g['training_steps'].median()):,} env steps "
            f"(+{int(g['tuning_steps'].median()):,} tuning)"
        )
    for algo in ("BC", "TD3+BC"):
        if method.startswith(algo + " ("):
            g = r.offline_training[r.offline_training["algo"] == algo]
            size = int(method.split("(")[1].rstrip("k)")) * 1000
            steps = int(g["gradient_steps"].median())
            return f"{size:,} logged transitions, {steps:,} gradient steps"
    return "none"


def main_table(r: Results) -> str:
    """Markdown table of E1 (nominal conditions) for every method."""
    lines = [
        "| Method | Success rate [95% CI] | Interactions to success, median [IQR] "
        "| Final excess e, median | Simulation training cost |",
        "|---|---|---|---|---|",
    ]
    b = r.budget
    for m in TABLE_METHODS:
        if m not in r.e1.index:
            continue
        x = r.e1.loc[m]
        lines.append(
            f"| {m} | {pct(x.success_rate)} [{pct(x.success_ci_low)}, {pct(x.success_ci_high)}] "
            f"| {its(x.its_median, b)} [{its(x.its_q25, b)}, {its(x.its_q75, b)}] "
            f"| {x.final_excess_median:.3f} | {_cost(r, m)} |"
        )
    return "\n".join(lines)


def robustness_table(r: Results) -> str:
    """Markdown table of success rate across conditions (E2, E2X, E3, E4)."""
    cols = [
        ("focal 0%", "e2", "focal_error_sigma", 0.0),
        ("focal 10%", "e2", "focal_error_sigma", 0.10),
        ("focal 20%*", "e2", "focal_error_sigma", 0.20),
        ("focal 30%*", "e2", "focal_error_sigma", 0.30),
        ("BPMs off 25%", "e3", "masked_fraction", 0.25),
        ("BPMs off 40%", "e3", "masked_fraction", 0.40),
    ]
    head = "| Method | " + " | ".join(c[0] for c in cols) + " | drift, last quarter |"
    lines = [head, "|---" * (len(cols) + 2) + "|"]
    e4_names = {"SAC": "SAC (frozen)"}
    for m in HEADLINE_METHODS + ("TD3", "TD3+BC (50k)", "Behaviour"):
        vals = [pct(r.sweep(t, c, lv, m, "success_rate")) for _, t, c, lv in cols]
        name = e4_names.get(m, m)
        drift = pct(r.e4_block(name, False, "success")) if name in set(r.e4["method"]) else "-"
        lines.append(f"| {m} | " + " | ".join(vals) + f" | {drift} |")
    if "SAC (fine-tuned)" in set(r.e4["method"]):
        lines.append(
            "| SAC (fine-tuned online) | "
            + " | ".join("-" for _ in cols)
            + f" | {pct(r.e4_block('SAC (fine-tuned)', False, 'success'))} |"
        )
    return "\n".join(lines)


def offline_table(r: Results) -> str:
    """Markdown table of the E5 offline-data ablation (success rate)."""
    e5 = r.e5
    noises = sorted(e5["behaviour_noise"].unique())
    lines = [
        "| Method | Dataset | " + " | ".join(f"noise {1000 * z:g} urad" for z in noises) + " |",
        "|---|---|" + "---|" * len(noises),
    ]
    for (method, size), g in e5.groupby(["method", "dataset_size"], sort=False):
        if method == "Behaviour" and size != e5["dataset_size"].min():
            continue
        g = g.set_index("behaviour_noise")
        label = "(logging policy)" if method == "Behaviour" else f"{size // 1000}k"
        lines.append(
            f"| {method} | {label} | "
            + " | ".join(pct(g.loc[z, "success_rate"]) for z in noises)
            + " |"
        )
    return "\n".join(lines)


def tuned_hyperparams(results_dir: str | Path) -> str:
    """Markdown list of the tuned hyperparameters."""
    hp = json.loads((Path(results_dir) / "hyperparams.json").read_text())

    def fmt(v: object) -> str:
        return f"{v:.3g}" if isinstance(v, float) else str(v)

    return "\n".join(
        f"- **{m}**: " + ", ".join(f"{k} = {fmt(v)}" for k, v in sorted(p.items()))
        for m, p in sorted(hp.items())
    )


def hypotheses(r: Results) -> str:
    """Markdown list of hypothesis verdicts (qualitative; guarded by :func:`check_claims`)."""
    check_claims(r)
    return "\n".join(
        [
            "- **H1 (SVD with an exact model is near-optimal): supported.** With only the design "
            "model, SVD matched the privileged true-model reference at nominal conditions.",
            "- **H2 (SVD degrades as model error grows): supported only beyond the specified "
            "range.** Iterative SVD with feedback gain kept succeeding throughout the 0-10% sweep, "
            "needing slightly more interactions as the error grew; it lost reliability only at "
            "20-30% focal error.",
            "- **H3 (adaptive response estimation recovers the loss): partly supported.** Within "
            "0-10% there was no loss to recover. At 30% RLS and response measurement held up "
            "where SVD degraded, and under drift RLS kept needing fewer interactions than static "
            "SVD; response measurement always paid its fixed measurement cost.",
            "- **H4 (simulation-trained RL is competitive in deployment interactions): not "
            "supported.** SAC and TD3 were less reliable than SVD at every model-error and "
            "missing-BPM level tested, on top of a large simulation-training cost; online "
            "fine-tuning on deployment data improved SAC under drift but left it far below SVD.",
            "- **H5 (offline RL improves on its behaviour policy only with adequate data): "
            "supported, with a caveat.** Every offline agent beat the noisy logging policy, the "
            "larger dataset helped markedly, but plain behaviour cloning beat TD3+BC: the gain "
            "came from imitating (and denoising) the behaviour policy, not from RL.",
        ]
    )


def setup_block(cfg: BenchmarkConfig) -> str:
    """Markdown description of the benchmark setup, read from the configuration."""
    lat, err, lim = cfg.machine.lattice, cfg.machine.errors, cfg.machine.limits
    ev, rl, off, dr = cfg.eval, cfg.rl, cfg.offline, cfg.drift
    from steerbench.experiments import E2_FOCAL_SIGMAS, E2X_FOCAL_SIGMAS, E3_MASKED_FRACTIONS

    def levels(xs: tuple[float, ...]) -> str:
        return ", ".join(f"{100 * x:g}%" for x in xs)

    def span(xs: list[float]) -> str:
        return f"{100 * xs[0]:g}-{100 * xs[1]:g}%"

    return "\n".join(
        [
            f"- **Line:** thin-lens FODO, {lat.n_cells} cells, {lat.drift_length:g} m between "
            f"quadrupoles, {lat.phase_advance_deg:g} deg phase advance per cell; "
            f"{2 * lat.n_cells} BPMs (one per quadrupole), {lat.n_cells} correctors (one per "
            f"focusing quadrupole).",
            f"- **Errors per machine:** focal-length errors (nominal sigma "
            f"{100 * err.focal_error_sigma:g}%), BPM gains ({100 * err.bpm_gain_sigma:g}%), "
            f"corrector gains ({100 * err.corrector_gain_sigma:g}%), quadrupole offsets "
            f"({1000 * err.quad_misalignment_mm:g} um), injection offset/angle "
            f"({1000 * err.injection_offset_mm:g} um / {1000 * err.injection_angle_mrad:g} urad); "
            f"BPM noise {1000 * cfg.machine.measurement.bpm_noise_mm:g} um.",
            f"- **Limits:** |theta| <= {1000 * lim.theta_max_mrad:g} urad, "
            f"|dtheta| <= {1000 * lim.dtheta_max_mrad:g} urad per interaction.",
            f"- **Protocol:** {ev.n_test_machines} held-out test machines, "
            f"{ev.n_val_machines} separate validation machines for tuning, {ev.max_interactions} "
            f"interactions per episode, success when the normalised excess RMS drops below "
            f"{ev.success_threshold:g}; {rl.n_seeds} training seeds per learned method; "
            f"{cfg.tuning.n_trials} tuning trials per method; {ev.n_bootstrap} bootstrap "
            f"resamples over machines for 95% CIs.",
            f"- **Sweeps:** focal error {levels(E2_FOCAL_SIGMAS)} (E2) and "
            f"{levels(E2X_FOCAL_SIGMAS)} (E2X extension); unavailable BPMs "
            f"{levels(E3_MASKED_FRACTIONS)} (E3); {dr.n_episodes} drifting episodes on "
            f"{dr.n_machines} machines (E4); offline datasets "
            f"{', '.join(f'{n // 1000}k' for n in off.e5_sizes)} x behaviour noise "
            f"{', '.join(f'{1000 * z:g} urad' for z in off.e5_noises)} (E5).",
            f"- **Online RL:** SAC and TD3 (stable-baselines3), {rl.total_steps:,} steps per agent "
            f"on machines with focal-error level drawn from "
            f"{span(cfg.randomisation.focal_error_sigma)} and "
            f"{span(cfg.randomisation.masked_fraction)} of BPMs unavailable.",
        ]
    )


def runtime_block(results_dir: str | Path) -> str:
    """Wall time of the benchmark steps, from ``runtime.csv``."""
    path = Path(results_dir) / "runtime.csv"
    if not path.exists():
        return "_runtime not recorded_"
    rt = pd.read_csv(path)
    parts = ", ".join(f"{r.step} {r.seconds / 60:.1f}" for r in rt.itertuples())
    return f"Full run: {rt['seconds'].sum() / 60:.0f} min on a laptop CPU ({parts} min)."


def e0_block(results_dir: str | Path) -> str:
    """Sanity-check results (E0): oracle within limits and one-step exact SVD."""
    d = Path(results_dir)
    lim = pd.read_csv(d / "e0_oracle_limits.csv")
    ex = pd.read_csv(d / "e0_svd_exact.csv")
    ok = ex[ex["oracle_within_limits"]]
    per_level = ", ".join(
        f"{pct(r.fraction_within_limits)} at {100 * r.focal_error_sigma:g}%"
        for r in lim.itertuples()
    )
    return (
        f"- The unconstrained oracle correction lies within the corrector limit for "
        f"{per_level} focal error ({int(lim['n_machines'].iloc[0])} sampled machines per level).\n"
        f"- SVD with the exact model, no noise and gain 1 reaches the oracle residual in one "
        f"interaction on {len(ok)} of {len(ex)} test machines (largest gap "
        f"{ok['abs_gap_after_1'].max():.1e} mm); on the remaining machine(s) the oracle "
        f"needs kicks beyond the limit."
    )

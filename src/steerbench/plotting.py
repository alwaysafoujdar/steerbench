"""Figures for the benchmark. Every function returns a Matplotlib figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from steerbench.config import BenchmarkConfig
from steerbench.optics import Lattice, Twiss, build_fodo, propagate_twiss, response_matrix


def save(fig: Figure, path: str | Path) -> Path:
    """Save a figure as PNG (150 dpi), creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_lattice(lattice: Lattice, twiss: Twiss, R: np.ndarray) -> Figure:
    """Beta function, phase advance and response-matrix heatmap of a lattice."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), gridspec_kw={"width_ratios": [1.3, 1.3, 1]})
    s = twiss.s
    ax = axes[0]
    ax.plot(s, twiss.beta, "o-", color="C0", ms=3)
    qf, cq = lattice.focal_lengths > 0, lattice.corrector_quads
    ax.vlines(s[qf], 0, twiss.beta.max() * 0.06, color="C3", lw=2, label="QF (+ corrector)")
    ax.vlines(s[~qf], 0, twiss.beta.max() * 0.06, color="C2", lw=2, label="QD")
    ax.set(xlabel="s [m]", ylabel=r"$\beta_x$ [m]", title="Beta function at quadrupoles")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.set_ylim(top=twiss.beta.max() * 1.25)
    ax = axes[1]
    ax.plot(s, twiss.mu / (2 * np.pi), "o-", color="C1", ms=3)
    ax.set(xlabel="s [m]", ylabel=r"$\mu_x / 2\pi$", title="Betatron phase")
    ax = axes[2]
    lim = np.abs(R).max()
    im = ax.imshow(R, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set(xlabel="corrector j", ylabel="BPM i", title="Response matrix R [mm/mrad]")
    ax.set_xticks(range(len(cq)))
    ax.set_yticks(range(0, R.shape[0], 2))
    fig.colorbar(im, ax=ax, fraction=0.05)
    fig.tight_layout()
    return fig


def plot_trajectories(
    s: np.ndarray, before: np.ndarray, after: dict[str, np.ndarray], oracle: np.ndarray
) -> Figure:
    """True trajectory before and after correction for several methods."""
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(s, before, "o-", color="0.6", ms=3, label="before correction")
    for name, x in after.items():
        st = _style(name)
        ax.plot(s, x, "o-", ms=3, color=st["color"], label=name)
    ax.plot(s, oracle, "k--", lw=1, label="oracle (best achievable)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set(
        xlabel="s [m]",
        ylabel="true horizontal position x [mm]",
        title="One test machine: trajectory at the BPMs before and after correction",
    )
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    return fig


def plot_oracle_kicks(samples: pd.DataFrame, theta_max: float) -> Figure:
    """Distribution of the largest unconstrained oracle kick per machine."""
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for sigma, g in samples.groupby("focal_error_sigma"):
        ax.hist(
            g["max_abs_theta"],
            bins=50,
            histtype="step",
            density=True,
            label=f"focal error {100 * sigma:.0f}%",
        )
    ax.axvline(theta_max, color="k", ls="--", label=r"$\theta_{max}$")
    ax.set(
        xlabel=r"max$_j |\theta^*_j|$ [mrad] (unconstrained oracle)",
        ylabel="density",
        title="Oracle corrector strength vs limit",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


#: Fixed colour/style per method so every figure uses the same legend.
METHOD_STYLE: dict[str, dict[str, object]] = {
    "SVD": {"color": "C0", "ls": "-"},
    "RM-SVD": {"color": "C1", "ls": "-"},
    "RLS": {"color": "C2", "ls": "-"},
    "SAC": {"color": "C3", "ls": "-"},
    "TD3": {"color": "C4", "ls": "-"},
    "BO": {"color": "C5", "ls": "-"},
    "BO-SVDmodes": {"color": "C6", "ls": "-"},
    "Behaviour": {"color": "C7", "ls": ":"},
    "BC (10k)": {"color": "C8", "ls": "--"},
    "BC (50k)": {"color": "C8", "ls": "-"},
    "TD3+BC (10k)": {"color": "C9", "ls": "--"},
    "TD3+BC (50k)": {"color": "C9", "ls": "-"},
    "BC": {"color": "C8", "ls": "-"},
    "TD3+BC": {"color": "C9", "ls": "-"},
    "SAC (frozen)": {"color": "C3", "ls": "--"},
    "SAC (fine-tuned)": {"color": "C3", "ls": "-"},
    "SVD (true R, reference)": {"color": "0.4", "ls": "--"},
}


def _style(method: str) -> dict[str, object]:
    return METHOD_STYLE.get(method, {"color": "k", "ls": ":"})


def plot_excess_vs_interactions(curves: pd.DataFrame, threshold: float = 0.05) -> Figure:
    """Median (IQR band) normalised excess RMS against deployment interactions.

    Parameters
    ----------
    curves : pandas.DataFrame
        Columns ``method, interaction, median, q25, q75``.
    threshold : float
        Success threshold, drawn as a horizontal line.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for method, g in curves.groupby("method", sort=False):
        st = _style(str(method))
        ax.plot(g["interaction"], g["median"], label=method, lw=1.8, **st)
        ax.fill_between(g["interaction"], g["q25"], g["q75"], color=st["color"], alpha=0.12)
    ax.axhline(threshold, color="k", ls=":", lw=1)
    ax.text(
        curves["interaction"].max(), threshold * 1.1, "success", ha="right", va="bottom", fontsize=8
    )
    ax.set_yscale("log")
    ax.set_ylim(bottom=max(1e-3, curves["q25"].clip(lower=1e-4).min() * 0.8))
    ax.set(
        xlabel="deployment interactions",
        ylabel="normalised excess RMS e (median, IQR)",
        title="E1: correction at nominal conditions (3% focal error)",
    )
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def plot_sweep(
    summary: pd.DataFrame,
    budget: int,
    column: str,
    xlabel: str,
    title: str,
    x_factor: float = 100.0,
) -> Figure:
    """Success rate and interactions-to-success against one swept non-ideality.

    Parameters
    ----------
    summary : pandas.DataFrame
        Summary with ``column`` and ``method`` columns.
    budget : int
        Interaction budget (failed episodes are drawn just above it).
    column : str
        Swept quantity.
    xlabel, title : str
        Axis label and figure title.
    x_factor : float
        Multiplier for the x values (100 shows fractions as percent).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    groups = list(summary.groupby("method", sort=False))
    span = x_factor * (summary[column].max() - summary[column].min())
    for k, (method, g) in enumerate(groups):
        st = _style(str(method))
        # Small horizontal offsets keep overlapping methods visible.
        x = x_factor * g[column] + 0.012 * span * (k - (len(groups) - 1) / 2)
        yerr = [g["success_rate"] - g["success_ci_low"], g["success_ci_high"] - g["success_rate"]]
        axes[0].errorbar(
            x, g["success_rate"], yerr=yerr, marker="o", ms=4, capsize=3, label=method, **st
        )
        med = g["its_median"].replace(np.inf, budget + 2)
        lo = g["its_q25"].replace(np.inf, budget + 2)
        hi = g["its_q75"].replace(np.inf, budget + 2)
        axes[1].errorbar(x, med, yerr=[med - lo, hi - med], marker="o", ms=4, capsize=3, **st)
    axes[0].set(
        xlabel=xlabel,
        ylabel="success rate (95% CI)",
        ylim=(-0.03, 1.03),
        title="Success within the interaction budget",
    )
    axes[1].axhline(budget, color="k", ls=":", lw=1)
    axes[1].set(
        xlabel=xlabel,
        ylabel="interactions to success (median, IQR)",
        title=f"Interactions to e < 0.05 (above {budget} = not reached)",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 7),
        fontsize=8,
        bbox_to_anchor=(0.5, -0.12),
    )
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def plot_headline(summary: pd.DataFrame, budget: int, methods: tuple[str, ...]) -> Figure:
    """README headline: E2 plus its extension for a readable subset of methods.

    The region beyond the specified 10% sweep is shaded; RL agents were trained
    with focal errors of at most 10%, so it is outside their training range.
    """
    sub = summary[summary["method"].isin(methods)].copy()
    sub["method"] = pd.Categorical(sub["method"], categories=list(methods), ordered=True)
    sub = sub.sort_values(["method", "focal_error_sigma"])
    sub["method"] = sub["method"].astype(str)
    fig = plot_sweep(
        sub,
        budget,
        "focal_error_sigma",
        "focal-length error sigma [%]",
        "Which correction strategy holds up as the optics model degrades?",
    )
    for ax in fig.axes[:2]:
        ax.axvspan(10.5, 31.5, color="0.92", zorder=0)
    fig.text(
        0.5,
        0.955,
        "shaded: beyond the specified 0-10% sweep (RL agents were trained on focal errors "
        "up to 10%)",
        ha="center",
        fontsize=9,
        color="0.35",
    )
    return fig


def plot_model_error_sweep(summary: pd.DataFrame, budget: int) -> Figure:
    """E2 headline: success and interactions-to-success against model error."""
    return plot_sweep(
        summary,
        budget,
        "focal_error_sigma",
        "focal-length error sigma [%]",
        "E2: how each method holds up as the optics model degrades",
    )


def plot_drift(episodes: pd.DataFrame, window: int = 10) -> Figure:
    """E4: per-episode quality on drifting machines (rolling median over episodes)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for method, g in episodes.groupby("method", sort=False):
        st = _style(str(method))
        per_ep = g.groupby("episode").agg(e=("final_excess", "median"), s=("success", "mean"))
        axes[0].plot(
            per_ep.index, per_ep["e"].rolling(window, min_periods=1).median(), label=method, **st
        )
        axes[1].plot(per_ep.index, per_ep["s"].rolling(window, min_periods=1).mean(), **st)
    drift = episodes.groupby("episode")["focal_error_rms"].median()
    axes[2].plot(drift.index, 100 * drift.values, color="k")
    axes[0].set(
        yscale="log",
        xlabel="episode",
        ylabel="final excess e (median, rolling)",
        title="End-of-episode quality",
    )
    axes[0].axhline(0.05, color="k", ls=":", lw=1)
    axes[1].set(
        xlabel="episode",
        ylabel="success rate (rolling)",
        ylim=(-0.03, 1.03),
        title="Success within the episode budget",
    )
    axes[2].set(
        xlabel="episode",
        ylabel="RMS focal error [%]",
        title="Drift of the true optics (median machine)",
    )
    axes[0].legend(fontsize=8)
    fig.suptitle("E4: consecutive episodes on drifting machines", y=1.02)
    fig.tight_layout()
    return fig


def plot_offline_ablation(summary: pd.DataFrame) -> Figure:
    """E5: offline methods vs behaviour policy across dataset size and behaviour noise."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    sizes = sorted(summary["dataset_size"].unique())
    for method, g in summary.groupby("method", sort=False):
        for j, size in enumerate(sizes):
            h = g[g["dataset_size"] == size].sort_values("behaviour_noise")
            st = dict(_style(str(method)))
            if method != "Behaviour":
                st["ls"] = ["--", "-", "-."][j % 3]
            elif j > 0:
                continue  # the behaviour policy does not depend on the dataset size
            label = method if method == "Behaviour" else f"{method} ({size // 1000}k)"
            x = 1000 * h["behaviour_noise"]
            axes[0].plot(x, h["success_rate"], marker="o", label=label, **st)
            axes[1].errorbar(
                x,
                h["final_excess_median"],
                yerr=[
                    h["final_excess_median"] - h["final_excess_q25"],
                    h["final_excess_q75"] - h["final_excess_median"],
                ],
                marker="o",
                capsize=3,
                **st,
            )
    axes[0].set(
        xlabel="behaviour noise [urad]",
        ylabel="success rate",
        ylim=(-0.03, 1.03),
        title="Success within the interaction budget",
    )
    axes[1].set(
        xlabel="behaviour noise [urad]",
        ylabel="final excess e (median, IQR)",
        yscale="log",
        title="Final quality",
    )
    axes[1].axhline(0.05, color="k", ls=":", lw=1)
    axes[0].legend(fontsize=8)
    fig.suptitle("E5: offline RL against its behaviour policy", y=1.02)
    fig.tight_layout()
    return fig


def make_figures(
    cfg: BenchmarkConfig, results_dir: Path, figures_dir: Path, experiments: list[str]
) -> list[Path]:
    """Build every figure for the given experiments from cached result CSVs.

    Parameters
    ----------
    cfg : BenchmarkConfig
        Configuration used for the run (lattice geometry and limits).
    results_dir, figures_dir : Path
        Input and output directories.
    experiments : list of str
        Experiments whose figures to build.

    Returns
    -------
    list of Path
        Written figure files.
    """
    lattice = build_fodo(cfg.machine.lattice)
    twiss = propagate_twiss(lattice)
    out = [
        save(
            plot_lattice(lattice, twiss, response_matrix(lattice, twiss)),
            figures_dir / "lattice.png",
        )
    ]
    if "E0" in experiments:
        samples = pd.read_csv(results_dir / "e0_oracle_samples.csv")
        fig = plot_oracle_kicks(samples, cfg.machine.limits.theta_max_mrad)
        out.append(save(fig, figures_dir / "e0_oracle_kicks.png"))
    if "E1" in experiments:
        curves = pd.read_csv(results_dir / "e1_curves.csv")
        fig = plot_excess_vs_interactions(curves, cfg.eval.success_threshold)
        out.append(save(fig, figures_dir / "e1_excess_vs_interactions.png"))
    budget = cfg.eval.max_interactions
    if "E2" in experiments:
        summary = pd.read_csv(results_dir / "e2_summary.csv")
        out.append(
            save(plot_model_error_sweep(summary, budget), figures_dir / "e2_model_error.png")
        )
    if "E2X" in experiments and (results_dir / "e2_summary.csv").exists():
        both = pd.concat(
            [pd.read_csv(results_dir / f) for f in ("e2_summary.csv", "e2x_summary.csv")]
        )
        fig = plot_sweep(
            both,
            budget,
            "focal_error_sigma",
            "focal-length error sigma [%]",
            "E2 extended: model error beyond the specified 10% (RL trained on <= 10%)",
        )
        out.append(save(fig, figures_dir / "e2x_model_error_extended.png"))
        from steerbench.report import HEADLINE_METHODS

        fig = plot_headline(both, budget, HEADLINE_METHODS)
        out.append(save(fig, figures_dir / "headline_model_error.png"))
    if "E3" in experiments:
        summary = pd.read_csv(results_dir / "e3_summary.csv")
        fig = plot_sweep(
            summary,
            budget,
            "masked_fraction",
            "unavailable BPMs [%]",
            "E3: correction with missing BPMs (3% focal error)",
        )
        out.append(save(fig, figures_dir / "e3_missing_bpms.png"))
    if "E4" in experiments:
        episodes = pd.read_csv(results_dir / "e4_episodes.csv")
        out.append(save(plot_drift(episodes), figures_dir / "e4_drift.png"))
    if "E5" in experiments:
        summary = pd.read_csv(results_dir / "e5_summary.csv")
        out.append(save(plot_offline_ablation(summary), figures_dir / "e5_offline_ablation.png"))
    return out

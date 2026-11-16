"""Run the steerbench experiments listed in a config and write results and figures.

Usage::

    python scripts/run_benchmark.py --config configs/quick.yaml
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from steerbench.config import load_config, to_dict  # noqa: E402
from steerbench.experiments import EXPERIMENTS  # noqa: E402
from steerbench.plotting import make_figures  # noqa: E402

logger = logging.getLogger("steerbench.run")


def record_runtime(path: Path, step: str, seconds: float) -> None:
    """Store the wall time of a step, replacing any earlier entry for it."""
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["step", "seconds"])
    new = pd.concat([old[old["step"] != step], pd.DataFrame([{"step": step, "seconds": seconds}])])
    new.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--only", nargs="*", help="subset of experiments to run, e.g. E0 E1")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    results_dir, figures_dir = Path(cfg.results_dir), Path(cfg.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "config_used.yaml", "w") as fh:
        yaml.safe_dump(to_dict(cfg), fh, sort_keys=False)

    todo = args.only or cfg.experiments
    t_all = time.perf_counter()
    runtime_path = results_dir / "runtime.csv"
    for name in todo:
        if name not in EXPERIMENTS:
            raise SystemExit(f"unknown experiment {name}; available: {sorted(EXPERIMENTS)}")
        t0 = time.perf_counter()
        logger.info("running %s", name)
        tables = EXPERIMENTS[name](cfg)
        for key, df in tables.items():
            df.to_csv(results_dir / f"{key}.csv", index=False)
        seconds = time.perf_counter() - t0
        logger.info("%s finished in %.1f s", name, seconds)
        record_runtime(runtime_path, name, seconds)
    for path in make_figures(cfg, results_dir, figures_dir, todo):
        logger.info("wrote %s", path)
    logger.info("benchmark finished in %.1f s", time.perf_counter() - t_all)


if __name__ == "__main__":
    main()

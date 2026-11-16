"""Regenerate the result blocks of README.md from cached results.

Every number in the README's generated blocks comes from ``results/`` through
:mod:`steerbench.report`; the script fails if the results no longer support the
qualitative statements in those blocks.

Usage::

    python scripts/make_readme_table.py [--config configs/full.yaml] [--readme README.md]
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

from steerbench.config import load_config
from steerbench.report import (
    Results,
    e0_block,
    findings,
    hypotheses,
    main_table,
    offline_table,
    robustness_table,
    runtime_block,
    setup_block,
    tuned_hyperparams,
)

logger = logging.getLogger("steerbench.readme")


def replace_block(text: str, name: str, body: str) -> str:
    """Replace the content between ``<!-- BEGIN:name -->`` and ``<!-- END:name -->``."""
    pattern = re.compile(rf"(<!-- BEGIN:{name} -->)\n.*?(<!-- END:{name} -->)", re.S)
    if not pattern.search(text):
        raise KeyError(f"marker block {name!r} not found in README")
    return pattern.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(2)}", text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/full.yaml"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    r = Results.load(cfg.results_dir, cfg.eval.max_interactions)
    blocks = {
        "summary": " ".join(findings(r)),
        "hypotheses": hypotheses(r),
        "table-main": main_table(r),
        "table-robustness": robustness_table(r),
        "table-offline": offline_table(r),
        "hyperparams": tuned_hyperparams(cfg.results_dir),
        "setup": setup_block(cfg),
        "runtime": runtime_block(cfg.results_dir),
        "e0": e0_block(cfg.results_dir),
    }
    text = args.readme.read_text()
    for name, body in blocks.items():
        text = replace_block(text, name, body)
    args.readme.write_text(text)
    logger.info("updated %d blocks in %s", len(blocks), args.readme)


if __name__ == "__main__":
    main()

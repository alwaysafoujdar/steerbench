import importlib.util
from pathlib import Path

import numpy as np
import pytest

from steerbench.report import its, pct, repo_root

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location("mrt", ROOT / "scripts" / "make_readme_table.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_replace_block_is_idempotent_and_strict():
    mrt = load_script()
    text = "a\n<!-- BEGIN:x -->\n<!-- END:x -->\nb\n"
    once = mrt.replace_block(text, "x", "| t |")
    assert once == "a\n<!-- BEGIN:x -->\n| t |\n<!-- END:x -->\nb\n"
    assert mrt.replace_block(once, "x", "| t |") == once
    assert "new" in mrt.replace_block(once, "x", "new")
    with pytest.raises(KeyError):
        mrt.replace_block(text, "missing", "body")


def test_formatting_helpers():
    assert pct(0.784) == "78%"
    assert its(np.inf, 30) == ">30"
    assert its(11.0, 30) == "11"


def test_repo_root_found_from_subdirectory():
    assert repo_root(ROOT / "src" / "steerbench") == ROOT

#!/usr/bin/env python
"""Run the preregistered structural-only Uniform-rooted bootstrap audit."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit import run_uniform_bootstrap_audit  # noqa: E402


if __name__ == "__main__":
    run_uniform_bootstrap_audit()


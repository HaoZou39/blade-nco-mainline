#!/usr/bin/env python
"""Compatibility entry point for the resumable external-v2 standard sweep."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).with_name("external_v2_oracle_standard_sweep.py")
    runpy.run_path(str(target), run_name="__main__")

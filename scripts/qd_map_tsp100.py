"""Stable entry point for D3-CVT-256 working map v1.

The mapping math lives in the versioned package. This file only selects that
package and supplies its manifest when --map is omitted.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAPPER = ROOT / "assets/qd_maps/d3-cvt256-v1/mapping/map_tsp100.py"
DEFAULT_MAP = ROOT / "assets/qd_maps/d3-cvt256-v1/map_manifest.json"


def main() -> None:
    if not MAPPER.is_file():
        raise SystemExit(f"frozen mapper is missing: {MAPPER}")
    if "--map" not in sys.argv:
        sys.argv[1:1] = ["--map", str(DEFAULT_MAP)]
    runpy.run_path(str(MAPPER), run_name="__main__")


if __name__ == "__main__":
    main()

"""
build_v5.py -- Phase 1: build v5 dataset variants.

Variants (config.DATASET_VARIANTS; v4_legacy is NOT rebuilt -- frozen):
  v5a: ohmic-corrected labels + first-sample decimation
  v5b: legacy labels        + mean-per-second decimation
  v5c: ohmic-corrected labels + mean-per-second decimation   <- v5 final

Window/stride/features/scenario-split logic identical to v4 (documented).
Outputs land in data/processed/<variant>_scenario_<X>; nothing legacy touched.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
for p in (str(BASE_DIR), str(BASE_DIR / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import DATASET_VARIANTS  # noqa: E402
from preprocessing_v4 import run_pipeline_v4  # noqa: E402

VARIANTS = ("v5a", "v5b", "v5c")


def main() -> None:
    t0 = time.time()
    for variant in VARIANTS:
        spec = DATASET_VARIANTS[variant]
        for scenario in ("A", "B"):
            out = BASE_DIR / "data" / "processed" / f"{variant}_scenario_{scenario}"
            if (out / "metadata_v4.json").exists():
                print(f"[skip] {out.name} already built")
                continue
            run_pipeline_v4(
                scenario=scenario,
                label_mode=spec["label_mode"],
                decimation_mode=spec["decimation_mode"],
                variant_name=variant,
            )
    print(f"\nAll v5 variants built in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()

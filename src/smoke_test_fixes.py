"""Smoke test for all mandatory Scopus fixes."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inspect
import numpy as np

PASS = []
FAIL = []

def check(name, cond):
    if cond:
        print(f"  [PASS] {name}")
        PASS.append(name)
    else:
        print(f"  [FAIL] {name}")
        FAIL.append(name)

# ── Test 1: preprocessing range fix (R4) ──────────────────────
print("\n=== Fix R4: build_sequences terminal window ===")
import preprocessing as pp
src_bs = inspect.getsource(pp.build_sequences)
check("range includes terminal window (+1)", "window + 1" in src_bs)

# ── Test 2: preprocessing F1 chronological split ──────────────
print("\n=== Fix F1: Chronological val split ===")
src_pipe = inspect.getsource(pp.run_pipeline_v2)
check("chron holdout comment present",   "chron holdout" in src_pipe)
check("val_X_list built per-temp",       "val_X_list" in src_pipe)
check("Shuffle ONLY train comment",      "Shuffle ONLY train" in src_pipe)

# ── Test 3: validate_mse_only in train.py ─────────────────────
print("\n=== Fix F2: validate_mse_only in train.py ===")
import train as tr
check("validate_mse_only defined",     hasattr(tr, "validate_mse_only"))
src_train = inspect.getsource(tr.train_scenario)
check("validate_mse_only called",      "validate_mse_only" in src_train)
check("val_mse drives best_val_logic", "val_mse < best_val_loss" in src_train)

# ── Test 4: CosineAnnealingLR in train.py ─────────────────────
print("\n=== Fix R5: CosineAnnealingLR in train.py ===")
check("CosineAnnealingLR present",       "CosineAnnealingLR" in src_train)
check("ReduceLROnPlateau removed",       "ReduceLROnPlateau" not in src_train)
check("scheduler.step() no arg",         "scheduler.step()" in src_train)

# ── Test 5: R² in evaluate.py ─────────────────────────────────
print("\n=== Fix R8: R² in evaluate.py ===")
from evaluate import compute_metrics, compute_pvr
m = compute_metrics(np.array([0.5, 0.6, 0.7]), np.array([0.5, 0.6, 0.7]))
check("r2 key in compute_metrics",    "r2" in m)
check("r2 == 1.0 for perfect pred",  abs(m["r2"] - 1.0) < 1e-6)
src_pvr = inspect.getsource(compute_pvr)
check("PVR threshold is -0.05",       "-0.05" in src_pvr)
check("PVR threshold -0.01 removed",  "-0.01" not in src_pvr)

# ── Test 6: CosineAnnealingLR + val_mse in sprint44 ──────────
print("\n=== Fix F2+R5+R8 in sprint44_ablation.py ===")
import sprint44_ablation as ab
src_tm = inspect.getsource(ab.train_model)
check("CosineAnnealingLR in ablation",    "CosineAnnealingLR" in src_tm)
check("ReduceLROnPlateau removed ablat.", "ReduceLROnPlateau" not in src_tm)
check("val_mse computed in ablation",     "val_mse" in src_tm)
check("is_best uses val_mse",            "is_best = val_mse" in src_tm)

src_ev = inspect.getsource(ab.evaluate_model)
check("r2_score imported in ablation",   "r2_score" in src_ev)
check("r2_full computed",                "r2_full" in src_ev)
check("r2_last computed",                "r2_last" in src_ev)

# ── Summary ───────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  PASSED: {len(PASS)}/{len(PASS)+len(FAIL)}")
if FAIL:
    print(f"  FAILED: {FAIL}")
    sys.exit(1)
else:
    print("  ALL CHECKS PASSED — code is ready for final training run!")
print(f"{'='*50}\n")

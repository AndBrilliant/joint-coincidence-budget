#!/usr/bin/env python3
"""
v0.2 PRODUCTION RUNNER — Joint Coincidence Budget
ENGINE_ID: amb
Runs all 12 cells with v0.2 fixes (wider T1 r-range, support gate).
"""
import numpy as np
import json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from joint_engine_amb import *

OUTDIR = "results/amb-20260811-v0.2"
OUTDIR_VARIANT = "results/amb-314160-v0.2"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(OUTDIR_VARIANT, exist_ok=True)

# ── Support gate ──
print("=" * 70)
print("v0.2 PRODUCTION — amb")
print(f"Production seed: 20260811 | Variant seed: 314160 | Calibration: 271828")
print("=" * 70)
assert check_support_gate(), "SUPPORT GATE FAILED!"

# ═══════════════════════════════════════════════════════════
# PHASE 1: Singletons at N=1e7 for cascade ordering
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 1: SINGLETONS (N=1e7) — cascade ordering")
print("=" * 70)

singleton_results = {}

for condition in ["T0", "T1"]:
    for prior in ["logU", "logN", "linU"]:
        seed = 20260811 if prior == "logU" else 314160
        outdir = OUTDIR if prior == "logU" else OUTDIR_VARIANT
        key = f"{condition}_{prior}"

        print(f"\n  Measuring singletons: {key} (seed={seed})")
        rng = np.random.default_rng(seed)
        result, N_eff = measure_singletons(rng, 10_000_000, prior, condition)

        singleton_results[key] = result
        outpath = os.path.join(outdir, f"singletons_{condition}_{prior}_seed{seed}.json")
        output = {
            "engine_id": "amb",
            "version": "v0.2",
            "seed": seed, "N_target": 10_000_000, "N_eff": int(N_eff),
            "prior": prior, "condition": condition, "results": result,
        }
        with open(outpath, "w") as f:
            json.dump(output, f, indent=2)
        print(f"    Saved: {outpath}")

# ═══════════════════════════════════════════════════════════
# PHASE 2: Tier-1 cascades for all 12 cells
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 2: TIER-1 CASCADES (12 cells)")
print("=" * 70)

all_cell_results = {}

for condition in ["T0", "T1"]:
    for u1_mode in ["fixed", "menu"]:
        for prior in ["logU", "logN", "linU"]:
            cell_key = f"{condition}_{u1_mode}_{prior}"
            seed = 20260811 if prior == "logU" else 314160
            N_target = 2_000_000_000 if prior == "logU" else 200_000_000
            outdir = OUTDIR if prior == "logU" else OUTDIR_VARIANT

            # Determine cascade order from singletons
            skey = f"{condition}_{prior}"
            rates = [(c, singleton_results[skey][c]["rate"])
                     for c in ALL_CLAIMS if singleton_results[skey][c]["rate"] > 0]
            # Remove L1 from T1 cascade (granted)
            if condition == "T1":
                rates = [(c, r) for c, r in rates if c != "L1"]
            # Remove wrong U1 variant
            if u1_mode == "fixed":
                rates = [(c, r) for c, r in rates if c != "U1_menu"]
            else:
                rates = [(c, r) for c, r in rates if c != "U1_fixed"]

            rates.sort(key=lambda x: x[1])  # rarest first
            claim_order = [c for c, _ in rates]

            print(f"\n  CELL: {cell_key} | seed={seed} | N>={N_target:,}")
            print(f"  Cascade order: {claim_order}")

            cell_rng = np.random.default_rng(seed)
            t0 = time.time()
            hits, N_eff, stage_counts = run_cascade(
                cell_rng, N_target, prior, condition, u1_mode, claim_order
            )
            elapsed = time.time() - t0

            f_rate = hits / N_eff if N_eff > 0 else 0.0
            lo, hi = clopper_pearson(int(hits), int(N_eff))
            label = "POINT" if hits >= 100 else "BOUND"

            print(f"  RESULT: {hits}/{N_eff} = {f_rate:.6e} CP95 [{lo:.6e}, {hi:.6e}] {label}")
            print(f"  Stage counts: {stage_counts}")
            print(f"  Elapsed: {elapsed:.1f}s, Rate: {N_eff/elapsed:.0f}/s")

            cell_result = {
                "engine_id": "amb", "version": "v0.2",
                "condition": condition, "u1_mode": u1_mode, "prior": prior,
                "seed": seed, "N_eff": int(N_eff), "N_eff_target": N_target,
                "hits": int(hits), "rate": float(f_rate),
                "cp95_lower": lo, "cp95_upper": hi,
                "cascade_order": claim_order,
                "stage_counts": {k: int(v) for k, v in stage_counts.items()},
                "tier": "Tier-1", "label": label,
            }
            all_cell_results[cell_key] = cell_result

            cell_path = os.path.join(outdir, f"cell_{cell_key}.json")
            with open(cell_path, "w") as f:
                json.dump(cell_result, f, indent=2)

            # Checkpoint mc_counts
            ckpt_path = os.path.join(outdir, "mc_counts.json")
            with open(ckpt_path, "w") as f:
                json.dump(all_cell_results, f, indent=2)

# ═══════════════════════════════════════════════════════════
# PHASE 3: Tier-2 adaptive calibration
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 3: TIER-2 ADAPTIVE-INFLATION CALIBRATION")
print("=" * 70)

calib_results, tier2_eligible = tier2_adaptive_inflation_calibrate(
    None, prior="logU", condition="T0",
    calibration_seed=271828, N_calib=100_000_000
)

os.makedirs("results/calibration-v0.2", exist_ok=True)
calib_path = "results/calibration-v0.2/adaptive_inflation_seed271828.json"
with open(calib_path, "w") as f:
    json.dump({
        "engine_id": "amb", "version": "v0.2",
        "calibration_seed": 271828, "N_calib": 100_000_000,
        "stage_pairs": calib_results,
        "tier2_eligible": tier2_eligible,
    }, f, indent=2)
print(f"Saved: {calib_path}")

# ═══════════════════════════════════════════════════════════
# PHASE 4: Summary
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("RESULTS SUMMARY — v0.2")
print("=" * 80)
print(f"{'Cell':<25} {'Hits':>10} {'N_eff':>14} {'Rate':>14} {'Label':>8} {'Tier':>8}")
print(f"{'─'*25} {'─'*10} {'─'*14} {'─'*14} {'─'*8} {'─'*8}")
for cell_key in sorted(all_cell_results.keys()):
    r = all_cell_results[cell_key]
    tier = r.get("tier", "Tier-1")
    print(f"{cell_key:<25} {r['hits']:>10} {r['N_eff']:>14,} "
          f"{r['rate']:>14.6e} {r['label']:>8} {tier:>8}")

# Determine headline cell
print(f"\nTier-2 eligible: {tier2_eligible}")
print(f"\nHeadline cell: T1_menu_logU (most conservative: widest tolerance × largest menu × most generous grant)")

# Save combined summary
summary_path = "results/all_cells_summary_v0.2.json"
with open(summary_path, "w") as f:
    json.dump({
        "version": "v0.2",
        "engine_id": "amb",
        "cells": all_cell_results,
        "tier2_calibration": calib_results,
        "tier2_eligible": tier2_eligible,
        "singletons": singleton_results,
    }, f, indent=2)
print(f"Saved: {summary_path}")
print("\nDONE")

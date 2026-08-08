#!/usr/bin/env python3
"""
PRIORS AND VARIANTS — v0.2 Calibration and validation utilities.
ENGINE_ID: amb
Calibration seed: 271828

v0.2: Tier-2 uses adaptive per-stage inflation calibration (§8 redesign).
Validates per stage-pair (L2^L3, +Q1, +Q2, +U1) rather than only at full joint.
Inflates tolerances by smallest factor in {1e2, 1e3, 1e4} reaching >=100 hits.
If no factor reaches 100 hits for a stage-pair, Tier-2 is ineligible for those cells.
"""

import numpy as np
import json
import os
import sys
import time
from scipy.stats import beta as beta_dist

# Import shared functions from main engine
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from joint_engine_amb import (
    kdist, Q_U, clopper_pearson,
    draw_mass_triple, draw_mass_pair, sample_t1_koide,
    check_L1, check_L2, check_L3, check_Q1, check_Q2,
    check_U1_fixed, check_U1_menu, check_support_gate,
    tier2_adaptive_inflation_calibrate,
    LEP_LO, LEP_HI, QUARK_LO, QUARK_HI,
)

# ═══════════════════════════════════════════════════════════════════════
# CALIBRATION CELL: all tolerances ×100
# ═══════════════════════════════════════════════════════════════════════

# Standard tolerances × 100
L1_TOL_X100 = 3.3049e-6 * 100   # = 3.3049e-4
L2_TOL_X100 = 1.00e-5 * 100     # = 1.00e-3
L3_TOL_X100 = 2.10e-5 * 100     # = 2.10e-3
B1_X100 = 3.00e-3 * 100         # = 0.30
B2_X100 = 1.18e-2 * 100         # = 1.18
U1_TOL_X100 = 1.1414e-2 * 100   # = 1.1414


def check_L1_x100(leptons):
    return kdist(leptons) <= L1_TOL_X100


def check_L2_x100(leptons):
    ratio = leptons[:, 1] / leptons[:, 0]
    return np.abs(ratio / 206.7703 - 1.0) <= L2_TOL_X100


def check_L3_x100(leptons):
    ratio = leptons[:, 2] / leptons[:, 1]
    return np.abs(ratio / 16.8180 - 1.0) <= L3_TOL_X100


def check_Q1_x100(light_q, leptons):
    mu, md, ms = light_q[:, 0], light_q[:, 1], light_q[:, 2]
    mu_star = leptons.sum(axis=1)
    val = np.log(ms * ms / (mu_star * md))
    return np.abs(val) <= B1_X100


def check_Q2_x100(light_q, leptons):
    mu, md = light_q[:, 0], light_q[:, 1]
    twome = 2.0 * leptons.min(axis=1)
    val = np.log(mu * mu / (md * twome))
    return np.abs(val) <= B2_X100


def check_U1_fixed_x100(light_q, up_s):
    mu, mc, mt = light_q[:, 0], up_s[:, 0], up_s[:, 1]
    triples = np.column_stack([mu, mc, mt])
    q_dir = Q_U(triples)
    q_inv = Q_U(1.0 / triples)
    best = np.minimum(np.abs(9.0 * q_dir - 8.0), np.abs(9.0 * q_inv - 8.0))
    return best <= U1_TOL_X100


def run_calibration_brute_force(seed=271828, N=100_000_000, prior="logU", condition="T0"):
    """Brute-force Tier-1 cascade at calibration cell (all tols ×100).
    Returns (hits, N_eff) for joint of all claims.
    """
    batch_size = 1_000_000
    n_batches = N // batch_size
    rng = np.random.default_rng(seed)

    claims = [
        ("L2", check_L2_x100),
        ("L3", check_L3_x100),
        ("L1", check_L1_x100),
        ("Q1", check_Q1_x100),
        ("Q2", check_Q2_x100),
        ("U1", check_U1_fixed_x100),
    ]

    total_eff = 0
    total_hits = 0
    stage_counts = {c[0]: 0 for c in claims}

    t0 = time.time()
    for b in range(n_batches):
        if condition == "T0":
            leptons = draw_mass_triple(rng, batch_size, LEP_LO, LEP_HI, prior, sort=True)
            n_acc = batch_size
            total_eff += batch_size
        else:
            leptons, attempted = sample_t1_koide(rng, batch_size)
            total_eff += attempted
            n_acc = len(leptons)
            if n_acc == 0:
                continue

        light_q = draw_mass_triple(rng, n_acc, QUARK_LO, QUARK_HI, prior, sort=True)
        up_s = draw_mass_pair(rng, n_acc, QUARK_LO, QUARK_HI, prior)

        survivors = np.ones(n_acc, dtype=bool)
        for cname, func in claims:
            if cname in ("L1", "L2", "L3"):
                mask = func(leptons[survivors])
            elif cname in ("Q1", "Q2"):
                mask = func(light_q[survivors], leptons[survivors])
            else:
                mask = func(light_q[survivors], up_s[survivors])

            survivors_indices = np.where(survivors)[0]
            survivors[survivors_indices[~mask]] = False
            stage_counts[cname] += mask.sum()

        total_hits += survivors.sum()

        if (b + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = total_eff / elapsed if elapsed > 0 else 0
            print(f"  [calib/{condition}/{prior}] batch {b+1}/{n_batches}, "
                  f"N_eff={total_eff:,}, hits={total_hits}, "
                  f"stages={stage_counts}", flush=True)

    elapsed = time.time() - t0
    print(f"  [calib] COMPLETE: N_eff={total_eff:,}, hits={total_hits}, "
          f"elapsed={elapsed:.1f}s, rate={total_eff/elapsed:.0f}/s")

    return total_hits, total_eff, stage_counts


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="calibrate_v2",
                        choices=["calibrate", "calibrate_v2", "tier2_validate"])
    parser.add_argument("--N", type=int, default=100_000_000)
    parser.add_argument("--seed", type=int, default=271828)
    args = parser.parse_args()

    if args.mode == "calibrate_v2":
        # v0.2 adaptive-inflation calibration
        print(f"{'='*70}")
        print(f"v0.2 ADAPTIVE-INFLATION TIER-2 CALIBRATION")
        print(f"Seed: {args.seed}, N: {args.N:,}")
        print(f"{'='*70}\n")

        check_support_gate()

        calib_results, eligible = tier2_adaptive_inflation_calibrate(
            None, prior="logU", condition="T0",
            calibration_seed=args.seed, N_calib=args.N
        )

        os.makedirs("results/calibration-v0.2", exist_ok=True)
        outpath = f"results/calibration-v0.2/adaptive_inflation_seed{args.seed}.json"
        result = {
            "engine_id": "amb",
            "version": "v0.2",
            "calibration_seed": args.seed,
            "N_calib": args.N,
            "stage_pairs": calib_results,
            "tier2_eligible": eligible,
        }
        with open(outpath, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved: {outpath}")
        print(f"Tier-2 eligible: {eligible}")

    elif args.mode == "calibrate":
        print(f"{'='*70}")
        print(f"LEGACY CALIBRATION CELL — Tolerances ×100 (v0.1)")
        print(f"Seed: {args.seed}, N: {args.N:,}")
        print(f"{'='*70}\n")

        hits, N_eff, stage_counts = run_calibration_brute_force(
            args.seed, args.N, "logU", "T0"
        )

        f_rate = hits / N_eff if N_eff > 0 else 0.0
        lo, hi = clopper_pearson(int(hits), int(N_eff))
        print(f"\nCalibration joint: {hits}/{N_eff} = {f_rate:.6e}")
        print(f"CP95: [{lo:.6e}, {hi:.6e}]")
        print(f"Stage counts: {stage_counts}")

        result = {
            "engine_id": "amb",
            "calibration_seed": args.seed,
            "N": int(N_eff),
            "hits": int(hits),
            "rate": float(f_rate),
            "cp95_lower": lo,
            "cp95_upper": hi,
            "stage_counts": {k: int(v) for k, v in stage_counts.items()},
            "tolerances_x100": True,
        }

        os.makedirs("results/calibration", exist_ok=True)
        outpath = f"results/calibration/brute_force_seed{args.seed}_N{args.N}.json"
        with open(outpath, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved: {outpath}")

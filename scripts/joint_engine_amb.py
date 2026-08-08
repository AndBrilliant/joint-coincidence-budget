#!/usr/bin/env python3
"""
JOINT COINCIDENCE BUDGET — Full Engine v0.2
ENGINE_ID: amb
Production seed: 20260811 | Prior-variant seed: 314160 | Calibration seed: 271828

Implements all 12 cells: {T0,T1} x {U1-fixed,U1-menu} x {P-logU,P-logN,P-linU}
Tier-1 cascade: singletons → cascade ordering → joint measurement.

v0.2 changes from v0.1:
- T1 sheet-sampler r-range widened to [1e-5, 1e-1] (was [1e-3, 1e-1])
- Support gate: verifies r_obs ∈ sampler support before production
- Output dirs keyed by version: results/<ENGINE_ID>-<seed>-v0.2/
"""

import numpy as np
import json
import os
import sys
import time
from scipy.stats import beta as beta_dist

# ═══════════════════════════════════════════════════════════════════════
# KDISK AND Q_U
# ═══════════════════════════════════════════════════════════════════════

ang = 2 * np.pi * np.arange(3) / 3
cos_ang = np.cos(ang)
sin_ang = np.sin(ang)


def kdist(m):
    """Koide distance. m shape (..., 3). Frame union: best of {sqrt, 1/sqrt}."""
    m = np.asarray(m)
    out = None
    for v in (np.sqrt(m), 1.0 / np.sqrt(m)):
        A = v.mean(axis=-1)
        X = (2.0 / 3.0) * (v * cos_ang).sum(axis=-1)
        Y = -(2.0 / 3.0) * (v * sin_ang).sum(axis=-1)
        d = np.abs(np.hypot(X, Y) / (np.sqrt(2.0) * A) - 1.0)
        out = d if out is None else np.minimum(out, d)
    return out


def Q_U(v):
    """Q_U[v] = sum(v) / sum(sqrt(v))^2. v shape (..., 3)."""
    v = np.asarray(v)
    return np.sum(v, axis=-1) / np.sum(np.sqrt(v), axis=-1) ** 2


def clopper_pearson(k, n, alpha=0.05):
    """CP95 interval. k=int, n=int."""
    if k <= 0:
        lower = 0.0
    else:
        lower = beta_dist.ppf(alpha / 2.0, k, n - k + 1)
    if k >= n:
        upper = 1.0
    else:
        upper = beta_dist.ppf(1.0 - alpha / 2.0, k + 1, n - k)
    return float(lower), float(upper)


# ═══════════════════════════════════════════════════════════════════════
# FROZEN CONSTANTS (§4, §7)
# ═══════════════════════════════════════════════════════════════════════

LEPTONS_OBS = np.array([0.51099895, 105.6583755, 1776.93])
L1_TOL = 3.3049e-6
L2_TARGET = 206.7703
L2_TOL = 1.00e-5
L3_TARGET = 16.8180
L3_TOL = 2.10e-5
B1 = 3.00e-3   # Q1 tolerance
B2 = 1.18e-2   # Q2 tolerance
U1_TOL = 1.1414e-2  # U1 linear window on 9Q (not log)

# U1-menu: 19 irreducible p/q with p,q <= 9 in (1/3, 1]
# Verified enumeration in ASSUMPTIONS.md §A5
U1_MENU = [
    (1, 1), (1, 2), (2, 3), (3, 4),
    (2, 5), (3, 5), (4, 5),
    (5, 6),
    (3, 7), (4, 7), (5, 7), (6, 7),
    (3, 8), (5, 8), (7, 8),
    (4, 9), (5, 9), (7, 9), (8, 9),
]
U1_MENU_TARGETS = np.array([9.0 * p / q for (p, q) in U1_MENU], dtype=np.float64)

# Ranges (MeV)
LEP_LO, LEP_HI = 0.3, 2000.0
QUARK_LO, QUARK_HI = 0.5, 2e5

# Hierarchy cutoff for T1: (4+sqrt(18))^2
HIERARCHY_MIN = (4.0 + np.sqrt(18.0)) ** 2  # ≈ 67.9


# ═══════════════════════════════════════════════════════════════════════
# DRAW FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def draw_mass_triple(rng, batch_size, lo, hi, prior, sort=True):
    """Draw a mass triple [lo, hi] under prior, optionally sort ascending."""
    if prior == "logU":
        x = np.exp(rng.uniform(np.log(lo), np.log(hi), size=(batch_size, 3)))
    elif prior == "logN":
        mid = np.exp((np.log(lo) + np.log(hi)) / 2.0)
        sigma_decades = 1.5
        x = np.exp(rng.normal(np.log(mid), sigma_decades * np.log(10),
                              size=(batch_size, 3)))
        # Clip to range
        x = np.clip(x, lo, hi)
    elif prior == "linU":
        x = rng.uniform(lo, hi, size=(batch_size, 3))
    else:
        raise ValueError(f"Unknown prior: {prior}")
    if sort:
        x.sort(axis=1)
    return x


def draw_mass_pair(rng, batch_size, lo, hi, prior):
    """Draw a mass pair (unsorted) under prior."""
    if prior == "logU":
        x = np.exp(rng.uniform(np.log(lo), np.log(hi), size=(batch_size, 2)))
    elif prior == "logN":
        mid = np.exp((np.log(lo) + np.log(hi)) / 2.0)
        sigma_decades = 1.5
        x = np.exp(rng.normal(np.log(mid), sigma_decades * np.log(10),
                              size=(batch_size, 2)))
        x = np.clip(x, lo, hi)
    elif prior == "linU":
        x = rng.uniform(lo, hi, size=(batch_size, 2))
    else:
        raise ValueError(f"Unknown prior: {prior}")
    return x


def check_support_gate():
    """§5 SUPPORT GATE: verify observed lepton ratio lies in sampler support.
    r_obs = m_e / m_tau = 0.51099895 / 1776.93 = 2.876e-4
    Must satisfy: 1e-5 <= r_obs <= 1e-1 AND pass branch admissibility test.
    Returns True if gate passes, prints and returns False on failure.
    """
    r_obs = 0.51099895 / 1776.93
    r_lo, r_hi = 1e-5, 1e-1

    if not (r_lo <= r_obs <= r_hi):
        print(f"*** SUPPORT GATE FAILED: r_obs={r_obs:.6e} not in [{r_lo}, {r_hi}]")
        return False

    # Branch admissibility: m3/m1 > (4+sqrt(18))^2
    hierarchy_ratio = 1776.93 / 0.51099895
    if hierarchy_ratio <= HIERARCHY_MIN:
        print(f"*** SUPPORT GATE FAILED: m3/m1={hierarchy_ratio:.1f} <= {HIERARCHY_MIN:.1f}")
        return False

    print(f"SUPPORT GATE PASSED: r_obs={r_obs:.6e} in [{r_lo}, {r_hi}], "
          f"m3/m1={hierarchy_ratio:.1f} > {HIERARCHY_MIN:.1f}")
    return True


def sample_t1_koide(rng, batch_size):
    """T1 Koide-conditioned lepton draws.
    Sheet sampler: m3 ~ logU[0.3, 2000], r = m1/m3 ~ logU[1e-5, 1e-1].
    Solve m2 from Q(m1,m2,m3)=2/3, minus branch.
    Returns (m1, m2, m3) as (N,3) array, plus attempted count.
    N_eff counts attempted (pre-rejection); accepted draw count = len(result).
    """
    m3 = np.exp(rng.uniform(np.log(LEP_LO), np.log(LEP_HI), size=batch_size))
    r = np.exp(rng.uniform(np.log(1e-5), np.log(1e-1), size=batch_size))
    m1 = r * m3

    # Solve for m2 from Q=2/3, minus branch
    # Q = (m1+m2+m3)/(s1+s2+s3)^2 = 2/3  where si = sqrt(mi)
    # => s2^2 - 4(s1+s3)s2 + (s1^2+s3^2-4*s1*s3) = 0
    s1 = np.sqrt(m1)
    s3 = np.sqrt(m3)
    b = -4.0 * (s1 + s3)
    c_coeff = s1**2 + s3**2 - 4.0 * s1 * s3

    disc = b**2 - 4.0 * c_coeff
    valid_disc = disc >= 0
    disc = np.maximum(disc, 0.0)

    # Minus branch (smaller root → m2 between m1 and m3)
    s2 = (-b - np.sqrt(disc)) / 2.0

    # Filters
    s2_ok = s2 > 0
    sorted_ok = (m1 < s2**2) & (s2**2 < m3)
    hierarchy_ok = (m3 / m1) > HIERARCHY_MIN

    keep = valid_disc & s2_ok & sorted_ok & hierarchy_ok

    m2 = s2[keep] ** 2
    result = np.column_stack([m1[keep], m2, m3[keep]])
    return result, batch_size


# ═══════════════════════════════════════════════════════════════════════
# CLAIM CHECK FUNCTIONS (vectorized, batch)
# ═══════════════════════════════════════════════════════════════════════

def check_L1(leptons):
    """L1: kdist(leptons) <= tol. Frame union built into kdist()."""
    return kdist(leptons) <= L1_TOL


def check_L2(leptons):
    """L2: |m2/m1 - target|/target <= tol."""
    ratio = leptons[:, 1] / leptons[:, 0]
    return np.abs(ratio / L2_TARGET - 1.0) <= L2_TOL


def check_L3(leptons):
    """L3: |m3/m2 - target|/target <= tol."""
    ratio = leptons[:, 2] / leptons[:, 1]
    return np.abs(ratio / L3_TARGET - 1.0) <= L3_TOL


def check_Q1(light_q, leptons):
    """Q1: |ln(ms^2 / (mu_star * md))| <= b1. Anchors from own draw."""
    mu, md, ms = light_q[:, 0], light_q[:, 1], light_q[:, 2]
    mu_star = leptons.sum(axis=1)
    val = np.log(ms * ms / (mu_star * md))
    return np.abs(val) <= B1


def check_Q2(light_q, leptons):
    """Q2: |ln(mu^2 / (md * twome))| <= b2. twome = 2*min(leptons)."""
    mu, md = light_q[:, 0], light_q[:, 1]
    twome = 2.0 * leptons.min(axis=1)
    val = np.log(mu * mu / (md * twome))
    return np.abs(val) <= B2


def check_U1_fixed(light_q, up_s):
    """U1-fixed: |9*Q_U - 8| <= tol, frame union {direct, inverse}."""
    mu, mc, mt = light_q[:, 0], up_s[:, 0], up_s[:, 1]
    triples = np.column_stack([mu, mc, mt])
    q_dir = Q_U(triples)
    q_inv = Q_U(1.0 / triples)
    best = np.minimum(np.abs(9.0 * q_dir - 8.0), np.abs(9.0 * q_inv - 8.0))
    return best <= U1_TOL


def check_U1_menu(light_q, up_s):
    """U1-menu: any of 19 targets, frame union.
    Hit if ANY p/q satisfies |9*Q_U - 9*(p/q)| <= U1_TOL in EITHER frame.
    """
    mu, mc, mt = light_q[:, 0], up_s[:, 0], up_s[:, 1]
    triples = np.column_stack([mu, mc, mt])
    q_dir = Q_U(triples)
    q_inv = Q_U(1.0 / triples)

    n = len(mu)
    hit = np.zeros(n, dtype=bool)
    for tgt in U1_MENU_TARGETS:
        hit |= (np.abs(9.0 * q_dir - tgt) <= U1_TOL)
        hit |= (np.abs(9.0 * q_inv - tgt) <= U1_TOL)
    return hit


# ═══════════════════════════════════════════════════════════════════════
# SINGLETON MEASUREMENT (§8 Tier-1 step 1)
# ═══════════════════════════════════════════════════════════════════════

ALL_CLAIMS = ["L1", "L2", "L3", "Q1", "Q2", "U1_fixed", "U1_menu"]


def measure_singletons(rng, N, prior="logU", condition="T0"):
    """Measure per-claim singleton rates at given N.
    Returns dict of {claim_id: {"counts": int, "rate": float, "cp95": [lo, hi]}}.
    """
    batch_size = 1_000_000
    n_batches = N // batch_size

    counts = {c: 0 for c in ALL_CLAIMS}
    total_eff = 0

    t0 = time.time()
    for b in range(n_batches):
        if condition == "T0":
            leptons = draw_mass_triple(rng, batch_size, LEP_LO, LEP_HI, prior, sort=True)
            n_acc = batch_size
        else:  # T1: Koide-conditioned
            leptons, attempted = sample_t1_koide(rng, batch_size)
            total_eff += attempted
            n_acc = len(leptons)
            if n_acc == 0:
                continue
            # L1 granted for T1
            counts["L1"] += n_acc

        if condition == "T0":
            total_eff += batch_size

        # Draw quarks (batch matches accepted leptons)
        light_q = draw_mass_triple(rng, n_acc, QUARK_LO, QUARK_HI, prior, sort=True)
        up_s = draw_mass_pair(rng, n_acc, QUARK_LO, QUARK_HI, prior)

        # Check claims
        if condition == "T0":
            counts["L1"] += check_L1(leptons).sum()
        counts["L2"] += check_L2(leptons).sum()
        counts["L3"] += check_L3(leptons).sum()
        counts["Q1"] += check_Q1(light_q, leptons).sum()
        counts["Q2"] += check_Q2(light_q, leptons).sum()
        counts["U1_fixed"] += check_U1_fixed(light_q, up_s).sum()
        counts["U1_menu"] += check_U1_menu(light_q, up_s).sum()

        if (b + 1) % 5 == 0:
            elapsed = time.time() - t0
            rate = total_eff / elapsed if elapsed > 0 else 0
            print(f"  [{condition}/{prior}] batch {b+1}/{n_batches}, "
                  f"N_eff={total_eff:,}, rate={rate:.0f}/s, "
                  f"L1={counts['L1']}, L2={counts['L2']}, U1m={counts['U1_menu']}",
                  flush=True)

    elapsed = time.time() - t0
    print(f"  [{condition}/{prior}] Done in {elapsed:.1f}s, "
          f"N_eff={total_eff:,}, rate={total_eff/elapsed:.0f}/s")

    result = {}
    for claim in ALL_CLAIMS:
        k = int(counts[claim])
        f = k / total_eff if total_eff > 0 else 0.0
        lo, hi = clopper_pearson(k, total_eff)
        result[claim] = {
            "count": k,
            "rate": float(f),
            "cp95_lower": lo,
            "cp95_upper": hi,
        }

    return result, total_eff


# ═══════════════════════════════════════════════════════════════════════
# CASCADE JOINT MEASUREMENT (§8 Tier-1 step 2)
# ═══════════════════════════════════════════════════════════════════════

def run_cascade(rng, N_eff_target, prior, condition, u1_mode, claim_order,
                batch_size=1_000_000):
    """Run cascade: filter by rarest claim first, survivors flow downstream.

    claim_order: list of claim function names in order of increasing rarity
                 (rarest first). Claims not in this run are skipped.

    Returns:
        hits: total joint hits
        N_eff: total effective draws
        path_counts: dict of surviving counts at each filter stage
    """
    # Map claim names to check functions
    check_funcs = {
        "L1": check_L1,
        "L2": check_L2,
        "L3": check_L3,
        "Q1": check_Q1,
        "Q2": check_Q2,
        "U1_fixed": check_U1_fixed,
        "U1_menu": check_U1_menu,
    }

    # Determine which claims are active
    active_claims = []
    for cname in claim_order:
        if cname == "U1_fixed" and u1_mode != "fixed":
            continue
        if cname == "U1_menu" and u1_mode != "menu":
            continue
        if cname == "L1" and condition == "T1":
            continue  # L1 granted for T1, skip filter
        active_claims.append(cname)

    total_eff = 0
    total_hits = 0
    stage_counts = {c: 0 for c in active_claims}

    t0 = time.time()
    batch_num = 0

    while total_eff < N_eff_target:
        batch_num += 1

        if condition == "T0":
            leptons = draw_mass_triple(rng, batch_size, LEP_LO, LEP_HI, prior, sort=True)
            n_acc = batch_size
        else:
            leptons, attempted = sample_t1_koide(rng, batch_size)
            total_eff += attempted
            n_acc = len(leptons)
            if n_acc == 0:
                continue

        if condition == "T0":
            total_eff += batch_size

        light_q = draw_mass_triple(rng, n_acc, QUARK_LO, QUARK_HI, prior, sort=True)
        up_s = draw_mass_pair(rng, n_acc, QUARK_LO, QUARK_HI, prior)

        # Apply cascade filters in order
        survivors = np.ones(n_acc, dtype=bool)

        for cname in active_claims:
            func = check_funcs[cname]
            # Determine which args the function needs
            if cname in ("L1", "L2", "L3"):
                mask = func(leptons[survivors])
            elif cname in ("Q1", "Q2"):
                mask = func(light_q[survivors], leptons[survivors])
            else:  # U1_fixed, U1_menu
                mask = func(light_q[survivors], up_s[survivors])

            # Apply to survivors
            survivors_indices = np.where(survivors)[0]
            survivors[survivors_indices[~mask]] = False
            stage_counts[cname] += mask.sum()

        total_hits += survivors.sum()

        if batch_num % 10 == 0:
            elapsed = time.time() - t0
            rate = total_eff / elapsed if elapsed > 0 else 0
            progress = 100.0 * total_eff / N_eff_target
            print(f"  [{condition}/{prior}/U1-{u1_mode}] "
                  f"N_eff={total_eff:,} ({progress:.1f}%), "
                  f"rate={rate:.0f}/s, "
                  f"joint_hits={total_hits}, "
                  f"stages={[{c: stage_counts[c]} for c in active_claims]}",
                  flush=True)

        if total_eff >= N_eff_target:
            break

    elapsed = time.time() - t0
    print(f"  [{condition}/{prior}/U1-{u1_mode}] COMPLETE: "
          f"N_eff={total_eff:,}, hits={total_hits}, "
          f"elapsed={elapsed:.1f}s", flush=True)

    return total_hits, total_eff, stage_counts


# ═══════════════════════════════════════════════════════════════════════
# TIER-2: CONDITIONAL-ANALYTIC (§8, v0.2 adaptive-inflation)
# ═══════════════════════════════════════════════════════════════════════

def tier2_adaptive_inflation_calibrate(rng, prior="logU", condition="T0",
                                        calibration_seed=271828, N_calib=100_000_000):
    """v0.2 Tier-2 calibration: adaptive per-stage inflation.

    Inflate tolerances by the smallest factor in {1e2, 1e3, 1e4} at which
    brute force (calibration seed, N=1e8) yields >= 100 hits, validating
    per stage-pair rather than only at the full joint.

    Stage pairs: (L2^L3), (+Q1), (+Q2), (+U1)

    Returns dict of {stage_pair: {"factor": int or None, "hits": int}},
    where factor=None means no factor reached 100 hits → Tier-2 ineligible.
    """
    factors = [1e2, 1e3, 1e4]
    batch_size = 1_000_000
    n_batches = N_calib // batch_size

    stage_pairs = [
        ("L2_L3", ["L2", "L3"]),
        ("+Q1", ["L2", "L3", "Q1"]),
        ("+Q2", ["L2", "L3", "Q1", "Q2"]),
        ("+U1", ["L2", "L3", "Q1", "Q2", "U1_fixed"]),
    ]

    results = {}

    for pair_name, claim_list in stage_pairs:
        print(f"\n  Calibrating stage-pair: {pair_name} (claims: {claim_list})")
        best_factor = None
        best_hits = 0

        for factor in factors:
            rng_calib = np.random.default_rng(calibration_seed)
            total_eff = 0
            total_hits = 0

            t0 = time.time()
            for b in range(n_batches):
                if condition == "T0":
                    leptons = draw_mass_triple(rng_calib, batch_size, LEP_LO, LEP_HI,
                                                prior, sort=True)
                    n_acc = batch_size
                    total_eff += batch_size
                else:
                    leptons, attempted = sample_t1_koide(rng_calib, batch_size)
                    total_eff += attempted
                    n_acc = len(leptons)
                    if n_acc == 0:
                        continue

                light_q = draw_mass_triple(rng_calib, n_acc, QUARK_LO, QUARK_HI,
                                            prior, sort=True)
                up_s = draw_mass_pair(rng_calib, n_acc, QUARK_LO, QUARK_HI, prior)

                survivors = np.ones(n_acc, dtype=bool)

                for claim in claim_list:
                    if claim == "L2":
                        ratio = leptons[:, 1] / leptons[:, 0]
                        mask = np.abs(ratio / L2_TARGET - 1.0) <= L2_TOL * factor
                    elif claim == "L3":
                        ratio = leptons[:, 2] / leptons[:, 1]
                        mask = np.abs(ratio / L3_TARGET - 1.0) <= L3_TOL * factor
                    elif claim == "Q1":
                        mu_q, md_q, ms_q = (light_q[:, 0], light_q[:, 1], light_q[:, 2])
                        mu_star = leptons.sum(axis=1)
                        val = np.log(ms_q * ms_q / (mu_star * md_q))
                        mask = np.abs(val) <= B1 * factor
                    elif claim == "Q2":
                        mu_q, md_q = light_q[:, 0], light_q[:, 1]
                        twome = 2.0 * leptons.min(axis=1)
                        val = np.log(mu_q * mu_q / (md_q * twome))
                        mask = np.abs(val) <= B2 * factor
                    elif claim == "U1_fixed":
                        mu_q, mc_q, mt_q = light_q[:, 0], up_s[:, 0], up_s[:, 1]
                        triples = np.column_stack([mu_q, mc_q, mt_q])
                        q_dir = Q_U(triples)
                        q_inv = Q_U(1.0 / triples)
                        best_u = np.minimum(np.abs(9.0 * q_dir - 8.0),
                                            np.abs(9.0 * q_inv - 8.0))
                        mask = best_u <= U1_TOL * factor
                    else:
                        continue

                    survivors_idx = np.where(survivors)[0]
                    survivors[survivors_idx[~mask[survivors_idx]]] = False

                total_hits += survivors.sum()

                if (b + 1) % 20 == 0:
                    print(f"    [{pair_name}/x{factor:.0f}] batch {b+1}/{n_batches}, "
                          f"hits={total_hits}", flush=True)

            print(f"    [{pair_name}/x{factor:.0f}] COMPLETE: "
                  f"hits={total_hits}, N_eff={total_eff}", flush=True)

            if total_hits >= 100:
                best_factor = int(factor)
                best_hits = int(total_hits)
                break  # smallest factor wins

        results[pair_name] = {
            "factor": best_factor,
            "hits": best_hits,
            "eligible": best_factor is not None,
        }
        status = f"ELIGIBLE (factor={best_factor}, hits={best_hits})" if best_factor else "INELIGIBLE"
        print(f"  Stage-pair {pair_name}: {status}")

    # Determine overall eligibility: all stage-pairs must be eligible
    all_eligible = all(r["eligible"] for r in results.values())
    print(f"\n  Tier-2 overall: {'ELIGIBLE' if all_eligible else 'INELIGIBLE'}")

    return results, all_eligible


# ═══════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Joint Coincidence Budget Engine")
    parser.add_argument("--mode", default="singletons",
                        choices=["singletons", "cascade", "full"])
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--prior", default="logU", choices=["logU", "logN", "linU"])
    parser.add_argument("--condition", default="T0", choices=["T0", "T1"])
    parser.add_argument("--u1-mode", default="fixed", choices=["fixed", "menu"])
    parser.add_argument("--N-singletons", type=int, default=10_000_000)
    parser.add_argument("--N-eff", type=int, default=2_000_000_000)
    parser.add_argument("--cascade-order", default=None,
                        help="Comma-separated claim order, rarest first")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    if args.outdir is None:
        args.outdir = f"results/amb-{args.seed}-v0.2"

    os.makedirs(args.outdir, exist_ok=True)

    # ── §5 SUPPORT GATE: verify before any production batch ──
    if not check_support_gate():
        print("\n*** STOP: support gate failed. Null excludes observation. ***")
        sys.exit(1)

    rng = np.random.default_rng(args.seed)

    print(f"{'='*70}")
    print(f"JOINT COINCIDENCE BUDGET ENGINE — amb")
    print(f"Mode: {args.mode} | Seed: {args.seed} | Prior: {args.prior}")
    print(f"Condition: {args.condition} | U1: {args.u1_mode}")
    print(f"N_singletons: {args.N_singletons:,} | N_eff: {args.N_eff:,}")
    print(f"{'='*70}\n")

    if args.mode == "singletons":
        result, N_eff = measure_singletons(
            rng, args.N_singletons, args.prior, args.condition
        )

        outpath = os.path.join(
            args.outdir,
            f"singletons_{args.condition}_{args.prior}_seed{args.seed}.json"
        )
        output = {
            "engine_id": "amb",
            "seed": args.seed,
            "N_target": args.N_singletons,
            "N_eff": int(N_eff),
            "prior": args.prior,
            "condition": args.condition,
            "results": result,
        }
        with open(outpath, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved: {outpath}")

    elif args.mode == "cascade":
        if args.cascade_order:
            claim_order = args.cascade_order.split(",")
        else:
            claim_order = ["L2", "L3", "L1", "Q1", "Q2", "U1_fixed", "U1_menu"]

        print(f"Cascade order: {claim_order}")

        hits, N_eff, stage_counts = run_cascade(
            rng, args.N_eff, args.prior, args.condition, args.u1_mode,
            claim_order
        )

        f_rate = hits / N_eff if N_eff > 0 else 0.0
        lo, hi = clopper_pearson(int(hits), int(N_eff))
        label = "POINT" if hits >= 100 else "BOUND"
        print(f"\nJoint: {hits}/{N_eff} = {f_rate:.6e}  CP95 [{lo:.6e}, {hi:.6e}]  {label}")

        # Save cell result
        cell_result = {
            "engine_id": "amb",
            "condition": args.condition,
            "u1_mode": args.u1_mode,
            "prior": args.prior,
            "seed": args.seed,
            "N_eff": int(N_eff),
            "N_eff_target": args.N_eff,
            "hits": int(hits),
            "rate": float(f_rate),
            "cp95_lower": lo,
            "cp95_upper": hi,
            "cascade_order": claim_order,
            "stage_counts": {k: int(v) for k, v in stage_counts.items()},
            "tier": "Tier-1",
            "label": label,
        }
        cell_path = os.path.join(
            args.outdir,
            f"cell_{args.condition}_{args.u1_mode}_{args.prior}.json"
        )
        with open(cell_path, "w") as f:
            json.dump(cell_result, f, indent=2)
        print(f"Saved: {cell_path}")

    elif args.mode == "full":
        # Run all 12 cells
        all_results = {}
        for condition in ["T0", "T1"]:
            for u1_mode in ["fixed", "menu"]:
                for prior in ["logU", "logN", "linU"]:
                    cell_key = f"{condition}_{u1_mode}_{prior}"
                    cell_seed = args.seed if prior == "logU" else 314160
                    N_target = args.N_eff if prior == "logU" else max(args.N_eff // 10, 200_000_000)

                    print(f"\n{'─'*60}")
                    print(f"CELL: {cell_key}  seed={cell_seed}  N_eff>={N_target:,}")
                    print(f"{'─'*60}")

                    cell_rng = np.random.default_rng(cell_seed)

                    # Step 1: singletons
                    singletons, N_sing = measure_singletons(
                        cell_rng, min(args.N_singletons, N_target),
                        prior, condition
                    )

                    # Step 2: cascade
                    # Order by measured rarity
                    claim_rates = [(c, singletons[c]["rate"]) for c in ALL_CLAIMS
                                   if singletons[c]["rate"] > 0]
                    claim_rates.sort(key=lambda x: x[1])  # rarest first
                    claim_order = [c for c, _ in claim_rates]

                    print(f"  Cascade order (rarest first): {claim_order}")

                    hits, N_eff, stage_counts = run_cascade(
                        cell_rng, N_target, prior, condition, u1_mode,
                        claim_order
                    )

                    f_rate = hits / N_eff if N_eff > 0 else 0.0
                    lo, hi = clopper_pearson(int(hits), int(N_eff))

                    all_results[cell_key] = {
                        "condition": condition,
                        "u1_mode": u1_mode,
                        "prior": prior,
                        "seed": cell_seed,
                        "N_eff": int(N_eff),
                        "hits": int(hits),
                        "rate": float(f_rate),
                        "cp95_lower": lo,
                        "cp95_upper": hi,
                        "singletons": singletons,
                        "cascade_order": claim_order,
                        "stage_counts": {k: int(v) for k, v in stage_counts.items()},
                        "tier": "Tier-1",
                        "label": "POINT" if hits >= 100 else "BOUND",
                    }

                    # Checkpoint
                    ckpt = os.path.join(args.outdir, f"checkpoint_{cell_key}.json")
                    with open(ckpt, "w") as fh:
                        json.dump(all_results[cell_key], fh, indent=2)

        # Save full results
        outpath = os.path.join(args.outdir, "mc_counts.json")
        with open(outpath, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nFull results saved: {outpath}")

        # Print summary table
        print(f"\n{'='*80}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"{'Cell':<25} {'Hits':>10} {'N_eff':>14} {'Rate':>14} {'Label':>8}")
        print(f"{'─'*25} {'─'*10} {'─'*14} {'─'*14} {'─'*8}")
        for cell_key, r in sorted(all_results.items()):
            print(f"{cell_key:<25} {r['hits']:>10} {r['N_eff']:>14,} "
                  f"{r['rate']:>14.6e} {r['label']:>8}")


if __name__ == "__main__":
    main()

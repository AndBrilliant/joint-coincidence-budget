#!/usr/bin/env python3
"""
JOINT COINCIDENCE BUDGET — Full Engine v0.5
ENGINE_ID: amb
Production seed: 20260811 | Prior-variant seed: 314160 | Calibration seed: 271828

v0.5 changes from v0.4 (validation-procedure fixes):
  (a) ANCHOR AVERAGING: at every calibration stage the analytic side must
      average the quark-block probability over the SAME inflated lepton window
      brute force samples — >= 1e4 lepton draws inside that window, anchors
      recomputed per draw. Fixes v0.4's +Q2 7.22σ fail (3.6× anchor-spread).
  (b) STAGE-LOCAL INFLATION: inflate ONLY the claims new to the stage under
      test; lepton windows run at the smallest factor giving >= 100 lepton-stage
      survivors in brute force, with (a) applied at that factor.
  (c) COMPOSITIONAL ELIGIBILITY for +U1: if no direct factor reaches 100 hits
      non-vacuously, validate instead (i) P(U1) singleton integral at factor 1
      and (ii) mu-coupled pair Q2^U1 at adaptive stage-local factor.
  (d) If every stage passes ((a)-(c)) => TIER-2 (VALIDATED); any residual
      fail => FAIL with numbers and stop.

All v0.1-v0.4 results preserved.
"""

import numpy as np
import json
import os
import sys
import time
from scipy.stats import beta as beta_dist
from scipy.optimize import bisect

# ═══════════════════════════════════════════════════════════════════════════
# KDISK, Q_U, CP95 (unchanged)
# ═══════════════════════════════════════════════════════════════════════════

ang = 2.0 * np.pi * np.arange(3) / 3.0
cos_ang = np.cos(ang)
sin_ang = np.sin(ang)

def kdist(m):
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
    v = np.asarray(v)
    return np.sum(v, axis=-1) / np.sum(np.sqrt(v), axis=-1) ** 2

def clopper_pearson(k, n, alpha=0.05):
    if k <= 0:
        lower = 0.0
    else:
        lower = beta_dist.ppf(alpha / 2.0, k, n - k + 1)
    if k >= n:
        upper = 1.0
    else:
        upper = beta_dist.ppf(1.0 - alpha / 2.0, k + 1, n - k)
    return float(lower), float(upper)

# ═══════════════════════════════════════════════════════════════════════════
# FROZEN CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

LEPTONS_OBS = np.array([0.51099895, 105.6583755, 1776.93])
L1_TOL = 3.3049e-6
L2_TARGET = 206.7703
L2_TOL = 1.00e-5
L3_TARGET = 16.8180
L3_TOL = 2.10e-5
B1 = 3.00e-3
B2 = 1.18e-2
U1_TOL = 1.1414e-2
U1_TARGET = 8.0 / 9.0

U1_MENU = [
    (1, 1), (1, 2), (2, 3), (3, 4),
    (2, 5), (3, 5), (4, 5),
    (5, 6),
    (3, 7), (4, 7), (5, 7), (6, 7),
    (3, 8), (5, 8), (7, 8),
    (4, 9), (5, 9), (7, 9), (8, 9),
]
U1_MENU_TARGETS = np.array([9.0 * p / q for (p, q) in U1_MENU], dtype=np.float64)

LEP_LO, LEP_HI = 0.3, 2000.0
QUARK_LO, QUARK_HI = 0.5, 2e5

LEP_LOG_LO, LEP_LOG_HI = np.log(LEP_LO), np.log(LEP_HI)
LEP_LOG_V = LEP_LOG_HI - LEP_LOG_LO
QUARK_LOG_LO, QUARK_LOG_HI = np.log(QUARK_LO), np.log(QUARK_HI)
QUARK_LOG_V = QUARK_LOG_HI - QUARK_LOG_LO
HIERARCHY_MIN = (4.0 + np.sqrt(18.0)) ** 2

INFLATION_FACTORS = [1, 3, 10, 30, 100, 300, 1000]

u_obs = np.log(L2_TARGET)
v_obs = np.log(L3_TARGET)

# ═══════════════════════════════════════════════════════════════════════════
# EXACT ANALYTIC DENSITY (verified v0.4)
# ═══════════════════════════════════════════════════════════════════════════

def density_uv(u, v):
    r = LEP_LOG_V - u - v
    return np.where(r > 0, 6.0 / (LEP_LOG_V ** 3) * r, 0.0)

def integrate_density_rectangle(u_a, u_b, v_a, v_b, n_sub=2000):
    Lv = LEP_LOG_V
    du = (u_b - u_a) / n_sub
    total = 0.0
    for i in range(n_sub):
        u_mid = u_a + (i + 0.5) * du
        u_cur = u_a + i * du
        u_next = u_a + (i + 1) * du
        v_upper_full = min(v_b, Lv - u_cur)
        v_lower = v_a
        if v_upper_full <= v_lower:
            if Lv - u_next > v_lower:
                v_upper_full = min(v_b, Lv - u_mid)
            else:
                continue
        if v_upper_full <= v_lower:
            continue
        r = Lv - u_mid
        v1, v2 = v_lower, v_upper_full
        contrib = r * (v2 - v1) - (v2 ** 2 - v1 ** 2) / 2.0
        if contrib > 0:
            total += contrib * du
    return total * 6.0 / (Lv ** 3)

# ═══════════════════════════════════════════════════════════════════════════
# T1 Koide sheet functions
# ═══════════════════════════════════════════════════════════════════════════

def m2_m1_from_r(r):
    x = np.sqrt(np.maximum(r, 1e-300))
    disc = 3.0 * (x ** 2 + 4.0 * x + 1.0)
    s2_over_s3 = 2.0 * (x + 1.0) - np.sqrt(np.maximum(0.0, disc))
    return s2_over_s3 ** 2 / x ** 2

def m3_m2_from_r(r):
    x = np.sqrt(np.maximum(r, 1e-300))
    disc = 3.0 * (x ** 2 + 4.0 * x + 1.0)
    s2_over_s3 = 2.0 * (x + 1.0) - np.sqrt(np.maximum(0.0, disc))
    return 1.0 / (s2_over_s3 ** 2)

def compute_r_intersection(tol_factor=1.0):
    """Find r-range satisfying BOTH L2 and L3 at given tolerance factor."""
    r_obs = LEPTONS_OBS[0] / LEPTONS_OBS[2]
    ln_r_obs = np.log(r_obs)
    for spread in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        r_grid = np.exp(ln_r_obs + np.linspace(-spread, spread, 200000))
        m2m1_grid = m2_m1_from_r(r_grid)
        m3m2_grid = m3_m2_from_r(r_grid)
        l2_ok = np.abs(m2m1_grid / L2_TARGET - 1.0) <= L2_TOL * tol_factor
        l3_ok = np.abs(m3m2_grid / L3_TARGET - 1.0) <= L3_TOL * tol_factor
        both_ok = l2_ok & l3_ok
        if both_ok.any():
            r_intersection = r_grid[both_ok]
            r_min, r_max = r_intersection.min(), r_intersection.max()
            delta_ln_r = np.log(r_max / r_min)
            return float(r_min), float(r_max), float(delta_ln_r)
    return None, None, None

# ═══════════════════════════════════════════════════════════════════════════
# DRAW FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def draw_mass_triple(rng, batch_size, lo, hi, prior, sort=True):
    if prior == "logU":
        x = np.exp(rng.uniform(np.log(lo), np.log(hi), size=(batch_size, 3)))
    elif prior == "logN":
        mid = np.exp((np.log(lo) + np.log(hi)) / 2.0)
        x = np.exp(rng.normal(np.log(mid), 1.5 * np.log(10), size=(batch_size, 3)))
        x = np.clip(x, lo, hi)
    elif prior == "linU":
        x = rng.uniform(lo, hi, size=(batch_size, 3))
    else:
        raise ValueError(f"Unknown prior: {prior}")
    if sort:
        x.sort(axis=1)
    return x

def draw_mass_pair(rng, batch_size, lo, hi, prior):
    if prior == "logU":
        x = np.exp(rng.uniform(np.log(lo), np.log(hi), size=(batch_size, 2)))
    elif prior == "logN":
        mid = np.exp((np.log(lo) + np.log(hi)) / 2.0)
        x = np.exp(rng.normal(np.log(mid), 1.5 * np.log(10), size=(batch_size, 2)))
        x = np.clip(x, lo, hi)
    elif prior == "linU":
        x = rng.uniform(lo, hi, size=(batch_size, 2))
    else:
        raise ValueError(f"Unknown prior: {prior}")
    return x

def sample_t1_koide(rng, batch_size):
    """T1 Koide-conditioned lepton draws. Returns (array, attempted_count)."""
    m3 = np.exp(rng.uniform(np.log(LEP_LO), np.log(LEP_HI), size=batch_size))
    r = np.exp(rng.uniform(np.log(1e-5), np.log(1e-1), size=batch_size))
    m1 = r * m3
    s1, s3 = np.sqrt(m1), np.sqrt(m3)
    b = -4.0 * (s1 + s3)
    c_coeff = s1**2 + s3**2 - 4.0 * s1 * s3
    disc = b**2 - 4.0 * c_coeff
    valid_disc = disc >= 0
    disc = np.maximum(disc, 0.0)
    s2 = (-b - np.sqrt(disc)) / 2.0
    s2_ok = s2 > 0
    sorted_ok = (m1 < s2**2) & (s2**2 < m3)
    hierarchy_ok = (m3 / m1) > HIERARCHY_MIN
    keep = valid_disc & s2_ok & sorted_ok & hierarchy_ok
    m2 = s2[keep] ** 2
    result = np.column_stack([m1[keep], m2, m3[keep]])
    return result, batch_size

# ═══════════════════════════════════════════════════════════════════════════
# CLAIM CHECK FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def check_L1(leptons, f=1.0):
    return kdist(leptons) <= L1_TOL * f

def check_L2(leptons, f=1.0):
    return np.abs(leptons[:, 1] / leptons[:, 0] / L2_TARGET - 1.0) <= L2_TOL * f

def check_L3(leptons, f=1.0):
    return np.abs(leptons[:, 2] / leptons[:, 1] / L3_TARGET - 1.0) <= L3_TOL * f

def check_Q1(light_q, leptons, f=1.0):
    mu, md, ms = light_q[:, 0], light_q[:, 1], light_q[:, 2]
    mu_star = leptons.sum(axis=1)
    return np.abs(np.log(ms * ms / (mu_star * md))) <= B1 * f

def check_Q2(light_q, leptons, f=1.0):
    mu, md = light_q[:, 0], light_q[:, 1]
    twome = 2.0 * leptons.min(axis=1)
    return np.abs(np.log(mu * mu / (md * twome))) <= B2 * f

def check_U1_fixed(light_q, up_s, f=1.0):
    mu, mc, mt = light_q[:, 0], up_s[:, 0], up_s[:, 1]
    t = np.column_stack([mu, mc, mt])
    qd, qi = Q_U(t), Q_U(1.0 / t)
    return np.minimum(np.abs(9.0 * qd - 8.0), np.abs(9.0 * qi - 8.0)) <= U1_TOL * f

def check_U1_menu(light_q, up_s, f=1.0):
    mu, mc, mt = light_q[:, 0], up_s[:, 0], up_s[:, 1]
    t = np.column_stack([mu, mc, mt])
    qd, qi = Q_U(t), Q_U(1.0 / t)
    hit = np.zeros(len(mu), dtype=bool)
    for tgt in U1_MENU_TARGETS:
        hit |= (np.abs(9.0 * qd - tgt) <= U1_TOL * f)
        hit |= (np.abs(9.0 * qi - tgt) <= U1_TOL * f)
    return hit

# ═══════════════════════════════════════════════════════════════════════════
# NON-VACUOUS GATE
# ═══════════════════════════════════════════════════════════════════════════

def check_nonvacuous(claim_id, factor, rng_seed=271828):
    """Check inflated window excludes >50% of prior mass."""
    if claim_id in ("L2", "L3"):
        V = LEP_LOG_V
        target = np.log(L2_TARGET) if claim_id == "L2" else np.log(L3_TARGET)
        tol = L2_TOL if claim_id == "L2" else L3_TOL
        hw = tol * factor
        lo, hi = max(0.0, target - hw), min(V, target + hw)
        if lo >= hi:
            return True, 0.0
        F = lambda x: (2.0 * x / V) - (x / V) ** 2
        frac = F(hi) - F(lo)
        return frac < 0.5, float(max(0, frac))
    elif claim_id == "L1":
        rng = np.random.default_rng(rng_seed)
        N = 500000
        x = rng.uniform(LEP_LOG_LO, LEP_LOG_HI, size=(N, 3))
        x.sort(axis=1)
        m = np.exp(x)
        frac = check_L1(m, factor).mean()
        return frac < 0.5, float(frac)
    elif claim_id in ("Q1", "Q2"):
        tol = B1 if claim_id == "Q1" else B2
        frac = 2.0 * tol * factor / QUARK_LOG_V
        return frac < 0.5, float(min(frac, 1.0))
    elif claim_id.startswith("U1"):
        rng = np.random.default_rng(rng_seed)
        N = 200000
        lq = np.exp(rng.uniform(QUARK_LOG_LO, QUARK_LOG_HI, size=(N, 3)))
        lq.sort(axis=1)
        us = np.exp(rng.uniform(QUARK_LOG_LO, QUARK_LOG_HI, size=(N, 2)))
        if "fixed" in claim_id:
            frac = check_U1_fixed(lq, us, factor).mean()
        else:
            frac = check_U1_menu(lq, us, factor).mean()
        return frac < 0.5, float(frac)
    return True, 0.0

# ═══════════════════════════════════════════════════════════════════════════
# ANCHOR COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_anchors(leptons):
    """Compute mu_star, twome, hinge from lepton draws.
    leptons: (N, 3) array of lepton masses in MeV.
    """
    mu_star = leptons.sum(axis=1)
    twome = 2.0 * leptons.min(axis=1)
    hinge = np.sqrt(twome * mu_star)
    return mu_star, twome, hinge

# ═══════════════════════════════════════════════════════════════════════════
# v0.5 ANCHOR-AVERAGED QUARK BLOCK PROBABILITY
# ═══════════════════════════════════════════════════════════════════════════

def quark_block_probability_anchored(lepton_samples, quark_claims, factors,
                                      prior="logU", N_quark_per_lepton=50000,
                                      rng_seed=271828):
    """Compute average P(quark claims | lepton) over lepton_samples with anchor averaging.

    lepton_samples: (N_lep, 3) array of lepton masses
    quark_claims: list of claim ids like ["Q1", "Q2", "U1_fixed"]
    factors: dict mapping claim_id -> inflation factor
    N_quark_per_lepton: quark draws per lepton sample
    rng_seed: base seed for RNG

    Returns: mean probability, individual probabilities array
    """
    N_lep = len(lepton_samples)
    anchors = compute_anchors(lepton_samples)  # mu_star, twome arrays of length N_lep

    rng = np.random.default_rng(rng_seed)
    probs = np.zeros(N_lep)

    for i in range(N_lep):
        mu_s, tw, hinge_i = anchors[0][i], anchors[1][i], anchors[2][i]

        # Draw quarks
        lq = np.exp(rng.uniform(QUARK_LOG_LO, QUARK_LOG_HI, size=(N_quark_per_lepton, 3)))
        lq.sort(axis=1)
        us_q = np.exp(rng.uniform(QUARK_LOG_LO, QUARK_LOG_HI, size=(N_quark_per_lepton, 2)))

        survivors = np.ones(N_quark_per_lepton, dtype=bool)
        # Tile the single lepton to match quark batch
        lep_tiled = np.tile(lepton_samples[i:i+1], (N_quark_per_lepton, 1))

        for claim in quark_claims:
            f = factors.get(claim, 1.0)
            if claim == "Q1":
                m = check_Q1(lq, lep_tiled, f)
            elif claim == "Q2":
                m = check_Q2(lq, lep_tiled, f)
            elif claim == "U1_fixed":
                m = check_U1_fixed(lq, us_q, f)
            elif claim == "U1_menu":
                m = check_U1_menu(lq, us_q, f)
            else:
                continue
            si = np.where(survivors)[0]
            survivors[si[~m[si]]] = False

        probs[i] = survivors.mean()

    return float(probs.mean()), probs

# ═══════════════════════════════════════════════════════════════════════════
# v0.5 TIER-2 LEPTON BLOCK (analytic)
# ═══════════════════════════════════════════════════════════════════════════

def tier2_lepton_block_prob(condition, lepton_factors):
    """Analytic P(lepton stage) at given inflation factors.

    condition: "T0" or "T1"
    lepton_factors: dict like {"L2": f2, "L3": f3} or {"L1": f1, "L2": f2, "L3": f3}
    Returns: probability, metadata dict
    """
    if condition == "T0":
        f_l1 = lepton_factors.get("L1", 1.0)
        f_l2 = lepton_factors.get("L2", 1.0)
        f_l3 = lepton_factors.get("L3", 1.0)

        V = LEP_LOG_V
        u_lo = np.log(L2_TARGET * (1.0 - L2_TOL * f_l2))
        u_hi = np.log(L2_TARGET * (1.0 + L2_TOL * f_l2))
        v_lo = np.log(L3_TARGET * (1.0 - L3_TOL * f_l3))
        v_hi = np.log(L3_TARGET * (1.0 + L3_TOL * f_l3))

        u_mid = (u_lo + u_hi) / 2.0
        v_mid = (v_lo + v_hi) / 2.0
        du = u_hi - u_lo
        dv = v_hi - v_lo
        f0 = density_uv(u_mid, v_mid)

        # L1 fraction within window
        if f_l1 > 1.0 or f_l2 > 1.0 or f_l3 > 1.0:
            N_l1 = 50000
            rng_l1 = np.random.default_rng(271828)
            u_t = rng_l1.uniform(u_lo, u_hi, N_l1)
            v_t = rng_l1.uniform(v_lo, v_hi, N_l1)
            m1_t = np.ones(N_l1)
            m2_t = m1_t * np.exp(u_t)
            m3_t = m2_t * np.exp(v_t)
            l_t = np.column_stack([m1_t, m2_t, m3_t])
            f_l1_frac = float(check_L1(l_t, f_l1).mean())
        else:
            # At factor 1, use the verified fraction from v0.4
            m_test = np.column_stack([np.ones(100000), np.ones(100000) * np.exp(u_mid),
                                       np.ones(100000) * np.exp(u_mid + v_mid)])
            f_l1_frac = float(check_L1(m_test, f_l1).mean())
            # Actually, use proper sampling for accuracy
            N_l1 = 100000
            rng_l1 = np.random.default_rng(271828)
            u_t = rng_l1.uniform(u_lo, u_hi, N_l1)
            v_t = rng_l1.uniform(v_lo, v_hi, N_l1)
            m1_t = np.ones(N_l1)
            m2_t = m1_t * np.exp(u_t)
            m3_t = m2_t * np.exp(v_t)
            l_t = np.column_stack([m1_t, m2_t, m3_t])
            f_l1_frac = float(check_L1(l_t, f_l1).mean())

        p_lep = f0 * du * dv * f_l1_frac
        info = {"method": "analytic_density", "u_mid": float(u_mid), "v_mid": float(v_mid),
                "du": float(du), "dv": float(dv), "f0": float(f0), "f_l1_frac": float(f_l1_frac)}

    else:  # T1
        f_l2 = lepton_factors.get("L2", 1.0)
        f_l3 = lepton_factors.get("L3", 1.0)
        V_r = np.log(1e-1 / 1e-5)
        r_min, r_max, delta_ln_r = compute_r_intersection(max(f_l2, f_l3))
        if delta_ln_r is not None and delta_ln_r > 0:
            p_lep = delta_ln_r / V_r
            info = {"method": "r_intersection", "r_min": r_min, "r_max": r_max,
                    "delta_ln_r": delta_ln_r, "V_r": V_r}
        else:
            p_lep = 0.0
            info = {"method": "r_intersection", "error": "no_intersection"}

    return p_lep, info

# ═══════════════════════════════════════════════════════════════════════════
# v0.5 PER-STAGE VALIDATION WITH ANCHOR AVERAGING
# ═══════════════════════════════════════════════════════════════════════════

def v05_per_stage_validation(prior="logU", condition="T0", u1_mode="fixed",
                               calibration_seed=271828, N_max=1_000_000_000):
    """v0.5 per-stage validation with anchor averaging and stage-local inflation.

    Returns: (validation_record, all_stages_validated)
    """
    batch_size = 1_000_000
    rng_bf = np.random.default_rng(calibration_seed)

    # Define stages with their lepton claims and new quark claims
    if condition == "T0":
        stages = [
            {"name": "L1_L2_L3", "lepton_claims": ["L1", "L2", "L3"], "new_claims": []},
            {"name": "+Q1", "lepton_claims": ["L1", "L2", "L3"], "new_claims": ["Q1"]},
            {"name": "+Q2", "lepton_claims": ["L1", "L2", "L3"], "new_claims": ["Q2"]},
            {"name": "+U1", "lepton_claims": ["L1", "L2", "L3"], "new_claims": [f"U1_{u1_mode}"]},
        ]
    else:
        stages = [
            {"name": "L2_L3", "lepton_claims": ["L2", "L3"], "new_claims": []},
            {"name": "+Q1", "lepton_claims": ["L2", "L3"], "new_claims": ["Q1"]},
            {"name": "+Q2", "lepton_claims": ["L2", "L3"], "new_claims": ["Q2"]},
            {"name": "+U1", "lepton_claims": ["L2", "L3"], "new_claims": [f"U1_{u1_mode}"]},
        ]

    validation_record = {}
    all_stages_validated = True
    # Track the lepton-stage factor (carried forward)
    lepton_stage_factor = None

    for stage_idx, stage in enumerate(stages):
        stage_name = stage["name"]
        lepton_claims = stage["lepton_claims"]
        new_claims = stage["new_claims"]

        print(f"\n{'='*60}")
        print(f"STAGE: {stage_name}  (lepton: {lepton_claims}, new: {new_claims})")
        print(f"{'='*60}")

        # ── Step 1: Find lepton-stage factor (only for first stage) ──
        if stage_idx == 0:
            # First stage: inflate ALL lepton claims together
            print(f"\n  Finding lepton-stage factor for {lepton_claims}...")
            lep_factor = None
            for f_lep in INFLATION_FACTORS:
                # Check non-vacuous
                vacuous = False
                for claim in lepton_claims:
                    nv, frac = check_nonvacuous(claim, f_lep)
                    if not nv:
                        print(f"    VACUOUS: {claim} at factor {f_lep}: {frac*100:.1f}%")
                        vacuous = True
                        break
                if vacuous:
                    continue

                # Brute force lepton stage
                rng_lep = np.random.default_rng(calibration_seed)
                lep_hits = 0
                lep_N = 0
                while lep_hits < 100 and lep_N < N_max:
                    if condition == "T0":
                        lep = draw_mass_triple(rng_lep, batch_size, LEP_LO, LEP_HI, prior, sort=True)
                        n_acc = batch_size
                        lep_N += batch_size
                    else:
                        lep, attempted = sample_t1_koide(rng_lep, batch_size)
                        lep_N += attempted
                        n_acc = len(lep)
                        if n_acc == 0:
                            continue

                    surv = np.ones(n_acc, dtype=bool)
                    for claim in lepton_claims:
                        f_dict = {"L1": f_lep, "L2": f_lep, "L3": f_lep}
                        if claim == "L1":
                            m = check_L1(lep, f_dict.get("L1", 1.0))
                        elif claim == "L2":
                            m = check_L2(lep, f_dict.get("L2", 1.0))
                        elif claim == "L3":
                            m = check_L3(lep, f_dict.get("L3", 1.0))
                        si = np.where(surv)[0]
                        surv[si[~m[si]]] = False
                    lep_hits += surv.sum()

                print(f"    Factor {f_lep}: {lep_hits} lepton-stage survivors in {lep_N:,} draws")
                if lep_hits >= 100:
                    lep_factor = f_lep
                    break

            if lep_factor is None:
                print(f"  *** LEPTON STAGE INELIGIBLE: no factor reached 100 hits ***")
                validation_record[stage_name] = {"eligible": False, "error": "no_lepton_factor"}
                all_stages_validated = False
                return validation_record, all_stages_validated

            lepton_stage_factor = lep_factor
            print(f"  → Lepton-stage factor: {lepton_stage_factor}")

            # Validate first stage (lepton-only, no quark claims)
            if not new_claims:
                # This stage has no new claims — validate the lepton stage itself
                # T2 analytic: lepton block probability at lepton_stage_factor
                lep_factors = {c: lepton_stage_factor for c in lepton_claims}
                p_lep, info = tier2_lepton_block_prob(condition, lep_factors)

                # Brute force: already computed above
                rng_bf2 = np.random.default_rng(calibration_seed)
                bf_hits = 0
                bf_N = 0
                while bf_hits < 100 and bf_N < N_max:
                    if condition == "T0":
                        lep = draw_mass_triple(rng_bf2, batch_size, LEP_LO, LEP_HI, prior, sort=True)
                        n_acc = batch_size
                        bf_N += batch_size
                    else:
                        lep, attempted = sample_t1_koide(rng_bf2, batch_size)
                        bf_N += attempted
                        n_acc = len(lep)
                        if n_acc == 0:
                            continue
                    surv = np.ones(n_acc, dtype=bool)
                    for claim in lepton_claims:
                        f_claim = lepton_stage_factor
                        if claim == "L1":
                            m = check_L1(lep, f_claim)
                        elif claim == "L2":
                            m = check_L2(lep, f_claim)
                        elif claim == "L3":
                            m = check_L3(lep, f_claim)
                        si = np.where(surv)[0]
                        surv[si[~m[si]]] = False
                    bf_hits += surv.sum()

                bf_rate = bf_hits / bf_N if bf_N > 0 else 0.0
                bf_sigma = np.sqrt(bf_hits) / bf_N if bf_N > 0 else 0.0
                dev = abs(p_lep - bf_rate)
                within_2sigma = dev <= 2.0 * bf_sigma if bf_sigma > 0 else False

                print(f"\n  ── VALIDATION: {stage_name} (lepton only) ──")
                print(f"  T2 analytic: {p_lep:.6e}")
                print(f"  BF rate:     {bf_rate:.6e} ± {bf_sigma:.6e}")
                print(f"  Deviation:   {dev:.6e} = {dev/bf_sigma:.2f}σ" if bf_sigma > 0 else f"  Deviation: {dev:.6e}")
                print(f"  Within 2σ:   {'✓ PASS' if within_2sigma else '✗ FAIL'}")

                validation_record[stage_name] = {
                    "eligible": True,
                    "lepton_factor": lepton_stage_factor,
                    "t2_analytic": float(p_lep),
                    "bf_hits": int(bf_hits),
                    "bf_N": int(bf_N),
                    "bf_rate": float(bf_rate),
                    "bf_sigma": float(bf_sigma),
                    "within_2sigma": bool(within_2sigma),
                }
                if not within_2sigma:
                    all_stages_validated = False
                continue

        # ── Step 2: For subsequent stages, try factors for new claims ──
        # Lepton window uses lepton_stage_factor (from first stage)
        # Anchor averaging is applied

        # First, draw lepton samples from the inflated window (for anchor averaging)
        print(f"\n  Drawing >= 1e4 lepton samples from window at factor {lepton_stage_factor}...")
        N_lep_samples = 10000
        lepton_samples_list = []
        rng_lep_samples = np.random.default_rng(calibration_seed + 1000 + stage_idx)

        if condition == "T0":
            # Sample from L1∧L2∧L3 window
            f2 = lepton_stage_factor
            f3 = lepton_stage_factor
            f1 = lepton_stage_factor
            u_lo = np.log(L2_TARGET * (1.0 - L2_TOL * f2))
            u_hi = np.log(L2_TARGET * (1.0 + L2_TOL * f2))
            v_lo = np.log(L3_TARGET * (1.0 - L3_TOL * f3))
            v_hi = np.log(L3_TARGET * (1.0 + L3_TOL * f3))

            while len(lepton_samples_list) < N_lep_samples:
                u_s = rng_lep_samples.uniform(u_lo, u_hi, batch_size)
                v_s = rng_lep_samples.uniform(v_lo, v_hi, batch_size)
                m1 = np.ones(batch_size)
                m2 = m1 * np.exp(u_s)
                m3 = m2 * np.exp(v_s)
                lep_cands = np.column_stack([m1, m2, m3])
                kd = kdist(lep_cands)
                ok = kd <= L1_TOL * f1
                if ok.any():
                    lepton_samples_list.append(lep_cands[ok])
        else:
            # T1: sample from r-range
            V_r = np.log(1e-1 / 1e-5)
            r_min, r_max, _ = compute_r_intersection(lepton_stage_factor)
            if r_min is None:
                print(f"  *** T1 r-intersection not found at factor {lepton_stage_factor} ***")
                validation_record[stage_name] = {"eligible": False, "error": "no_r_intersection"}
                all_stages_validated = False
                continue

            while len(lepton_samples_list) < N_lep_samples:
                r_s = np.exp(rng_lep_samples.uniform(np.log(r_min), np.log(r_max), batch_size))
                m3_s = np.exp(rng_lep_samples.uniform(LEP_LOG_LO, LEP_LOG_HI, batch_size))
                m1_s = r_s * m3_s
                s1_s, s3_s = np.sqrt(m1_s), np.sqrt(m3_s)
                b_s = -4.0 * (s1_s + s3_s)
                c_s = s1_s**2 + s3_s**2 - 4.0 * s1_s * s3_s
                disc_s = b_s**2 - 4.0 * c_s
                ok_disc = disc_s >= 0
                disc_s = np.maximum(disc_s, 0.0)
                s2_s = (-b_s - np.sqrt(disc_s)) / 2.0
                s2_ok = s2_s > 0
                sort_ok = (m1_s < s2_s**2) & (s2_s**2 < m3_s)
                hier_ok = (m3_s / m1_s) > HIERARCHY_MIN
                keep = ok_disc & s2_ok & sort_ok & hier_ok
                if keep.any():
                    m2_s = s2_s[keep] ** 2
                    lep_ok = np.column_stack([m1_s[keep], m2_s, m3_s[keep]])
                    lepton_samples_list.append(lep_ok)

        lepton_samples = np.vstack(lepton_samples_list)[:N_lep_samples]
        print(f"  Got {len(lepton_samples)} lepton samples for anchor averaging")

        # ── Step 3: Try inflation factors for new claims only ──
        best_factor = None
        best_result = None

        for f_new in INFLATION_FACTORS:
            if best_factor is not None:
                break

            print(f"\n  Testing new-claim factor {f_new}...", flush=True)

            # Non-vacuous gate for new claims
            vacuous = False
            for claim in new_claims:
                nv, frac = check_nonvacuous(claim, f_new)
                if not nv:
                    print(f"    VACUOUS: {claim} at factor {f_new}: {frac*100:.1f}%")
                    vacuous = True
                    break
            if vacuous:
                continue

            # ── Analytic side: anchor-averaged quark-block probability ──
            t0_anchor = time.time()
            claim_factors = {c: f_new for c in new_claims}
            p_quark_avg, quark_probs = quark_block_probability_anchored(
                lepton_samples, new_claims, claim_factors,
                prior=prior, N_quark_per_lepton=50000,
                rng_seed=calibration_seed + stage_idx * 10000
            )
            anchor_time = time.time() - t0_anchor

            # Lepton block probability at lepton_stage_factor
            lep_factors = {c: lepton_stage_factor for c in lepton_claims}
            p_lep, _ = tier2_lepton_block_prob(condition, lep_factors)

            # Joint analytic estimate
            t2_analytic = p_lep * p_quark_avg

            print(f"    Anchor averaging: {anchor_time:.1f}s")
            print(f"    p_lep = {p_lep:.6e}, p_quark_avg = {p_quark_avg:.6e}")
            print(f"    T2 analytic = {t2_analytic:.6e}")

            # ── Brute force side ──
            t0_bf = time.time()
            rng_stage = np.random.default_rng(calibration_seed + stage_idx * 7777)
            bf_hits = 0
            bf_N = 0
            n_batches_done = 0

            while bf_hits < 100 and bf_N < N_max:
                n_batches_done += 1
                if condition == "T0":
                    lep = draw_mass_triple(rng_stage, batch_size, LEP_LO, LEP_HI, prior, sort=True)
                    n_acc = batch_size
                    bf_N += batch_size
                else:
                    lep, attempted = sample_t1_koide(rng_stage, batch_size)
                    bf_N += attempted
                    n_acc = len(lep)
                    if n_acc == 0:
                        continue

                # Filter by lepton claims (at lepton_stage_factor)
                surv = np.ones(n_acc, dtype=bool)
                for claim in lepton_claims:
                    f_lep = lepton_stage_factor
                    if claim == "L1":
                        m = check_L1(lep, f_lep)
                    elif claim == "L2":
                        m = check_L2(lep, f_lep)
                    elif claim == "L3":
                        m = check_L3(lep, f_lep)
                    si = np.where(surv)[0]
                    surv[si[~m[si]]] = False

                n_surv = surv.sum()
                if n_surv == 0:
                    continue

                # Draw quarks for survivors
                lep_surv = lep[surv]
                lq = draw_mass_triple(rng_stage, n_surv, QUARK_LO, QUARK_HI, prior, sort=True)
                us_bf = draw_mass_pair(rng_stage, n_surv, QUARK_LO, QUARK_HI, prior)

                # Check new claims (at f_new, stage-local)
                surv_q = np.ones(n_surv, dtype=bool)
                for claim in new_claims:
                    if claim == "Q1":
                        m = check_Q1(lq, lep_surv, f_new)
                    elif claim == "Q2":
                        m = check_Q2(lq, lep_surv, f_new)
                    elif claim == "U1_fixed":
                        m = check_U1_fixed(lq, us_bf, f_new)
                    elif claim == "U1_menu":
                        m = check_U1_menu(lq, us_bf, f_new)
                    si = np.where(surv_q)[0]
                    surv_q[si[~m[si]]] = False

                bf_hits += surv_q.sum()

                if n_batches_done % 20 == 0:
                    print(f"    BF [{stage_name}/f_new={f_new}] N={bf_N:,}, hits={bf_hits}", flush=True)

            bf_time = time.time() - t0_bf
            bf_rate = bf_hits / bf_N if bf_N > 0 else 0.0
            print(f"    BF DONE: hits={bf_hits}, N={bf_N:,}, rate={bf_rate:.6e}, {bf_time:.0f}s")

            if bf_hits >= 100:
                # Validation
                bf_sigma = np.sqrt(bf_hits) / bf_N if bf_N > 0 else 0.0
                deviation = abs(t2_analytic - bf_rate)
                within_2sigma = deviation <= 2.0 * bf_sigma if bf_sigma > 0 else False

                best_factor = f_new
                best_result = {
                    "new_claim_factor": f_new,
                    "lepton_stage_factor": lepton_stage_factor,
                    "t2_analytic": float(t2_analytic),
                    "p_lep": float(p_lep),
                    "p_quark_avg": float(p_quark_avg),
                    "bf_hits": int(bf_hits),
                    "bf_N": int(bf_N),
                    "bf_rate": float(bf_rate),
                    "bf_sigma": float(bf_sigma),
                    "deviation_sigma": float(deviation / bf_sigma) if bf_sigma > 0 else float('inf'),
                    "within_2sigma": bool(within_2sigma),
                    "anchor_samples": N_lep_samples,
                    "N_quark_per_lepton": 50000,
                }

                print(f"\n  ── VALIDATION: {stage_name} ──")
                print(f"  T2 analytic: {t2_analytic:.6e} (p_lep={p_lep:.6e} × p_quark_avg={p_quark_avg:.6e})")
                print(f"  BF rate:     {bf_rate:.6e} ± {bf_sigma:.6e}")
                print(f"  Deviation:   {deviation:.6e} = {deviation/bf_sigma:.2f}σ" if bf_sigma > 0 else "")
                print(f"  Within 2σ:   {'✓ PASS' if within_2sigma else '✗ FAIL'}")

                if not within_2sigma:
                    all_stages_validated = False

        # ── Handle +U1 compositional eligibility ──
        if best_factor is None and stage_name == "+U1":
            print(f"\n  +U1: No direct factor reached 100 hits — trying compositional validation (§v0.5 c)...")

            # (i) P(U1) singleton at factor 1
            print(f"  (i) P(U1) singleton at factor 1...")
            rng_u1 = np.random.default_rng(calibration_seed + 55555)
            # U1 singleton: Q_U on (mu, mc, mt) with direct/inverse union
            N_u1_sing = 10_000_000
            lq_sing = np.exp(rng_u1.uniform(QUARK_LOG_LO, QUARK_LOG_HI, size=(N_u1_sing // 100, 3)))
            lq_sing.sort(axis=1)
            us_sing = np.exp(rng_u1.uniform(QUARK_LOG_LO, QUARK_LOG_HI, size=(N_u1_sing // 100, 2)))
            u1_sing_hits = 0
            u1_sing_N = 0
            batch_u1 = 100000
            n_b_u1 = N_u1_sing // batch_u1
            for _ in range(n_b_u1):
                lq_b = np.exp(rng_u1.uniform(QUARK_LOG_LO, QUARK_LOG_HI, size=(batch_u1, 3)))
                lq_b.sort(axis=1)
                us_b = np.exp(rng_u1.uniform(QUARK_LOG_LO, QUARK_LOG_HI, size=(batch_u1, 2)))
                if u1_mode == "fixed":
                    hits_b = check_U1_fixed(lq_b, us_b, 1.0).sum()
                else:
                    hits_b = check_U1_menu(lq_b, us_b, 1.0).sum()
                u1_sing_hits += hits_b
                u1_sing_N += batch_u1

            u1_sing_rate = u1_sing_hits / u1_sing_N
            u1_sing_sigma = np.sqrt(u1_sing_hits) / u1_sing_N
            print(f"    P(U1) singleton: {u1_sing_rate:.6e} ± {u1_sing_sigma:.6e} ({u1_sing_hits}/{u1_sing_N})")

            # (ii) mu-coupled Q2^U1 pair at adaptive stage-local factor
            # Find smallest factor for Q2^U1 that's non-vacuous and gives >=100 hits
            print(f"  (ii) Q2^U1 pair at adaptive factor...")
            q2u1_factor = None
            q2u1_result = None
            for f_pair in INFLATION_FACTORS:
                # Check non-vacuous
                nv_q2, _ = check_nonvacuous("Q2", f_pair)
                nv_u1, _ = check_nonvacuous(f"U1_{u1_mode}", f_pair)
                if not nv_q2 or not nv_u1:
                    print(f"    Factor {f_pair}: VACUOUS (Q2={nv_q2}, U1={nv_u1})")
                    continue

                # Brute force: start from lepton stage, then Q2, then U1
                rng_pair = np.random.default_rng(calibration_seed + 66666)
                pair_hits = 0
                pair_N = 0
                while pair_hits < 100 and pair_N < N_max:
                    if condition == "T0":
                        lep = draw_mass_triple(rng_pair, batch_size, LEP_LO, LEP_HI, prior, sort=True)
                        n_acc = batch_size
                        pair_N += batch_size
                    else:
                        lep, attempted = sample_t1_koide(rng_pair, batch_size)
                        pair_N += attempted
                        n_acc = len(lep)
                        if n_acc == 0:
                            continue

                    # Lepton stage filter
                    surv = np.ones(n_acc, dtype=bool)
                    for claim in lepton_claims:
                        if claim == "L1":
                            m = check_L1(lep, lepton_stage_factor)
                        elif claim == "L2":
                            m = check_L2(lep, lepton_stage_factor)
                        elif claim == "L3":
                            m = check_L3(lep, lepton_stage_factor)
                        si = np.where(surv)[0]
                        surv[si[~m[si]]] = False
                    n_surv = surv.sum()
                    if n_surv == 0:
                        continue

                    lep_s = lep[surv]
                    lq = draw_mass_triple(rng_pair, n_surv, QUARK_LO, QUARK_HI, prior, sort=True)
                    us_p = draw_mass_pair(rng_pair, n_surv, QUARK_LO, QUARK_HI, prior)

                    q2_ok = check_Q2(lq, lep_s, f_pair)
                    q2_idx = np.where(q2_ok)[0]
                    if len(q2_idx) == 0:
                        continue

                    if u1_mode == "fixed":
                        u1_ok = check_U1_fixed(lq[q2_idx], us_p[q2_idx], f_pair)
                    else:
                        u1_ok = check_U1_menu(lq[q2_idx], us_p[q2_idx], f_pair)
                    pair_hits += u1_ok.sum()

                print(f"    Q2^U1 at factor {f_pair}: {pair_hits} hits in {pair_N:,} draws")
                if pair_hits >= 100:
                    q2u1_factor = f_pair
                    q2u1_result = {"factor": f_pair, "hits": int(pair_hits), "N": int(pair_N),
                                   "rate": float(pair_hits/pair_N)}
                    break

            # ── Compositional validation verdict ──
            if q2u1_factor is not None:
                # Valid: both (i) and (ii) succeeded
                # Compute analytic for comparison
                # P(Q2^U1 | lepton) anchored-averaged
                p_q2u1_anchored, _ = quark_block_probability_anchored(
                    lepton_samples, ["Q2", f"U1_{u1_mode}"],
                    {"Q2": q2u1_factor, f"U1_{u1_mode}": q2u1_factor},
                    prior=prior, N_quark_per_lepton=50000,
                    rng_seed=calibration_seed + 77777
                )
                lep_factors = {c: lepton_stage_factor for c in lepton_claims}
                p_lep, _ = tier2_lepton_block_prob(condition, lep_factors)
                t2_q2u1 = p_lep * p_q2u1_anchored

                q2u1_rate = q2u1_result["rate"]
                q2u1_sigma = np.sqrt(q2u1_result["hits"]) / q2u1_result["N"]
                dev_q2u1 = abs(t2_q2u1 - q2u1_rate)
                within_2sigma_q2u1 = dev_q2u1 <= 2.0 * q2u1_sigma

                u1_dev = abs(u1_sing_rate - 3.4e-3)  # expected ~3.4e-3
                u1_within_2sigma = u1_dev <= 2.0 * u1_sing_sigma

                compositional_ok = within_2sigma_q2u1 and u1_within_2sigma

                print(f"\n  ── COMPOSITIONAL VALIDATION: +U1 ──")
                print(f"  (i)  P(U1) singleton at f=1: {u1_sing_rate:.6e} ± {u1_sing_sigma:.6e}")
                print(f"       Expected ~3.4e-3, within 2σ: {'✓' if u1_within_2sigma else '✗'}")
                print(f"  (ii) Q2^U1 at factor {q2u1_factor}:")
                print(f"       T2: {t2_q2u1:.6e}, BF: {q2u1_rate:.6e} ± {q2u1_sigma:.6e}")
                print(f"       Deviation: {dev_q2u1/q2u1_sigma:.2f}σ, within 2σ: {'✓' if within_2sigma_q2u1 else '✗'}")
                print(f"  VALIDATED-BY-COMPOSITION: {'✓' if compositional_ok else '✗ FAIL'}")

                best_result = {
                    "method": "compositional",
                    "lepton_stage_factor": lepton_stage_factor,
                    "u1_singleton": {"rate": float(u1_sing_rate), "hits": int(u1_sing_hits), "N": int(u1_sing_N),
                                     "within_2sigma": bool(u1_within_2sigma)},
                    "q2u1_pair": {"factor": q2u1_factor, "t2_analytic": float(t2_q2u1),
                                  "bf_rate": float(q2u1_rate), "bf_hits": q2u1_result["hits"],
                                  "bf_N": q2u1_result["N"], "within_2sigma": bool(within_2sigma_q2u1)},
                    "validated_by_composition": bool(compositional_ok),
                    "within_2sigma": bool(compositional_ok),
                }
                validation_record[stage_name] = {"eligible": True, **best_result}
                if not compositional_ok:
                    all_stages_validated = False
            else:
                print(f"\n  +U1: COMPOSITIONAL VALIDATION FAILED — Q2^U1 no factor reached 100 hits")
                validation_record[stage_name] = {"eligible": False, "error": "compositional_no_factor"}
                all_stages_validated = False
        elif best_factor is None:
            print(f"\n  Stage {stage_name}: INELIGIBLE — no factor reached 100 hits")
            validation_record[stage_name] = {"eligible": False, "error": "no_factor_reached_100"}
            all_stages_validated = False
        else:
            validation_record[stage_name] = {"eligible": True, **best_result}

    return validation_record, all_stages_validated

# ═══════════════════════════════════════════════════════════════════════════
# TIER-1: SINGLETONS + CASCADE
# ═══════════════════════════════════════════════════════════════════════════

ALL_CLAIMS = ["L1", "L2", "L3", "Q1", "Q2", "U1_fixed", "U1_menu"]

def measure_singletons(rng, N, prior="logU", condition="T0"):
    batch_size = 1_000_000
    n_batches = N // batch_size
    counts = {c: 0 for c in ALL_CLAIMS}
    total_eff = 0
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
            counts["L1"] += n_acc

        light_q = draw_mass_triple(rng, n_acc, QUARK_LO, QUARK_HI, prior, sort=True)
        up_s = draw_mass_pair(rng, n_acc, QUARK_LO, QUARK_HI, prior)

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
        f_rate = k / total_eff if total_eff > 0 else 0.0
        lo, hi = clopper_pearson(k, total_eff)
        result[claim] = {"count": k, "rate": float(f_rate),
                         "cp95_lower": lo, "cp95_upper": hi}
    return result, total_eff


def run_tier1_cascade(rng, N_eff, prior="logU", condition="T0", u1_mode="fixed",
                       singleton_rates=None):
    """Tier-1 rarity-ordered cascade."""
    batch_size = 1_000_000
    total_eff = 0
    total_hits = 0
    stage_counts = {}
    t0 = time.time()

    if condition == "T0":
        cascade_order = ["L1", "L2", "L3", "Q1", "Q2", f"U1_{u1_mode}"]
    else:
        cascade_order = ["L2", "L3", "Q1", "Q2", f"U1_{u1_mode}"]

    # Reorder by measured rarity if singleton rates available
    if singleton_rates:
        claim_order = [(c, singleton_rates.get(c, {}).get("rate", 1.0)) for c in cascade_order]
        claim_order.sort(key=lambda x: x[1])
        cascade_order = [c for c, _ in claim_order]
        print(f"  Cascade order (rarest first): {cascade_order}")

    for claim in cascade_order:
        stage_counts[claim] = 0

    n_batches = N_eff // batch_size
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

        for claim in cascade_order:
            si = np.where(survivors)[0]
            if len(si) == 0:
                break
            if claim == "L1":
                m = check_L1(leptons[si])
            elif claim == "L2":
                m = check_L2(leptons[si])
            elif claim == "L3":
                m = check_L3(leptons[si])
            elif claim == "Q1":
                m = check_Q1(light_q[si], leptons[si])
            elif claim == "Q2":
                m = check_Q2(light_q[si], leptons[si])
            elif claim == "U1_fixed":
                m = check_U1_fixed(light_q[si], up_s[si])
            elif claim == "U1_menu":
                m = check_U1_menu(light_q[si], up_s[si])
            else:
                m = np.ones(len(si), dtype=bool)
            survivors[si[~m]] = False
            stage_counts[claim] += m.sum()

        total_hits += survivors.sum()
        if (b + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = total_eff / elapsed if elapsed > 0 else 0
            print(f"  [T1/{condition}/{u1_mode}/{prior}] batch {b+1}/{n_batches}, "
                  f"N_eff={total_eff:,}, hits={total_hits}, rate={rate:.0f}/s",
                  flush=True)

    elapsed = time.time() - t0
    f_rate = total_hits / total_eff if total_eff > 0 else 0.0
    lo, hi = clopper_pearson(int(total_hits), int(total_eff))
    print(f"  DONE: hits={total_hits}, N_eff={total_eff:,}, f={f_rate:.6e}, "
          f"CP95=[{lo:.6e},{hi:.6e}], {elapsed:.0f}s")

    return {
        "hits": int(total_hits), "N_eff": int(total_eff),
        "rate": float(f_rate), "cp95_lower": lo, "cp95_upper": hi,
        "stage_counts": {k: int(v) for k, v in stage_counts.items()},
    }

# ═══════════════════════════════════════════════════════════════════════════
# SUPPORT GATE
# ═══════════════════════════════════════════════════════════════════════════

def check_support_gate():
    r_obs = 0.51099895 / 1776.93
    r_lo, r_hi = 1e-5, 1e-1
    if not (r_lo <= r_obs <= r_hi):
        print(f"*** SUPPORT GATE FAILED: r_obs={r_obs:.6e} not in [{r_lo}, {r_hi}]")
        return False
    hierarchy_ratio = 1776.93 / 0.51099895
    if hierarchy_ratio <= HIERARCHY_MIN:
        print(f"*** SUPPORT GATE FAILED: m3/m1={hierarchy_ratio:.1f} <= {HIERARCHY_MIN:.1f}")
        return False
    print(f"SUPPORT GATE PASSED: r_obs={r_obs:.6e} in [{r_lo}, {r_hi}], "
          f"m3/m1={hierarchy_ratio:.1f} > {HIERARCHY_MIN:.1f}")
    return True

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Joint Coincidence Budget Engine v0.5")
    parser.add_argument("--mode", default="full",
                        choices=["singletons", "tier1", "tier2_validate", "full", "report"])
    parser.add_argument("--prior", default="logU", choices=["logU", "logN", "linU"])
    parser.add_argument("--condition", default="T1", choices=["T0", "T1"])
    parser.add_argument("--u1-mode", default="menu", choices=["fixed", "menu"])
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--N-tier1", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.outdir is None:
        args.outdir = "results/amb-20260811-v0.5"
    os.makedirs(args.outdir, exist_ok=True)

    if not check_support_gate():
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"JOINT COINCIDENCE BUDGET ENGINE v0.5 — amb")
    print(f"Mode: {args.mode} | Prior: {args.prior}")
    print(f"Condition: {args.condition} | U1: {args.u1_mode}")
    print(f"Output: {args.outdir}")
    print(f"{'='*70}\n")

    seed = args.seed if args.seed else (20260811 if args.prior == "logU" else 314160)
    rng = np.random.default_rng(seed)

    if args.mode == "singletons":
        N_sing = args.N_tier1 if args.N_tier1 else 10_000_000
        print(f"Measuring singletons at N={N_sing:,}")
        rates, N = measure_singletons(rng, N_sing, args.prior, args.condition)
        outpath = os.path.join(args.outdir,
            f"singletons_{args.condition}_{args.u1_mode}_{args.prior}_seed{seed}.json")
        with open(outpath, "w") as f:
            json.dump({"seed": seed, "N": N, "rates": rates, "engine_id": "amb",
                       "spec_version": "v0.5"}, f, indent=2)
        print(f"Saved: {outpath}")
        for c in ALL_CLAIMS:
            r = rates[c]
            print(f"  {c}: {r['count']}/{N} = {r['rate']:.6e} CP95=[{r['cp95_lower']:.6e},{r['cp95_upper']:.6e}]")

    elif args.mode == "tier1":
        N_t1 = args.N_tier1 if args.N_tier1 else 2_000_000_000
        print(f"Running Tier-1 cascade at N_eff={N_t1:,}")
        result = run_tier1_cascade(rng, N_t1, args.prior, args.condition, args.u1_mode)
        outpath = os.path.join(args.outdir,
            f"tier1_{args.condition}_{args.u1_mode}_{args.prior}_seed{seed}.json")
        with open(outpath, "w") as f:
            json.dump({"seed": seed, **result, "engine_id": "amb",
                       "spec_version": "v0.5"}, f, indent=2)
        print(f"Saved: {outpath}")
        print(f"  Joint: {result['hits']}/{result['N_eff']} = {result['rate']:.6e}")
        print(f"  CP95: [{result['cp95_lower']:.6e}, {result['cp95_upper']:.6e}]")

    elif args.mode == "tier2_validate":
        print("Running v0.5 per-stage validation with anchor averaging...")
        val_record, all_validated = v05_per_stage_validation(
            args.prior, args.condition, args.u1_mode,
            calibration_seed=271828, N_max=1_000_000_000
        )
        outpath = os.path.join(args.outdir,
            f"tier2_validation_{args.condition}_{args.u1_mode}_{args.prior}_v0.5.json")
        output = {
            "engine_id": "amb", "spec_version": "v0.5",
            "condition": args.condition, "u1_mode": args.u1_mode, "prior": args.prior,
            "calibration_seed": 271828,
            "all_stages_validated": bool(all_validated),
            "validation_record": val_record,
        }
        with open(outpath, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved: {outpath}")
        if all_validated:
            print("\n✓ ALL STAGES VALIDATED — Tier-2 estimates are TIER-2 (VALIDATED)")
        else:
            print("\n✗ VALIDATION FAILED — Tier-2 estimates UNVALIDATED")

    elif args.mode == "full":
        # Run all cells
        all_cell_results = {}
        conditions = ["T0", "T1"]
        u1_modes = ["fixed", "menu"]
        priors = ["logU", "logN", "linU"]
        primary_N = 2_000_000_000
        variant_N = 200_000_000
        primary_prior = "logU"

        for prior in priors:
            N_eff = primary_N if prior == primary_prior else variant_N
            seed_p = 20260811 if prior == primary_prior else 314160
            rng_p = np.random.default_rng(seed_p)

            for condition in conditions:
                for u1_mode in u1_modes:
                    cell_key = f"{condition}_{u1_mode}_{prior}"
                    print(f"\n{'─'*60}")
                    print(f"CELL: {cell_key}  (N_eff={N_eff:,}, seed={seed_p})")
                    print(f"{'─'*60}")

                    # Tier-1 cascade
                    result = run_tier1_cascade(rng_p, N_eff, prior, condition, u1_mode)
                    # Save
                    cell_outpath = os.path.join(args.outdir,
                        f"tier1_{cell_key}_seed{seed_p}.json")
                    with open(cell_outpath, "w") as f:
                        json.dump({"seed": seed_p, "cell": cell_key, **result,
                                   "engine_id": "amb", "spec_version": "v0.5"}, f, indent=2)

                    bound_type = "BOUND" if result["hits"] < 100 else "POINT"
                    all_cell_results[cell_key] = {**result, "tier": "Tier-1", "type": bound_type}
                    print(f"  → {bound_type}: f={result['rate']:.6e} "
                          f"CP95=[{result['cp95_lower']:.6e},{result['cp95_upper']:.6e}]")

        # Save summary
        summary_path = os.path.join(args.outdir, "tier1_summary_v0.5.json")
        with open(summary_path, "w") as f:
            json.dump({"engine_id": "amb", "spec_version": "v0.5",
                       "cells": all_cell_results}, f, indent=2)
        print(f"\nSummary saved to {summary_path}")

        # Tier-2 validation for primary cells
        print(f"\n{'='*70}")
        print("v0.5 TIER-2 VALIDATION (anchor averaging)")
        print(f"{'='*70}")
        t2_results = {}
        for condition in conditions:
            for u1_mode in u1_modes:
                cell_key = f"{condition}_{u1_mode}_logU"
                print(f"\n  T2 validation for {cell_key}...")
                val_record, all_val = v05_per_stage_validation(
                    "logU", condition, u1_mode,
                    calibration_seed=271828, N_max=1_000_000_000
                )
                t2_outpath = os.path.join(args.outdir,
                    f"tier2_validation_{cell_key}_v0.5.json")
                with open(t2_outpath, "w") as f:
                    json.dump({"engine_id": "amb", "spec_version": "v0.5",
                               "cell": cell_key, "all_stages_validated": bool(all_val),
                               "validation_record": val_record}, f, indent=2)
                t2_results[cell_key] = {"all_validated": bool(all_val), "record": val_record}
                print(f"  {cell_key}: {'✓ VALIDATED' if all_val else '✗ FAILED'}")

        # Save tier-2 summary
        t2_summary_path = os.path.join(args.outdir, "tier2_summary_v0.5.json")
        with open(t2_summary_path, "w") as f:
            json.dump({"engine_id": "amb", "spec_version": "v0.5",
                       "tier2_results": {k: {"all_validated": v["all_validated"]}
                                        for k, v in t2_results.items()}}, f, indent=2)

    elif args.mode == "report":
        print("Generating report from saved results...")
        # This will be handled separately
        pass

    print(f"\nDONE — engine v0.5")
    sys.exit(0)


if __name__ == "__main__":
    main()

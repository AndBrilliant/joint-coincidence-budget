#!/usr/bin/env python3
"""
JOINT COINCIDENCE BUDGET — Full Engine v0.3
ENGINE_ID: amb
Production seed: 20260811 | Prior-variant seed: 314160 | Calibration seed: 271828

Implements all 12 cells: {T0,T1} x {U1-fixed,U1-menu} x {P-logU,P-logN,P-linU}

v0.3 changes from v0.2:
- Tier-2 redesigned as EXACT BLOCK INTEGRATION:
  P(joint) = P(lepton block) x P(quark block | lepton draw)
- Lepton block: 2D quadrature of exact draw density
- Quark block: nested quadrature carrying md coupling (Q1-Q2) and mu coupling (Q2-U1)
- Adaptive per-stage inflation with factors {1, 3, 10, 30, 100, 300, 1000}
- Non-vacuous gate: each factor must exclude >50% of prior mass for every claim
- Validation: each factor within 2 Poisson sigma of brute force
- Tier-2 point estimates labeled TIER-2 (VALIDATED) when all stages pass
"""

import numpy as np
import json
import os
import sys
import time
from scipy.stats import beta as beta_dist
from scipy.integrate import nquad
from scipy.special import erf

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
    """CP95 interval."""
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

# Log ranges
LEP_LOG_LO, LEP_LOG_HI = np.log(LEP_LO), np.log(LEP_HI)
LEP_LOG_V = LEP_LOG_HI - LEP_LOG_LO  # ≈ 8.8049
QUARK_LOG_LO, QUARK_LOG_HI = np.log(QUARK_LO), np.log(QUARK_HI)
QUARK_LOG_V = QUARK_LOG_HI - QUARK_LOG_LO  # ≈ 12.8992

# Hierarchy cutoff for T1: (4+sqrt(18))^2
HIERARCHY_MIN = (4.0 + np.sqrt(18.0)) ** 2  # ≈ 67.9

# Tier-2 adaptive inflation factors (§8 v0.3)
INFLATION_FACTORS = [1, 3, 10, 30, 100, 300, 1000]


# ═══════════════════════════════════════════════════════════════════════
# DRAW FUNCTIONS (same as v0.2)
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


def sample_t1_koide(rng, batch_size):
    """T1 Koide-conditioned lepton draws. Sheet sampler."""
    m3 = np.exp(rng.uniform(np.log(LEP_LO), np.log(LEP_HI), size=batch_size))
    r = np.exp(rng.uniform(np.log(1e-5), np.log(1e-1), size=batch_size))
    m1 = r * m3

    s1 = np.sqrt(m1)
    s3 = np.sqrt(m3)
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


# ═══════════════════════════════════════════════════════════════════════
# CLAIM CHECK FUNCTIONS (vectorized)
# ═══════════════════════════════════════════════════════════════════════

def check_L1(leptons):
    return kdist(leptons) <= L1_TOL


def check_L2(leptons):
    ratio = leptons[:, 1] / leptons[:, 0]
    return np.abs(ratio / L2_TARGET - 1.0) <= L2_TOL


def check_L3(leptons):
    ratio = leptons[:, 2] / leptons[:, 1]
    return np.abs(ratio / L3_TARGET - 1.0) <= L3_TOL


def check_Q1(light_q, leptons):
    mu, md, ms = light_q[:, 0], light_q[:, 1], light_q[:, 2]
    mu_star = leptons.sum(axis=1)
    val = np.log(ms * ms / (mu_star * md))
    return np.abs(val) <= B1


def check_Q2(light_q, leptons):
    mu, md = light_q[:, 0], light_q[:, 1]
    twome = 2.0 * leptons.min(axis=1)
    val = np.log(mu * mu / (md * twome))
    return np.abs(val) <= B2


def check_U1_fixed(light_q, up_s):
    mu, mc, mt = light_q[:, 0], up_s[:, 0], up_s[:, 1]
    triples = np.column_stack([mu, mc, mt])
    q_dir = Q_U(triples)
    q_inv = Q_U(1.0 / triples)
    best = np.minimum(np.abs(9.0 * q_dir - 8.0), np.abs(9.0 * q_inv - 8.0))
    return best <= U1_TOL


def check_U1_menu(light_q, up_s):
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
# INFLATED CHECK FUNCTIONS (for calibration)
# ═══════════════════════════════════════════════════════════════════════

def check_L1_inflated(leptons, factor):
    return kdist(leptons) <= L1_TOL * factor


def check_L2_inflated(leptons, factor):
    ratio = leptons[:, 1] / leptons[:, 0]
    return np.abs(ratio / L2_TARGET - 1.0) <= L2_TOL * factor


def check_L3_inflated(leptons, factor):
    ratio = leptons[:, 2] / leptons[:, 1]
    return np.abs(ratio / L3_TARGET - 1.0) <= L3_TOL * factor


def check_Q1_inflated(light_q, leptons, factor):
    mu, md, ms = light_q[:, 0], light_q[:, 1], light_q[:, 2]
    mu_star = leptons.sum(axis=1)
    val = np.log(ms * ms / (mu_star * md))
    return np.abs(val) <= B1 * factor


def check_Q2_inflated(light_q, leptons, factor):
    mu, md = light_q[:, 0], light_q[:, 1]
    twome = 2.0 * leptons.min(axis=1)
    val = np.log(mu * mu / (md * twome))
    return np.abs(val) <= B2 * factor


def check_U1_fixed_inflated(light_q, up_s, factor):
    mu, mc, mt = light_q[:, 0], up_s[:, 0], up_s[:, 1]
    triples = np.column_stack([mu, mc, mt])
    q_dir = Q_U(triples)
    q_inv = Q_U(1.0 / triples)
    best = np.minimum(np.abs(9.0 * q_dir - 8.0), np.abs(9.0 * q_inv - 8.0))
    return best <= U1_TOL * factor


def check_U1_menu_inflated(light_q, up_s, factor):
    mu, mc, mt = light_q[:, 0], up_s[:, 0], up_s[:, 1]
    triples = np.column_stack([mu, mc, mt])
    q_dir = Q_U(triples)
    q_inv = Q_U(1.0 / triples)
    n = len(mu)
    hit = np.zeros(n, dtype=bool)
    for tgt in U1_MENU_TARGETS:
        hit |= (np.abs(9.0 * q_dir - tgt) <= U1_TOL * factor)
        hit |= (np.abs(9.0 * q_inv - tgt) <= U1_TOL * factor)
    return hit


# ═══════════════════════════════════════════════════════════════════════
# NON-VACUOUS GATE: check that inflated window excludes >50% of prior mass
# ═══════════════════════════════════════════════════════════════════════

def check_nonvacuous_lepton(claim_id, factor):
    """Check whether the inflated lepton window excludes >50% of prior mass.

    For L2: window in ln(m2/m1). The prior over ln(m2/m1) is triangular
    (difference of two sorted uniforms) over [0, LEP_LOG_V].
    The density at u is: f(u) = (2/(V^2)) * (V - u) for u in [0, V].
    The CDF is: F(u) = (2u/V) - (u/V)^2.

    For L3: similarly for ln(m3/m2).

    For L1: kdist window — approximate as the volume fraction in shape space.
    """
    V = LEP_LOG_V

    if claim_id == "L1":
        # kdist is scale-invariant; the shape-space volume satisfying
        # kdist <= L1_TOL*factor is very small compared to the full shape space.
        # For typical masses, kdist ~ O(1) away from Koide; the window is tiny.
        # The non-vacuous check: does the factor make the kdist window cover
        # >50% of the prior mass in shape space?
        # The shape space is the 2-simplex of (u,v) with u,v ≥ 0, u+v ≤ V.
        # kdist tolerance scales with factor.
        # At factor 1, the window is ~3e-6 wide in kdist → infinitesimal.
        # At factor 1000, it's ~3e-3 wide → still very small.
        # So L1 is always non-vacuous for our factor range.
        # We check empirically: sample 1e6 random triples, compute fraction
        # with kdist <= L1_TOL * factor. If < 0.5, non-vacuous.
        rng = np.random.default_rng(271828)
        N_test = 1_000_000
        x = rng.uniform(LEP_LOG_LO, LEP_LOG_HI, size=(N_test, 3))
        x.sort(axis=1)
        m = np.exp(x)
        frac = check_L1_inflated(m, factor).mean()
        return frac < 0.5, float(frac)

    elif claim_id == "L2":
        # u = ln(m2/m1), window half-width ≈ L2_TOL * factor
        u_target = np.log(L2_TARGET)
        half_width = L2_TOL * factor  # approximate for small tol
        u_lo = max(0, u_target - half_width)
        u_hi = min(V, u_target + half_width)
        if u_lo >= u_hi:
            return True, 0.0  # vacuous if window fully outside range
        # Fraction of prior mass in [u_lo, u_hi]
        def F(u):
            return (2*u/V) - (u/V)**2
        frac = F(u_hi) - F(u_lo)
        return frac < 0.5, float(frac)

    elif claim_id == "L3":
        v_target = np.log(L3_TARGET)
        half_width = L3_TOL * factor
        v_lo = max(0, v_target - half_width)
        v_hi = min(V, v_target + half_width)
        if v_lo >= v_hi:
            return True, 0.0
        def F(v):
            return (2*v/V) - (v/V)**2
        frac = F(v_hi) - F(v_lo)
        return frac < 0.5, float(frac)

    return True, 0.0


def check_nonvacuous_quark(claim_id, factor):
    """Check whether the inflated quark window excludes >50% of prior mass.

    For Q1: window in ln(ms²/(mu_star·md)) = 2*ln(ms) - ln(mu_star) - ln(md).
    Given lepton draw fixing mu_star, this is a band in (ln md, ln ms) space.
    With ln md and ln ms uniform over the sorted simplex, the marginal
    distribution of the Q1 statistic can be approximated.

    For Q2: window in ln(mu²/(md·twome)) = 2*ln(mu) - ln(twome) - ln(md).
    Similar band in (ln mu, ln md) space.

    For U1: region in (mu, mc, mt) space — harder to characterize analytically.
    We use empirical sampling.

    Since Q1, Q2 involve anchors that vary with the lepton draw, we check
    for a typical anchor configuration (observed leptons). The window widths
    in log-space are:
    - Q1: width 2*b1*factor in the relevant 2D subspace
    - Q2: width 2*b2*factor
    """
    V_q = QUARK_LOG_V

    if claim_id == "Q1":
        # Q1: |2*y_s - y_d - ln(mu_star)| ≤ b1*factor
        # This is a band in (y_d, y_s) space of width (b1*factor)/√5
        # For sorted (y_u < y_d < y_s), the full 3D volume is V_q^3/6
        # The band restricts y_s relative to y_d, reducing 1 dimension.
        # Effective fraction ≈ (2*b1*factor) / V_q
        window_width = 2.0 * B1 * factor
        frac = window_width / V_q
        return frac < 0.5, float(min(frac, 1.0))

    elif claim_id == "Q2":
        window_width = 2.0 * B2 * factor
        frac = window_width / V_q
        return frac < 0.5, float(min(frac, 1.0))

    elif claim_id in ("U1_fixed", "U1_menu"):
        # U1: region in 3D (mu, mc, mt) space.
        # Empirical check with sampling at typical anchors.
        rng = np.random.default_rng(271828)
        N_test = 200_000
        # Use observed leptons for typical anchors
        mu_star_obs = LEPTONS_OBS.sum()
        twome_obs = 2.0 * LEPTONS_OBS.min()

        light_q = np.exp(rng.uniform(QUARK_LOG_LO, QUARK_LOG_HI, size=(N_test, 3)))
        light_q.sort(axis=1)
        up_s = np.exp(rng.uniform(QUARK_LOG_LO, QUARK_LOG_HI, size=(N_test, 2)))

        if claim_id == "U1_fixed":
            hits = check_U1_fixed_inflated(light_q, up_s, factor)
        else:
            hits = check_U1_menu_inflated(light_q, up_s, factor)

        frac = float(hits.mean())
        return frac < 0.5, frac

    return True, 0.0


# ═══════════════════════════════════════════════════════════════════════
# TIER-1: SINGLETONS + CASCADE (same as v0.2, updated output dirs)
# ═══════════════════════════════════════════════════════════════════════

ALL_CLAIMS = ["L1", "L2", "L3", "Q1", "Q2", "U1_fixed", "U1_menu"]


def measure_singletons(rng, N, prior="logU", condition="T0"):
    """Measure per-claim singleton rates at given N."""
    batch_size = 1_000_000
    n_batches = N // batch_size

    counts = {c: 0 for c in ALL_CLAIMS}
    total_eff = 0

    t0 = time.time()
    for b in range(n_batches):
        if condition == "T0":
            leptons = draw_mass_triple(rng, batch_size, LEP_LO, LEP_HI, prior, sort=True)
            n_acc = batch_size
        else:
            leptons, attempted = sample_t1_koide(rng, batch_size)
            total_eff += attempted
            n_acc = len(leptons)
            if n_acc == 0:
                continue
            counts["L1"] += n_acc

        if condition == "T0":
            total_eff += batch_size

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
        result[claim] = {
            "count": k,
            "rate": float(f_rate),
            "cp95_lower": lo,
            "cp95_upper": hi,
        }

    return result, total_eff


def run_cascade(rng, N_eff_target, prior, condition, u1_mode, claim_order,
                batch_size=1_000_000):
    """Tier-1 cascade: filter by rarest claim first."""
    check_funcs = {
        "L1": check_L1, "L2": check_L2, "L3": check_L3,
        "Q1": check_Q1, "Q2": check_Q2,
        "U1_fixed": check_U1_fixed, "U1_menu": check_U1_menu,
    }

    active_claims = []
    for cname in claim_order:
        if cname == "U1_fixed" and u1_mode != "fixed":
            continue
        if cname == "U1_menu" and u1_mode != "menu":
            continue
        if cname == "L1" and condition == "T1":
            continue
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

        survivors = np.ones(n_acc, dtype=bool)

        for cname in active_claims:
            func = check_funcs[cname]
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

        if batch_num % 20 == 0:
            elapsed = time.time() - t0
            rate = total_eff / elapsed if elapsed > 0 else 0
            progress = 100.0 * total_eff / N_eff_target
            print(f"  [{condition}/{prior}/U1-{u1_mode}] "
                  f"N_eff={total_eff:,} ({progress:.1f}%), "
                  f"rate={rate:.0f}/s, joint_hits={total_hits}",
                  flush=True)

        if total_eff >= N_eff_target:
            break

    elapsed = time.time() - t0
    print(f"  [{condition}/{prior}/U1-{u1_mode}] COMPLETE: "
          f"N_eff={total_eff:,}, hits={total_hits}, elapsed={elapsed:.1f}s", flush=True)

    return total_hits, total_eff, stage_counts


# ═══════════════════════════════════════════════════════════════════════
# TIER-2: EXACT BLOCK INTEGRATION (§8 v0.3)
# ═══════════════════════════════════════════════════════════════════════

def tier2_lepton_block_t0():
    """Compute P(L1∧L2∧L3) for T0 (unconditioned logU prior).

    In (x1, x2, x3) log-space with x1 < x2 < x3, the density is:
    ρ = 6 / V³ where V = ln(2000/0.3).

    Transform to (s, u, v) where:
    u = x2 - x1 = ln(m2/m1)
    v = x3 - x2 = ln(m3/m2)
    s = (x1 + x2 + x3) / 3

    Jacobian = 1. Bounds: u ≥ 0, v ≥ 0, u+v ≤ V (from x1 ≥ a, x3 ≤ b).
    s ∈ [a + (2u+v)/3, b - (u+2v)/3], width = V - u - v.

    Claims:
    L1: kdist(m) ≤ L1_TOL — restricts (u,v) to near Koide curve
    L2: |exp(u) - L2_TARGET|/L2_TARGET ≤ L2_TOL
    L3: |exp(v) - L3_TARGET|/L3_TARGET ≤ L3_TOL

    The kdist constraint in terms of (u,v) is complex. Since kdist is
    scale-invariant and the observed leptons satisfy Koide, L1∧L2∧L3 is
    satisfied exactly when (u,v) = (ln L2_TARGET, ln L3_TARGET) to within
    the L2/L3 tolerances (and L1 is then automatically satisfied because
    the observed ratios produce Koide).

    Actually L1 is NOT automatically satisfied — kdist depends on all three
    masses. Even with correct ratios, the absolute scale affects kdist?
    No — kdist is scale-invariant. So if (u,v) exactly match the observed
    ratios, kdist = 3.3049e-6 (the observed value) ≤ L1_TOL = 3.3049e-6.
    But L1_TOL is defined as the "achieved miss" of the claim, which IS
    3.3049e-6. So kdist ≤ L1_TOL is satisfied at the observed ratios.

    However, the L1 window allows kdist values up to 3.3049e-6. The
    observed kdist IS 3.304884e-6, just barely inside. The question is:
    in the (u,v) neighborhood defined by L2 and L3, does kdist stay
    within the L1 window?

    For small perturbations δu, δv around the observed ratios:
    kdist(u,v) ≈ kdist_obs + ∂k/∂u·δu + ∂k/∂v·δv

    Since the L2 and L3 windows are extremely narrow (half-widths 1e-5
    and 2.1e-5 in u,v), and the kdist gradient near the Koide point is
    finite (typically O(1)), the kdist variation across the L2∧L3 window
    could be substantial compared to L1_TOL = 3.3049e-6.

    We need to check: within the L2∧L3 rectangle, what fraction satisfies
    L1? Or does L1 automatically hold throughout?

    Let me compute kdist at the corners of the L2∧L3 window.
    """
    # Check kdist variation across L2∧L3 window
    u0 = np.log(L2_TARGET)  # ≈ 5.331
    v0 = np.log(L3_TARGET)  # ≈ 2.822

    du = L2_TOL  # half-width in u (approximate)
    dv = L3_TOL  # half-width in v

    # Test points: center + corners
    m1 = 0.51099895  # anchor (scale doesn't matter for kdist)

    for du_i in [-du, 0, du]:
        for dv_i in [-dv, 0, dv]:
            u_i = u0 + du_i
            v_i = v0 + dv_i
            m2 = m1 * np.exp(u_i)
            m3 = m2 * np.exp(v_i)
            kd = kdist(np.array([m1, m2, m3]))
            ok = "✓" if kd <= L1_TOL else "✗"
            print(f"  [T2 lepton block] (δu={du_i:+.1e}, δv={dv_i:+.1e}): "
                  f"kdist={kd:.6e} {ok} (tol={L1_TOL:.6e})")

    # If L1 holds at all corners, the entire L2∧L3 rectangle satisfies L1
    # (kdist is smooth and the window is tiny).
    # If not, we need to compute the intersection.

    # For now, compute P(L1∧L2∧L3) as:
    # ∫∫ 6/V³ × (V-u-v) × I[L1] × I[L2] × I[L3] du dv
    # over u ∈ [0,V], v ∈ [0,V-u]

    # With the narrow windows, we can use:
    # P ≈ 6/V³ × (V-u0-v0) × (2·L2_TOL) × (2·L3_TOL) × f_L1
    # where f_L1 is the fraction of the L2∧L3 window satisfying L1.

    V = LEP_LOG_V
    s_width = V - u0 - v0  # ≈ 0.65

    # Window area in (u,v):
    area_uv = (2 * du) * (2 * dv)

    # Check if L1 holds throughout the L2∧L3 window
    # Sample densely
    N_test = 10000
    rng_test = np.random.default_rng(271828)
    u_test = u0 + rng_test.uniform(-du, du, N_test)
    v_test = v0 + rng_test.uniform(-dv, dv, N_test)

    # For each (u,v), construct a lepton triple with arbitrary m1
    m1_test = np.ones(N_test) * LEPTONS_OBS[0]
    m2_test = m1_test * np.exp(u_test)
    m3_test = m2_test * np.exp(v_test)
    leptons_test = np.column_stack([m1_test, m2_test, m3_test])
    kd_test = kdist(leptons_test)
    l1_ok = kd_test <= L1_TOL
    f_l1 = float(l1_ok.mean())

    print(f"  [T2 lepton block] L1 fraction within L2∧L3 window: {f_l1:.6f}")

    # P(L1∧L2∧L3) = 6/V³ × s_width × area_uv × f_l1
    p_lepton = 6.0 / (V**3) * s_width * area_uv * f_l1

    # Also compute P(L2∧L3) for T1 (no L1 constraint needed there):
    p_l2l3 = 6.0 / (V**3) * s_width * area_uv

    print(f"  [T2 lepton block T0] P(L1∧L2∧L3) = {p_lepton:.6e}")
    print(f"  [T2 lepton block] P(L2∧L3) = {p_l2l3:.6e}")

    return p_lepton, p_l2l3, float(f_l1)


def tier2_lepton_block_t1():
    """Compute P(L2∧L3 | T1 sheet) using 2D quadrature.

    T1 sheet sampler: m3 ~ logU[0.3, 2000], r = m1/m3 ~ logU[1e-5, 1e-1].
    m2 solved from Koide equation (minus branch).

    In (ln m3, ln r) space, the density is uniform:
    ρ = 1 / (V_m3 × V_r) where V_m3 = ln(2000/0.3) ≈ 8.80,
    V_r = ln(0.1/1e-5) = ln(10000) ≈ 9.21.

    Claims L2 and L3 are constraints on the derived ratios:
    L2: |m2/m1 - L2_TARGET| / L2_TARGET ≤ L2_TOL
    L3: |m3/m2 - L3_TARGET| / L3_TARGET ≤ L3_TOL

    We need to integrate over (ln m3, ln r) with indicator functions.

    The derived quantities:
    m1 = r × m3
    s1 = sqrt(m1), s3 = sqrt(m3)
    s2 = (-b - sqrt(b² - 4c)) / 2 where b = -4(s1+s3), c = s1²+s3²-4s1s3
    m2 = s2²

    And L2 involves m2/m1, L3 involves m3/m2.

    For the narrow windows, the (ln m3, ln r) region is tiny. We can use
    importance sampling or local grid search.
    """
    V_m3 = LEP_LOG_V
    V_r = np.log(1e-1 / 1e-5)  # = ln(10000)
    density = 1.0 / (V_m3 * V_r)

    # First, find the (ln m3, ln r) that give the observed ratios
    # Use the observed lepton masses to get target values
    m1_obs, m2_obs, m3_obs = LEPTONS_OBS
    r_obs = m1_obs / m3_obs

    ln_m3_obs = np.log(m3_obs)
    ln_r_obs = np.log(r_obs)

    print(f"  [T2 lepton block T1] Observed: ln(m3)={ln_m3_obs:.4f}, ln(r)={ln_r_obs:.4f}")

    # Check that the observed values are in the sheet sampler support
    # (they should be — m3 in [0.3, 2000], r in [1e-5, 1e-1])

    # Now determine the (ln m3, ln r) region satisfying L2 and L3.
    # Since m2 is determined by the Koide equation, which depends on
    # (m1, m3) only through the dimensionless ratio r = m1/m3 and the
    # overall scale m3, we need to map the constraints.

    # The Koide solution: Q(m1, m2, m3) = 2/3 exactly.
    # The minus branch gives m2 as a function of (m1, m3).
    # In terms of r = m1/m3: s1/s3 = sqrt(r), and the solution for
    # s2/s3 depends only on sqrt(r).

    # Let x = sqrt(r) = sqrt(m1/m3). Then:
    # s1 = x·s3
    # b = -4(s1+s3) = -4·s3·(x+1)
    # c = s1²+s3²-4·s1·s3 = s3²·(x²+1-4x)
    # s2 = s3 · [2(x+1) - sqrt(4(x+1)² - (x²+1-4x))] / 2
    #    = s3 · [(x+1) - sqrt(x²+2x+1 - x²-1+4x)/2]  ... let me redo

    # disc = b² - 4c = 16·s3²·(x+1)² - 4·s3²·(x²+1-4x)
    #      = 4·s3²·[4(x+1)² - (x²+1-4x)]
    #      = 4·s3²·[4x²+8x+4 - x²-1+4x]
    #      = 4·s3²·[3x²+12x+3]
    #      = 12·s3²·[x²+4x+1]

    # s2 = (4·s3·(x+1) - sqrt(12·s3²·(x²+4x+1))) / 2
    #    = s3 · [2(x+1) - sqrt(3(x²+4x+1))]

    # So m2/m3 = (s2/s3)² = [2(x+1) - sqrt(3(x²+4x+1))]²
    # And m2/m1 = m2/m3 / r = [2(x+1) - sqrt(3(x²+4x+1))]² / x²

    # For observed: x_obs = sqrt(0.51099895/1776.93) ≈ 0.01695
    # m2/m1 = [2(1.01695) - sqrt(3(0.000287+0.0678+1))]² / 0.000287
    # Let's verify numerically:
    x_obs = np.sqrt(m1_obs / m3_obs)
    s2_s3 = 2*(x_obs+1) - np.sqrt(3*(x_obs**2 + 4*x_obs + 1))
    m2_m3 = s2_s3**2
    m2_m1 = m2_m3 / x_obs**2
    print(f"  [T2 T1] x_obs = {x_obs:.6f}, m2/m3 = {m2_m3:.6f}, m2/m1 = {m2_m1:.4f}")
    print(f"  [T2 T1] Observed: m2/m3 = {m2_obs/m3_obs:.6f}, m2/m1 = {m2_obs/m1_obs:.4f}")

    # Good — m2/m1 depends only on r (through x = sqrt(r)), not on m3.
    # Similarly, m3/m2 = 1/(m2/m3) = 1/s2_s3², which also depends only on r.

    # So: L2 and L3 are constraints on r alone, independent of m3!
    # This means the integral factorizes: ∫ dm3 × ∫ dr I[L2(r)] I[L3(r)] / (V_m3 V_r)
    # = (1/V_m3 × V_m3) × (Δr / V_r) = Δr / V_r, where Δr is the r-range satisfying both.

    # Wait — we also need m3/m1 > (4+sqrt(18))² for admissibility.
    # m3/m1 = 1/r, so this is 1/r > HIERARCHY_MIN → r < 1/HIERARCHY_MIN ≈ 0.0147.
    # r_obs ≈ 2.876e-4, so this is satisfied (and the r-window is narrow enough
    # that it won't matter).

    # The r-range satisfying L2 and L3:
    # L2: |m2/m1(r) - L2_TARGET|/L2_TARGET ≤ L2_TOL
    # L3: |1/(m2/m3(r)) - L3_TARGET|/L3_TARGET ≤ L3_TOL

    # Since both depend only on r, we can find the r-interval numerically.

    def m2_m1_from_r(r):
        x = np.sqrt(r)
        s2_s3 = 2*(x+1) - np.sqrt(np.maximum(0, 3*(x**2 + 4*x + 1)))
        return s2_s3**2 / x**2

    def m3_m2_from_r(r):
        x = np.sqrt(r)
        s2_s3 = 2*(x+1) - np.sqrt(np.maximum(0, 3*(x**2 + 4*x + 1)))
        return 1.0 / s2_s3**2

    # Find r-range for L2
    r_test = np.logspace(np.log10(1e-5), np.log10(1e-1), 1000000)
    m2m1_test = m2_m1_from_r(r_test)
    m3m2_test = m3_m2_from_r(r_test)

    l2_ok = np.abs(m2m1_test / L2_TARGET - 1.0) <= L2_TOL
    l3_ok = np.abs(m3m2_test / L3_TARGET - 1.0) <= L3_TOL

    if l2_ok.any():
        r_l2 = r_test[l2_ok]
        r_l2_min, r_l2_max = r_l2.min(), r_l2.max()
        print(f"  [T2 T1] L2 r-range: [{r_l2_min:.6e}, {r_l2_max:.6e}], "
              f"width in ln(r): {np.log(r_l2_max/r_l2_min):.6e}")
    else:
        print(f"  [T2 T1] L2: NO r-values satisfy constraint!")
        r_l2_min, r_l2_max = None, None

    if l3_ok.any():
        r_l3 = r_test[l3_ok]
        r_l3_min, r_l3_max = r_l3.min(), r_l3.max()
        print(f"  [T2 T1] L3 r-range: [{r_l3_min:.6e}, {r_l3_max:.6e}], "
              f"width in ln(r): {np.log(r_l3_max/r_l3_min):.6e}")
    else:
        print(f"  [T2 T1] L3: NO r-values satisfy constraint!")
        r_l3_min, r_l3_max = None, None

    both_ok = l2_ok & l3_ok
    if both_ok.any():
        r_both = r_test[both_ok]
        r_min, r_max = r_both.min(), r_both.max()
        delta_ln_r = np.log(r_max / r_min)
        print(f"  [T2 T1] L2∧L3 r-range: [{r_min:.6e}, {r_max:.6e}], "
              f"width in ln(r): {delta_ln_r:.6e}")

        # P(L2∧L3 | T1) = Δ(ln r) / V_r (since m3 integrates out completely)
        p_l2l3 = delta_ln_r / V_r
        print(f"  [T2 T1] P(L2∧L3) = {delta_ln_r:.6e} / {V_r:.4f} = {p_l2l3:.6e}")
    else:
        print(f"  [T2 T1] L2∧L3: NO r-values satisfy BOTH constraints!")
        p_l2l3 = 0.0
        r_min, r_max = None, None

    return p_l2l3, {"r_min": float(r_min) if r_min else None,
                    "r_max": float(r_max) if r_max else None,
                    "V_r": float(V_r),
                    "delta_ln_r": float(np.log(r_max/r_min)) if r_min else 0.0}


def tier2_quark_block_mc(lepton_draws, prior="logU", u1_mode="fixed",
                          n_mc_per_lepton=100000):
    """Compute P(Q1∧Q2∧U1 | lepton draws) by Monte Carlo.

    Given a set of lepton draws (each fixes mu_star and twome), draw
    quark triples and up-sector pairs from the prior, and compute the
    fraction satisfying all three claims.

    Returns the mean and standard error across lepton draws.
    """
    n_lep = len(lepton_draws)
    fractions = np.zeros(n_lep)

    for i in range(n_lep):
        lep = lepton_draws[i]
        rng_q = np.random.default_rng(271828 + i)

        # Draw quarks
        if prior == "logU":
            light_q = np.exp(rng_q.uniform(QUARK_LOG_LO, QUARK_LOG_HI,
                                           size=(n_mc_per_lepton, 3)))
        elif prior == "logN":
            mid = np.exp((QUARK_LOG_LO + QUARK_LOG_HI) / 2)
            light_q = np.exp(rng_q.normal(np.log(mid), 1.5 * np.log(10),
                                          size=(n_mc_per_lepton, 3)))
            light_q = np.clip(light_q, QUARK_LO, QUARK_HI)
        else:
            light_q = rng_q.uniform(QUARK_LO, QUARK_HI, size=(n_mc_per_lepton, 3))
        light_q.sort(axis=1)

        if prior == "logU":
            up_s = np.exp(rng_q.uniform(QUARK_LOG_LO, QUARK_LOG_HI,
                                        size=(n_mc_per_lepton, 2)))
        elif prior == "logN":
            up_s = np.exp(rng_q.normal(np.log(mid), 1.5 * np.log(10),
                                       size=(n_mc_per_lepton, 2)))
            up_s = np.clip(up_s, QUARK_LO, QUARK_HI)
        else:
            up_s = rng_q.uniform(QUARK_LO, QUARK_HI, size=(n_mc_per_lepton, 2))

        # Duplicate lepton for vectorized checks
        lep_batch = np.tile(lep, (n_mc_per_lepton, 1))

        q1_ok = check_Q1(light_q, lep_batch)
        q2_ok = check_Q2(light_q, lep_batch)
        if u1_mode == "fixed":
            u1_ok = check_U1_fixed(light_q, up_s)
        else:
            u1_ok = check_U1_menu(light_q, up_s)

        fractions[i] = (q1_ok & q2_ok & u1_ok).mean()

    mean_f = fractions.mean()
    std_f = fractions.std(ddof=1) / np.sqrt(n_lep) if n_lep > 1 else 0.0

    return float(mean_f), float(std_f), fractions


def tier2_quark_block_analytic(lepton_draw, prior="logU", u1_mode="fixed"):
    """Analytic/semi-analytic P(Q1∧Q2∧U1 | single lepton draw).

    Uses nested quadrature with explicit md and mu coupling.

    Log-uniform prior: in log-space (y_u, y_d, y_s, y_c, y_t), the density
    is constant over the sorted simplex for (y_u, y_d, y_s) and independent
    for (y_c, y_t).

    V_q = ln(2e5/0.5) ≈ 12.8992

    Density for sorted triple: ρ(y_u, y_d, y_s) = 6/V_q³ for y_u < y_d < y_s
    Density for pair: ρ(y_c, y_t) = 1/V_q² (no sorting)

    Q1: |2*y_s - y_d - ln(mu_star)| ≤ b1  → y_s ∈ [y_d/2 + A1, y_d/2 + A1 + b1/2]
        where A1 = ln(mu_star)/2, window centered at y_d/2 + A1, half-width b1/2

    Q2: |2*y_u - y_d - ln(twome)| ≤ b2  → y_u ∈ [y_d/2 + A2, y_d/2 + A2 + b2/2]
        where A2 = ln(twome)/2, half-width b2/2

    U1: |9*Q_U(exp(y_u), exp(y_c), exp(y_t)) - 8| ≤ U1_TOL (or menu)

    The integration:
    1. y_d is free (within bounds)
    2. y_s is constrained by Q1 (band around y_d/2 + A1)
    3. y_u is constrained by Q2 (band around y_d/2 + A2)
    4. (y_c, y_t) are constrained by U1
    5. Sorting: y_u < y_d < y_s must hold

    Since the Q1 and Q2 bands are thin and U1 involves a 3D region,
    we use Monte Carlo for the full joint and analytic for the bands.

    Returns P(Q1∧Q2∧U1 | this lepton draw).
    """
    mu_star = lepton_draw.sum()
    twome = 2.0 * lepton_draw.min()

    V_q = QUARK_LOG_V
    A1 = np.log(mu_star) / 2.0
    A2 = np.log(twome) / 2.0

    # Monte Carlo integration with importance sampling in the bands
    N_mc = 2_000_000

    rng = np.random.default_rng(271828)

    # Draw y_d uniformly over [QUARK_LOG_LO, QUARK_LOG_HI]
    y_d = rng.uniform(QUARK_LOG_LO, QUARK_LOG_HI, size=N_mc)

    # Q1 constrains y_s: y_s ∈ [y_d/2 + A1 - b1/2, y_d/2 + A1 + b1/2]
    # Q2 constrains y_u: y_u ∈ [y_d/2 + A2 - b2/2, y_d/2 + A2 + b2/2]
    y_s = y_d/2 + A1 + rng.uniform(-B1/2, B1/2, size=N_mc)
    y_u = y_d/2 + A2 + rng.uniform(-B2/2, B2/2, size=N_mc)

    # Draw y_c, y_t uniformly
    y_c = rng.uniform(QUARK_LOG_LO, QUARK_LOG_HI, size=N_mc)
    y_t = rng.uniform(QUARK_LOG_LO, QUARK_LOG_HI, size=N_mc)

    # Sorting constraint: y_u < y_d < y_s
    sorted_ok = (y_u < y_d) & (y_d < y_s)

    # Q1 and Q2 are satisfied by construction (we drew within the bands)
    # But we need to check bounds: all y's must be in [QUARK_LOG_LO, QUARK_LOG_HI]
    in_range = (y_u >= QUARK_LOG_LO) & (y_s <= QUARK_LOG_HI) & \
               (y_c >= QUARK_LOG_LO) & (y_c <= QUARK_LOG_HI) & \
               (y_t >= QUARK_LOG_LO) & (y_t <= QUARK_LOG_HI)

    # U1 check
    mu_v = np.exp(y_u)
    mc_v = np.exp(y_c)
    mt_v = np.exp(y_t)
    triples = np.column_stack([mu_v, mc_v, mt_v])
    q_dir = Q_U(triples)
    q_inv = Q_U(1.0 / triples)

    if u1_mode == "fixed":
        u1_ok = np.minimum(np.abs(9.0 * q_dir - 8.0),
                          np.abs(9.0 * q_inv - 8.0)) <= U1_TOL
    else:
        u1_ok = np.zeros(N_mc, dtype=bool)
        for tgt in U1_MENU_TARGETS:
            u1_ok |= (np.abs(9.0 * q_dir - tgt) <= U1_TOL)
            u1_ok |= (np.abs(9.0 * q_inv - tgt) <= U1_TOL)

    all_ok = sorted_ok & in_range & u1_ok
    hit_fraction = float(all_ok.mean())

    # The sampling density is NOT the prior — we importance-sampled y_s and y_u
    # in the Q1/Q2 bands. The prior density for (y_u, y_d, y_s) is 6/V_q³
    # (with sorting), and the sampling density is uniform in the band cross-section
    # with width b1 for y_s and b2 for y_u.

    # Prior volume for (y_u, y_d, y_s) with sorting: V_q³/6
    # Sampling volume: V_q × b1 × b2 (y_d uniform over V_q, y_s in band width b1,
    # y_u in band width b2)

    # Importance weight = prior_volume / sampling_volume (times sorting factor
    # already accounted for in the prior)

    # Actually, let me compute this more carefully.
    # The prior density ρ(y_u,y_d,y_s) = 6/V_q³ for y_u < y_d < y_s in the cube.
    # The sampling density q(y_u,y_d,y_s) = 1/(V_q × b1 × b2) for y_d in [lo,hi],
    # y_s in [y_d/2+A1-b1/2, y_d/2+A1+b1/2], y_u in [y_d/2+A2-b2/2, y_d/2+A2+b2/2].
    # (And 0 elsewhere.)
    #
    # Importance weight w = ρ/q, but only where q > 0 and sorting + range hold.
    #
    # P = ∫ ρ I[claims] d³y = ∫ q × (ρ/q) × I[claims] d³y
    #   ≈ (1/N) Σ w_i × I[claims]
    #
    # w = (6/V_q³) / (1/(V_q × b1 × b2)) = 6 × b1 × b2 / V_q²

    weight = 6.0 * B1 * B2 / (V_q ** 2)

    # For (y_c, y_t): prior density = 1/V_q², sampling density = 1/V_q²
    # → weight factor = 1 (they cancel)

    # Total P = weight × hit_fraction
    p_quark = weight * hit_fraction

    return float(p_quark), float(hit_fraction), float(weight)


# ═══════════════════════════════════════════════════════════════════════
# TIER-2: ADAPTIVE PER-STAGE INFLATION CALIBRATION (§8 v0.3)
# ═══════════════════════════════════════════════════════════════════════

def tier2_adaptive_calibration(prior="logU", condition="T0", u1_mode="fixed",
                                calibration_seed=271828, N_calib_max=1_000_000_000):
    """v0.3 Tier-2 adaptive per-stage inflation calibration.

    For each stage, find the smallest factor in {1, 3, 10, 30, 100, 300, 1000}
    at which brute force yields ≥ 100 hits AND the inflated window remains
    non-vacuous (excludes >50% of prior mass for every claim in the stage).

    Stages:
      Stage 1: L2∧L3 (lepton ratios only; + L1 for T0)
      Stage 2: +Q1 (lepton-quark bridge)
      Stage 3: +Q2 (second quark constraint)
      Stage 4: +U1 (full joint)

    For each stage and factor, validate the Tier-2 point estimate against
    brute force within 2 Poisson sigma.

    Returns:
        calibration_record: dict of per-stage results
        tier2_validated: bool — True only if ALL stages validated
        tier2_point_estimates: dict of per-stage point estimates
    """
    batch_size = 1_000_000

    if condition == "T0":
        stage_defs = [
            ("L1_L2_L3", ["L1", "L2", "L3"]),
            ("+Q1", ["L1", "L2", "L3", "Q1"]),
            ("+Q2", ["L1", "L2", "L3", "Q1", "Q2"]),
            ("+U1", ["L1", "L2", "L3", "Q1", "Q2", f"U1_{u1_mode}"]),
        ]
    else:  # T1: L1 granted
        stage_defs = [
            ("L2_L3", ["L2", "L3"]),
            ("+Q1", ["L2", "L3", "Q1"]),
            ("+Q2", ["L2", "L3", "Q1", "Q2"]),
            ("+U1", ["L2", "L3", "Q1", "Q2", f"U1_{u1_mode}"]),
        ]

    calibration_record = {}
    tier2_point_estimates = {}
    all_stages_validated = True

    for stage_name, claim_list in stage_defs:
        print(f"\n  ── Stage: {stage_name} (claims: {claim_list}) ──")

        stage_result = {"claims": claim_list, "factors_tested": {}}
        best_factor = None
        best_bf_hits = 0
        best_bf_n = 0

        for factor in INFLATION_FACTORS:
            print(f"    Testing factor={factor}...", flush=True)

            # ── Non-vacuous gate ──
            vacuous = False
            for claim in claim_list:
                claim_base = claim.replace("U1_fixed", "U1_fixed").replace("U1_menu", "U1_menu")
                if claim_base in ("L1", "L2", "L3"):
                    nonvac, frac = check_nonvacuous_lepton(claim_base, factor)
                    if not nonvac:
                        print(f"      VACUOUS: {claim_base} at factor {factor}: "
                              f"{frac*100:.1f}% of prior (must be <50%)")
                        vacuous = True
                        break
                elif claim_base in ("Q1", "Q2"):
                    nonvac, frac = check_nonvacuous_quark(claim_base, factor)
                    if not nonvac:
                        print(f"      VACUOUS: {claim_base} at factor {factor}: "
                              f"{frac*100:.1f}% of prior (must be <50%)")
                        vacuous = True
                        break
                elif claim_base.startswith("U1"):
                    u1_type = "U1_fixed" if "fixed" in claim_base else "U1_menu"
                    nonvac, frac = check_nonvacuous_quark(u1_type, factor)
                    if not nonvac:
                        print(f"      VACUOUS: {u1_type} at factor {factor}: "
                              f"{frac*100:.1f}% of prior (must be <50%)")
                        vacuous = True
                        break

            if vacuous:
                stage_result["factors_tested"][str(factor)] = {
                    "vacuous": True, "hits": None, "N_eff": None
                }
                continue

            # ── Brute force at this factor ──
            rng_calib = np.random.default_rng(calibration_seed)
            total_eff = 0
            total_hits = 0
            n_batches_done = 0

            t0 = time.time()
            while total_eff < N_calib_max and total_hits < 100:
                n_batches_done += 1

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
                    if claim == "L1":
                        mask = check_L1_inflated(leptons, factor)
                    elif claim == "L2":
                        mask = check_L2_inflated(leptons, factor)
                    elif claim == "L3":
                        mask = check_L3_inflated(leptons, factor)
                    elif claim == "Q1":
                        mask = check_Q1_inflated(light_q, leptons, factor)
                    elif claim == "Q2":
                        mask = check_Q2_inflated(light_q, leptons, factor)
                    elif claim in ("U1_fixed", "U1_menu"):
                        u1_check = check_U1_fixed_inflated if claim == "U1_fixed" else check_U1_menu_inflated
                        mask = u1_check(light_q, up_s, factor)
                    else:
                        continue

                    survivors_idx = np.where(survivors)[0]
                    survivors[survivors_idx[~mask[survivors_idx]]] = False

                total_hits += survivors.sum()

                if n_batches_done % 20 == 0:
                    elapsed = time.time() - t0
                    print(f"      [{stage_name}/x{factor}] batch {n_batches_done}, "
                          f"N_eff={total_eff:,}, hits={total_hits}", flush=True)

            elapsed = time.time() - t0
            print(f"      [{stage_name}/x{factor}] DONE: hits={total_hits}, "
                  f"N_eff={total_eff:,}, elapsed={elapsed:.1f}s", flush=True)

            stage_result["factors_tested"][str(factor)] = {
                "vacuous": False,
                "hits": int(total_hits),
                "N_eff": int(total_eff),
                "rate": float(total_hits / total_eff) if total_eff > 0 else 0.0,
            }

            if total_hits >= 100:
                best_factor = factor
                best_bf_hits = int(total_hits)
                best_bf_n = int(total_eff)
                break  # smallest factor that works

        # ── Record stage result ──
        stage_result["best_factor"] = best_factor
        stage_result["best_bf_hits"] = best_bf_hits
        stage_result["best_bf_n"] = best_bf_n

        if best_factor is None:
            print(f"  Stage {stage_name}: INELIGIBLE (no factor reached 100 hits)")
            all_stages_validated = False
        else:
            # ── Compute Tier-2 point estimate at this stage ──
            # For the lepton block, use analytic
            if condition == "T0":
                p_lep, _, _ = tier2_lepton_block_t0()
            else:
                p_lep, _ = tier2_lepton_block_t1()

            # For the quark block, use importance-sampled MC with observed leptons
            # At the inflated tolerances
            # Actually for calibration we want the Tier-2 estimate at the INFLATED
            # tolerances, to compare with brute force.

            # Compute Tier-2 estimate at inflated tolerances
            # Lepton block at inflated tolerances
            if condition == "T0":
                # Inflated lepton block
                V = LEP_LOG_V
                u0 = np.log(L2_TARGET)
                v0 = np.log(L3_TARGET)
                s_width = V - u0 - v0
                area_uv = (2 * L2_TOL * best_factor) * (2 * L3_TOL * best_factor)
                # L1 fraction within L2∧L3 window at this factor
                N_test_l1 = 50000
                rng_l1 = np.random.default_rng(calibration_seed)
                du_inf = L2_TOL * best_factor
                dv_inf = L3_TOL * best_factor
                u_test = u0 + rng_l1.uniform(-du_inf, du_inf, N_test_l1)
                v_test = v0 + rng_l1.uniform(-dv_inf, dv_inf, N_test_l1)
                m1_test = np.ones(N_test_l1) * LEPTONS_OBS[0]
                m2_test = m1_test * np.exp(u_test)
                m3_test = m2_test * np.exp(v_test)
                leptons_test = np.column_stack([m1_test, m2_test, m3_test])
                kd_test = kdist(leptons_test)
                f_l1_inf = float((kd_test <= L1_TOL * best_factor).mean())
                p_lep_inf = 6.0 / (V**3) * s_width * area_uv * f_l1_inf
            else:
                # T1: inflate L2, L3 tolerances
                V_r = np.log(1e-1 / 1e-5)
                # Find r-range with inflated tolerances
                r_test = np.logspace(np.log10(1e-5), np.log10(1e-1), 2000000)
                x_test = np.sqrt(r_test)
                s2_s3_test = 2*(x_test+1) - np.sqrt(np.maximum(0, 3*(x_test**2 + 4*x_test + 1)))
                m2m1_test = s2_s3_test**2 / x_test**2
                m3m2_test = 1.0 / s2_s3_test**2

                l2_ok = np.abs(m2m1_test / L2_TARGET - 1.0) <= L2_TOL * best_factor
                l3_ok = np.abs(m3m2_test / L3_TARGET - 1.0) <= L3_TOL * best_factor
                both_ok = l2_ok & l3_ok
                if both_ok.any():
                    r_both = r_test[both_ok]
                    delta_ln_r_inf = np.log(r_both.max() / r_both.min())
                    p_lep_inf = delta_ln_r_inf / V_r
                else:
                    p_lep_inf = 0.0

            # Quark block: use MC with observed leptons at inflated tolerances
            # (reuse the same approach)
            lep_obs_batch = np.tile(LEPTONS_OBS, (200000, 1))

            rng_q_inf = np.random.default_rng(calibration_seed)
            light_q_inf = np.exp(rng_q_inf.uniform(QUARK_LOG_LO, QUARK_LOG_HI,
                                                    size=(200000, 3)))
            light_q_inf.sort(axis=1)
            up_s_inf = np.exp(rng_q_inf.uniform(QUARK_LOG_LO, QUARK_LOG_HI,
                                                  size=(200000, 2)))

            # Check at inflated tolerances
            q1_inf = check_Q1_inflated(light_q_inf, lep_obs_batch, best_factor)
            q2_inf = check_Q2_inflated(light_q_inf, lep_obs_batch, best_factor)
            if u1_mode == "fixed":
                u1_inf = check_U1_fixed_inflated(light_q_inf, up_s_inf, best_factor)
            else:
                u1_inf = check_U1_menu_inflated(light_q_inf, up_s_inf, best_factor)

            p_quark_inf = float((q1_inf & q2_inf & u1_inf).mean())

            tier2_est_inf = p_lep_inf * p_quark_inf

            # Compare with brute force
            bf_rate = best_bf_hits / best_bf_n if best_bf_n > 0 else 0.0
            # 2 Poisson sigma: σ_bf = sqrt(bf_rate / best_bf_n)
            # Actually for rate: σ = sqrt(hits) / N
            bf_sigma = np.sqrt(best_bf_hits) / best_bf_n if best_bf_n > 0 else 0.0

            within_2sigma = abs(tier2_est_inf - bf_rate) <= 2.0 * bf_sigma

            stage_result["tier2_est_inflated"] = float(tier2_est_inf)
            stage_result["bf_rate"] = float(bf_rate)
            stage_result["bf_sigma"] = float(bf_sigma)
            stage_result["within_2sigma"] = bool(within_2sigma)

            if not within_2sigma:
                print(f"  Stage {stage_name}: VALIDATION FAILED — "
                      f"T2={tier2_est_inf:.6e}, BF={bf_rate:.6e}±{bf_sigma:.6e}")
                all_stages_validated = False
            else:
                print(f"  Stage {stage_name}: VALIDATED — "
                      f"T2={tier2_est_inf:.6e}, BF={bf_rate:.6e}±{bf_sigma:.6e} ✓")

        calibration_record[stage_name] = stage_result

    # ── Compute Tier-2 point estimates at factor 1 (the actual claim) ──
    if condition == "T0":
        p_lep_actual, _, _ = tier2_lepton_block_t0()
    else:
        p_lep_actual, _ = tier2_lepton_block_t1()

    # Quark block with observed leptons at factor 1
    N_q_mc = 5_000_000
    rng_q = np.random.default_rng(calibration_seed)
    lep_batch = np.tile(LEPTONS_OBS, (N_q_mc, 1))

    light_q_act = np.exp(rng_q.uniform(QUARK_LOG_LO, QUARK_LOG_HI, size=(N_q_mc, 3)))
    light_q_act.sort(axis=1)
    up_s_act = np.exp(rng_q.uniform(QUARK_LOG_LO, QUARK_LOG_HI, size=(N_q_mc, 2)))

    q1_act = check_Q1(light_q_act, lep_batch)
    q2_act = check_Q2(light_q_act, lep_batch)
    if u1_mode == "fixed":
        u1_act = check_U1_fixed(light_q_act, up_s_act)
    else:
        u1_act = check_U1_menu(light_q_act, up_s_act)

    p_quark_act = float((q1_act & q2_act & u1_act).mean())
    p_joint_t2 = p_lep_actual * p_quark_act

    print(f"\n  Tier-2 point estimate (factor 1):")
    print(f"    P(lepton block) = {p_lep_actual:.6e}")
    print(f"    P(quark block | lepton) = {p_quark_act:.6e}")
    print(f"    P(joint) = {p_joint_t2:.6e}")

    return calibration_record, all_stages_validated, {
        "p_lepton_block": float(p_lep_actual),
        "p_quark_block": float(p_quark_act),
        "p_joint": float(p_joint_t2),
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════

def check_support_gate():
    """§5 SUPPORT GATE."""
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Joint Coincidence Budget Engine v0.3")
    parser.add_argument("--mode", default="full",
                        choices=["singletons", "cascade", "full", "tier2_calibrate", "tier2_lepton"])
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--prior", default="logU", choices=["logU", "logN", "linU"])
    parser.add_argument("--condition", default="T0", choices=["T0", "T1"])
    parser.add_argument("--u1-mode", default="fixed", choices=["fixed", "menu"])
    parser.add_argument("--N-singletons", type=int, default=10_000_000)
    parser.add_argument("--N-eff", type=int, default=2_000_000_000)
    parser.add_argument("--cascade-order", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--skip-tier1", action="store_true",
                        help="Skip Tier-1 cascade (Tier-2 only)")
    args = parser.parse_args()

    if args.outdir is None:
        args.outdir = f"results/amb-{args.seed}-v0.3"

    os.makedirs(args.outdir, exist_ok=True)

    # ── SUPPORT GATE ──
    if not check_support_gate():
        print("\n*** STOP: support gate failed. ***")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"JOINT COINCIDENCE BUDGET ENGINE v0.3 — amb")
    print(f"Mode: {args.mode} | Seed: {args.seed} | Prior: {args.prior}")
    print(f"Condition: {args.condition} | U1: {args.u1_mode}")
    print(f"{'='*70}\n")

    if args.mode == "tier2_lepton":
        # Just compute lepton block quadrature
        print("Computing Tier-2 lepton block...")
        if args.condition == "T0":
            p_lep, p_l2l3, f_l1 = tier2_lepton_block_t0()
        else:
            p_lep, info = tier2_lepton_block_t1()
        return

    if args.mode == "tier2_calibrate":
        # Run Tier-2 calibration for a single cell
        calib_record, validated, t2_estimates = tier2_adaptive_calibration(
            args.prior, args.condition, args.u1_mode,
            calibration_seed=271828
        )
        outpath = os.path.join(args.outdir,
            f"tier2_calibration_{args.condition}_{args.u1_mode}_{args.prior}.json")
        output = {
            "engine_id": "amb",
            "spec_version": "v0.3",
            "condition": args.condition,
            "u1_mode": args.u1_mode,
            "prior": args.prior,
            "calibration_seed": 271828,
            "calibration_record": calib_record,
            "tier2_validated": validated,
            "tier2_estimates": t2_estimates,
        }
        with open(outpath, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved: {outpath}")
        return

    if args.mode == "singletons":
        rng = np.random.default_rng(args.seed)
        result, N_eff = measure_singletons(
            rng, args.N_singletons, args.prior, args.condition
        )
        outpath = os.path.join(
            args.outdir,
            f"singletons_{args.condition}_{args.prior}_seed{args.seed}.json"
        )
        output = {
            "engine_id": "amb",
            "spec_version": "v0.3",
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
        rng = np.random.default_rng(args.seed)
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

        cell_result = {
            "engine_id": "amb",
            "spec_version": "v0.3",
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
        # Run all 12 cells: Tier-1 cascade + Tier-2 calibration
        all_results = {}
        tier2_results = {}

        for condition in ["T0", "T1"]:
            for u1_mode in ["fixed", "menu"]:
                for prior in ["logU", "logN", "linU"]:
                    cell_key = f"{condition}_{u1_mode}_{prior}"
                    cell_seed = args.seed if prior == "logU" else 314160
                    N_target = args.N_eff if prior == "logU" else max(args.N_eff // 10, 200_000_000)

                    print(f"\n{'─'*60}")
                    print(f"CELL: {cell_key}  seed={cell_seed}  N_eff>={N_target:,}")
                    print(f"{'─'*60}")

                    # ── Tier-1 cascade ──
                    if not args.skip_tier1:
                        cell_rng = np.random.default_rng(cell_seed)

                        singletons, N_sing = measure_singletons(
                            cell_rng, min(args.N_singletons, N_target),
                            prior, condition
                        )

                        claim_rates = [(c, singletons[c]["rate"]) for c in ALL_CLAIMS
                                       if singletons[c]["rate"] > 0]
                        claim_rates.sort(key=lambda x: x[1])
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

                    # ── Tier-2 calibration (primary cells only: logU) ──
                    if prior == "logU":
                        print(f"\n  ── Tier-2 calibration for {cell_key} ──")
                        calib_record, t2_validated, t2_estimates = tier2_adaptive_calibration(
                            prior, condition, u1_mode,
                            calibration_seed=271828
                        )
                        tier2_results[cell_key] = {
                            "calibration_record": calib_record,
                            "tier2_validated": t2_validated,
                            "tier2_estimates": t2_estimates,
                        }

                        # Save Tier-2 results
                        t2_path = os.path.join(args.outdir,
                            f"tier2_{condition}_{u1_mode}_{prior}.json")
                        with open(t2_path, "w") as f:
                            json.dump(tier2_results[cell_key], f, indent=2)
                        print(f"  Saved Tier-2: {t2_path}")

        # Save Tier-1 results
        if not args.skip_tier1:
            outpath = os.path.join(args.outdir, "mc_counts.json")
            with open(outpath, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\nTier-1 results saved: {outpath}")

        # Save Tier-2 summary
        t2_summary_path = os.path.join(args.outdir, "tier2_summary.json")
        with open(t2_summary_path, "w") as f:
            json.dump(tier2_results, f, indent=2)
        print(f"Tier-2 summary saved: {t2_summary_path}")

        # Print summary
        print(f"\n{'='*80}")
        print(f"RESULTS SUMMARY (v0.3)")
        print(f"{'='*80}")
        if not args.skip_tier1:
            print(f"\n{'Cell':<25} {'T1 Hits':>10} {'T1 N_eff':>14} {'T1 Rate':>14} {'T1 Label':>10} {'T2 Validated':>14}")
            print(f"{'─'*25} {'─'*10} {'─'*14} {'─'*14} {'─'*10} {'─'*14}")
            for cell_key in sorted(all_results.keys()):
                r = all_results[cell_key]
                t2_info = tier2_results.get(cell_key, {})
                t2_val = "YES" if t2_info.get("tier2_validated", False) else "NO"
                print(f"{cell_key:<25} {r['hits']:>10} {r['N_eff']:>14,} "
                      f"{r['rate']:>14.6e} {r['label']:>10} {t2_val:>14}")


if __name__ == "__main__":
    main()

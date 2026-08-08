#!/usr/bin/env python3
"""
v0.5 TIER-2 VALIDATION — RECOVERY BUILD (post-crash)
ENGINE_ID: amb | Calibration seed: 271828

Supersedes scripts/tier2_optimized_v0.5.py, which is PRESERVED but cannot
complete. Three defects found during the 2026-08-08 crash recovery:

  (1) HANG: `while len(lep_samples_list) < N_lep_samples` counts list CHUNKS,
      not samples. Each append adds ~1e6 draws but increments len() by 1, so
      the loop demanded 10,000 x 1e6 = 1e10 draws. The `[:N_lep_samples]`
      truncation immediately after proves 10,000 TOTAL was the intent.
      This is where the pre-crash run was stuck.
  (2) UNIT MISMATCH: `bf_rate = bf_hits / N_cached` is the CONDITIONAL rate
      P(new claim | lepton survivor) ~ 1e-1, but `t2_analytic = p_lep *
      p_quark_avg` is the JOINT ~ 1e-7. Comparing them fails by ~6 orders of
      magnitude regardless of the physics. joint_engine_v0.5.py uses
      `bf_hits / bf_N` (total attempted draws) which is correctly joint.
  (3) INFEASIBLE ANCHOR LOOP: quark_block_anchored_batched loops
      10,000 leptons x 20 batches x 50,000 quarks x 38 U1-menu comparisons
      ~ 4e11 element-ops per factor.

Fix for (3) preserves the spec's semantics EXACTLY. Per §v0.5(a) the analytic
side must average P(quark block) over >=1e4 lepton draws from the same
inflated window with anchors recomputed per draw. Both anchored claims are
one-sided bands in a single quark statistic:

    Q1: |ln(ms^2/(mu_star*md))| <= B1*f  <=>  |w1 - ln(mu_star)| <= B1*f,
        w1 = 2*ln(ms) - ln(md)
    Q2: |ln(mu^2/(md*twome))|  <= B2*f  <=>  |w2 - ln(twome)|   <= B2*f,
        w2 = 2*ln(mu) - ln(md)
    U1: depends only on the quark multiset {mu, mc, mt} — no lepton anchor.

So P(claim | anchor) is the fraction of a fixed quark pool whose statistic
falls in a window centred on that draw's own anchor. Sorting the statistic
once and running two binary searches per anchor returns the IDENTICAL MC
fraction in O(N log N) instead of O(N_lep * N_quark). Anchors are still
recomputed per lepton draw; nothing is evaluated at observed leptons.

All v0.1-v0.4 results and the original v0.5 scripts remain untouched.
"""

import numpy as np
import json
import os
import sys
import time
import argparse
from scipy.stats import beta as beta_dist

# ═══════════════════════════════════════════════════════════════════════════
# KDIST, Q_U, CP95
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
    lower = 0.0 if k <= 0 else float(beta_dist.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k >= n else float(beta_dist.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lower, upper


# ═══════════════════════════════════════════════════════════════════════════
# FROZEN CONSTANTS (§4, §7)
# ═══════════════════════════════════════════════════════════════════════════

LEPTONS_OBS = np.array([0.51099895, 105.6583755, 1776.93])
L1_TOL = 3.3049e-6
L2_TARGET, L2_TOL = 206.7703, 1.00e-5
L3_TARGET, L3_TOL = 16.8180, 2.10e-5
B1, B2 = 3.00e-3, 1.18e-2
U1_TOL = 1.1414e-2

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
R_LO, R_HI = 1e-5, 1e-1
V_R = np.log(R_HI / R_LO)


# ═══════════════════════════════════════════════════════════════════════════
# CLAIM CHECKS
# ═══════════════════════════════════════════════════════════════════════════

def check_L1(lep, f=1.0):
    return kdist(lep) <= L1_TOL * f


def check_L2(lep, f=1.0):
    return np.abs(lep[:, 1] / lep[:, 0] / L2_TARGET - 1.0) <= L2_TOL * f


def check_L3(lep, f=1.0):
    return np.abs(lep[:, 2] / lep[:, 1] / L3_TARGET - 1.0) <= L3_TOL * f


def check_Q1(lq, lep, f=1.0):
    ms, md = lq[:, 2], lq[:, 1]
    mu_star = lep.sum(axis=1)
    return np.abs(np.log(ms * ms / (mu_star * md))) <= B1 * f


def check_Q2(lq, lep, f=1.0):
    mu, md = lq[:, 0], lq[:, 1]
    twome = 2.0 * lep.min(axis=1)
    return np.abs(np.log(mu * mu / (md * twome))) <= B2 * f


def check_U1_fixed(lq, us, f=1.0):
    t = np.column_stack([lq[:, 0], us[:, 0], us[:, 1]])
    qd, qi = Q_U(t), Q_U(1.0 / t)
    return np.minimum(np.abs(9.0 * qd - 8.0), np.abs(9.0 * qi - 8.0)) <= U1_TOL * f


def check_U1_menu(lq, us, f=1.0):
    t = np.column_stack([lq[:, 0], us[:, 0], us[:, 1]])
    qd, qi = Q_U(t), Q_U(1.0 / t)
    hit = np.zeros(len(t), dtype=bool)
    for tgt in U1_MENU_TARGETS:
        hit |= (np.abs(9.0 * qd - tgt) <= U1_TOL * f)
        hit |= (np.abs(9.0 * qi - tgt) <= U1_TOL * f)
    return hit


LEP_CHECK = {"L1": check_L1, "L2": check_L2, "L3": check_L3}


# ═══════════════════════════════════════════════════════════════════════════
# DRAWS (§5 order: leptons first, then light quarks, then up-sector pair)
# ═══════════════════════════════════════════════════════════════════════════

def draw_quark_triple(rng, N):
    x = np.exp(rng.uniform(QUARK_LOG_LO, QUARK_LOG_HI, size=(N, 3)))
    x.sort(axis=1)
    return x


def draw_quark_pair(rng, N):
    return np.exp(rng.uniform(QUARK_LOG_LO, QUARK_LOG_HI, size=(N, 2)))


def draw_leptons_T0(rng, N):
    x = np.exp(rng.uniform(LEP_LOG_LO, LEP_LOG_HI, size=(N, 3)))
    x.sort(axis=1)
    return x


def sample_t1_koide(rng, N):
    """Koide sheet sampler (§5 T1). Returns (accepted_triples, attempted)."""
    m3 = np.exp(rng.uniform(LEP_LOG_LO, LEP_LOG_HI, size=N))
    r = np.exp(rng.uniform(np.log(R_LO), np.log(R_HI), size=N))
    m1 = r * m3
    s1, s3 = np.sqrt(m1), np.sqrt(m3)
    b = -4.0 * (s1 + s3)
    c = s1 ** 2 + s3 ** 2 - 4.0 * s1 * s3
    disc = b ** 2 - 4.0 * c
    ok_disc = disc >= 0
    disc = np.maximum(disc, 0.0)
    s2 = (-b - np.sqrt(disc)) / 2.0          # minus branch
    keep = ok_disc & (s2 > 0) & (m1 < s2 ** 2) & (s2 ** 2 < m3) & ((m3 / m1) > HIERARCHY_MIN)
    return np.column_stack([m1[keep], s2[keep] ** 2, m3[keep]]), N


# ═══════════════════════════════════════════════════════════════════════════
# NON-VACUOUS GATE (§8: window must exclude >50% of prior mass)
# ═══════════════════════════════════════════════════════════════════════════

def check_nonvacuous(claim, factor, seed=271828):
    if claim in ("L2", "L3"):
        target = np.log(L2_TARGET) if claim == "L2" else np.log(L3_TARGET)
        tol = L2_TOL if claim == "L2" else L3_TOL
        hw = tol * factor
        lo, hi = max(0.0, target - hw), min(LEP_LOG_V, target + hw)
        if lo >= hi:
            return True, 0.0
        F = lambda x: (2.0 * x / LEP_LOG_V) - (x / LEP_LOG_V) ** 2
        frac = F(hi) - F(lo)
        return frac < 0.5, float(max(0.0, frac))
    if claim == "L1":
        rng = np.random.default_rng(seed)
        m = draw_leptons_T0(rng, 500_000)
        frac = float(check_L1(m, factor).mean())
        return frac < 0.5, frac
    if claim in ("Q1", "Q2"):
        tol = B1 if claim == "Q1" else B2
        frac = float(min(2.0 * tol * factor / QUARK_LOG_V, 1.0))
        return frac < 0.5, frac
    if claim.startswith("U1"):
        rng = np.random.default_rng(seed)
        lq = draw_quark_triple(rng, 200_000)
        us = draw_quark_pair(rng, 200_000)
        fn = check_U1_fixed if "fixed" in claim else check_U1_menu
        frac = float(fn(lq, us, factor).mean())
        return frac < 0.5, frac
    return True, 0.0


# ═══════════════════════════════════════════════════════════════════════════
# LEPTON BLOCK (analytic)
# ═══════════════════════════════════════════════════════════════════════════

def m_ratios_from_r(r):
    x = np.sqrt(np.maximum(r, 1e-300))
    disc = 3.0 * (x ** 2 + 4.0 * x + 1.0)
    s2_over_s3 = 2.0 * (x + 1.0) - np.sqrt(np.maximum(0.0, disc))
    return s2_over_s3 ** 2 / x ** 2, 1.0 / (s2_over_s3 ** 2)


def compute_r_intersection(factor=1.0):
    """r-range satisfying BOTH L2 and L3 at the given inflation factor."""
    ln_r_obs = np.log(LEPTONS_OBS[0] / LEPTONS_OBS[2])
    for spread in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        r_grid = np.exp(ln_r_obs + np.linspace(-spread, spread, 400_000))
        m2m1, m3m2 = m_ratios_from_r(r_grid)
        both = (np.abs(m2m1 / L2_TARGET - 1.0) <= L2_TOL * factor) & \
               (np.abs(m3m2 / L3_TARGET - 1.0) <= L3_TOL * factor)
        if both.any():
            ri = r_grid[both]
            r_min, r_max = float(ri.min()), float(ri.max())
            if r_max > r_min:
                return r_min, r_max, float(np.log(r_max / r_min))
    return None, None, None


def lepton_block_prob(condition, factor):
    """Analytic P(lepton stage) at the given stage-local inflation factor."""
    if condition == "T1":
        r_min, r_max, dlnr = compute_r_intersection(factor)
        if dlnr is None or dlnr <= 0:
            return 0.0, {"method": "r_intersection", "error": "no_intersection"}
        return dlnr / V_R, {"method": "r_intersection", "r_min": r_min,
                            "r_max": r_max, "delta_ln_r": dlnr, "V_r": float(V_R)}
    # T0: exact density f(u,v) = 6/L^3 * (L-u-v), verified in v0.4
    u_lo = np.log(L2_TARGET * (1.0 - L2_TOL * factor))
    u_hi = np.log(L2_TARGET * (1.0 + L2_TOL * factor))
    v_lo = np.log(L3_TARGET * (1.0 - L3_TOL * factor))
    v_hi = np.log(L3_TARGET * (1.0 + L3_TOL * factor))
    u_mid, v_mid = (u_lo + u_hi) / 2.0, (v_lo + v_hi) / 2.0
    du, dv = u_hi - u_lo, v_hi - v_lo
    resid = LEP_LOG_V - u_mid - v_mid
    f0 = 6.0 / (LEP_LOG_V ** 3) * resid if resid > 0 else 0.0
    # L1 overlap fraction inside the L2^L3 window
    rng = np.random.default_rng(271828)
    N = 200_000
    u_t = rng.uniform(u_lo, u_hi, N)
    v_t = rng.uniform(v_lo, v_hi, N)
    m1 = np.ones(N)
    trip = np.column_stack([m1, np.exp(u_t), np.exp(u_t + v_t)])
    f_l1 = float(check_L1(trip, factor).mean())
    return f0 * du * dv * f_l1, {"method": "analytic_density", "f0": float(f0),
                                 "du": float(du), "dv": float(dv), "f_l1_frac": f_l1}


# ═══════════════════════════════════════════════════════════════════════════
# LEPTON WINDOW SAMPLING for anchor averaging (§v0.5 a)
# ═══════════════════════════════════════════════════════════════════════════

def sample_lepton_window(condition, factor, n_want, seed):
    """Draw n_want lepton triples from INSIDE the inflated lepton window.

    T0: sample the two shape dofs (u,v) inside the window, then sample the
        overall scale m1 log-uniformly over its admissible range
        [ln a, ln b - u - v]. The scale MUST be sampled: mu_star and twome
        are extensive in it, so fixing m1=1 would collapse the anchor
        distribution the averaging exists to capture.
    T1: sample r inside the L2^L3 intersection and m3 log-uniform over the
        full prior range, then solve the Koide minus branch.
    """
    rng = np.random.default_rng(seed)
    out = []
    got = 0
    batch = 200_000

    if condition == "T0":
        u_lo = np.log(L2_TARGET * (1.0 - L2_TOL * factor))
        u_hi = np.log(L2_TARGET * (1.0 + L2_TOL * factor))
        v_lo = np.log(L3_TARGET * (1.0 - L3_TOL * factor))
        v_hi = np.log(L3_TARGET * (1.0 + L3_TOL * factor))
        while got < n_want:
            u = rng.uniform(u_lo, u_hi, batch)
            v = rng.uniform(v_lo, v_hi, batch)
            hi_scale = LEP_LOG_HI - u - v
            ok = hi_scale > LEP_LOG_LO
            if not ok.any():
                raise RuntimeError("T0 lepton window has no admissible scale range")
            u, v, hi_scale = u[ok], v[ok], hi_scale[ok]
            ln_m1 = rng.uniform(LEP_LOG_LO, hi_scale)
            m1 = np.exp(ln_m1)
            trip = np.column_stack([m1, m1 * np.exp(u), m1 * np.exp(u + v)])
            trip = trip[check_L1(trip, factor)]
            if len(trip):
                out.append(trip)
                got += len(trip)
    else:
        r_min, r_max, _ = compute_r_intersection(factor)
        if r_min is None:
            raise RuntimeError("T1 r-intersection empty")
        while got < n_want:
            r = np.exp(rng.uniform(np.log(r_min), np.log(r_max), batch))
            m3 = np.exp(rng.uniform(LEP_LOG_LO, LEP_LOG_HI, batch))
            m1 = r * m3
            s1, s3 = np.sqrt(m1), np.sqrt(m3)
            b = -4.0 * (s1 + s3)
            c = s1 ** 2 + s3 ** 2 - 4.0 * s1 * s3
            disc = b ** 2 - 4.0 * c
            ok = disc >= 0
            disc = np.maximum(disc, 0.0)
            s2 = (-b - np.sqrt(disc)) / 2.0
            keep = ok & (s2 > 0) & (m1 < s2 ** 2) & (s2 ** 2 < m3) & ((m3 / m1) > HIERARCHY_MIN)
            if keep.any():
                out.append(np.column_stack([m1[keep], s2[keep] ** 2, m3[keep]]))
                got += int(keep.sum())

    return np.vstack(out)[:n_want]


# ═══════════════════════════════════════════════════════════════════════════
# ANCHOR-AVERAGED QUARK BLOCK (§v0.5 a) — band-statistic formulation
# ═══════════════════════════════════════════════════════════════════════════

def anchored_quark_prob(lepton_samples, new_claims, factor, u1_mode,
                        n_pool=2_000_000, n_pools=5, seed=271828):
    """Average P(new quark claims | anchors) over the supplied lepton draws.

    Anchors are recomputed from each lepton draw's OWN masses (§2). Returns
    (mean, stderr over independent pools, per-anchor spread diagnostics).
    """
    mu_star = lepton_samples.sum(axis=1)          # own-draw anchor
    twome = 2.0 * lepton_samples.min(axis=1)      # own-draw floor
    ln_mu_star = np.log(mu_star)
    ln_twome = np.log(twome)

    has_q1 = "Q1" in new_claims
    has_q2 = "Q2" in new_claims
    u1_claim = next((c for c in new_claims if c.startswith("U1")), None)

    pool_means = []
    last_per_anchor = None

    for p in range(n_pools):
        rng = np.random.default_rng(seed + 991 * p)
        lq = draw_quark_triple(rng, n_pool)
        us = draw_quark_pair(rng, n_pool)

        mask = np.ones(n_pool, dtype=bool)
        if u1_claim is not None:
            fn = check_U1_fixed if u1_claim.endswith("fixed") else check_U1_menu
            mask &= fn(lq, us, factor)

        if not (has_q1 or has_q2):
            # No lepton-anchored claim: probability is anchor-independent.
            per_anchor = np.full(len(lepton_samples), mask.mean())
        elif has_q1 and has_q2:
            # Both anchored bands: md is shared, so evaluate jointly per anchor.
            w1 = 2.0 * np.log(lq[:, 2]) - np.log(lq[:, 1])
            w2 = 2.0 * np.log(lq[:, 0]) - np.log(lq[:, 1])
            w1, w2 = w1[mask], w2[mask]
            per_anchor = np.empty(len(lepton_samples))
            for i in range(len(lepton_samples)):
                per_anchor[i] = np.count_nonzero(
                    (np.abs(w1 - ln_mu_star[i]) <= B1 * factor) &
                    (np.abs(w2 - ln_twome[i]) <= B2 * factor)
                ) / n_pool
        else:
            # Exactly one anchored band -> sort once, two binary searches
            # per anchor. Identical MC fraction, O(N log N).
            if has_q1:
                w = 2.0 * np.log(lq[:, 2]) - np.log(lq[:, 1])
                centres, hw = ln_mu_star, B1 * factor
            else:
                w = 2.0 * np.log(lq[:, 0]) - np.log(lq[:, 1])
                centres, hw = ln_twome, B2 * factor
            w = np.sort(w[mask])
            left = np.searchsorted(w, centres - hw, side="left")
            right = np.searchsorted(w, centres + hw, side="right")
            per_anchor = (right - left) / n_pool

        pool_means.append(float(per_anchor.mean()))
        last_per_anchor = per_anchor

    mean = float(np.mean(pool_means))
    stderr = float(np.std(pool_means) / np.sqrt(n_pools)) if n_pools > 1 else 0.0
    spread = {
        "mu_star_min": float(mu_star.min()), "mu_star_max": float(mu_star.max()),
        "mu_star_ratio": float(mu_star.max() / mu_star.min()),
        "twome_ratio": float(twome.max() / twome.min()),
        "p_anchor_min": float(last_per_anchor.min()),
        "p_anchor_max": float(last_per_anchor.max()),
        "p_anchor_mean": float(last_per_anchor.mean()),
        "p_anchor_std": float(last_per_anchor.std()),
    }
    if last_per_anchor.mean() > 0:
        spread["p_anchor_rel_std"] = float(last_per_anchor.std() / last_per_anchor.mean())
    return mean, stderr, spread


# ═══════════════════════════════════════════════════════════════════════════
# BRUTE FORCE
# ═══════════════════════════════════════════════════════════════════════════

def find_lepton_factor(condition, lepton_claims, seed, N_max, batch=1_000_000):
    """Smallest non-vacuous factor giving >=100 lepton-stage survivors."""
    for f in INFLATION_FACTORS:
        vac = False
        for c in lepton_claims:
            nv, frac = check_nonvacuous(c, f)
            if not nv:
                print(f"    factor {f}: VACUOUS ({c}: {frac*100:.1f}% of prior mass)", flush=True)
                vac = True
                break
        if vac:
            continue
        rng = np.random.default_rng(seed)
        hits, N = 0, 0
        while hits < 100 and N < N_max:
            if condition == "T0":
                lep = draw_leptons_T0(rng, batch)
                N += batch
            else:
                lep, att = sample_t1_koide(rng, batch)
                N += att
                if len(lep) == 0:
                    continue
            surv = np.ones(len(lep), dtype=bool)
            for c in lepton_claims:
                idx = np.where(surv)[0]
                if len(idx) == 0:
                    break
                surv[idx[~LEP_CHECK[c](lep[idx], f)]] = False
            hits += int(surv.sum())
        print(f"    factor {f}: {hits} lepton-stage survivors in {N:,} draws", flush=True)
        if hits >= 100:
            return f, hits, N
    return None, 0, 0


def build_survivor_pool(condition, lepton_claims, factor, seed, N_max, batch=1_000_000):
    """Collect lepton-stage survivors and the TOTAL attempted draws.

    total_eff is the denominator for the joint brute-force rate — this is the
    unit the analytic side (p_lep * p_quark) is expressed in.
    """
    rng = np.random.default_rng(seed)
    chunks = []
    total_eff = 0
    n_surv = 0
    t0 = time.time()
    while total_eff < N_max:
        if condition == "T0":
            lep = draw_leptons_T0(rng, batch)
            total_eff += batch
        else:
            lep, att = sample_t1_koide(rng, batch)
            total_eff += att
            if len(lep) == 0:
                continue
        surv = np.ones(len(lep), dtype=bool)
        for c in lepton_claims:
            idx = np.where(surv)[0]
            if len(idx) == 0:
                break
            surv[idx[~LEP_CHECK[c](lep[idx], factor)]] = False
        if surv.any():
            chunks.append(lep[surv])
            n_surv += int(surv.sum())
        if total_eff % 100_000_000 == 0:
            print(f"      pool: {n_surv} survivors / {total_eff:,} draws "
                  f"({time.time()-t0:.0f}s)", flush=True)
    pool = np.vstack(chunks) if chunks else np.empty((0, 3))
    print(f"    pool: {len(pool)} survivors from {total_eff:,} draws "
          f"({time.time()-t0:.0f}s)", flush=True)
    return pool, total_eff


def brute_force_stage(pool, total_eff, new_claims, factor, seed, chunk=2_000_000):
    """Joint brute-force hits: lepton-stage survivors that also pass new claims."""
    rng = np.random.default_rng(seed)
    hits = 0
    for s in range(0, len(pool), chunk):
        lep_s = pool[s:s + chunk]
        n = len(lep_s)
        lq = draw_quark_triple(rng, n)
        us = draw_quark_pair(rng, n)
        surv = np.ones(n, dtype=bool)
        for c in new_claims:
            idx = np.where(surv)[0]
            if len(idx) == 0:
                break
            if c == "Q1":
                m = check_Q1(lq[idx], lep_s[idx], factor)
            elif c == "Q2":
                m = check_Q2(lq[idx], lep_s[idx], factor)
            elif c == "U1_fixed":
                m = check_U1_fixed(lq[idx], us[idx], factor)
            elif c == "U1_menu":
                m = check_U1_menu(lq[idx], us[idx], factor)
            else:
                continue
            surv[idx[~m]] = False
        hits += int(surv.sum())
    return hits, hits / total_eff if total_eff else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# STAGE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def validate_stage(condition, stage_name, new_claims, u1_mode, pool, total_eff,
                   lepton_samples, p_lep, calib_seed):
    """Adaptive stage-local factor search + 2-Poisson-sigma comparison."""
    print(f"\n  ── STAGE {stage_name}  (new claims: {new_claims}) ──", flush=True)
    attempts = []
    for f in INFLATION_FACTORS:
        vac = False
        for c in new_claims:
            nv, frac = check_nonvacuous(c, f)
            if not nv:
                print(f"    factor {f}: VACUOUS ({c}: {frac*100:.1f}% of prior mass)", flush=True)
                attempts.append({"factor": f, "vacuous": True, "claim": c,
                                 "prior_mass_frac": frac})
                vac = True
                break
        if vac:
            continue

        t0 = time.time()
        p_q, p_q_err, spread = anchored_quark_prob(
            lepton_samples, new_claims, f, u1_mode,
            seed=calib_seed + 13 * INFLATION_FACTORS.index(f))
        t2 = p_lep * p_q
        hits, bf_rate = brute_force_stage(
            pool, total_eff, new_claims, f,
            seed=calib_seed + 7777 * INFLATION_FACTORS.index(f))
        sigma = np.sqrt(hits) / total_eff if hits > 0 else 0.0
        dev_sigma = abs(t2 - bf_rate) / sigma if sigma > 0 else float("inf")
        within = bool(dev_sigma <= 2.0)

        print(f"    factor {f}: T2={t2:.4e}  BF={bf_rate:.4e}±{sigma:.4e} "
              f"({hits} hits)  {dev_sigma:.2f}σ  {'PASS' if within else 'FAIL'} "
              f"[{time.time()-t0:.0f}s]", flush=True)

        attempts.append({
            "factor": f, "vacuous": False, "t2_analytic": float(t2),
            "p_lep": float(p_lep), "p_quark_anchor_avg": float(p_q),
            "p_quark_stderr": float(p_q_err), "bf_hits": int(hits),
            "bf_rate": float(bf_rate), "bf_sigma": float(sigma),
            "deviation_sigma": float(dev_sigma), "within_2sigma": within,
            "anchor_spread": spread,
        })

        if hits >= 100:
            return {
                "eligible": True, "stage": stage_name,
                "chosen_factor": f, "t2_analytic": float(t2),
                "p_lep": float(p_lep), "p_quark_anchor_avg": float(p_q),
                "p_quark_stderr": float(p_q_err),
                "bf_hits": int(hits), "bf_N_eff": int(total_eff),
                "bf_rate": float(bf_rate), "bf_sigma": float(sigma),
                "deviation_sigma": float(dev_sigma), "within_2sigma": within,
                "anchor_samples": int(len(lepton_samples)),
                "anchor_spread": spread, "attempts": attempts,
            }, within

    return {"eligible": False, "stage": stage_name,
            "reason": "no non-vacuous factor reached 100 hits",
            "attempts": attempts}, False


def validate_cell(condition, u1_mode, calib_seed=271828, N_max=1_000_000_000,
                  n_anchor=10_000):
    t_start = time.time()
    lepton_claims = ["L1", "L2", "L3"] if condition == "T0" else ["L2", "L3"]
    lep_stage_name = "L1_L2_L3" if condition == "T0" else "L2_L3"
    u1_claim = f"U1_{u1_mode}"

    print(f"\n{'='*70}")
    print(f"TIER-2 v0.5 VALIDATION — {condition} x U1-{u1_mode} x P-logU")
    print(f"calibration seed {calib_seed}, N_max {N_max:,}, anchors {n_anchor}")
    print(f"{'='*70}")

    # ── Step 1: stage-local lepton factor (§v0.5 b) ──
    print("\n  Step 1: lepton-stage factor", flush=True)
    f_lep, lep_hits, lep_N = find_lepton_factor(condition, lepton_claims,
                                                calib_seed, N_max)
    if f_lep is None:
        return {"cell": f"{condition}_{u1_mode}_logU", "all_stages_validated": False,
                "error": "no lepton factor reached 100 hits"}, False
    print(f"  → lepton-stage factor = {f_lep}", flush=True)

    # ── Step 2: analytic lepton block + brute-force survivor pool ──
    p_lep, p_lep_info = lepton_block_prob(condition, f_lep)
    print(f"\n  Step 2: lepton block  P = {p_lep:.6e}  ({p_lep_info['method']})", flush=True)

    print("  building survivor pool...", flush=True)
    pool, total_eff = build_survivor_pool(condition, lepton_claims, f_lep,
                                          calib_seed + 1, N_max)

    bf_lep_rate = len(pool) / total_eff if total_eff else 0.0
    bf_lep_sigma = np.sqrt(len(pool)) / total_eff if len(pool) else 0.0
    lep_dev = abs(p_lep - bf_lep_rate) / bf_lep_sigma if bf_lep_sigma > 0 else float("inf")
    lep_within = bool(lep_dev <= 2.0)
    print(f"    T2={p_lep:.4e}  BF={bf_lep_rate:.4e}±{bf_lep_sigma:.4e} "
          f"({len(pool)} hits)  {lep_dev:.2f}σ  {'PASS' if lep_within else 'FAIL'}", flush=True)

    record = {lep_stage_name: {
        "eligible": True, "stage": lep_stage_name, "chosen_factor": f_lep,
        "t2_analytic": float(p_lep), "lepton_block_info": p_lep_info,
        "bf_hits": int(len(pool)), "bf_N_eff": int(total_eff),
        "bf_rate": float(bf_lep_rate), "bf_sigma": float(bf_lep_sigma),
        "deviation_sigma": float(lep_dev), "within_2sigma": lep_within,
    }}
    all_ok = lep_within

    # ── Step 3: anchor samples from inside the inflated window (§v0.5 a) ──
    print(f"\n  Step 3: drawing {n_anchor} lepton draws from the inflated window", flush=True)
    lepton_samples = sample_lepton_window(condition, f_lep, n_anchor, calib_seed + 2)
    mu_star = lepton_samples.sum(axis=1)
    print(f"    got {len(lepton_samples)}; mu_star spans "
          f"[{mu_star.min():.4g}, {mu_star.max():.4g}] MeV "
          f"(ratio {mu_star.max()/mu_star.min():.4g})", flush=True)

    # ── Step 4: per-stage validation ──
    for stage_name, new_claims in [("+Q1", ["Q1"]), ("+Q2", ["Q2"]), ("+U1", [u1_claim])]:
        res, ok = validate_stage(condition, stage_name, new_claims, u1_mode,
                                 pool, total_eff, lepton_samples, p_lep, calib_seed)

        # §v0.5(c): compositional eligibility for +U1
        if stage_name == "+U1" and not res.get("eligible"):
            print("\n    +U1 not directly eligible → compositional route (§v0.5 c)", flush=True)
            comp = {"method": "compositional", "stage": "+U1"}

            # (i) P(U1) singleton at factor 1
            rng = np.random.default_rng(calib_seed + 55555)
            hits_u1, N_u1 = 0, 0
            fn = check_U1_fixed if u1_mode == "fixed" else check_U1_menu
            for _ in range(100):
                lq = draw_quark_triple(rng, 100_000)
                us = draw_quark_pair(rng, 100_000)
                hits_u1 += int(fn(lq, us, 1.0).sum())
                N_u1 += 100_000
            rate_u1 = hits_u1 / N_u1
            sig_u1 = np.sqrt(hits_u1) / N_u1
            p_u1_analytic, _, _ = anchored_quark_prob(
                lepton_samples, [u1_claim], 1.0, u1_mode, seed=calib_seed + 4242)
            dev_u1 = abs(p_u1_analytic - rate_u1) / sig_u1 if sig_u1 > 0 else float("inf")
            ok_u1 = bool(dev_u1 <= 2.0)
            print(f"    (i)  P(U1) f=1: T2={p_u1_analytic:.6e} "
                  f"BF={rate_u1:.6e}±{sig_u1:.6e} ({hits_u1}/{N_u1}) "
                  f"{dev_u1:.2f}σ {'PASS' if ok_u1 else 'FAIL'}", flush=True)
            comp["u1_singleton"] = {
                "factor": 1, "t2_analytic": float(p_u1_analytic),
                "bf_hits": int(hits_u1), "bf_N": int(N_u1),
                "bf_rate": float(rate_u1), "bf_sigma": float(sig_u1),
                "deviation_sigma": float(dev_u1), "within_2sigma": ok_u1}

            # (ii) mu-coupled Q2^U1 pair at adaptive stage-local factor
            print("    (ii) Q2^U1 pair", flush=True)
            res_pair, ok_pair = validate_stage(
                condition, "+U1_pair(Q2^U1)", ["Q2", u1_claim], u1_mode,
                pool, total_eff, lepton_samples, p_lep, calib_seed + 66666)
            comp["q2_u1_pair"] = res_pair

            comp_ok = bool(ok_u1 and res_pair.get("eligible") and ok_pair)
            comp["validated_by_composition"] = comp_ok
            comp["within_2sigma"] = comp_ok
            comp["eligible"] = True
            print(f"    VALIDATED-BY-COMPOSITION: {'YES' if comp_ok else 'NO'}", flush=True)
            res, ok = comp, comp_ok

        record[stage_name] = res
        if not ok:
            all_ok = False

    out = {
        "engine_id": "amb", "spec_version": "v0.5",
        "script": "tier2_recovery_v0.5.py",
        "cell": f"{condition}_{u1_mode}_logU",
        "condition": condition, "u1_mode": u1_mode, "prior": "logU",
        "calibration_seed": calib_seed, "N_max": N_max,
        "anchor_samples": int(n_anchor),
        "lepton_stage_factor": f_lep,
        "all_stages_validated": bool(all_ok),
        "validation_record": record,
        "runtime_seconds": float(time.time() - t_start),
    }
    return out, all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="T1", choices=["T0", "T1"])
    ap.add_argument("--u1-mode", default="menu", choices=["fixed", "menu"])
    ap.add_argument("--outdir", default="results/amb-20260811-v0.5")
    ap.add_argument("--N-max", type=int, default=1_000_000_000)
    ap.add_argument("--anchors", type=int, default=10_000)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out, ok = validate_cell(args.condition, args.u1_mode,
                            calib_seed=271828, N_max=args.N_max,
                            n_anchor=args.anchors)
    path = os.path.join(
        args.outdir,
        f"tier2_validation_{args.condition}_{args.u1_mode}_logU_v0.5.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {path}")
    print("ALL STAGES VALIDATED" if ok else "VALIDATION INCOMPLETE/FAILED")


if __name__ == "__main__":
    main()

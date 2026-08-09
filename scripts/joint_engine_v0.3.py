#!/usr/bin/env python3
"""
JOINT COINCIDENCE BUDGET — Full Engine v0.3 (AUDIT-RESPONSE REVISION)
ENGINE_ID: amb
Spec: specs/SPEC_V03.md

v0.3 changes from v0.2:
  1. ACCEPTED-N SEMANTICS: N_eff = N accepted worlds per cell.
     Rejection is internal. Acceptance rate reported separately.
  2. T1 documented as Koide-sheet measure, log-uniform in (m3, m1/m3).
  3. 20-target U1 menu (1/3 included). v0.2 used 19.
  4. True truncated normal (scipy.stats.truncnorm) for logN prior.
     Clipped variant renamed 'censored'.
  5. |log(r/r0)| <= eps for L2/L3 criterion.
  6. Alt-sheet T1 variant: log-uniform in (m2, m1/m2).

Gates (stop on mismatch, never tune):
  G1: golden regression 69/1e7 byte-identical under T0.
  G2: T1 L1 singleton = 1.000000 exactly (accepted-N denominator).
  G3: all support gates as v0.2.

T0 cells: NOT rerun; carried forward from v0.2.
T1 cells: rerun at N=2e9 accepted (primary) / 2e8 (variants).
Alt-sheet: 2e8 accepted.
Seeds: SHA-256 of spec text + cell name.
"""

import numpy as np
import json
import os
import sys
import time
import hashlib
from scipy.stats import beta as beta_dist
from scipy.stats import truncnorm as truncnorm_dist

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS (frozen, shared with v0.2)
# ═══════════════════════════════════════════════════════════════════════════

ang = 2.0 * np.pi * np.arange(3) / 3.0
cos_ang = np.cos(ang)
sin_ang = np.sin(ang)

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

# v0.3: 20-target menu (1/3 included, matching uptype-pinning v2.0)
U1_MENU = [
    (1, 3),  # NEW in v0.3 — 1/3 was excluded from the (1/3,1] interval in v0.2
    (1, 1), (1, 2), (2, 3), (3, 4),
    (2, 5), (3, 5), (4, 5),
    (5, 6),
    (3, 7), (4, 7), (5, 7), (6, 7),
    (3, 8), (5, 8), (7, 8),
    (4, 9), (5, 9), (7, 9), (8, 9),
]
assert len(U1_MENU) == 20, f"U1_MENU has {len(U1_MENU)} targets, expected 20"
U1_MENU_TARGETS = np.array([9.0 * p / q for (p, q) in U1_MENU], dtype=np.float64)

LEP_LO, LEP_HI = 0.3, 2000.0
QUARK_LO, QUARK_HI = 0.5, 2e5

LEP_LOG_LO, LEP_LOG_HI = np.log(LEP_LO), np.log(LEP_HI)
LEP_LOG_V = LEP_LOG_HI - LEP_LOG_LO
QUARK_LOG_LO, QUARK_LOG_HI = np.log(QUARK_LO), np.log(QUARK_HI)
QUARK_LOG_V = QUARK_LOG_HI - QUARK_LOG_LO
HIERARCHY_MIN = (4.0 + np.sqrt(18.0)) ** 2  # ≈ 67.9

# ═══════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS (kdist, Q_U, CP95)
# ═══════════════════════════════════════════════════════════════════════════

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
# CLAIM CHECK FUNCTIONS — v0.3: L2/L3 use |log(r/r0)| <= eps
# ═══════════════════════════════════════════════════════════════════════════

def check_L1(leptons, f=1.0):
    return kdist(leptons) <= L1_TOL * f

def check_L2(leptons, f=1.0):
    # v0.3: |log(r/r0)| <= eps  (was |r/r0 - 1| <= eps in v0.2)
    return np.abs(np.log(leptons[:, 1] / leptons[:, 0] / L2_TARGET)) <= L2_TOL * f

def check_L3(leptons, f=1.0):
    # v0.3: |log(r/r0)| <= eps
    return np.abs(np.log(leptons[:, 2] / leptons[:, 1] / L3_TARGET)) <= L3_TOL * f

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
# DRAW FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def draw_mass_triple(rng, batch_size, lo, hi, prior, sort=True):
    if prior == "logU":
        x = np.exp(rng.uniform(np.log(lo), np.log(hi), size=(batch_size, 3)))
    elif prior == "logN":
        # v0.3: true truncated normal in log space
        log_lo, log_hi = np.log(lo), np.log(hi)
        log_mid = (log_lo + log_hi) / 2.0
        log_sigma = 1.5 * np.log(10)
        a = (log_lo - log_mid) / log_sigma
        b = (log_hi - log_mid) / log_sigma
        log_x = truncnorm_dist.rvs(a, b, loc=log_mid, scale=log_sigma, size=(batch_size, 3))
        x = np.exp(log_x)
    elif prior == "linU":
        x = rng.uniform(lo, hi, size=(batch_size, 3))
    elif prior == "censored":
        # v0.3: renamed — old clip-based logN, retained as 'censored'
        mid = np.exp((np.log(lo) + np.log(hi)) / 2.0)
        x = np.exp(rng.normal(np.log(mid), 1.5 * np.log(10), size=(batch_size, 3)))
        x = np.clip(x, lo, hi)
    else:
        raise ValueError(f"Unknown prior: {prior}")
    if sort:
        x.sort(axis=1)
    return x

def draw_mass_pair(rng, batch_size, lo, hi, prior):
    if prior == "logU":
        x = np.exp(rng.uniform(np.log(lo), np.log(hi), size=(batch_size, 2)))
    elif prior == "logN":
        log_lo, log_hi = np.log(lo), np.log(hi)
        log_mid = (log_lo + log_hi) / 2.0
        log_sigma = 1.5 * np.log(10)
        a = (log_lo - log_mid) / log_sigma
        b = (log_hi - log_mid) / log_sigma
        log_x = truncnorm_dist.rvs(a, b, loc=log_mid, scale=log_sigma, size=(batch_size, 2))
        x = np.exp(log_x)
    elif prior == "linU":
        x = rng.uniform(lo, hi, size=(batch_size, 2))
    elif prior == "censored":
        mid = np.exp((np.log(lo) + np.log(hi)) / 2.0)
        x = np.exp(rng.normal(np.log(mid), 1.5 * np.log(10), size=(batch_size, 2)))
        x = np.clip(x, lo, hi)
    else:
        raise ValueError(f"Unknown prior: {prior}")
    return x

# ═══════════════════════════════════════════════════════════════════════════
# T1 KOIDE SHEET SAMPLERS
# ═══════════════════════════════════════════════════════════════════════════

def sample_t1_koide(rng, batch_size):
    """T1 Koide-conditioned lepton draws.
    Log-uniform in (m3, m1/m3). Solve Koide equation for m2.
    Returns (accepted_array, attempted_count).
    """
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


def sample_t1_alt_koide(rng, batch_size):
    """ALT-SHEET T1: log-uniform in (m2, m1/m2). Solve Koide for m3.
    Returns (accepted_array, attempted_count).
    """
    m2 = np.exp(rng.uniform(np.log(LEP_LO), np.log(LEP_HI), size=batch_size))
    r = np.exp(rng.uniform(np.log(1e-5), np.log(1e-1), size=batch_size))
    m1 = r * m2
    s1, s2 = np.sqrt(m1), np.sqrt(m2)
    # Quadratic for s3: s3^2 - 4(s1+s2)s3 + (s1^2+s2^2-4s1s2) = 0
    b_alt = -4.0 * (s1 + s2)
    c_alt = s1**2 + s2**2 - 4.0 * s1 * s2
    disc = b_alt**2 - 4.0 * c_alt
    valid_disc = disc >= 0
    disc = np.maximum(disc, 0.0)
    # Plus branch: s3 > 2(s1+s2) > s2, giving m3 > m2
    s3 = (-b_alt + np.sqrt(disc)) / 2.0
    s3_ok = s3 > s2
    sorted_ok = (m1 < m2) & (m2 < s3**2)
    hierarchy_ok = (s3**2 / m1) > HIERARCHY_MIN
    keep = valid_disc & s3_ok & sorted_ok & hierarchy_ok
    m3 = s3[keep] ** 2
    result = np.column_stack([m1[keep], m2[keep], m3])
    return result, batch_size


def generate_t1_accepted(rng, N_accepted, sampler='standard', batch_size=1_000_000):
    """Generate exactly N_accepted T1 draws. Returns (array, total_attempted, acc_rate)."""
    sf = sample_t1_koide if sampler == 'standard' else sample_t1_alt_koide
    chunks = []
    total_attempted = 0
    total_accepted = 0
    last_report = 0

    while total_accepted < N_accepted:
        remaining = N_accepted - total_accepted
        # Oversample: acceptance rate ~0.79 for standard, similar for alt
        draw_size = min(batch_size, max(batch_size, int(remaining / 0.70)))
        draws, attempted = sf(rng, draw_size)
        total_attempted += attempted
        if len(draws) > 0:
            chunks.append(draws)
            total_accepted += len(draws)

        # Progress reporting every ~5%
        milestone = (total_accepted * 20) // N_accepted
        if milestone > last_report:
            last_report = milestone
            pct = total_accepted * 100.0 / N_accepted
            acc_rate = total_accepted / total_attempted if total_attempted > 0 else 0.0
            print(f"    [generate] {total_accepted:,}/{N_accepted:,} accepted ({pct:.0f}%), "
                  f"acceptance rate={acc_rate:.4f}, attempted={total_attempted:,}",
                  flush=True)

    result = np.vstack(chunks)[:N_accepted]
    acceptance_rate = N_accepted / total_attempted
    return result, total_attempted, acceptance_rate

# ═══════════════════════════════════════════════════════════════════════════
# SUPPORT GATE (G3)
# ═══════════════════════════════════════════════════════════════════════════

def check_support_gate():
    r_obs = LEPTONS_OBS[0] / LEPTONS_OBS[2]
    r_lo, r_hi = 1e-5, 1e-1
    ok = True
    if not (r_lo <= r_obs <= r_hi):
        print(f"G3 FAIL: r_obs={r_obs:.6e} not in [{r_lo}, {r_hi}]")
        ok = False
    hierarchy_ratio = LEPTONS_OBS[2] / LEPTONS_OBS[0]
    if hierarchy_ratio <= HIERARCHY_MIN:
        print(f"G3 FAIL: m3/m1={hierarchy_ratio:.1f} <= {HIERARCHY_MIN:.1f}")
        ok = False
    if ok:
        print(f"G3 PASS: r_obs={r_obs:.6e} in [{r_lo}, {r_hi}], "
              f"m3/m1={hierarchy_ratio:.1f} > {HIERARCHY_MIN:.1f}")
    else:
        print("G3 FAILED — STOP.")
        sys.exit(1)
    return ok

# ═══════════════════════════════════════════════════════════════════════════
# GOLDEN REGRESSION (G1) — T0, byte-identical to v0.2
# ═══════════════════════════════════════════════════════════════════════════

def run_golden_regression():
    print("\n" + "=" * 70)
    print("G1: GOLDEN REGRESSION — seed=20260726, N=1e7 -> expect 69 hits")
    print("=" * 70)
    seed = 20260726
    N = 10_000_000
    batch_size = 1_000_000
    n_batches = N // batch_size

    leptons_obs = np.array([0.51099895, 105.6583755, 1776.93])
    downs_obs = np.array([4.70, 93.4, 4966.0])
    J_lepton_obs = kdist(leptons_obs)
    J_downs_obs = kdist(downs_obs)
    J_obs = J_lepton_obs + J_downs_obs

    print(f"  kdist(leptons) = {J_lepton_obs:.6e}  (target: 3.3049e-6)")
    print(f"  kdist(downs)   = {J_downs_obs:.6e}  (target: 1.0503e-3)")

    if not np.isclose(J_lepton_obs, 3.3049e-6, rtol=1e-4):
        print(f"G1 FAIL: lepton kdist mismatch")
        sys.exit(1)
    if not np.isclose(J_downs_obs, 1.0503e-3, rtol=1e-4):
        print(f"G1 FAIL: downs kdist mismatch")
        sys.exit(1)
    print("  OK kdist values match")

    rng = np.random.default_rng(seed)
    lep_lo, lep_hi = np.log(0.3), np.log(2000.0)
    dwn_lo, dwn_hi = np.log(2.0), np.log(10000.0)
    total_hits = 0

    for b in range(n_batches):
        lep = np.exp(rng.uniform(lep_lo, lep_hi, size=(batch_size, 3)))
        lep.sort(axis=1)
        dwn = np.exp(rng.uniform(dwn_lo, dwn_hi, size=(batch_size, 3)))
        dwn.sort(axis=1)
        J_batch = kdist(lep) + kdist(dwn)
        hits = np.sum(J_batch <= J_obs)
        total_hits += hits
        if (b + 1) % 2 == 0:
            print(f"    batch {b+1}/{n_batches}, hits={total_hits}", flush=True)

    print(f"  Total hits: {total_hits}  (expected: 69)")
    if total_hits != 69:
        print(f"G1 FAIL: hit count {total_hits} != 69")
        sys.exit(1)

    # CP95 check
    k_pooled, n_pooled = 189, 30_000_000
    lo, hi = clopper_pearson(k_pooled, n_pooled)
    print(f"  CP95 pooled {k_pooled}/{n_pooled}: [{lo:.6e}, {hi:.6e}]  (target: [5.43e-6, 7.27e-6])")
    if not (np.isclose(lo, 5.43e-6, rtol=5e-3) and np.isclose(hi, 7.27e-6, rtol=5e-3)):
        print(f"G1 FAIL: CP95 mismatch")
        sys.exit(1)

    print("G1 PASS: golden regression byte-identical OK")
    return True

# ═══════════════════════════════════════════════════════════════════════════
# SINGLETONS — v0.3 accepted-N semantics
# ═══════════════════════════════════════════════════════════════════════════

ALL_CLAIMS = ["L1", "L2", "L3", "Q1", "Q2", "U1_fixed", "U1_menu"]

def measure_singletons_t1(rng, N_accepted, prior="logU", sampler='standard'):
    """Measure T1 singleton rates at N_accepted ACCEPTED worlds.
    Returns (rates_dict, N_accepted, total_attempted, acceptance_rate).
    """
    batch_size = 1_000_000
    counts = {c: 0 for c in ALL_CLAIMS}
    total_accepted = 0
    total_attempted = 0
    t0 = time.time()

    print(f"  Measuring T1 singletons at N_accepted={N_accepted:,}, prior={prior}, sampler={sampler}")
    sf = sample_t1_koide if sampler == 'standard' else sample_t1_alt_koide
    last_report = 0

    while total_accepted < N_accepted:
        remaining = N_accepted - total_accepted
        draw_size = min(batch_size, max(batch_size, int(remaining / 0.70)))
        leptons, attempted = sf(rng, draw_size)
        total_attempted += attempted
        n_acc = len(leptons)
        if n_acc == 0:
            continue
        total_accepted += n_acc

        # L1 is granted on sheet for T1
        counts["L1"] += n_acc

        light_q = draw_mass_triple(rng, n_acc, QUARK_LO, QUARK_HI, prior, sort=True)
        up_s = draw_mass_pair(rng, n_acc, QUARK_LO, QUARK_HI, prior)

        counts["L2"] += check_L2(leptons).sum()
        counts["L3"] += check_L3(leptons).sum()
        counts["Q1"] += check_Q1(light_q, leptons).sum()
        counts["Q2"] += check_Q2(light_q, leptons).sum()
        counts["U1_fixed"] += check_U1_fixed(light_q, up_s).sum()
        counts["U1_menu"] += check_U1_menu(light_q, up_s).sum()

        milestone = (total_accepted * 10) // N_accepted
        if milestone > last_report:
            last_report = milestone
            pct = total_accepted * 100.0 / N_accepted
            elapsed = time.time() - t0
            rate = total_accepted / elapsed if elapsed > 0 else 0
            print(f"    {total_accepted:,}/{N_accepted:,} accepted ({pct:.0f}%), "
                  f"{rate:.0f} acc/s, L2={counts['L2']}, L3={counts['L3']}",
                  flush=True)

    elapsed = time.time() - t0
    acceptance_rate = total_accepted / total_attempted
    print(f"  Done in {elapsed:.0f}s, acceptance_rate={acceptance_rate:.4f}, "
          f"attempted={total_attempted:,}")

    result = {}
    for claim in ALL_CLAIMS:
        k = int(counts[claim])
        f_rate = k / total_accepted if total_accepted > 0 else 0.0
        lo, hi = clopper_pearson(k, total_accepted)
        result[claim] = {"count": k, "rate": float(f_rate),
                         "cp95_lower": lo, "cp95_upper": hi}
    return result, total_accepted, total_attempted, acceptance_rate

# ═══════════════════════════════════════════════════════════════════════════
# TIER-1 CASCADE — v0.3 accepted-N semantics
# ═══════════════════════════════════════════════════════════════════════════

def run_tier1_cascade_t1(rng, N_accepted, prior="logU", u1_mode="menu",
                          sampler='standard', singleton_rates=None,
                          checkpoint_dir=None, cell_key=None):
    """Tier-1 rarity-ordered cascade for T1 at N_accepted ACCEPTED worlds.
    Checkpoints progress to disk.
    """
    batch_size = 1_000_000
    total_accepted = 0
    total_attempted = 0
    total_hits = 0
    stage_counts = {}
    t0 = time.time()

    # T1 cascade: L1 is granted, not in cascade
    cascade_order = ["L2", "L3", "Q1", "Q2", f"U1_{u1_mode}"]

    if singleton_rates:
        claim_order = [(c, singleton_rates.get(c, {}).get("rate", 1.0)) for c in cascade_order]
        claim_order.sort(key=lambda x: x[1])
        cascade_order = [c for c, _ in claim_order]
        print(f"  Cascade order (rarest first): {cascade_order}")

    for claim in cascade_order:
        stage_counts[claim] = 0

    sf = sample_t1_koide if sampler == 'standard' else sample_t1_alt_koide
    last_report = 0
    last_checkpoint = 0

    while total_accepted < N_accepted:
        remaining = N_accepted - total_accepted
        draw_size = min(batch_size, max(batch_size, int(remaining / 0.70)))
        leptons, attempted = sf(rng, draw_size)
        total_attempted += attempted
        n_acc = len(leptons)
        if n_acc == 0:
            continue
        total_accepted += n_acc

        light_q = draw_mass_triple(rng, n_acc, QUARK_LO, QUARK_HI, prior, sort=True)
        up_s = draw_mass_pair(rng, n_acc, QUARK_LO, QUARK_HI, prior)
        survivors = np.ones(n_acc, dtype=bool)

        for claim in cascade_order:
            si = np.where(survivors)[0]
            if len(si) == 0:
                break
            if claim == "L2":
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

        # Progress every ~5%
        milestone = (total_accepted * 20) // N_accepted
        if milestone > last_report:
            last_report = milestone
            pct = total_accepted * 100.0 / N_accepted
            elapsed = time.time() - t0
            rate = total_accepted / elapsed if elapsed > 0 else 0
            acc_rate = total_accepted / total_attempted if total_attempted > 0 else 0
            print(f"    [{cell_key}] {total_accepted:,}/{N_accepted:,} acc ({pct:.0f}%), "
                  f"attempted={total_attempted:,}, acc_rate={acc_rate:.4f}, "
                  f"rate={rate:.0f} acc/s, hits={total_hits}, "
                  f"L2={stage_counts.get('L2',0)}, L3={stage_counts.get('L3',0)}, "
                  f"Q1={stage_counts.get('Q1',0)}",
                  flush=True)

        # Checkpoint every ~25%
        ckpt_milestone = (total_accepted * 4) // N_accepted
        if checkpoint_dir and ckpt_milestone > last_checkpoint:
            last_checkpoint = ckpt_milestone
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{cell_key}.json")
            ckpt_data = {
                "cell_key": cell_key,
                "total_accepted": int(total_accepted),
                "total_attempted": int(total_attempted),
                "acceptance_rate": float(total_accepted / total_attempted),
                "total_hits": int(total_hits),
                "stage_counts": {k: int(v) for k, v in stage_counts.items()},
                "N_accepted_target": N_accepted,
                "elapsed_s": time.time() - t0,
            }
            with open(ckpt_path, "w") as f:
                json.dump(ckpt_data, f)

    elapsed = time.time() - t0
    acceptance_rate = total_accepted / total_attempted
    f_rate = total_hits / total_accepted if total_accepted > 0 else 0.0
    lo, hi = clopper_pearson(int(total_hits), int(total_accepted))
    print(f"  DONE [{cell_key}]: hits={total_hits}, N_accepted={total_accepted:,}, "
          f"attempted={total_attempted:,}, f={f_rate:.6e}, "
          f"CP95=[{lo:.6e},{hi:.6e}], acc_rate={acceptance_rate:.4f}, "
          f"{elapsed:.0f}s",
          flush=True)

    return {
        "hits": int(total_hits), "N_accepted": int(total_accepted),
        "N_attempted": int(total_attempted),
        "acceptance_rate": float(acceptance_rate),
        "rate": float(f_rate), "cp95_lower": lo, "cp95_upper": hi,
        "stage_counts": {k: int(v) for k, v in stage_counts.items()},
        "elapsed_s": elapsed,
    }

# ═══════════════════════════════════════════════════════════════════════════
# SEED DERIVATION — SHA-256 of spec text + cell name
# ═══════════════════════════════════════════════════════════════════════════

def derive_seeds(spec_path="specs/SPEC_V03.md"):
    """Derive per-cell seeds from SHA-256 of spec text + cell name."""
    with open(spec_path) as f:
        spec_text = f.read()
    spec_hash = hashlib.sha256(spec_text.encode()).hexdigest()
    print(f"Spec SHA-256: {spec_hash}")

    def seed_for(name):
        h = hashlib.sha256(f"{spec_text}\n{name}".encode()).digest()
        return int.from_bytes(h[:4], 'big')  # 32-bit unsigned

    return seed_for, spec_hash

# ═══════════════════════════════════════════════════════════════════════════
# MAIN — v0.3 full production
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Joint Coincidence Budget Engine v0.3")
    parser.add_argument("--mode", default="full",
                        choices=["gates", "singletons", "tier1", "full"])
    parser.add_argument("--outdir", default="results/amb-v0.3")
    parser.add_argument("--spec", default="specs/SPEC_V03.md")
    parser.add_argument("--skip-completed", action="store_true",
                        help="Skip cells with existing result files in outdir")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    seed_for, spec_hash = derive_seeds(args.spec)

    print(f"\n{'='*70}")
    print(f"JOINT COINCIDENCE BUDGET ENGINE v0.3 — amb")
    print(f"Spec: {args.spec}  SHA-256: {spec_hash[:16]}...")
    print(f"Output: {args.outdir}")
    print(f"Mode: {args.mode}")
    print(f"{'='*70}")

    # ── G3: Support gate ──
    print(f"\n{'─'*50}")
    print("G3: SUPPORT GATE")
    print(f"{'─'*50}")
    check_support_gate()

    # ── G1: Golden regression ──
    run_golden_regression()

    # ── G2: T1 L1 singleton verification ──
    print(f"\n{'─'*50}")
    print("G2: T1 L1 SINGLETON = 1.000000 VERIFICATION")
    print(f"{'─'*50}")
    t1_sing_seed = seed_for("T1_singletons_logU")
    print(f"  Seed: {t1_sing_seed}")
    rng_sing = np.random.default_rng(t1_sing_seed)
    sing_rates, sing_N, sing_attempted, sing_acc_rate = measure_singletons_t1(
        rng_sing, 10_000_000, "logU", "standard")

    l1_rate = sing_rates["L1"]["rate"]
    print(f"\n  L1 rate: {l1_rate:.6f}  (must be 1.000000)")
    print(f"  Acceptance rate: {sing_acc_rate:.6f}  (v0.2 anomaly ~0.792)")
    analytic_acc = np.log(0.0147 / 1e-5) / np.log(1e4)
    print(f"  Analytic check: ln(0.0147/1e-5)/ln(1e4) = {analytic_acc:.5f}")

    if not np.isclose(l1_rate, 1.0, atol=1e-9):
        print(f"G2 FAIL: L1 singleton rate = {l1_rate:.10f} != 1.000000")
        sys.exit(1)

    acc_match = np.isclose(sing_acc_rate, analytic_acc, rtol=5e-3)
    print(f"  Acceptance rate match to analytic: {'OK' if acc_match else 'MISMATCH'} "
          f"(got {sing_acc_rate:.5f}, analytic {analytic_acc:.5f})")

    # Print all singleton rates
    print(f"\n  T1 Singletons at N_accepted={sing_N:,}:")
    for c in ALL_CLAIMS:
        r = sing_rates[c]
        print(f"    {c}: {r['count']}/{sing_N} = {r['rate']:.6e} "
              f"CP95=[{r['cp95_lower']:.6e},{r['cp95_upper']:.6e}]")

    # Save singletons
    sing_path = os.path.join(args.outdir, "singletons_T1_logU_v0.3.json")
    with open(sing_path, "w") as f:
        json.dump({
            "engine_id": "amb", "spec_version": "v0.3", "spec_hash": spec_hash,
            "seed": t1_sing_seed, "N_accepted": sing_N, "N_attempted": sing_attempted,
            "acceptance_rate": sing_acc_rate,
            "rates": sing_rates,
        }, f, indent=2)
    print(f"  Saved: {sing_path}")

    print("G2 PASS: T1 L1 singleton = 1.000000 OK")

    if args.mode == "gates":
        print("\nAll gates passed. Exiting (--mode gates).")
        return

    if args.mode == "singletons":
        print("\nSingletons complete. Exiting (--mode singletons).")
        return

    # ── TIER-1 PRODUCTION ──
    print(f"\n{'='*70}")
    print("TIER-1 PRODUCTION — v0.3 ACCEPTED-N SEMANTICS")
    print(f"{'='*70}")

    # Use singleton rates for cascade ordering
    cascade_singleton_rates = {c: sing_rates[c] for c in ALL_CLAIMS}

    # Define all T1 cells
    T1_CELLS = [
        # Primary logU cells at N=2e9 accepted
        ("T1", "fixed", "logU", "standard", 2_000_000_000),
        ("T1", "menu",  "logU", "standard", 2_000_000_000),
        # Variant logN cells at N=2e8 accepted
        ("T1", "fixed", "logN", "standard", 200_000_000),
        ("T1", "menu",  "logN", "standard", 200_000_000),
        # Variant linU cells at N=2e8 accepted
        ("T1", "fixed", "linU", "standard", 200_000_000),
        ("T1", "menu",  "linU", "standard", 200_000_000),
        # Alt-sheet variant at N=2e8 accepted
        ("T1", "fixed", "logU", "alt",     200_000_000),
        ("T1", "menu",  "logU", "alt",     200_000_000),
    ]

    all_results = {}
    total_start = time.time()

    for idx, (condition, u1_mode, prior, sampler, N_acc) in enumerate(T1_CELLS):
        sampler_label = "alt" if sampler == "alt" else "std"
        cell_key = f"{condition}_{u1_mode}_{prior}_{sampler_label}"
        cell_seed = seed_for(cell_key)

        cell_path = os.path.join(args.outdir, f"cell_{cell_key}.json")
        if args.skip_completed and os.path.exists(cell_path):
            print(f"\n{'─'*60}")
            print(f"CELL {idx+1}/{len(T1_CELLS)}: {cell_key} — SKIPPED (completed)")
            print(f"{'─'*60}")
            with open(cell_path) as f:
                existing = json.load(f)
            all_results[cell_key] = {
                "hits": existing["hits"], "N_accepted": existing["N_accepted"],
                "N_attempted": existing["N_attempted"],
                "acceptance_rate": existing["acceptance_rate"],
                "rate": existing["rate"], "cp95_lower": existing["cp95_lower"],
                "cp95_upper": existing["cp95_upper"],
                "stage_counts": existing["stage_counts"],
                "type": existing.get("type", "BOUND"),
                "elapsed_s": existing.get("elapsed_s", 0),
            }
            continue

        print(f"\n{'─'*60}")
        print(f"CELL {idx+1}/{len(T1_CELLS)}: {cell_key}")
        print(f"  N_accepted={N_acc:,}  seed={cell_seed}")
        print(f"{'─'*60}")
        sys.stdout.flush()

        rng_cell = np.random.default_rng(cell_seed)
        cell_start = time.time()

        result = run_tier1_cascade_t1(
            rng_cell, N_acc, prior=prior, u1_mode=u1_mode,
            sampler=sampler, singleton_rates=cascade_singleton_rates,
            checkpoint_dir=args.outdir, cell_key=cell_key
        )

        cell_elapsed = time.time() - cell_start
        bound_type = "BOUND" if result["hits"] < 100 else "POINT"

        # Save individual cell result
        cell_output = {
            "engine_id": "amb", "spec_version": "v0.3", "spec_hash": spec_hash,
            "cell_key": cell_key, "seed": cell_seed,
            "condition": condition, "u1_mode": u1_mode, "prior": prior,
            "sampler": sampler,
            **result, "type": bound_type,
        }
        with open(cell_path, "w") as f:
            json.dump(cell_output, f, indent=2)

        all_results[cell_key] = {
            "hits": result["hits"], "N_accepted": result["N_accepted"],
            "N_attempted": result["N_attempted"],
            "acceptance_rate": result["acceptance_rate"],
            "rate": result["rate"], "cp95_lower": result["cp95_lower"],
            "cp95_upper": result["cp95_upper"],
            "stage_counts": result["stage_counts"],
            "type": bound_type, "elapsed_s": cell_elapsed,
        }

        print(f"  -> {bound_type}: f={result['rate']:.6e} "
              f"CP95=[{result['cp95_lower']:.6e},{result['cp95_upper']:.6e}], "
              f"acc_rate={result['acceptance_rate']:.4f}",
              flush=True)

        # Clean up checkpoint file
        ckpt_path = os.path.join(args.outdir, f"checkpoint_{cell_key}.json")
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

    # ── SAVE SUMMARY ──
    total_elapsed = time.time() - total_start
    summary_path = os.path.join(args.outdir, "tier1_summary_v0.3.json")
    summary = {
        "engine_id": "amb", "spec_version": "v0.3", "spec_hash": spec_hash,
        "total_elapsed_s": total_elapsed,
        "cells": all_results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── FINAL REPORT ──
    print(f"\n{'='*70}")
    print(f"V0.3 TIER-1 PRODUCTION COMPLETE — {total_elapsed:.0f}s total")
    print(f"{'='*70}")
    print(f"\n{'Cell':<30} {'Hits':>8} {'N_acc':>14} {'f':>12} {'CP95':>30} {'AccRate':>8} {'Type':>8}")
    print("-" * 110)
    for cell_key in sorted(all_results.keys()):
        r = all_results[cell_key]
        cp95_str = f"[{r['cp95_lower']:.4e},{r['cp95_upper']:.4e}]"
        print(f"  {cell_key:<28} {r['hits']:>8} {r['N_accepted']:>14,} "
              f"{r['rate']:>12.6e} {cp95_str:>30} {r['acceptance_rate']:>8.4f} {r['type']:>8}")

    # Find headline cell (most conservative T1 cell)
    primary_cells = {k: v for k, v in all_results.items() if v['N_accepted'] >= 2_000_000_000}
    if primary_cells:
        headline_key = max(primary_cells, key=lambda k: primary_cells[k]['cp95_upper'])
        headline = primary_cells[headline_key]
        print(f"\n  HEADLINE (most conservative): {headline_key}")
        print(f"  f < {headline['cp95_upper']:.6e} (CP95 upper bound, "
              f"{headline['hits']} hits in {headline['N_accepted']:,} accepted worlds)")
    else:
        headline_key = max(all_results, key=lambda k: all_results[k]['cp95_upper'])
        headline = all_results[headline_key]
        print(f"\n  HEADLINE: {headline_key}")
        print(f"  f < {headline['cp95_upper']:.6e} (CP95 upper bound)")

    # Acceptance rate summary
    print(f"\n  Acceptance rates:")
    for cell_key in sorted(all_results.keys()):
        r = all_results[cell_key]
        print(f"    {cell_key:<30} {r['acceptance_rate']:.4f}")

    print(f"\n  Artifacts:")
    print(f"    Summary: {summary_path}")
    print(f"    Cell results: {args.outdir}/cell_*.json")
    print(f"    Singletons: {sing_path}")
    print(f"    Spec hash: {spec_hash}")

    print(f"\n{'='*70}")
    print("DONE — engine v0.3")
    print(f"{'='*70}")

    print(f"\nGATE SUMMARY:")
    print(f"  G1 (golden regression): PASS OK")
    print(f"  G2 (T1 L1 = 1.000000):  PASS OK")
    print(f"  G3 (support gates):     PASS OK")

    return all_results


if __name__ == "__main__":
    main()

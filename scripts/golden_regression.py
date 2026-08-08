#!/usr/bin/env python3
"""
GOLDEN REGRESSION (§6) — GATING CHECK
Certify RNG conventions, kdist implementation, and CP95 code.
Must reproduce published results EXACTLY before any production run.

ENGINE_ID: amb
"""

import numpy as np
from scipy.stats import beta as beta_dist
import sys
import json

# ── kdist implementation (verbatim from spec §6) ──────────────────────
ang = 2 * np.pi * np.arange(3) / 3
c = np.cos(ang)
s = np.sin(ang)


def kdist(m):
    """Koide distance. m shape: (..., 3) or (3,) for single triple."""
    m = np.asarray(m)
    out = None
    for v in (np.sqrt(m), 1 / np.sqrt(m)):
        A = v.mean(axis=-1)
        X = (2 / 3) * (v * c).sum(-1)
        Y = -(2 / 3) * (v * s).sum(-1)
        d = np.abs(np.hypot(X, Y) / (np.sqrt(2) * A) - 1)
        out = d if out is None else np.minimum(out, d)
    return out


def clopper_pearson(k, n, alpha=0.05):
    """Clopper-Pearson CP95 interval.
    lower solves P(X>=k) = alpha/2  → Beta(α=k, β=n-k+1)
    upper solves P(X<=k) = alpha/2  → Beta(α=k+1, β=n-k)
    """
    if k == 0:
        lower = 0.0
    else:
        lower = beta_dist.ppf(alpha / 2, k, n - k + 1)
    if k == n:
        upper = 1.0
    else:
        upper = beta_dist.ppf(1 - alpha / 2, k + 1, n - k)
    return lower, upper


def run_golden_regression():
    """Run the full golden regression check from §6."""
    print("=" * 70)
    print("GOLDEN REGRESSION (§6) — GATING CHECK")
    print("=" * 70)

    # Observed values
    leptons_obs = np.array([0.51099895, 105.6583755, 1776.93])
    downs_obs = np.array([4.70, 93.4, 4966.0])

    J_lepton_obs = kdist(leptons_obs)
    J_downs_obs = kdist(downs_obs)
    J_obs = J_lepton_obs + J_downs_obs

    print(f"\nObserved kdist checks:")
    print(f"  kdist(leptons pole) = {J_lepton_obs:.6e}")
    print(f"  kdist(downs 2-GeV)  = {J_downs_obs:.6e}")
    print(f"  J_obs               = {J_obs:.6e}")

    target_J_obs = 1.0536e-3
    target_lepton_kdist = 3.3049e-6
    target_downs_kdist = 1.0503e-3

    # Check to 4 significant figures
    if not np.isclose(J_lepton_obs, target_lepton_kdist, rtol=1e-4):
        print(f"\n*** GATING FAILURE: lepton kdist mismatch ***")
        print(f"    Got {J_lepton_obs:.6e}, expected {target_lepton_kdist:.6e}")
        return False

    if not np.isclose(J_downs_obs, target_downs_kdist, rtol=1e-4):
        print(f"\n*** GATING FAILURE: downs kdist mismatch ***")
        print(f"    Got {J_downs_obs:.6e}, expected {target_downs_kdist:.6e}")
        return False

    if not np.isclose(J_obs, target_J_obs, rtol=1e-4):
        print(f"\n*** GATING FAILURE: J_obs mismatch ***")
        print(f"    Got {J_obs:.6e}, expected {target_J_obs:.6e}")
        return False

    print(f"  ✓ All kdist values match to 4 significant figures")

    # ── Primary gate: seed 20260726, N=1e7 → 69 hits ──────────────────
    seed = 20260726
    N = 10_000_000
    batch_size = 1_000_000
    n_batches = N // batch_size

    rng = np.random.default_rng(seed)

    # Ranges
    lep_lo, lep_hi = np.log(0.3), np.log(2000.0)
    dwn_lo, dwn_hi = np.log(2.0), np.log(10000.0)

    total_hits = 0

    print(f"\nRunning primary gate: seed={seed}, N={N:,}, batch_size={batch_size:,}")
    print(f"  (leptons drawn first per §5 draw order)")

    for b in range(n_batches):
        # Draw leptons first (log-uniform [0.3, 2000], sorted ascending)
        lep_log = rng.uniform(lep_lo, lep_hi, size=(batch_size, 3))
        lep = np.exp(lep_log)
        lep.sort(axis=1)  # sorted ascending

        # Draw downs (log-uniform [2, 10000], sorted ascending)
        dwn_log = rng.uniform(dwn_lo, dwn_hi, size=(batch_size, 3))
        dwn = np.exp(dwn_log)
        dwn.sort(axis=1)

        # Compute J = kdist(leptons) + kdist(downs)
        J_lepton_batch = kdist(lep)
        J_downs_batch = kdist(dwn)
        J_batch = J_lepton_batch + J_downs_batch

        hits = np.sum(J_batch <= J_obs)
        total_hits += hits

        if (b + 1) % 2 == 0:
            print(f"  batch {b+1}/{n_batches}, hits so far: {total_hits}", flush=True)

    print(f"\n  Total hits: {total_hits}")
    print(f"  Expected:    69")

    if total_hits != 69:
        print(f"\n*** GATING FAILURE: hit count mismatch ***")
        print(f"    Got {total_hits}, expected 69")
        print(f"    Difference: {total_hits - 69:+d}")
        return False

    print(f"  ✓ Hit count matches: 69")

    # ── CP95 check: 189/3e7 → [5.43, 7.27]e-6 ────────────────────────
    k_pooled = 189
    n_pooled = 30_000_000
    lower, upper = clopper_pearson(k_pooled, n_pooled)

    print(f"\nCP95 check: {k_pooled}/{n_pooled}")
    print(f"  f = {k_pooled/n_pooled:.6e}")
    print(f"  CP95 lower: {lower:.6e}  (target: 5.43e-6)")
    print(f"  CP95 upper: {upper:.6e}  (target: 7.27e-6)")

    # Check to 3 significant figures (allowing small float differences)
    if not np.isclose(lower, 5.43e-6, rtol=5e-3):
        print(f"\n*** GATING FAILURE: CP95 lower bound mismatch ***")
        print(f"    Got {lower:.6e}, expected 5.43e-6")
        return False

    if not np.isclose(upper, 7.27e-6, rtol=5e-3):
        print(f"\n*** GATING FAILURE: CP95 upper bound mismatch ***")
        print(f"    Got {upper:.6e}, expected 7.27e-6")
        return False

    print(f"  ✓ CP95 interval matches [5.43, 7.27]e-6")

    print(f"\n{'=' * 70}")
    print(f"GOLDEN REGRESSION: ALL GATES PASSED ✓")
    print(f"{'=' * 70}")

    return True


if __name__ == "__main__":
    success = run_golden_regression()
    if not success:
        print("\n*** GATING FAILED — STOP. Do not proceed to production. ***")
        sys.exit(1)
    else:
        print("\nProceed to observed-side checks (§7) and production.")
        sys.exit(0)

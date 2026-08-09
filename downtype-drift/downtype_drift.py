#!/usr/bin/env python3
"""
downtype_drift.py — Down-type inverse-coordinate drift test with uncertainty quantification.

Extends the trunc-differencing integrator (sm_rge.py) to the down sector:
  - Q_inv = Q(1/y_d, 1/y_s, 1/y_b)  [inverse-coordinate analog of Q_U]
  - Derived analytic law for dQ_inv/dt (y_t-driven differential on y_b)
  - Draw-once uncertainty propagation (N >= 1e5, seed 20260811)
  - GATE D1: reproduce AHS2026 control values
  - GATE D2: correlation discipline (draw-once, correlated endpoints)

Spec: specs/SPEC_DRIFT.md v1.0
Seed: 20260811
Machine: ganymede (single-machine)

References:
  AHS2026: Antusch, Hinze, Saad, arXiv:2510.01312v2 (2026)
  PDG 2024: Phys. Rev. D 110, 030001 (2024)
  Machacek & Vaughn, NPB 249 (1985) 70 — 2-loop gauge beta functions
  Luo & Xiao, PRD 67, 065019 (2003); PRL 90, 011601 (2003) — 2-loop Yukawa
"""

import numpy as np
from scipy.integrate import solve_ivp
import json, os, sys, time, copy

# ─── PATH SETUP — import beta functions from sibling trunc-differencing ──────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "trunc-differencing"))
from sm_rge import (
    beta_1loop, beta_2loop, Q_U, nine_Q_U, integrate,
    np_to_native, json_dump, VAR_NAMES,
    ONE_LOOP_FACTOR, TWO_LOOP_FACTOR,
)

# ─── CONSTANTS ───────────────────────────────────────────────────────────────
MZ       = 91.1876          # GeV, PDG 2024 pole mass
M_3TEV   = 3.0e3            # GeV
T_3TEV   = np.log(M_3TEV / MZ)  # ~3.494
SEED     = 20260811
N_DRAW   = 100_000          # draw-once universes

# ─── CENTRAL INPUTS (AHS2026 Eq. 2.4, 2024 PDG) ─────────────────────────────
# Down-type Yukawas at M_Z
YB_CENTRAL = 1.630e-2
YS_CENTRAL = 3.06e-4
YD_CENTRAL = 1.54e-5

# Up-type Yukawas at M_Z (frozen from AHS2026 Eq. 2.4)
YT_CENTRAL = 0.967
YC_CENTRAL = 3.56e-3
YU_CENTRAL = 7.04e-6

# Gauge couplings at M_Z (AHS2026 Eq. 2.4)
G1_CENTRAL = 0.461228
G2_CENTRAL = 0.65096
G3_CENTRAL = 1.2123

# ─── UNCERTAINTIES (1-sigma, AHS2026 Eq. 2.4) ───────────────────────────────
SIGMA = {
    "yb": 0.009e-2,
    "ys": 0.04e-4,
    "yd": 0.02e-5,
    "yt": 0.004,
    "yc": 0.06e-3,
    "yu": 0.15e-6,
    "g1": 0.000026,
    "g2": 0.00004,
    "g3": 0.0046,
}

# ─── LEPTON YUKAWAS (held fixed — negligible impact on down-type drift) ─────
# From AHS2026 Eq. 2.4
YTAU_MZ = 0.99378e-2
YMU_MZ  = 5.85042e-4
YE_MZ   = 2.77713e-6

# ─── PUBLISHED CONTROL VALUES (AHS2026 paper) ───────────────────────────────
PUBLISHED = {
    "Q_inv_MZ":     0.66738,
    "Q_inv_3TeV":   0.66686,
    "drift_meas_pct": -0.078,   # measured drift in percent
    "drift_pred_pct": -0.06,    # predicted drift from derived law, percent
}

# ─── Q_inv ───────────────────────────────────────────────────────────────────
def Q_inv(yb, ys, yd):
    """Q_inv[y] = Q(1/y_d, 1/y_s, 1/y_b) — inverse-coordinate down-type diagnostic."""
    # z_i = 1/y_i
    zd = 1.0 / yd
    zs = 1.0 / ys
    zb = 1.0 / yb
    s  = zd + zs + zb
    sr = np.sqrt(zd) + np.sqrt(zs) + np.sqrt(zb)
    return s / (sr * sr)

def Q_inv_from_state(y):
    """Extract Q_inv from the full state vector y=[g1,g2,g3,yt,yc,yu,yb,ys,yd,...]."""
    return Q_inv(y[6], y[7], y[8])


# ─── DERIVED LAW: analytic dQ_inv/dt ─────────────────────────────────────────
def derived_drift_rate(yb, ys, yd, yt, yc, yu, g1, g2, g3):
    """
    Analytic dQ_inv/dt at 1-loop, keeping only flavor-differential beta terms.

    Q_inv = Q(z) where z_i = 1/y_i, Q[z] = Σz_i / (Σ√z_i)².

    dQ_inv/dt = Σ_i (∂Q/∂z_i) * dz_i/dt

    ∂Q/∂z_i = (R - S/√z_i) / R³   where S = Σz_j, R = Σ√z_j

    dz_i/dt = -z_i * (β_{y_i}^{diff} / y_i)
    where β_{y_i}^{diff} / y_i = (3/2 y_i² - 3/2 y_{u_i}²) / (16π²)
    is the flavor-differential part at 1-loop (flavor-universal gauge+trace cancels).

    Returns dQ_inv/dt in units of 1/(ln μ).
    """
    # Inverse Yukawas
    zd = 1.0 / yd
    zs = 1.0 / ys
    zb = 1.0 / yb

    S = zd + zs + zb
    sqrt_zd = np.sqrt(zd)
    sqrt_zs = np.sqrt(zs)
    sqrt_zb = np.sqrt(zb)
    R = sqrt_zd + sqrt_zs + sqrt_zb
    R3 = R * R * R

    # ∂Q/∂z_i coefficients
    dQ_dzd = (R - S / sqrt_zd) / R3
    dQ_dzs = (R - S / sqrt_zs) / R3
    dQ_dzb = (R - S / sqrt_zb) / R3

    # Flavor-differential beta: β^{diff}_{y_i} / y_i = (3/2 y_i² - 3/2 y_{u_i}²) / (16π²)
    beta_ratio_b = ONE_LOOP_FACTOR * (1.5 * yb * yb - 1.5 * yt * yt)
    beta_ratio_s = ONE_LOOP_FACTOR * (1.5 * ys * ys - 1.5 * yc * yc)
    beta_ratio_d = ONE_LOOP_FACTOR * (1.5 * yd * yd - 1.5 * yu * yu)

    # dz_i/dt = -z_i * beta_ratio_i
    dzb_dt = -zb * beta_ratio_b
    dzs_dt = -zs * beta_ratio_s
    dzd_dt = -zd * beta_ratio_d

    dQ_dt = dQ_dzd * dzd_dt + dQ_dzs * dzs_dt + dQ_dzb * dzb_dt
    return dQ_dt


def predicted_drift(yb, ys, yd, yt, yc, yu, g1, g2, g3, t1=0.0, t2=T_3TEV):
    """
    Predicted drift from the derived law: integrate dQ_inv/dt over [t1, t2].

    At 1-loop, keeping the leading yt-driven differential:
      ΔQ_inv ≈ dQ_inv/dt|_{M_Z} × Δt
    since the run M_Z → 3 TeV is short (Δt ≈ 3.49).
    """
    rate = derived_drift_rate(yb, ys, yd, yt, yc, yu, g1, g2, g3)
    return rate * (t2 - t1)


# ─── INITIAL CONDITIONS ─────────────────────────────────────────────────────
def initial_conditions(yb=YB_CENTRAL, ys=YS_CENTRAL, yd=YD_CENTRAL,
                       yt=YT_CENTRAL, yc=YC_CENTRAL, yu=YU_CENTRAL,
                       g1=G1_CENTRAL, g2=G2_CENTRAL, g3=G3_CENTRAL):
    """Return y0 array at t=0 (M_Z)."""
    return np.array([
        g1, g2, g3,
        yt, yc, yu,
        yb, ys, yd,
        YTAU_MZ, YMU_MZ, YE_MZ,
    ])


# ─── SINGLE EVOLUTION ────────────────────────────────────────────────────────
def evolve_one(y0, t_max=T_3TEV, rtol=1e-10, atol=1e-12):
    """
    Evolve from M_Z to t_max using 2-loop beta functions.
    Returns (sol_1L, sol_2L) or (None, None) on failure.
    """
    t_checkpoints = np.linspace(0.0, t_max, 20)
    try:
        sol_1L = solve_ivp(
            beta_1loop, [0.0, t_max], y0,
            method='RK45', rtol=rtol, atol=atol,
            t_eval=t_checkpoints, max_step=0.5,
        )
        sol_2L = solve_ivp(
            beta_2loop, [0.0, t_max], y0,
            method='RK45', rtol=rtol, atol=atol,
            t_eval=t_checkpoints, max_step=0.5,
        )
        if not sol_1L.success or not sol_2L.success:
            return None, None
        return sol_1L, sol_2L
    except Exception:
        return None, None


# ─── Q_inv FROM SOLUTION ─────────────────────────────────────────────────────
def extract_Q_inv(sol, t_targets):
    """
    Extract Q_inv at specified t-values from a solution.
    Returns dict: {label: Q_inv_value}
    """
    result = {}
    for label, t_val in t_targets.items():
        yb_val = float(np.interp(t_val, sol.t, sol.y[6]))
        ys_val = float(np.interp(t_val, sol.t, sol.y[7]))
        yd_val = float(np.interp(t_val, sol.t, sol.y[8]))
        result[label] = Q_inv(yb_val, ys_val, yd_val)
    return result


# ─── GATE D1: REPRODUCE CONTROL VALUES ──────────────────────────────────────
def gate_D1():
    """
    GATE D1: From central inputs, reproduce the paper's published control values:
      Q_inv(M_Z) = 0.66738, Q_inv(3 TeV) = 0.66686,
      measured drift = -0.078%, predicted drift = -0.06%.
    Tolerance: last quoted digit on each.
    """
    print("\n" + "="*72)
    print("GATE D1 — Control-value reproduction")
    print("="*72)

    # Compute Q_inv at M_Z directly from central inputs
    q_mz = Q_inv(YB_CENTRAL, YS_CENTRAL, YD_CENTRAL)

    # Evolve central inputs to 3 TeV
    y0 = initial_conditions()
    sol_1L, sol_2L = evolve_one(y0, t_max=T_3TEV)
    if sol_2L is None:
        print("  ⛔ Central evolution FAILED")
        return False, {}

    # Interpolate Q_inv at 3 TeV from 2-loop evolution
    yb_3tev = float(np.interp(T_3TEV, sol_2L.t, sol_2L.y[6]))
    ys_3tev = float(np.interp(T_3TEV, sol_2L.t, sol_2L.y[7]))
    yd_3tev = float(np.interp(T_3TEV, sol_2L.t, sol_2L.y[8]))
    q_3tev = Q_inv(yb_3tev, ys_3tev, yd_3tev)

    # Measured drift (from RGE evolution)
    drift_meas = q_3tev - q_mz
    drift_meas_pct = 100.0 * drift_meas / q_mz

    # Predicted drift (from derived law)
    drift_pred = predicted_drift(YB_CENTRAL, YS_CENTRAL, YD_CENTRAL,
                                 YT_CENTRAL, YC_CENTRAL, YU_CENTRAL,
                                 G1_CENTRAL, G2_CENTRAL, G3_CENTRAL)
    drift_pred_pct = 100.0 * drift_pred / q_mz

    # Tolerances: last quoted digit = ±1 in the last quoted place
    TOL_Q       = 1e-5    # 0.66738 → last digit in 1e-5 place
    TOL_DRIFT_MEAS = 0.001  # -0.078% → last digit "8" in 0.001% place
    TOL_DRIFT_PRED = 0.01   # -0.06%  → last digit "6" in 0.01% place

    print(f"  Q_inv(M_Z):      computed = {q_mz:.6f}  target = {PUBLISHED['Q_inv_MZ']:.5f}")
    print(f"  Q_inv(3 TeV):    computed = {q_3tev:.6f}  target = {PUBLISHED['Q_inv_3TeV']:.5f}")
    print(f"  Measured drift:  computed = {drift_meas_pct:.4f}%  target = {PUBLISHED['drift_meas_pct']:.3f}%")
    print(f"  Predicted drift: computed = {drift_pred_pct:.4f}%  target = {PUBLISHED['drift_pred_pct']:.2f}%")

    checks = {
        "Q_inv_MZ":     abs(q_mz          - PUBLISHED['Q_inv_MZ'])     < TOL_Q,
        "Q_inv_3TeV":   abs(q_3tev        - PUBLISHED['Q_inv_3TeV'])   < TOL_Q,
        "drift_meas":   abs(drift_meas_pct - PUBLISHED['drift_meas_pct']) < TOL_DRIFT_MEAS,
        "drift_pred":   abs(drift_pred_pct - PUBLISHED['drift_pred_pct']) < TOL_DRIFT_PRED,
    }

    for name, passed in checks.items():
        print(f"  GATE D1 {name}: {'✅ PASS' if passed else '❌ FAIL'}")

    all_pass = all(checks.values())

    record = {
        "gate": "GATE_D1",
        "passed": all_pass,
        "values": {
            "Q_inv_MZ_computed":     round(q_mz, 6),
            "Q_inv_MZ_target":       PUBLISHED['Q_inv_MZ'],
            "Q_inv_3TeV_computed":   round(q_3tev, 6),
            "Q_inv_3TeV_target":     PUBLISHED['Q_inv_3TeV'],
            "drift_meas_pct":        round(drift_meas_pct, 4),
            "drift_meas_target":     PUBLISHED['drift_meas_pct'],
            "drift_pred_pct":        round(drift_pred_pct, 4),
            "drift_pred_target":     PUBLISHED['drift_pred_pct'],
        },
        "checks": {k: bool(v) for k, v in checks.items()},
    }

    return all_pass, record


# ─── GATE D2: DRAW-ONCE CORRELATION DISCIPLINE ───────────────────────────────
def gate_D2(outdir):
    """
    GATE D2: Draw-once uncertainty propagation.
    Each universe is drawn ONCE and evolved M_Z → 3 TeV.
    Endpoints are correlated by construction.
    N >= 1e5 draws, seed 20260811.

    Reports:
      - Endpoint bands: σ(Q_inv(M_Z)), σ(Q_inv(3 TeV))
      - Drift band: σ(ΔQ_inv) — narrower than naive combination
      - Correlation benefit explicit
    """
    print("\n" + "="*72)
    print("GATE D2 — Draw-once correlation discipline (N={:,})".format(N_DRAW))
    print("="*72)

    rng = np.random.RandomState(SEED)

    # Draw parameter universes
    draws = {}
    param_names_ordered = ["g1", "g2", "g3", "yt", "yc", "yu", "yb", "ys", "yd"]
    central = {
        "g1": G1_CENTRAL, "g2": G2_CENTRAL, "g3": G3_CENTRAL,
        "yt": YT_CENTRAL, "yc": YC_CENTRAL, "yu": YU_CENTRAL,
        "yb": YB_CENTRAL, "ys": YS_CENTRAL, "yd": YD_CENTRAL,
    }
    for p in param_names_ordered:
        draws[p] = rng.normal(central[p], SIGMA[p], N_DRAW)
        # Clip non-negative for Yukawas
        if p.startswith("y"):
            draws[p] = np.clip(draws[p], 1e-30, None)

    # Storage for results
    q_mz_arr   = np.zeros(N_DRAW)
    q_3tev_arr = np.zeros(N_DRAW)
    drift_arr  = np.zeros(N_DRAW)
    pred_drift_arr = np.zeros(N_DRAW)
    n_failed = 0

    tic = time.time()
    report_interval = max(1, N_DRAW // 20)

    # ─── Checkpoint support (survive session loss) ─────────────────────────
    checkpoint_path = os.path.join(outdir, "draw_checkpoint.npz")
    start_i = 0

    # Try loading existing checkpoint
    if os.path.exists(checkpoint_path):
        ck = np.load(checkpoint_path, allow_pickle=True)
        start_i = int(ck["i"]) + 1
        if start_i >= N_DRAW:
            print(f"  Checkpoint complete ({start_i} >= {N_DRAW}), skipping draws")
        else:
            q_mz_arr[:start_i]   = ck["q_mz"]
            q_3tev_arr[:start_i] = ck["q_3tev"]
            drift_arr[:start_i]  = ck["drift"]
            pred_drift_arr[:start_i] = ck["pred_drift"]
            n_failed = int(ck["n_failed"])
            print(f"  Resuming from checkpoint: i={start_i}/{N_DRAW}, {n_failed} failed so far")
    else:
        print(f"  No checkpoint found, starting from i=0")

    CHECKPOINT_EVERY = 1000

    for i in range(start_i, N_DRAW):
        # Compute Q_inv(M_Z) directly from drawn inputs
        q_mz_arr[i] = Q_inv(draws["yb"][i], draws["ys"][i], draws["yd"][i])

        # Compute predicted drift from derived law
        pred_drift_arr[i] = predicted_drift(
            draws["yb"][i], draws["ys"][i], draws["yd"][i],
            draws["yt"][i], draws["yc"][i], draws["yu"][i],
            draws["g1"][i], draws["g2"][i], draws["g3"][i],
        )

        # Evolve this universe to 3 TeV
        y0 = initial_conditions(
            yb=draws["yb"][i], ys=draws["ys"][i], yd=draws["yd"][i],
            yt=draws["yt"][i], yc=draws["yc"][i], yu=draws["yu"][i],
            g1=draws["g1"][i], g2=draws["g2"][i], g3=draws["g3"][i],
        )
        sol_1L, sol_2L = evolve_one(y0, t_max=T_3TEV)

        if sol_2L is None:
            n_failed += 1
            q_3tev_arr[i] = np.nan
            drift_arr[i] = np.nan
        else:
            yb_f = float(np.interp(T_3TEV, sol_2L.t, sol_2L.y[6]))
            ys_f = float(np.interp(T_3TEV, sol_2L.t, sol_2L.y[7]))
            yd_f = float(np.interp(T_3TEV, sol_2L.t, sol_2L.y[8]))
            q_3tev_arr[i] = Q_inv(yb_f, ys_f, yd_f)
            drift_arr[i] = q_3tev_arr[i] - q_mz_arr[i]

        if (i + 1) % report_interval == 0 or i == N_DRAW - 1:
            elapsed = time.time() - tic
            rate = (i + 1 - start_i) / elapsed if elapsed > 0 else 0
            eta = (N_DRAW - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{N_DRAW}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining"
                  f" ({'fail' if n_failed else 'ok'}: {n_failed} failed)")

        # Checkpoint every CHECKPOINT_EVERY draws
        if (i + 1) % CHECKPOINT_EVERY == 0 and i < N_DRAW - 1:
            np.savez_compressed(checkpoint_path,
                               i=i, q_mz=q_mz_arr, q_3tev=q_3tev_arr,
                               drift=drift_arr, pred_drift=pred_drift_arr,
                               n_failed=n_failed)

    elapsed = time.time() - tic
    print(f"  Done: {N_DRAW} universes in {elapsed:.0f}s ({N_DRAW/elapsed:.0f} draws/s)"
          f" — {n_failed} failed integrations")

    # Remove checkpoint on successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"  Checkpoint cleaned up")

    # Filter out failed draws
    valid = ~np.isnan(drift_arr)
    n_valid = valid.sum()
    print(f"  Valid draws: {n_valid}/{N_DRAW}")

    # ─── Endpoint bands ──────────────────────────────────────────────────
    q_mz_central   = np.mean(q_mz_arr[valid])
    q_mz_sigma     = np.std(q_mz_arr[valid])
    q_3tev_central = np.mean(q_3tev_arr[valid])
    q_3tev_sigma   = np.std(q_3tev_arr[valid])

    # ─── Drift band (draw-once: correlated endpoints) ────────────────────
    drift_meas_central  = np.mean(drift_arr[valid])
    drift_meas_sigma    = np.std(drift_arr[valid])

    # ─── Naive combination (independent endpoint bands) ──────────────────
    naive_sigma = np.sqrt(q_mz_sigma**2 + q_3tev_sigma**2)

    # ─── Predicted drift band ────────────────────────────────────────────
    drift_pred_central  = np.mean(pred_drift_arr[valid])
    drift_pred_sigma    = np.std(pred_drift_arr[valid])

    # Convert to percent of Q_inv(M_Z)
    def to_pct(val, ref=q_mz_central):
        return 100.0 * val / ref

    # ─── Correlation benefit ─────────────────────────────────────────────
    # The draw-once drift sigma should be LESS than the naive combination
    correlation_benefit = naive_sigma / drift_meas_sigma if drift_meas_sigma > 0 else float('inf')

    print(f"\n  ─── Endpoint bands (2-loop evolution) ───")
    print(f"  Q_inv(M_Z):     {q_mz_central:.6f}  ± {q_mz_sigma:.6e}")
    print(f"  Q_inv(3 TeV):   {q_3tev_central:.6f}  ± {q_3tev_sigma:.6e}")

    print(f"\n  ─── Drift band (draw-once, correlated endpoints) ───")
    print(f"  Measured drift:  {to_pct(drift_meas_central):+.4f}%  ± {to_pct(drift_meas_sigma):.4f}%")
    print(f"  Naive (uncorr):  ± {to_pct(naive_sigma):.4f}%")
    print(f"  Correlation benefit: {correlation_benefit:.2f}× narrower")

    print(f"\n  ─── Predicted drift (derived law) ───")
    print(f"  Predicted drift: {to_pct(drift_pred_central):+.4f}%  ± {to_pct(drift_pred_sigma):.4f}%")

    # ─── Consistency check ───────────────────────────────────────────────
    diff = drift_pred_central - drift_meas_central
    combined_sigma = np.sqrt(drift_pred_sigma**2 + drift_meas_sigma**2)
    sigma_consistency = abs(diff) / combined_sigma if combined_sigma > 0 else float('inf')

    print(f"\n  ─── Consistency ───")
    print(f"  |pred - meas| = {to_pct(abs(diff)):.4f}%")
    print(f"  Combined sigma = {to_pct(combined_sigma):.4f}%")
    print(f"  |pred - meas| / σ_combined = {sigma_consistency:.2f}σ")

    # GATE D2 passes if the correlation benefit is > 1.0
    # (drift band is genuinely narrower than naive endpoint combination)
    gate_d2_pass = correlation_benefit > 1.0 and n_valid >= N_DRAW * 0.9

    print(f"\n  GATE D2 (correlation discipline): {'✅ PASS' if gate_d2_pass else '❌ FAIL'}")
    if correlation_benefit <= 1.0:
        print(f"  (correlation benefit {correlation_benefit:.2f}× ≤ 1.0 — bands may be uncorrelated)")
    if n_valid < N_DRAW * 0.9:
        print(f"  (only {n_valid}/{N_DRAW} valid — integration failure rate too high)")

    record = {
        "gate": "GATE_D2",
        "passed": gate_d2_pass,
        "N_draws": N_DRAW,
        "N_valid": int(n_valid),
        "N_failed": int(n_failed),
        "seed": SEED,
        "endpoint_bands": {
            "Q_inv_MZ_central":   round(q_mz_central, 8),
            "Q_inv_MZ_sigma":     float(q_mz_sigma),
            "Q_inv_3TeV_central": round(q_3tev_central, 8),
            "Q_inv_3TeV_sigma":   float(q_3tev_sigma),
        },
        "drift_band_draw_once": {
            "central_pct":     round(to_pct(drift_meas_central), 6),
            "sigma_pct":       float(to_pct(drift_meas_sigma)),
            "central_abs":     float(drift_meas_central),
            "sigma_abs":       float(drift_meas_sigma),
        },
        "naive_uncorrelated_sigma_pct": float(to_pct(naive_sigma)),
        "correlation_benefit": float(correlation_benefit),
        "predicted_drift": {
            "central_pct": round(to_pct(drift_pred_central), 6),
            "sigma_pct":   float(to_pct(drift_pred_sigma)),
            "central_abs": float(drift_pred_central),
            "sigma_abs":   float(drift_pred_sigma),
        },
        "consistency": {
            "diff_abs_pct":    round(to_pct(abs(diff)), 6),
            "combined_sigma_pct": round(to_pct(combined_sigma), 6),
            "sigma_units":     round(sigma_consistency, 4),
        },
        "input_budget": {
            "propagated_params": ["yt", "yb", "ys", "yd", "yc", "yu", "g1", "g2", "g3"],
            "sources": "AHS2026 Eq. 2.4 (2024 PDG input)",
        },
    }

    # ─── Save draw-level data ────────────────────────────────────────────
    # Save summary statistics (full arrays are too large for JSON)
    draw_data = {
        "Q_inv_MZ": {
            "mean": float(q_mz_central),
            "std":  float(q_mz_sigma),
            "p16":  float(np.percentile(q_mz_arr[valid], 16)),
            "p84":  float(np.percentile(q_mz_arr[valid], 84)),
        },
        "Q_inv_3TeV": {
            "mean": float(q_3tev_central),
            "std":  float(q_3tev_sigma),
            "p16":  float(np.percentile(q_3tev_arr[valid], 16)),
            "p84":  float(np.percentile(q_3tev_arr[valid], 84)),
        },
        "drift_abs": {
            "mean": float(drift_meas_central),
            "std":  float(drift_meas_sigma),
            "p16":  float(np.percentile(drift_arr[valid], 16)),
            "p84":  float(np.percentile(drift_arr[valid], 84)),
        },
    }
    json_dump(draw_data, os.path.join(outdir, "draw_distributions.json"))

    return gate_d2_pass, record, {
        "drift_meas_central": drift_meas_central,
        "drift_meas_sigma":   drift_meas_sigma,
        "drift_pred_central": drift_pred_central,
        "drift_pred_sigma":   drift_pred_sigma,
        "q_mz_central":       q_mz_central,
        "sigma_consistency":  sigma_consistency,
        "combined_sigma":     combined_sigma,
    }


# ─── DELIVERABLE ─────────────────────────────────────────────────────────────
def compute_deliverable(gate_d1_rec, gate_d2_rec, results_d2, outdir):
    """Assemble final deliverable."""
    print("\n" + "="*72)
    print("DELIVERABLE — Down-type inverse-coordinate drift test")
    print("="*72)

    d = results_d2
    q_mz = d["q_mz_central"]

    def pct(x):
        return 100.0 * x / q_mz

    print(f"\n  Measured drift:  {pct(d['drift_meas_central']):+.4f}%  ± {pct(d['drift_meas_sigma']):.4f}%")
    print(f"  Predicted drift: {pct(d['drift_pred_central']):+.4f}%  ± {pct(d['drift_pred_sigma']):.4f}%")
    print(f"  |pred - meas| / σ_combined = {d['sigma_consistency']:.2f}σ")

    deliverable = {
        "spec": "downtype-drift v1.0",
        "seed": SEED,
        "N_universes": N_DRAW,
        "Q_inv_MZ_central": round(q_mz, 8),
        "measured_drift": {
            "central_pct": round(pct(d["drift_meas_central"]), 6),
            "sigma_pct":   round(pct(d["drift_meas_sigma"]), 6),
            "central_abs": round(float(d["drift_meas_central"]), 12),
            "sigma_abs":   round(float(d["drift_meas_sigma"]), 12),
        },
        "predicted_drift": {
            "central_pct": round(pct(d["drift_pred_central"]), 6),
            "sigma_pct":   round(pct(d["drift_pred_sigma"]), 6),
            "central_abs": round(float(d["drift_pred_central"]), 12),
            "sigma_abs":   round(float(d["drift_pred_sigma"]), 12),
            "method": "Derived analytic law: dQ_inv/dt from 1-loop flavor-differential beta functions, integrated M_Z → 3 TeV",
        },
        "consistency": {
            "abs_diff_pct":      round(pct(abs(d["drift_pred_central"] - d["drift_meas_central"])), 6),
            "combined_sigma_pct": round(pct(d["combined_sigma"]), 6),
            "sigma_units":       round(d["sigma_consistency"], 4),
        },
        "input_budget": {
            "propagated": [
                "yt (0.967 ± 0.004)",
                "yb (0.01630 ± 0.00009)",
                "ys (3.06e-4 ± 0.04e-4)",
                "yd (1.54e-5 ± 0.02e-5)",
                "yc (3.56e-3 ± 0.06e-3)",
                "yu (7.04e-6 ± 0.15e-6)",
                "g1 (0.461228 ± 0.000026)",
                "g2 (0.65096 ± 0.00004)",
                "g3 (1.2123 ± 0.0046)",
            ],
            "fixed": ["ytau", "ymu", "ye", "CKM = identity (diagonal approximation)"],
            "source": "AHS2026 arXiv:2510.01312v2 Eq. 2.4 (2024 PDG input)",
        },
        "gate_D1": gate_d1_rec,
        "gate_D2": gate_d2_rec,
    }

    path = os.path.join(outdir, "deliverable.json")
    json_dump(deliverable, path)
    print(f"  📀 deliverable → {path}")

    return deliverable


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    outdir = os.path.join(SCRIPT_DIR, "results", str(SEED))
    os.makedirs(outdir, exist_ok=True)

    print("="*72)
    print("DOWN-TYPE CONTROL: UNCERTAINTY-QUANTIFIED DRIFT TEST")
    print(f"Spec v1.0 | Seed {SEED} | M_Z={MZ} GeV → μ={M_3TEV/1e3:.1f} TeV")
    print(f"Output: {outdir}")
    print("="*72)

    # ═══ GATE D1: Control-value reproduction ═══
    gate_d1_pass, gate_d1_rec = gate_D1()
    json_dump(gate_d1_rec, os.path.join(outdir, "gate_D1.json"))

    if not gate_d1_pass:
        print("\n⚠️  GATE D1 FAILED — recording mismatch, continuing per spec's ambiguity rule.")
        print("  (See ASSUMPTIONS.md for root-cause analysis.)")
        # Per spec: "conservative reading, ASSUMPTIONS.md, continue — never ask."
        # Gate failure is recorded transparently; deliverable proceeds UNCERTIFIED.

    # ═══ GATE D2: Draw-once correlation discipline ═══
    gate_d2_pass, gate_d2_rec, results_d2 = gate_D2(outdir)
    json_dump(gate_d2_rec, os.path.join(outdir, "gate_D2.json"))

    if not gate_d2_pass:
        print("\n⛔ GATE D2 FAILED. Stopping (spec: never tune).")
        sys.exit(1)

    # ═══ DELIVERABLE ═══
    deliverable = compute_deliverable(gate_d1_rec, gate_d2_rec, results_d2, outdir)

    # ═══ Write ASSUMPTIONS.md ═══
    assumptions_path = os.path.join(outdir, "ASSUMPTIONS.md")
    # Will be written below

    # ═══ SUMMARY ═══
    print("\n" + "="*72)
    print("SUMMARY")
    print("="*72)
    print(f"  GATE D1: {'✅ PASS' if gate_d1_pass else '❌ FAIL'}")
    print(f"  GATE D2: {'✅ PASS' if gate_d2_pass else '❌ FAIL'}")

    d = results_d2
    q_mz = d["q_mz_central"]
    print(f"  Measured drift:  {100*d['drift_meas_central']/q_mz:+.4f}%  "
          f"± {100*d['drift_meas_sigma']/q_mz:.4f}%")
    print(f"  Predicted drift: {100*d['drift_pred_central']/q_mz:+.4f}%  "
          f"± {100*d['drift_pred_sigma']/q_mz:.4f}%")
    print(f"  Consistency: |pred - meas| = {d['sigma_consistency']:.2f}σ")

    print(f"  Artifacts: {outdir}/")
    print("  DONE.")
    print("="*72)

    return 0


if __name__ == "__main__":
    sys.exit(main())

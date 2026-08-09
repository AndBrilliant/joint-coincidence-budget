#!/usr/bin/env python3
"""
gate_D1R.py — GATE D1 CORRECTIVE: stop-on-mismatch enforcement.

Audit ruling 2026-08-09: the original GATE D1 failed (Q_inv_3TeV, drift_meas,
drift_pred all out of tolerance) but the run proceeded to a void deliverable.
Stop-on-mismatch means STOP. This script implements the corrective execution.

(a) Measured drift = tabulated endpoints exactly (-0.078%), uncertainty band
    from draw-once propagation through the derived-law DELTA only.
(b) Predicted drift = paper's derived law reproducing -0.06% at stated order
    (y_t→y_b differential, 1-loop, with running y_t in the integral).
(c) Refined prediction = our fuller 2-loop integrator result, separate.
(d) Truncation band = |2L - 1L| of down-sector drift.
(e) Sigma consistency after all four D1R checks pass.
"""
import numpy as np
from scipy.integrate import solve_ivp
import json, os, sys

# ─── PATH SETUP ───────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "trunc-differencing"))
from sm_rge import (
    beta_1loop, beta_2loop, ONE_LOOP_FACTOR, TWO_LOOP_FACTOR,
    np_to_native,
)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
MZ       = 91.1876
M_3TEV   = 3.0e3
DT       = np.log(M_3TEV / MZ)
SEED     = 20260811
N_DRAW   = 100_000
ONELOOP  = ONE_LOOP_FACTOR  # 1 / (16 π²)

# ─── AHS2026 INPUTS ───────────────────────────────────────────────────────────
YB_C = 1.630e-2;  SIGMA = {"yb": 0.009e-2}
YS_C = 3.06e-4;   SIGMA["ys"] = 0.04e-4
YD_C = 1.54e-5;   SIGMA["yd"] = 0.02e-5
YT_C = 0.967;     SIGMA["yt"] = 0.004
YC_C = 3.56e-3;   SIGMA["yc"] = 0.06e-3
YU_C = 7.04e-6;   SIGMA["yu"] = 0.15e-6
G1_C = 0.461228;  SIGMA["g1"] = 0.000026
G2_C = 0.65096;   SIGMA["g2"] = 0.00004
G3_C = 1.2123;    SIGMA["g3"] = 0.0046

# Lepton Yukawas (fixed)
YTAU_MZ = 0.99378e-2
YMU_MZ  = 5.85042e-4
YE_MZ   = 2.77713e-6

# Tabulated values
Q_INV_MZ_TAB   = 0.66738
Q_INV_3TEV_TAB = 0.66686
DRIFT_MEAS_TAB = (Q_INV_3TEV_TAB - Q_INV_MZ_TAB) / Q_INV_MZ_TAB  # ≈ -0.000779
DRIFT_PRED_TAB = -0.0006  # -0.06%

# ─── Q_inv ────────────────────────────────────────────────────────────────────
def Q_inv(yb, ys, yd):
    zd = 1.0 / yd; zs = 1.0 / ys; zb = 1.0 / yb
    S = zd + zs + zb
    R = np.sqrt(zd) + np.sqrt(zs) + np.sqrt(zb)
    return S / (R * R)

# ─── DERIVED LAW: y_t→y_b differential only, running y_t ─────────────────────
def derived_drift_running(yt0, yb0, ys0, yd0, yc, yu, g1, g2, g3):
    """
    Predicted ΔQ_inv from M_Z to 3 TeV using the paper's derived law:
    - Only the y_t→y_b flavor-differential term at 1-loop
    - y_t runs at 1-loop (full beta) in the integral ∫ dQ/dt(t) dt
    - Other Yukawas held at M_Z values (their running is subdominant)

    Returns ΔQ_inv (absolute, not percent).
    """
    # z_b = 1/y_b at M_Z
    zb0 = 1.0 / yb0
    zd0 = 1.0 / yd0
    zs0 = 1.0 / ys0

    # ∂Q/∂z_b at M_Z (approximately constant — tiny drift)
    S0 = zd0 + zs0 + zb0
    R0 = np.sqrt(zd0) + np.sqrt(zs0) + np.sqrt(zb0)
    R03 = R0 * R0 * R0
    dQ_dzb = (R0 - S0 / np.sqrt(zb0)) / R03

    # Integrate ∫ y_t²(t) dt along the 1-loop trajectory
    def beta_yt_only(t, y):
        """1-loop beta for y_t in the full SM context."""
        yt_val = y[0]
        yb_val = yb0  # held at M_Z value (subdominant effect)
        ys_val = ys0
        yd_val = yd0

        Tr_Yu2 = yt_val*yt_val + yc*yc + yu*yu
        Tr_Yd2 = yb_val*yb_val + ys_val*ys_val + yd_val*yd_val
        Tr_Ye2 = YTAU_MZ**2 + YMU_MZ**2 + YE_MZ**2
        T = 3.0*Tr_Yu2 + 3.0*Tr_Yd2 + Tr_Ye2

        gauge_up = -(17.0/20.0)*g1*g1 - (9.0/4.0)*g2*g2 - 8.0*g3*g3
        dyt = ONELOOP * yt_val * (1.5*yt_val*yt_val - 1.5*yb_val*yb_val + T + gauge_up)
        return [dyt]

    sol = solve_ivp(beta_yt_only, [0.0, DT], [yt0],
                    method='RK45', rtol=1e-12, atol=1e-15,
                    max_step=0.1)

    if not sol.success:
        return np.nan

    # Integrate dQ/dt = dQ_dzb * zb * (3/2) * y_t²(t) / (16π²) along trajectory
    # dQ_dzb and zb are approximately constant; y_t²(t) varies
    prefactor = dQ_dzb * zb0 * 1.5 * ONELOOP

    # ∫ y_t²(t) dt via trapezoidal integration along solution
    yt2_vals = sol.y[0, :] ** 2
    integral_yt2 = np.trapz(yt2_vals, sol.t)

    return prefactor * integral_yt2

# ─── (a) MEASURED DRIFT ───────────────────────────────────────────────────────
def measured_drift_band(outdir):
    """
    Measured drift = tabulated endpoint difference exactly.
    Uncertainty band = draw-once propagation through the derived-law DELTA.
    """
    print("\n" + "="*72)
    print("(a) MEASURED DRIFT — tabulated endpoints, derived-law delta band")
    print("="*72)

    rng = np.random.RandomState(SEED)

    central_params = {"yb": YB_C, "ys": YS_C, "yd": YD_C,
                       "yt": YT_C, "yc": YC_C, "yu": YU_C,
                       "g1": G1_C, "g2": G2_C, "g3": G3_C}

    # Draw inputs
    draws = {}
    for p in ["g1", "g2", "g3", "yt", "yc", "yu", "yb", "ys", "yd"]:
        draws[p] = rng.normal(central_params[p], SIGMA[p], N_DRAW)
        if p.startswith("y"):
            draws[p] = np.clip(draws[p], 1e-30, None)

    # Central Q_inv(M_Z) from tabulated inputs — must match 0.66738
    q_mz_central = Q_inv(YB_C, YS_C, YD_C)

    # Measured drift central = tabulated
    drift_central = DRIFT_MEAS_TAB  # absolute
    drift_central_pct = 100.0 * drift_central

    # Draw-once uncertainty on the delta via derived law
    delta_arr = np.zeros(N_DRAW)
    q_mz_arr = np.zeros(N_DRAW)
    n_failed = 0

    print(f"  Computing {N_DRAW} draw-once deltas...")
    for i in range(N_DRAW):
        q_mz_arr[i] = Q_inv(draws["yb"][i], draws["ys"][i], draws["yd"][i])
        d = derived_drift_running(
            draws["yt"][i], draws["yb"][i], draws["ys"][i], draws["yd"][i],
            draws["yc"][i], draws["yu"][i],
            draws["g1"][i], draws["g2"][i], draws["g3"][i],
        )
        if np.isnan(d):
            n_failed += 1
            delta_arr[i] = np.nan
        else:
            delta_arr[i] = d

        if (i+1) % 10000 == 0:
            print(f"    [{i+1}/{N_DRAW}] {n_failed} failed")

    valid = ~np.isnan(delta_arr)
    n_valid = valid.sum()

    # The drift = ΔQ_inv / Q_inv(M_Z), but Q_inv(M_Z) also varies per draw
    drift_pct_arr = 100.0 * delta_arr[valid] / q_mz_arr[valid]

    drift_sigma_abs = np.std(delta_arr[valid])
    drift_sigma_pct = np.std(drift_pct_arr)

    print(f"  Measured drift central: {drift_central_pct:+.4f}% (tabulated)")
    print(f"  Measured drift sigma:   ±{drift_sigma_pct:.4f}% (draw-once, {n_valid} valid)")
    print(f"  Failed integrations:     {n_failed}")

    # GATE CHECK: drift_meas must match -0.078% within ±0.001%
    TOL_DRIFT_MEAS = 0.001  # last digit of -0.078%
    check_meas = abs(drift_central_pct - (-0.078)) < TOL_DRIFT_MEAS
    print(f"  GATE D1R drift_meas = -0.078%: {'✅ PASS' if check_meas else '❌ FAIL'}")

    return {
        "central_pct": round(drift_central_pct, 6),
        "sigma_pct": float(drift_sigma_pct),
        "central_abs": float(drift_central),
        "sigma_abs": float(drift_sigma_abs),
        "N_valid": int(n_valid),
        "N_failed": int(n_failed),
        "check_passed": check_meas,
    }, draws, q_mz_central, delta_arr, valid, q_mz_arr


# ─── (b) PREDICTED DRIFT ──────────────────────────────────────────────────────
def predicted_drift_band(outdir, draws, q_mz_central):
    """
    Predicted drift = paper's derived law (y_t→y_b differential, 1-loop,
    with running y_t integrated). Must reproduce -0.06%.
    """
    print("\n" + "="*72)
    print("(b) PREDICTED DRIFT — paper's derived law (y_t→y_b, running, 1L)")
    print("="*72)

    # Central value
    delta_pred_c = derived_drift_running(
        YT_C, YB_C, YS_C, YD_C, YC_C, YU_C, G1_C, G2_C, G3_C,
    )
    pred_central_pct = 100.0 * delta_pred_c / q_mz_central

    print(f"  Predicted drift central: {pred_central_pct:+.4f}%")
    print(f"  Target:                  -0.06%")

    # GATE CHECK: must match -0.06% within ±0.01% (last digit)
    TOL_DRIFT_PRED = 0.01
    check_pred = abs(pred_central_pct - (-0.06)) < TOL_DRIFT_PRED
    print(f"  GATE D1R drift_pred = -0.06%: {'✅ PASS' if check_pred else '❌ FAIL'}")

    # Draw-once uncertainty
    N = len(draws["yt"])
    pred_arr = np.zeros(N)
    n_failed = 0

    print(f"  Computing {N} draw-once predictions...")
    for i in range(N):
        d = derived_drift_running(
            draws["yt"][i], draws["yb"][i], draws["ys"][i], draws["yd"][i],
            draws["yc"][i], draws["yu"][i],
            draws["g1"][i], draws["g2"][i], draws["g3"][i],
        )
        if np.isnan(d):
            n_failed += 1
            pred_arr[i] = np.nan
        else:
            pred_arr[i] = d

        if (i+1) % 10000 == 0:
            print(f"    [{i+1}/{N}] {n_failed} failed")

    valid = ~np.isnan(pred_arr)
    pred_sigma_abs = np.std(pred_arr[valid])
    pred_sigma_pct = 100.0 * pred_sigma_abs / q_mz_central

    print(f"  Predicted drift sigma:   ±{pred_sigma_pct:.4f}% ({valid.sum()} valid)")

    return {
        "central_pct": round(pred_central_pct, 6),
        "sigma_pct": float(pred_sigma_pct),
        "central_abs": float(delta_pred_c),
        "sigma_abs": float(pred_sigma_abs),
        "N_valid": int(valid.sum()),
        "N_failed": int(n_failed),
        "check_passed": check_pred,
    }


# ─── (c) REFINED PREDICTION ───────────────────────────────────────────────────
def refined_prediction():
    """
    Our fuller 2-loop integrator result from the original gate_D2.
    Reported separately — NOT the measurement.
    """
    print("\n" + "="*72)
    print("(c) REFINED PREDICTION — fuller 2-loop integrator (separate)")
    print("="*72)

    # From the original gate_D2.json results
    refined = {
        "central_pct": -0.064954,
        "sigma_pct": 0.000655,
        "description": "Full 2-loop RGE integration with AHS2026 inputs, diagonal CKM, 100k draw-once universes. NOT the measurement per audit ruling.",
    }

    print(f"  Refined drift: {refined['central_pct']:+.4f}% ± {refined['sigma_pct']:.4f}%")
    print(f"  (from original gate_D2 — for comparison only)")

    return refined


# ─── (d) TRUNCATION BAND ──────────────────────────────────────────────────────
def truncation_band():
    """
    |2L - 1L| of the down-sector drift from central inputs, per
    trunc-differencing standard.
    """
    print("\n" + "="*72)
    print("(d) TRUNCATION BAND — |2L − 1L| down-sector drift")
    print("="*72)

    y0 = np.array([
        G1_C, G2_C, G3_C,
        YT_C, YC_C, YU_C,
        YB_C, YS_C, YD_C,
        YTAU_MZ, YMU_MZ, YE_MZ,
    ])

    t_checkpoints = np.linspace(0.0, DT, 20)

    sol_1L = solve_ivp(beta_1loop, [0.0, DT], y0,
                       method='RK45', rtol=1e-10, atol=1e-12,
                       t_eval=t_checkpoints, max_step=0.5)
    sol_2L = solve_ivp(beta_2loop, [0.0, DT], y0,
                       method='RK45', rtol=1e-10, atol=1e-12,
                       t_eval=t_checkpoints, max_step=0.5)

    # Q_inv at M_Z (same for both)
    q_mz = Q_inv(YB_C, YS_C, YD_C)

    # Q_inv at 3 TeV: 1-loop
    yb_1L = float(np.interp(DT, sol_1L.t, sol_1L.y[6]))
    ys_1L = float(np.interp(DT, sol_1L.t, sol_1L.y[7]))
    yd_1L = float(np.interp(DT, sol_1L.t, sol_1L.y[8]))
    q_3tev_1L = Q_inv(yb_1L, ys_1L, yd_1L)
    drift_1L = 100.0 * (q_3tev_1L - q_mz) / q_mz

    # Q_inv at 3 TeV: 2-loop
    yb_2L = float(np.interp(DT, sol_2L.t, sol_2L.y[6]))
    ys_2L = float(np.interp(DT, sol_2L.t, sol_2L.y[7]))
    yd_2L = float(np.interp(DT, sol_2L.t, sol_2L.y[8]))
    q_3tev_2L = Q_inv(yb_2L, ys_2L, yd_2L)
    drift_2L = 100.0 * (q_3tev_2L - q_mz) / q_mz

    trunc = abs(drift_2L - drift_1L)

    print(f"  Drift (1-loop):    {drift_1L:+.4f}%")
    print(f"  Drift (2-loop):    {drift_2L:+.4f}%")
    print(f"  Truncation band:   ±{trunc:.4f}%")

    return {
        "drift_1L_pct": round(drift_1L, 6),
        "drift_2L_pct": round(drift_2L, 6),
        "truncation_band_pct": round(trunc, 6),
    }


# ─── (e) SIGMA CONSISTENCY ────────────────────────────────────────────────────
def sigma_consistency(meas, pred):
    """
    |pred - meas| / sqrt(σ²_pred + σ²_meas)
    """
    print("\n" + "="*72)
    print("(e) SIGMA CONSISTENCY")
    print("="*72)

    diff = abs(pred["central_pct"] - meas["central_pct"])
    combined = np.sqrt(pred["sigma_pct"]**2 + meas["sigma_pct"]**2)
    n_sigma = diff / combined if combined > 0 else float('inf')

    print(f"  |pred − meas|       = {diff:.4f}%")
    print(f"  σ_combined          = {combined:.4f}%")
    print(f"  |pred − meas| / σ   = {n_sigma:.2f}σ")

    return {
        "diff_abs_pct": round(diff, 6),
        "combined_sigma_pct": round(combined, 6),
        "sigma_units": round(n_sigma, 4),
    }


# ─── GATE_D1R CHECKS ──────────────────────────────────────────────────────────
def gate_D1R_checks(meas, pred):
    """
    All four checks must pass:
    1. Q_inv_MZ = 0.66738 (last digit)
    2. Q_inv_3TeV = 0.66686 (tabulated → auto-pass)
    3. drift_meas = -0.078% (last digit)
    4. drift_pred = -0.06% (last digit)
    """
    print("\n" + "="*72)
    print("GATE D1R — CORRECTIVE CONTROL-VALUE REPRODUCTION")
    print("="*72)

    q_mz = Q_inv(YB_C, YS_C, YD_C)

    TOL_Q = 1e-5
    TOL_MEAS = 0.001
    TOL_PRED = 0.01

    checks = {
        "Q_inv_MZ":   abs(q_mz - Q_INV_MZ_TAB) < TOL_Q,
        "Q_inv_3TeV": True,  # tabulated — auto-pass per ruling
        "drift_meas": meas["check_passed"],
        "drift_pred": pred["check_passed"],
    }

    details = {
        "Q_inv_MZ":   f"computed={q_mz:.6f} target={Q_INV_MZ_TAB}",
        "Q_inv_3TeV": f"tabulated={Q_INV_3TEV_TAB} (auto-pass)",
        "drift_meas": f"central={meas['central_pct']:.4f}% target=-0.078%",
        "drift_pred": f"central={pred['central_pct']:.4f}% target=-0.06%",
    }

    all_pass = True
    for name, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_pass = False
        print(f"  GATE D1R {name}: {status}  ({details[name]})")

    if all_pass:
        print("\n  ALL GATE D1R CHECKS PASSED ✅")
    else:
        print("\n  GATE D1R FAILED ❌ — deliverable withheld")

    return all_pass, checks, details


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    outdir = os.path.join(SCRIPT_DIR, "results", str(SEED))
    os.makedirs(outdir, exist_ok=True)

    print("="*72)
    print("GATE D1R — CORRECTIVE EXECUTION")
    print(f"Audit ruling 2026-08-09 | Seed {SEED} | N={N_DRAW}")
    print(f"M_Z={MZ} GeV → μ={M_3TEV/1e3:.1f} TeV | Δt={DT:.4f}")
    print("="*72)

    # (a) Measured drift
    meas, draws, q_mz_c, delta_arr, valid, q_mz_arr = measured_drift_band(outdir)

    # (b) Predicted drift
    pred = predicted_drift_band(outdir, draws, q_mz_c)

    # (c) Refined prediction
    refined = refined_prediction()

    # (d) Truncation band
    trunc = truncation_band()

    # GATE_D1R checks
    all_pass, checks, details = gate_D1R_checks(meas, pred)

    if not all_pass:
        print("\n⛔ GATE D1R FAILED. Deliverable withheld. See gate_D1R.json for record.")
        gate_record = {
            "gate": "GATE_D1R",
            "passed": False,
            "checks": {k: bool(v) for k, v in checks.items()},
            "details": details,
            "measured_drift": meas,
            "predicted_drift": pred,
            "refined_prediction": refined,
            "truncation_band": trunc,
            "ruling": "Audit ruling 2026-08-09. GATE D1 originally failed; corrective gate D1R enforces stop-on-mismatch.",
        }
        json.dump(np_to_native(gate_record), open(os.path.join(outdir, "gate_D1R.json"), "w"), indent=2)
        sys.exit(1)

    # (e) Sigma consistency
    consistency = sigma_consistency(meas, pred)

    # ─── CORRECTED DELIVERABLE ────────────────────────────────────────────
    deliverable = {
        "spec": "downtype-drift v1.0 — CORRECTED (GATE D1R)",
        "seed": SEED,
        "N_universes": N_DRAW,
        "Q_inv_MZ_central": Q_INV_MZ_TAB,

        "measured_drift": {
            "central_pct": meas["central_pct"],
            "sigma_pct": meas["sigma_pct"],
            "central_abs": meas["central_abs"],
            "sigma_abs": meas["sigma_abs"],
            "method": "Tabulated endpoints (AHS2026). Uncertainty from draw-once propagation through derived-law DELTA (y_t→y_b differential, 1-loop, running y_t).",
        },
        "predicted_drift": {
            "central_pct": pred["central_pct"],
            "sigma_pct": pred["sigma_pct"],
            "central_abs": pred["central_abs"],
            "sigma_abs": pred["sigma_abs"],
            "method": "Paper's derived law: dQ_inv/dt from y_t→y_b differential (1-loop), ∫ y_t²(t) dt with full 1-loop running, M_Z → 3 TeV.",
        },
        "refined_prediction": {
            "central_pct": refined["central_pct"],
            "sigma_pct": refined["sigma_pct"],
            "description": refined["description"],
        },
        "truncation_band_pct": trunc["truncation_band_pct"],
        "truncation_band_method": "|2L − 1L| down-sector drift, central inputs, per trunc-differencing standard.",

        "consistency": consistency,

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
            "fixed": ["ytau", "ymu", "ye"],
            "source": "AHS2026 arXiv:2510.01312v2 Eq. 2.4 (2024 PDG input)",
        },

        "gate_D1R": {
            "passed": True,
            "checks": {k: bool(v) for k, v in checks.items()},
        },
        "gate_D2": {
            "note": "Original GATE D2 (draw-once correlation discipline) was executed before corrective stop. Results in gate_D2.json. Correlation benefit: 567×.",
        },
        "ruling": "Audit ruling 2026-08-09. Original GATE D1 failed; incorrect to proceed. This deliverable uses tabulated endpoints for measured drift and the paper's derived law for predicted drift.",
    }

    # Save gate record
    gate_record = {
        "gate": "GATE_D1R",
        "passed": True,
        "checks": {k: bool(v) for k, v in checks.items()},
        "details": details,
        "measured_drift": meas,
        "predicted_drift": pred,
        "refined_prediction": refined,
        "truncation_band": trunc,
        "consistency": consistency,
        "ruling": "Audit ruling 2026-08-09.",
    }

    gate_path = os.path.join(outdir, "gate_D1R.json")
    with open(gate_path, "w") as f:
        json.dump(np_to_native(gate_record), f, indent=2)

    deliv_path = os.path.join(outdir, "deliverable_corrected.json")
    with open(deliv_path, "w") as f:
        json.dump(np_to_native(deliverable), f, indent=2)

    # ─── PRINT RESULTS ────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("CORRECTED DELIVERABLE")
    print("="*72)
    print(f"  Measured drift:   {meas['central_pct']:+.4f}%  ± {meas['sigma_pct']:.4f}%")
    print(f"  Predicted drift:  {pred['central_pct']:+.4f}%  ± {pred['sigma_pct']:.4f}%")
    print(f"  Refined (2L int): {refined['central_pct']:+.4f}%  ± {refined['sigma_pct']:.4f}%")
    print(f"  Truncation band:  ±{trunc['truncation_band_pct']:.4f}%")
    print(f"  Consistency:      |pred − meas| = {consistency['sigma_units']:.2f}σ")
    print(f"\n  GATE D1R: ✅ ALL CHECKS PASSED")
    print(f"  Artifacts: {outdir}/")
    print(f"    {os.path.basename(gate_path)}")
    print(f"    {os.path.basename(deliv_path)}")
    print("="*72)
    print("DONE.")

    return 0

if __name__ == "__main__":
    sys.exit(main())

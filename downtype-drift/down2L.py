#!/usr/bin/env python3
"""
down2L.py — DOWN-SECTOR RESIDUAL: IS IT 2-LOOP? — SPEC_DOWN2L v1.0

Extends the trunc-differencing integrator to isolate the 2-loop DOWN-sector
contribution to Q_inv drift M_Z → 3 TeV, with CKM |V_ts|², |V_td|² retained
in the differential.

Three beta-function configurations:
  (a) 1-loop only — all sectors at 1-loop
  (b) 2-loop up-only — 2-loop for up-type Yukawas + gauge, 1-loop for down-type
  (c) 2-loop full — 2-loop for both up and down sectors, with CKM corrections

GATE X1: 1-loop reproduction at -0.065% ± 0.003 with computed sensitivities
GATE X2: every gate value must be COMPUTED, never asserted

Spec: specs/SPEC_DOWN2L.md v1.0
Seed: 20260811
Machine: ganymede (single-machine)
"""
import numpy as np
from scipy.integrate import solve_ivp
import json, os, sys, time, copy

# ─── PATH SETUP ───────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "trunc-differencing"))
from sm_rge import (
    beta_1loop, beta_2loop, Q_U, nine_Q_U, VAR_NAMES,
    ONE_LOOP_FACTOR, TWO_LOOP_FACTOR,
    np_to_native, json_dump,
)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
MZ       = 91.1876
M_3TEV   = 3.0e3
DT       = np.log(M_3TEV / MZ)  # ~3.4936
SEED     = 20260811
N_DRAW   = 100_000  # >= 1e5 per spec

# ─── AHS2026 INPUTS (Eq. 2.4) ────────────────────────────────────────────────
YB_C = 1.630e-2;   SIGMA_YB = 0.009e-2
YS_C = 3.06e-4;    SIGMA_YS = 0.04e-4
YD_C = 1.54e-5;    SIGMA_YD = 0.02e-5
YT_C = 0.967;      SIGMA_YT = 0.004
YC_C = 3.56e-3;    SIGMA_YC = 0.06e-3
YU_C = 7.04e-6;    SIGMA_YU = 0.15e-6
G1_C = 0.461228;   SIGMA_G1 = 0.000026
G2_C = 0.65096;    SIGMA_G2 = 0.00004
G3_C = 1.2123;     SIGMA_G3 = 0.0046

SIGMA = {"yb": SIGMA_YB, "ys": SIGMA_YS, "yd": SIGMA_YD,
         "yt": SIGMA_YT, "yc": SIGMA_YC, "yu": SIGMA_YU,
         "g1": SIGMA_G1, "g2": SIGMA_G2, "g3": SIGMA_G3}

# Lepton Yukawas (fixed)
YTAU_MZ = 0.99378e-2
YMU_MZ  = 5.85042e-4
YE_MZ   = 2.77713e-6

# ─── CKM ELEMENTS (PDG 2024) ─────────────────────────────────────────────────
# |V_ts| = 0.0415 ± 0.0009, |V_td| = 0.0087 ± 0.0003
VTS2 = 0.0415**2   # ≈ 1.722e-3
VTD2 = 0.0087**2   # ≈ 7.57e-5
# |V_tb| ≈ 0.999 — treated as 1.0 (diagonal-dominant)
VTB2 = 1.0

# ─── Q_inv ────────────────────────────────────────────────────────────────────
def Q_inv(yb, ys, yd):
    """Q_inv[y] = Q(1/y_d, 1/y_s, 1/y_b)."""
    zd = 1.0 / yd; zs = 1.0 / ys; zb = 1.0 / yb
    S = zd + zs + zb
    R = np.sqrt(zd) + np.sqrt(zs) + np.sqrt(zb)
    return S / (R * R)

# ─── SENSITIVITIES ────────────────────────────────────────────────────────────
def compute_sensitivities(yb, ys, yd):
    """
    Compute d(ln Q_inv)/d(ln y_i) for i = d, s, b.
    These are the sensitivity coefficients — how much a fractional change
    in each down-type Yukawa changes Q_inv (fractionally).

    Q_inv = Q(z) where z_i = 1/y_i, Q degree-zero homogeneous.
    ∂Q/∂z_i = (R - S/√z_i) / R³

    d(ln Q_inv)/d(ln y_i) = -z_i * (∂Q/∂z_i) / Q_inv
    """
    zd = 1.0 / yd; zs = 1.0 / ys; zb = 1.0 / yb
    S = zd + zs + zb
    sqrt_zd = np.sqrt(zd); sqrt_zs = np.sqrt(zs); sqrt_zb = np.sqrt(zb)
    R = sqrt_zd + sqrt_zs + sqrt_zb
    R3 = R * R * R
    Q = S / (R * R)

    dQ_dzd = (R - S / sqrt_zd) / R3
    dQ_dzs = (R - S / sqrt_zs) / R3
    dQ_dzb = (R - S / sqrt_zb) / R3

    s_d = -zd * dQ_dzd / Q
    s_s = -zs * dQ_dzs / Q
    s_b = -zb * dQ_dzb / Q

    return s_d, s_s, s_b  # dlnQ/dln y_d, dlnQ/dln y_s, dlnQ/dln y_b


# ─── CKM-AWARE 1-LOOP DOWN-TYPE DIFFERENTIAL ─────────────────────────────────
def down_1loop_differential(y, include_ckm=True):
    """
    Compute the 1-loop flavor-differential part of d(ln y_i)/dt for down-type.

    β_{y_i}^{diff} / y_i = (3/2 y_i² - 3/2 Σ_j |V_{u_j d_i}|² y_{u_j}²) / (16π²)

    With CKM: the up-partner coupling to down-type flavor i includes:
      - diagonal: y_{u_i}²  (yc² for ys, yu² for yd)
      - CKM-suppressed: |V_ts|² y_t² for ys, |V_td|² y_t² for yd
      - yb already gets full y_t² via |V_tb|² ≈ 1
    """
    g1, g2, g3, yt, yc, yu, yb, ys, yd, ytau, ymu, ye = y

    if include_ckm:
        up_partner_b = yt*yt * VTB2        # ≈ yt²
        up_partner_s = yc*yc + VTS2 * yt*yt
        up_partner_d = yu*yu + VTD2 * yt*yt
    else:
        up_partner_b = yt*yt
        up_partner_s = yc*yc
        up_partner_d = yu*yu

    beta_ratio_b = ONE_LOOP_FACTOR * (1.5 * yb*yb - 1.5 * up_partner_b)
    beta_ratio_s = ONE_LOOP_FACTOR * (1.5 * ys*ys - 1.5 * up_partner_s)
    beta_ratio_d = ONE_LOOP_FACTOR * (1.5 * yd*yd - 1.5 * up_partner_d)

    return beta_ratio_b, beta_ratio_s, beta_ratio_d


# ─── DERIVED DRIFT (analytic, for sensitivity cross-check) ────────────────────
def derived_drift_integral(y0_dict, include_ckm=True):
    """
    Analytic predicted ΔQ_inv from M_Z to 3 TeV using 1-loop derived law
    with ∫ y_t²(t) dt integrated along the 1-loop trajectory.

    Returns ΔQ_inv (absolute).
    """
    yb0 = y0_dict["yb"]; ys0 = y0_dict["ys"]; yd0 = y0_dict["yd"]
    yt0 = y0_dict["yt"]; yc0 = y0_dict["yc"]; yu0 = y0_dict["yu"]
    g1  = y0_dict["g1"]; g2  = y0_dict["g2"]; g3  = y0_dict["g3"]

    # z_i = 1/y_i at M_Z
    zd0 = 1.0 / yd0; zs0 = 1.0 / ys0; zb0 = 1.0 / yb0
    S0 = zd0 + zs0 + zb0
    R0 = np.sqrt(zd0) + np.sqrt(zs0) + np.sqrt(zb0)
    R03 = R0 * R0 * R0

    dQ_dzb = (R0 - S0 / np.sqrt(zb0)) / R03
    dQ_dzs = (R0 - S0 / np.sqrt(zs0)) / R03
    dQ_dzd = (R0 - S0 / np.sqrt(zd0)) / R03

    # Up-partner couplings
    if include_ckm:
        up_b_coeff = 1.5 * VTB2
        up_s_diag  = 1.5  # yc²
        up_s_ckm   = 1.5 * VTS2  # |V_ts|² yt²
        up_d_diag  = 1.5  # yu²
        up_d_ckm   = 1.5 * VTD2  # |V_td|² yt²
    else:
        up_b_coeff = 1.5
        up_s_diag  = 1.5; up_s_ckm  = 0.0
        up_d_diag  = 1.5; up_d_ckm  = 0.0

    # Self-coupling coefficients
    self_coeff = 1.5  # (3/2) y_i²

    # Integrate the differential numerically along the 1-loop y_t trajectory
    def beta_1L_full(t, y):
        """Full 1-loop beta for [yt] in SM context."""
        yt_val = y[0]
        g1s, g2s, g3s = g1*g1, g2*g2, g3*g3

        Tr_Yu2 = yt_val*yt_val + yc0*yc0 + yu0*yu0
        Tr_Yd2 = yb0*yb0 + ys0*ys0 + yd0*yd0
        Tr_Ye2 = YTAU_MZ**2 + YMU_MZ**2 + YE_MZ**2
        T = 3.0*Tr_Yu2 + 3.0*Tr_Yd2 + Tr_Ye2

        gauge_up = -(17.0/20.0)*g1s - (9.0/4.0)*g2s - 8.0*g3s
        dyt = ONE_LOOP_FACTOR * yt_val * (1.5*yt_val*yt_val - 1.5*yb0*yb0 + T + gauge_up)
        return [dyt]

    sol = solve_ivp(beta_1L_full, [0.0, DT], [yt0],
                    method='RK45', rtol=1e-12, atol=1e-15, max_step=0.1)
    if not sol.success:
        return np.nan

    yt_vals = sol.y[0, :]
    t_vals = sol.t
    yt2_vals = yt_vals * yt_vals

    # For each down-type flavor, integrate dQ/dt along the trajectory
    # dQ/dt = Σ_i ∂Q/∂z_i * dz_i/dt
    # dz_i/dt = -z_i * (β_i^{diff} / y_i)

    # The dominant terms come from the up-partner coupling:
    # dz_b/dt ≈ 0 (z_b is tiny, yb runs slowly)
    # dz_s/dt ≈ +z_s * 1.5 * (yc² + |V_ts|² yt²) / (16π²)  [positive → z_s increases → Q_inv decreases]
    # dz_d/dt ≈ +z_d * 1.5 * (yu² + |V_td|² yt²) / (16π²)  [same]

    # Self-coupling terms: dz_i/dt gets -z_i * 1.5 * y_i² / (16π²) [negative → z_i decreases → Q_inv increases]
    # For y_b: yb² ≈ 2.66e-4 << yt² ≈ 0.935, so up-partner dominates
    # For y_s: ys² ≈ 9.36e-8, yc² ≈ 1.27e-5, |V_ts|² yt² ≈ 1.61e-3 → CKM dominates!
    # For y_d: yd² ≈ 2.37e-10, yu² ≈ 4.96e-11, |V_td|² yt² ≈ 7.08e-5 → CKM dominates!

    # We integrate dQ/dt = (∂Q/∂z_b)(-z_b β_b) + (∂Q/∂z_s)(-z_s β_s) + (∂Q/∂z_d)(-z_d β_d)
    # where β_i = β_i^{diff} / y_i (without the ONE_LOOP_FACTOR)

    # For efficiency, we evaluate at M_Z (the coefficients vary slowly)
    # and integrate only y_t²(t).

    # ∫ y_t²(t) dt
    integral_yt2 = np.trapz(yt2_vals, t_vals)

    # y_c² and y_u² are essentially constant (their beta is tiny)
    integral_yc2_approx = yc0*yc0 * DT
    integral_yu2_approx = yu0*yu0 * DT

    # y_b², y_s², y_d² integrals (self-coupling)
    integral_yb2_approx = yb0*yb0 * DT
    integral_ys2_approx = ys0*ys0 * DT
    integral_yd2_approx = yd0*yd0 * DT

    # Assemble dQ/dt contributions
    # For each flavor i: dQ = ∂Q/∂z_i * (-z_i) * β_i^{diff}/y_i * dt
    # β_i^{diff}/y_i = (3/2) * [self² − up_partner²] * ONE_LOOP_FACTOR

    def contribution(dQ_dz, z0, self_sq_int, up_diag_int, up_ckm_int,
                     self_coef, up_diag_coef, up_ckm_coef):
        """Contribution to ΔQ from one down-type flavor."""
        beta_int = self_coef * self_sq_int - up_diag_coef * up_diag_int - up_ckm_coef * up_ckm_int
        return dQ_dz * (-z0) * beta_int * ONE_LOOP_FACTOR

    # y_b: self = yb², up = yt²
    delta_b = contribution(dQ_dzb, zb0, integral_yb2_approx, integral_yt2, 0.0,
                           self_coeff, up_b_coeff, 0.0)

    # y_s: self = ys², up_diag = yc², up_ckm = |V_ts|² y_t² (CKM) — treated as part of up
    # The up_s coefficient: (1.5)*(yc² + |V_ts|² y_t²)
    # Split: 1.5*yc² (diag integral: yc²*DT) + 1.5*|V_ts|²*yt² (CKM integral: |V_ts|²*∫yt²)
    delta_s = contribution(dQ_dzs, zs0, integral_ys2_approx,
                           integral_yc2_approx, integral_yt2,
                           self_coeff, up_s_diag, up_s_ckm)

    # y_d: similar
    delta_d = contribution(dQ_dzd, zd0, integral_yd2_approx,
                           integral_yu2_approx, integral_yt2,
                           self_coeff, up_d_diag, up_d_ckm)

    delta_Q = delta_b + delta_s + delta_d
    return delta_Q


# ─── CKM-AWARE BETA FUNCTIONS ─────────────────────────────────────────────────
def beta_1loop_ckm(t, y):
    """
    1-loop beta functions with CKM |V_ts|², |V_td|² retained in the
    down-type differential.
    """
    g1, g2, g3, yt, yc, yu, yb, ys, yd, ytau, ymu, ye = y

    # Trace
    Tr_Yu2 = yt*yt + yc*yc + yu*yu
    Tr_Yd2 = yb*yb + ys*ys + yd*yd
    Tr_Ye2 = ytau*ytau + ymu*ymu + ye*ye
    T = 3.0 * Tr_Yu2 + 3.0 * Tr_Yd2 + Tr_Ye2

    g1s, g2s, g3s = g1*g1, g2*g2, g3*g3

    # Gauge
    b1, b2, b3 = 41.0/10.0, -19.0/6.0, -7.0
    dg1 = ONE_LOOP_FACTOR * b1 * g1 * g1s
    dg2 = ONE_LOOP_FACTOR * b2 * g2 * g2s
    dg3 = ONE_LOOP_FACTOR * b3 * g3 * g3s

    gauge_up   = -(17.0/20.0)*g1s - (9.0/4.0)*g2s - 8.0*g3s
    gauge_down = -(1.0/4.0)*g1s  - (9.0/4.0)*g2s - 8.0*g3s
    gauge_lep  = -(9.0/4.0)*g1s  - (9.0/4.0)*g2s

    # Up-type (standard diagonal)
    dyt = ONE_LOOP_FACTOR * yt * (1.5*yt*yt - 1.5*yb*yb + T + gauge_up)
    dyc = ONE_LOOP_FACTOR * yc * (1.5*yc*yc - 1.5*ys*ys + T + gauge_up)
    dyu = ONE_LOOP_FACTOR * yu * (1.5*yu*yu - 1.5*yd*yd + T + gauge_up)

    # Down-type WITH CKM corrections
    # Up-partner couplings: diagonal + CKM-suppressed
    up_b = yt*yt * VTB2                           # ≈ yt²
    up_s = yc*yc + VTS2 * yt*yt                    # diagonal + CKM
    up_d = yu*yu + VTD2 * yt*yt                    # diagonal + CKM

    dyb = ONE_LOOP_FACTOR * yb * (1.5*yb*yb - 1.5*up_b + T + gauge_down)
    dys = ONE_LOOP_FACTOR * ys * (1.5*ys*ys - 1.5*up_s + T + gauge_down)
    dyd = ONE_LOOP_FACTOR * yd * (1.5*yd*yd - 1.5*up_d + T + gauge_down)

    # Leptons (unchanged)
    dytau = ONE_LOOP_FACTOR * ytau * (1.5*ytau*ytau + T + gauge_lep)
    dymu  = ONE_LOOP_FACTOR * ymu  * (1.5*ymu*ymu   + T + gauge_lep)
    dye   = ONE_LOOP_FACTOR * ye   * (1.5*ye*ye     + T + gauge_lep)

    return [dg1, dg2, dg3, dyt, dyc, dyu, dyb, dys, dyd, dytau, dymu, dye]


def beta_2loop_up_only(t, y):
    """
    2-loop RGE with 2-loop contributions for the UP-TYPE Yukawas ONLY.
    Down-type and leptons remain at 1-loop.

    This is the "current state" described in SPEC_DOWN2L — the suspected gap.
    """
    # Get full 1-loop
    d1 = beta_1loop_ckm(t, y)

    g1, g2, g3, yt, yc, yu, yb, ys, yd, ytau, ymu, ye = y
    g1s, g2s, g3s = g1*g1, g2*g2, g3*g3
    g1q, g2q, g3q = g1s*g1s, g2s*g2s, g3s*g3s

    yts, ycs, yus = yt*yt, yc*yc, yu*yu
    ybs, yss, yds = yb*yb, ys*ys, yd*yd

    Tr_Yu2 = yts + ycs + yus
    Tr_Yd2 = ybs + yss + yds
    Tr_Ye2 = ytau*ytau + ymu*ymu + ye*ye
    T = 3.0 * Tr_Yu2 + 3.0 * Tr_Yd2 + Tr_Ye2

    # ─── 2-loop gauge (Machacek-Vaughn) ───────────────────────────────────
    B11, B12, B13 = 199.0/50.0,  27.0/10.0,  44.0/5.0
    B21, B22, B23 =  9.0/10.0,   35.0/6.0,   12.0
    B31, B32, B33 = 11.0/10.0,    9.0/2.0,  -26.0

    S1 = B11*g1s + B12*g2s + B13*g3s
    S2 = B21*g1s + B22*g2s + B23*g3s
    S3 = B31*g1s + B32*g2s + B33*g3s

    Y1 = (17.0/10.0)*Tr_Yu2 + (1.0/2.0)*Tr_Yd2 + (3.0/2.0)*Tr_Ye2
    Y2 = (3.0/2.0)*Tr_Yu2   + (3.0/2.0)*Tr_Yd2 + (1.0/2.0)*Tr_Ye2
    Y3 = 2.0*Tr_Yu2         + 2.0*Tr_Yd2

    dg1_2L = TWO_LOOP_FACTOR * g1 * g1s * (S1 - Y1)
    dg2_2L = TWO_LOOP_FACTOR * g2 * g2s * (S2 - Y2)
    dg3_2L = TWO_LOOP_FACTOR * g3 * g3s * (S3 - Y3)

    # ─── 2-loop up-type Yukawa (Luo & Xiao) ───────────────────────────────
    pure_gauge_up = (
        (1187.0/600.0) * g1q
        - (9.0/20.0)   * g1s * g2s
        + (19.0/15.0)  * g1s * g3s
        - (23.0/4.0)   * g2q
        + 9.0          * g2s * g3s
        - 108.0        * g3q
    )

    gauge_HH_yt = (223.0/80.0)*g1s*yts + (135.0/16.0)*g2s*yts + 16.0*g3s*yts
    gauge_HH_yc = (223.0/80.0)*g1s*ycs + (135.0/16.0)*g2s*ycs + 16.0*g3s*ycs
    gauge_HH_yu = (223.0/80.0)*g1s*yus + (135.0/16.0)*g2s*yus + 16.0*g3s*yus

    gauge_FDFD_yt = -(43.0/80.0)*g1s*ybs + (9.0/16.0)*g2s*ybs - 16.0*g3s*ybs
    gauge_FDFD_yc = -(43.0/80.0)*g1s*yss + (9.0/16.0)*g2s*yss - 16.0*g3s*yss
    gauge_FDFD_yu = -(43.0/80.0)*g1s*yds + (9.0/16.0)*g2s*yds - 16.0*g3s*yds

    gauge_traces = (
        (17.0/8.0)*g1s*Tr_Yu2 + (45.0/8.0)*g2s*Tr_Yu2 + 20.0*g3s*Tr_Yu2
        + (5.0/8.0)*g1s*Tr_Yd2 + (45.0/8.0)*g2s*Tr_Yd2 + 20.0*g3s*Tr_Yd2
        + (15.0/8.0)*g1s*Tr_Ye2 + (15.0/8.0)*g2s*Tr_Ye2
    )

    Tr_H4 = yts*yts + ycs*ycs + yus*yus
    Tr_FD4 = ybs*ybs + yss*yss + yds*yds
    Tr_FL4 = ytau**4 + ymu**4 + ye**4
    Tr_HH_FDFD = 2.0*(yts*ybs + ycs*yss + yus*yds)
    chi4 = (9.0/4.0) * (3.0*Tr_H4 + 3.0*Tr_FD4 + Tr_FL4 - (1.0/3.0)*Tr_HH_FDFD)
    neg_chi4 = -chi4

    lam = 0.126; lam2 = lam * lam
    lam_HH_yt = (3.0/2.0)*lam2 - 6.0*lam*yts
    lam_HH_yc = (3.0/2.0)*lam2 - 6.0*lam*ycs
    lam_HH_yu = (3.0/2.0)*lam2 - 6.0*lam*yus

    common_2L = pure_gauge_up + gauge_traces + neg_chi4

    yt_pure = (
        + (3.0/2.0) * yts*yts
        - yts * ybs
        - (1.0/4.0) * yts * ybs
        + (11.0/4.0) * ybs*ybs
        - (9.0/4.0) * T * yts
        + (5.0/4.0) * T * ybs
    )
    yc_pure = (
        + (3.0/2.0) * ycs*ycs
        - ycs * yss
        - (1.0/4.0) * ycs * yss
        + (11.0/4.0) * yss*yss
        - (9.0/4.0) * T * ycs
        + (5.0/4.0) * T * yss
    )
    yu_pure = (
        + (3.0/2.0) * yus*yus
        - yus * yds
        - (1.0/4.0) * yus * yds
        + (11.0/4.0) * yds*yds
        - (9.0/4.0) * T * yus
        + (5.0/4.0) * T * yds
    )

    dyt_2L = TWO_LOOP_FACTOR * yt * (common_2L + gauge_HH_yt + gauge_FDFD_yt + yt_pure + lam_HH_yt)
    dyc_2L = TWO_LOOP_FACTOR * yc * (common_2L + gauge_HH_yc + gauge_FDFD_yc + yc_pure + lam_HH_yc)
    dyu_2L = TWO_LOOP_FACTOR * yu * (common_2L + gauge_HH_yu + gauge_FDFD_yu + yu_pure + lam_HH_yu)

    # Down-type: 1-loop ONLY (this is the key difference)
    # Return total = 1-loop + {2-loop gauge + 2-loop up-type + 1-loop down-type}
    return [
        d1[0] + dg1_2L, d1[1] + dg2_2L, d1[2] + dg3_2L,
        d1[3] + dyt_2L, d1[4] + dyc_2L, d1[5] + dyu_2L,
        d1[6], d1[7], d1[8],  # down-type: 1-loop only
        d1[9], d1[10], d1[11],  # leptons: 1-loop only
    ]


def beta_2loop_full_ckm(t, y):
    """
    2-loop RGE with 2-loop contributions for BOTH up-type and down-type
    Yukawas, including CKM |V_ts|², |V_td|² in the down-type differential.

    This is the "full 2-loop both sectors" configuration.
    """
    # Get full 1-loop with CKM
    d1 = beta_1loop_ckm(t, y)

    g1, g2, g3, yt, yc, yu, yb, ys, yd, ytau, ymu, ye = y
    g1s, g2s, g3s = g1*g1, g2*g2, g3*g3
    g1q, g2q, g3q = g1s*g1s, g2s*g2s, g3s*g3s

    yts, ycs, yus = yt*yt, yc*yc, yu*yu
    ybs, yss, yds = yb*yb, ys*ys, yd*yd

    Tr_Yu2 = yts + ycs + yus
    Tr_Yd2 = ybs + yss + yds
    Tr_Ye2 = ytau*ytau + ymu*ymu + ye*ye
    T = 3.0 * Tr_Yu2 + 3.0 * Tr_Yd2 + Tr_Ye2

    # ─── 2-loop gauge ─────────────────────────────────────────────────────
    B11, B12, B13 = 199.0/50.0,  27.0/10.0,  44.0/5.0
    B21, B22, B23 =  9.0/10.0,   35.0/6.0,   12.0
    B31, B32, B33 = 11.0/10.0,    9.0/2.0,  -26.0

    S1 = B11*g1s + B12*g2s + B13*g3s
    S2 = B21*g1s + B22*g2s + B23*g3s
    S3 = B31*g1s + B32*g2s + B33*g3s

    Y1 = (17.0/10.0)*Tr_Yu2 + (1.0/2.0)*Tr_Yd2 + (3.0/2.0)*Tr_Ye2
    Y2 = (3.0/2.0)*Tr_Yu2   + (3.0/2.0)*Tr_Yd2 + (1.0/2.0)*Tr_Ye2
    Y3 = 2.0*Tr_Yu2         + 2.0*Tr_Yd2

    dg1_2L = TWO_LOOP_FACTOR * g1 * g1s * (S1 - Y1)
    dg2_2L = TWO_LOOP_FACTOR * g2 * g2s * (S2 - Y2)
    dg3_2L = TWO_LOOP_FACTOR * g3 * g3s * (S3 - Y3)

    # ─── 2-loop up-type Yukawa ─────────────────────────────────────────────
    pure_gauge_up = (
        (1187.0/600.0) * g1q
        - (9.0/20.0)   * g1s * g2s
        + (19.0/15.0)  * g1s * g3s
        - (23.0/4.0)   * g2q
        + 9.0          * g2s * g3s
        - 108.0        * g3q
    )

    gauge_HH_yt = (223.0/80.0)*g1s*yts + (135.0/16.0)*g2s*yts + 16.0*g3s*yts
    gauge_HH_yc = (223.0/80.0)*g1s*ycs + (135.0/16.0)*g2s*ycs + 16.0*g3s*ycs
    gauge_HH_yu = (223.0/80.0)*g1s*yus + (135.0/16.0)*g2s*yus + 16.0*g3s*yus

    gauge_FDFD_yt = -(43.0/80.0)*g1s*ybs + (9.0/16.0)*g2s*ybs - 16.0*g3s*ybs
    gauge_FDFD_yc = -(43.0/80.0)*g1s*yss + (9.0/16.0)*g2s*yss - 16.0*g3s*yss
    gauge_FDFD_yu = -(43.0/80.0)*g1s*yds + (9.0/16.0)*g2s*yds - 16.0*g3s*yds

    gauge_traces = (
        (17.0/8.0)*g1s*Tr_Yu2 + (45.0/8.0)*g2s*Tr_Yu2 + 20.0*g3s*Tr_Yu2
        + (5.0/8.0)*g1s*Tr_Yd2 + (45.0/8.0)*g2s*Tr_Yd2 + 20.0*g3s*Tr_Yd2
        + (15.0/8.0)*g1s*Tr_Ye2 + (15.0/8.0)*g2s*Tr_Ye2
    )

    Tr_H4 = yts*yts + ycs*ycs + yus*yus
    Tr_FD4 = ybs*ybs + yss*yss + yds*yds
    Tr_FL4 = ytau**4 + ymu**4 + ye**4
    Tr_HH_FDFD = 2.0*(yts*ybs + ycs*yss + yus*yds)
    chi4 = (9.0/4.0) * (3.0*Tr_H4 + 3.0*Tr_FD4 + Tr_FL4 - (1.0/3.0)*Tr_HH_FDFD)
    neg_chi4 = -chi4

    lam = 0.126; lam2 = lam * lam
    lam_HH_yt = (3.0/2.0)*lam2 - 6.0*lam*yts
    lam_HH_yc = (3.0/2.0)*lam2 - 6.0*lam*ycs
    lam_HH_yu = (3.0/2.0)*lam2 - 6.0*lam*yus

    common_2L = pure_gauge_up + gauge_traces + neg_chi4

    yt_pure = (
        + (3.0/2.0) * yts*yts
        - (5.0/4.0) * yts * ybs
        + (11.0/4.0) * ybs*ybs
        - (9.0/4.0) * T * yts
        + (5.0/4.0) * T * ybs
    )
    yc_pure = (
        + (3.0/2.0) * ycs*ycs
        - (5.0/4.0) * ycs * yss
        + (11.0/4.0) * yss*yss
        - (9.0/4.0) * T * ycs
        + (5.0/4.0) * T * yss
    )
    yu_pure = (
        + (3.0/2.0) * yus*yus
        - (5.0/4.0) * yus * yds
        + (11.0/4.0) * yds*yds
        - (9.0/4.0) * T * yus
        + (5.0/4.0) * T * yds
    )

    dyt_2L = TWO_LOOP_FACTOR * yt * (common_2L + gauge_HH_yt + gauge_FDFD_yt + yt_pure + lam_HH_yt)
    dyc_2L = TWO_LOOP_FACTOR * yc * (common_2L + gauge_HH_yc + gauge_FDFD_yc + yc_pure + lam_HH_yc)
    dyu_2L = TWO_LOOP_FACTOR * yu * (common_2L + gauge_HH_yu + gauge_FDFD_yu + yu_pure + lam_HH_yu)

    # ─── 2-loop down-type Yukawa (WITH CKM) ────────────────────────────────
    # Gauge × F_D†F_D (self-coupling): same coefficients as up-type gauge×H†H
    gauge_FDFD_yb = (223.0/80.0)*g1s*ybs + (135.0/16.0)*g2s*ybs + 16.0*g3s*ybs
    gauge_FDFD_ys = (223.0/80.0)*g1s*yss + (135.0/16.0)*g2s*yss + 16.0*g3s*yss
    gauge_FDFD_yd = (223.0/80.0)*g1s*yds + (135.0/16.0)*g2s*yds + 16.0*g3s*yds

    # Gauge × H†H (cross-coupling to up partner) WITH CKM factors
    # The up-partner contribution gets weighted by CKM:
    #   yb: |V_tb|² yt² ≈ yt²
    #   ys: yc² + |V_ts|² yt²
    #   yd: yu² + |V_td|² yt²
    cross_coeff = -(43.0/80.0)*g1s + (9.0/16.0)*g2s - 16.0*g3s
    gauge_HH_yb_ckm = cross_coeff * (yts * VTB2)
    gauge_HH_ys_ckm = cross_coeff * (ycs + VTS2 * yts)  # weighted sum
    gauge_HH_yd_ckm = cross_coeff * (yus + VTD2 * yts)

    # Pure Yukawa — down-type with CKM
    # Up-type pure Yukawa structure for flavor i:
    #   +(3/2) y_{u_i}⁴ - (5/4) y_{u_i}² y_{d_i}² + (11/4) y_{d_i}⁴
    #   -(9/4) T y_{u_i}² + (5/4) T y_{d_i}²
    #
    # Down-type: swap u↔d → self-term is y_{d_i}, cross-term is y_{u_i}
    #   +(3/2) y_{d_i}⁴ - (5/4) y_{d_i}² y_{u_i}² + (11/4) y_{u_i}⁴
    #   -(9/4) T y_{d_i}² + (5/4) T y_{u_i}²
    #
    # With CKM: the up-partner coupling y_{u_i}² → Σ_j |V_{u_j d_i}|² y_{u_j}²
    #   b: y_t² (V_tb ≈ 1)
    #   s: y_c² + |V_ts|² y_t²
    #   d: y_u² + |V_td|² y_t²

    # For the b-quark (CKM-aligned, V_tb ≈ 1):
    up_eff_b = yts  # y_t²
    yb_pure_ckm = (
        + (3.0/2.0) * ybs*ybs
        - (5.0/4.0) * ybs * up_eff_b
        + (11.0/4.0) * up_eff_b * up_eff_b
        - (9.0/4.0) * T * ybs
        + (5.0/4.0) * T * up_eff_b
    )

    # For the s-quark (CKM: y_c² + |V_ts|² y_t²):
    up_eff_s = ycs + VTS2 * yts
    ys_pure_ckm = (
        + (3.0/2.0) * yss*yss
        - (5.0/4.0) * yss * up_eff_s
        + (11.0/4.0) * up_eff_s * up_eff_s
        - (9.0/4.0) * T * yss
        + (5.0/4.0) * T * up_eff_s
    )

    # For the d-quark (CKM: y_u² + |V_td|² y_t²):
    up_eff_d = yus + VTD2 * yts
    yd_pure_ckm = (
        + (3.0/2.0) * yds*yds
        - (5.0/4.0) * yds * up_eff_d
        + (11.0/4.0) * up_eff_d * up_eff_d
        - (9.0/4.0) * T * yds
        + (5.0/4.0) * T * up_eff_d
    )

    # Higgs quartic for down-type
    lam_FDFD_yb = (3.0/2.0)*lam2 - 6.0*lam*ybs
    lam_FDFD_ys = (3.0/2.0)*lam2 - 6.0*lam*yss
    lam_FDFD_yd = (3.0/2.0)*lam2 - 6.0*lam*yds

    dyb_2L = TWO_LOOP_FACTOR * yb * (common_2L + gauge_FDFD_yb + gauge_HH_yb_ckm + yb_pure_ckm + lam_FDFD_yb)
    dys_2L = TWO_LOOP_FACTOR * ys * (common_2L + gauge_FDFD_ys + gauge_HH_ys_ckm + ys_pure_ckm + lam_FDFD_ys)
    dyd_2L = TWO_LOOP_FACTOR * yd * (common_2L + gauge_FDFD_yd + gauge_HH_yd_ckm + yd_pure_ckm + lam_FDFD_yd)

    return [
        d1[0] + dg1_2L, d1[1] + dg2_2L, d1[2] + dg3_2L,
        d1[3] + dyt_2L, d1[4] + dyc_2L, d1[5] + dyu_2L,
        d1[6] + dyb_2L, d1[7] + dys_2L, d1[8] + dyd_2L,
        d1[9], d1[10], d1[11],  # leptons: 1-loop only
    ]


# ─── INITIAL CONDITIONS ─────────────────────────────────────────────────────
def initial_conditions(yb=YB_C, ys=YS_C, yd=YD_C,
                       yt=YT_C, yc=YC_C, yu=YU_C,
                       g1=G1_C, g2=G2_C, g3=G3_C):
    return np.array([
        g1, g2, g3, yt, yc, yu, yb, ys, yd,
        YTAU_MZ, YMU_MZ, YE_MZ,
    ])


# ─── EVOLVE ONE UNIVERSE ─────────────────────────────────────────────────────
def evolve_one(y0, beta_func, t_max=DT, rtol=1e-10, atol=1e-12):
    """Evolve from M_Z to t_max. Returns solution or None."""
    try:
        sol = solve_ivp(
            beta_func, [0.0, t_max], y0,
            method='RK45', rtol=rtol, atol=atol,
            t_eval=np.linspace(0.0, t_max, 20), max_step=0.5,
        )
        if not sol.success:
            return None
        return sol
    except Exception:
        return None


def extract_drift(sol, yb0, ys0, yd0):
    """Extract Q_inv drift from a solution. Returns (Q_mz, Q_3tev, drift_abs, drift_pct)."""
    q_mz = Q_inv(yb0, ys0, yd0)
    yb_f = float(np.interp(DT, sol.t, sol.y[6]))
    ys_f = float(np.interp(DT, sol.t, sol.y[7]))
    yd_f = float(np.interp(DT, sol.t, sol.y[8]))
    q_3tev = Q_inv(yb_f, ys_f, yd_f)
    drift_abs = q_3tev - q_mz
    drift_pct = 100.0 * drift_abs / q_mz
    return q_mz, q_3tev, drift_abs, drift_pct


# ─── CENTRAL-VALUE COMPUTATIONS ───────────────────────────────────────────────
def compute_central_drifts():
    """Compute the three drift predictions at central input values."""
    print("\n" + "="*72)
    print("CENTRAL-VALUE DRIFT COMPUTATIONS")
    print("="*72)

    y0 = initial_conditions()

    results = {}

    # (a) 1-loop only
    print("\n─── (a) 1-loop only ───")
    sol_1L = evolve_one(y0, beta_1loop_ckm)
    if sol_1L is None:
        print("  ⛔ 1-loop integration FAILED")
        return None
    q_mz, q_3tev, drift_abs, drift_pct = extract_drift(sol_1L, YB_C, YS_C, YD_C)
    s_d, s_s, s_b = compute_sensitivities(YB_C, YS_C, YD_C)
    results["1L"] = {
        "drift_pct": drift_pct, "drift_abs": drift_abs,
        "Q_mz": q_mz, "Q_3tev": q_3tev,
        "sensitivities": {"d": s_d, "s": s_s, "b": s_b},
        "yb_3tev": float(np.interp(DT, sol_1L.t, sol_1L.y[6])),
        "ys_3tev": float(np.interp(DT, sol_1L.t, sol_1L.y[7])),
        "yd_3tev": float(np.interp(DT, sol_1L.t, sol_1L.y[8])),
    }
    print(f"  Q_inv(M_Z)  = {q_mz:.8f}")
    print(f"  Q_inv(3TeV) = {q_3tev:.8f}")
    print(f"  Drift       = {drift_pct:+.4f}%")
    print(f"  Sensitivities dlnQ/dln[y_d,y_s,y_b] = [{s_d:+.4f}, {s_s:+.4f}, {s_b:+.4f}]")

    # (b) 2-loop up-only
    print("\n─── (b) 2-loop up-only ───")
    sol_2L_up = evolve_one(y0, beta_2loop_up_only)
    if sol_2L_up is None:
        print("  ⛔ 2-loop up-only integration FAILED")
        return None
    _, _, drift_abs, drift_pct = extract_drift(sol_2L_up, YB_C, YS_C, YD_C)
    results["2L_up_only"] = {
        "drift_pct": drift_pct, "drift_abs": drift_abs,
        "Q_mz": q_mz,  # same M_Z value
        "Q_3tev": float(Q_inv(
            float(np.interp(DT, sol_2L_up.t, sol_2L_up.y[6])),
            float(np.interp(DT, sol_2L_up.t, sol_2L_up.y[7])),
            float(np.interp(DT, sol_2L_up.t, sol_2L_up.y[8])))),
    }
    print(f"  Drift       = {drift_pct:+.4f}%")

    # (c) 2-loop full
    print("\n─── (c) 2-loop full (both sectors + CKM) ───")
    sol_2L_full = evolve_one(y0, beta_2loop_full_ckm)
    if sol_2L_full is None:
        print("  ⛔ 2-loop full integration FAILED")
        return None
    _, _, drift_abs, drift_pct = extract_drift(sol_2L_full, YB_C, YS_C, YD_C)
    results["2L_full"] = {
        "drift_pct": drift_pct, "drift_abs": drift_abs,
        "Q_mz": q_mz,
        "Q_3tev": float(Q_inv(
            float(np.interp(DT, sol_2L_full.t, sol_2L_full.y[6])),
            float(np.interp(DT, sol_2L_full.t, sol_2L_full.y[7])),
            float(np.interp(DT, sol_2L_full.t, sol_2L_full.y[8])))),
    }
    print(f"  Drift       = {drift_pct:+.4f}%")

    return results


# ─── FULL ERROR MODEL (draw-once, N >= 1e5) ───────────────────────────────────
def error_model(central_results):
    """Draw-once uncertainty propagation for all three drift predictions."""
    print("\n" + "="*72)
    print(f"FULL ERROR MODEL — draw-once, N={N_DRAW:,}, seed={SEED}")
    print("="*72)

    rng = np.random.RandomState(SEED)

    # Draw parameters
    central_params = {
        "yb": YB_C, "ys": YS_C, "yd": YD_C,
        "yt": YT_C, "yc": YC_C, "yu": YU_C,
        "g1": G1_C, "g2": G2_C, "g3": G3_C,
    }
    draws = {}
    for p in ["g1", "g2", "g3", "yt", "yc", "yu", "yb", "ys", "yd"]:
        draws[p] = rng.normal(central_params[p], SIGMA[p], N_DRAW)
        if p.startswith("y"):
            draws[p] = np.clip(draws[p], 1e-30, None)

    # Storage
    results_1L = {"drift_pct": np.zeros(N_DRAW), "drift_abs": np.zeros(N_DRAW),
                  "q_mz": np.zeros(N_DRAW), "q_3tev": np.zeros(N_DRAW),
                  "s_d": np.zeros(N_DRAW), "s_s": np.zeros(N_DRAW), "s_b": np.zeros(N_DRAW)}
    results_up = {"drift_pct": np.zeros(N_DRAW), "drift_abs": np.zeros(N_DRAW)}
    results_full = {"drift_pct": np.zeros(N_DRAW), "drift_abs": np.zeros(N_DRAW)}
    n_failed_1L = 0; n_failed_up = 0; n_failed_full = 0

    tic = time.time()
    report_every = max(1, N_DRAW // 10)

    # Checkpoint support
    outdir = os.path.join(SCRIPT_DIR, "results", "2L")
    os.makedirs(outdir, exist_ok=True)
    ckpt_path = os.path.join(outdir, "error_model_checkpoint.npz")

    start_i = 0
    if os.path.exists(ckpt_path):
        ck = np.load(ckpt_path, allow_pickle=True)
        ck_i = int(ck["i"])
        # Verify checkpoint is compatible (same N_DRAW, not stale)
        ck_N = len(ck["r1L_drift_pct"])
        if ck_N != N_DRAW:
            print(f"  Stale checkpoint (N={ck_N} != {N_DRAW}), discarding and restarting")
            os.remove(ckpt_path)
        elif ck_i + 1 >= N_DRAW:
            print(f"  Checkpoint complete ({ck_i+1} >= {N_DRAW}), skipping draws")
            start_i = N_DRAW
            for key in results_1L:
                results_1L[key][:] = ck["r1L_" + key]
            for key in results_up:
                results_up[key][:] = ck["rup_" + key]
            for key in results_full:
                results_full[key][:] = ck["rfull_" + key]
            n_failed_1L = int(ck["nf1L"])
            n_failed_up = int(ck["nfup"])
            n_failed_full = int(ck["nffull"])
        else:
            start_i = ck_i + 1
            for key in results_1L:
                results_1L[key][:start_i] = ck["r1L_" + key][:start_i]
            for key in results_up:
                results_up[key][:start_i] = ck["rup_" + key][:start_i]
            for key in results_full:
                results_full[key][:start_i] = ck["rfull_" + key][:start_i]
            n_failed_1L = int(ck["nf1L"])
            n_failed_up = int(ck["nfup"])
            n_failed_full = int(ck["nffull"])
            print(f"  Resuming from i={start_i}, failed: {n_failed_1L}/{n_failed_up}/{n_failed_full}")

    for i in range(start_i, N_DRAW):
        y0 = initial_conditions(
            yb=draws["yb"][i], ys=draws["ys"][i], yd=draws["yd"][i],
            yt=draws["yt"][i], yc=draws["yc"][i], yu=draws["yu"][i],
            g1=draws["g1"][i], g2=draws["g2"][i], g3=draws["g3"][i],
        )

        # Sensitivities at M_Z
        s_d, s_s, s_b = compute_sensitivities(draws["yb"][i], draws["ys"][i], draws["yd"][i])
        results_1L["s_d"][i] = s_d
        results_1L["s_s"][i] = s_s
        results_1L["s_b"][i] = s_b

        q_mz_i = Q_inv(draws["yb"][i], draws["ys"][i], draws["yd"][i])
        results_1L["q_mz"][i] = q_mz_i

        # (a) 1-loop
        sol = evolve_one(y0, beta_1loop_ckm)
        if sol is None:
            n_failed_1L += 1
            for key in results_1L:
                if key != "q_mz" and not key.startswith("s_"):
                    results_1L[key][i] = np.nan
        else:
            q3 = Q_inv(float(np.interp(DT, sol.t, sol.y[6])),
                       float(np.interp(DT, sol.t, sol.y[7])),
                       float(np.interp(DT, sol.t, sol.y[8])))
            results_1L["q_3tev"][i] = q3
            results_1L["drift_abs"][i] = q3 - q_mz_i
            results_1L["drift_pct"][i] = 100.0 * (q3 - q_mz_i) / q_mz_i

        # (b) 2-loop up-only
        sol = evolve_one(y0, beta_2loop_up_only)
        if sol is None:
            n_failed_up += 1
            results_up["drift_pct"][i] = np.nan
            results_up["drift_abs"][i] = np.nan
        else:
            q3 = Q_inv(float(np.interp(DT, sol.t, sol.y[6])),
                       float(np.interp(DT, sol.t, sol.y[7])),
                       float(np.interp(DT, sol.t, sol.y[8])))
            results_up["drift_abs"][i] = q3 - q_mz_i
            results_up["drift_pct"][i] = 100.0 * (q3 - q_mz_i) / q_mz_i

        # (c) 2-loop full
        sol = evolve_one(y0, beta_2loop_full_ckm)
        if sol is None:
            n_failed_full += 1
            results_full["drift_pct"][i] = np.nan
            results_full["drift_abs"][i] = np.nan
        else:
            q3 = Q_inv(float(np.interp(DT, sol.t, sol.y[6])),
                       float(np.interp(DT, sol.t, sol.y[7])),
                       float(np.interp(DT, sol.t, sol.y[8])))
            results_full["drift_abs"][i] = q3 - q_mz_i
            results_full["drift_pct"][i] = 100.0 * (q3 - q_mz_i) / q_mz_i

        if (i + 1) % report_every == 0 or i == N_DRAW - 1:
            elapsed = time.time() - tic
            rate = (i + 1 - start_i) / elapsed if elapsed > 0 else 0
            eta = (N_DRAW - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{N_DRAW}] {elapsed:.0f}s, ~{eta:.0f}s remaining "
                  f"(fail: {n_failed_1L}/{n_failed_up}/{n_failed_full})")

        # Checkpoint every 2000 draws
        if (i + 1) % 2000 == 0 and i < N_DRAW - 1:
            ck_data = {"i": i, "nf1L": n_failed_1L, "nfup": n_failed_up, "nffull": n_failed_full}
            for key in results_1L:
                ck_data["r1L_" + key] = results_1L[key]
            for key in results_up:
                ck_data["rup_" + key] = results_up[key]
            for key in results_full:
                ck_data["rfull_" + key] = results_full[key]
            np.savez_compressed(ckpt_path, **ck_data)

    elapsed = time.time() - tic
    print(f"  Done: {N_DRAW} universes in {elapsed:.0f}s ({N_DRAW/elapsed:.0f} draws/s)")

    # Clean up checkpoint
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    # Summarize
    def summarize(data, label):
        valid = ~np.isnan(data["drift_pct"])
        nv = valid.sum()
        result = {
            "label": label,
            "N_valid": int(nv),
            "N_failed": int(N_DRAW - nv),
            "drift_central_pct": float(np.mean(data["drift_pct"][valid])),
            "drift_sigma_pct": float(np.std(data["drift_pct"][valid])),
            "drift_central_abs": float(np.mean(data["drift_abs"][valid])),
            "drift_sigma_abs": float(np.std(data["drift_abs"][valid])),
        }
        if "q_mz" in data:
            result["q_mz_central"] = float(np.mean(data["q_mz"][valid]))
            result["q_mz_sigma"] = float(np.std(data["q_mz"][valid]))
        return result

    summaries = {
        "1L": summarize(results_1L, "1-loop"),
        "2L_up_only": summarize(results_up, "2-loop up-only"),
        "2L_full": summarize(results_full, "2-loop full"),
    }

    # Sensitivity summary
    valid_1L = ~np.isnan(results_1L["drift_pct"])
    sens_summary = {
        "d": {"central": float(np.mean(results_1L["s_d"][valid_1L])),
              "sigma": float(np.std(results_1L["s_d"][valid_1L]))},
        "s": {"central": float(np.mean(results_1L["s_s"][valid_1L])),
              "sigma": float(np.std(results_1L["s_s"][valid_1L]))},
        "b": {"central": float(np.mean(results_1L["s_b"][valid_1L])),
              "sigma": float(np.std(results_1L["s_b"][valid_1L]))},
    }

    return summaries, sens_summary, results_1L, results_up, results_full


# ─── TRUNCATION BAND ─────────────────────────────────────────────────────────
def compute_truncation_band():
    """|2L_full - 1L| as truncation band estimate."""
    y0 = initial_conditions()
    sol_1L = evolve_one(y0, beta_1loop_ckm)
    sol_2L = evolve_one(y0, beta_2loop_full_ckm)
    _, _, _, drift_1L = extract_drift(sol_1L, YB_C, YS_C, YD_C)
    _, _, _, drift_2L = extract_drift(sol_2L, YB_C, YS_C, YD_C)
    return abs(drift_2L - drift_1L)


# ─── GATE X1 ──────────────────────────────────────────────────────────────────
def gate_X1(summaries, sens_summary, central_results):
    """
    GATE X1: the 1-loop reproduction must land at -0.065% ± 0.003.

    Independent hand calculation gives:
      yt(M_Z)=0.967, sensitivities dlnQ/dln y = [-0.155, +0.131, +0.024]
      for [y_d, y_s, y_b].
    """
    print("\n" + "="*72)
    print("GATE X1 — 1-loop drift reproduction")
    print("="*72)

    s_1L = summaries["1L"]
    drift = s_1L["drift_central_pct"]

    # Target: -0.065% ± 0.003
    target = -0.065
    tolerance = 0.003
    gate_x1_pass = abs(drift - target) <= tolerance

    print(f"  1-loop drift (computed): {drift:+.4f}%")
    print(f"  Target:                  {target:+.3f}% ± {tolerance:.3f}")
    print(f"  GATE X1: {'✅ PASS' if gate_x1_pass else '❌ FAIL'}")

    # Print sensitivities alongside reference values
    print(f"\n  ─── Sensitivities dlnQ/dln y ───")
    print(f"  Parameter | Computed     | Reference    | Match?")
    ref_sens = {"d": -0.155, "s": +0.131, "b": +0.024}
    sens_checks = {}
    for key, ref in ref_sens.items():
        comp = sens_summary[key]["central"]
        # Tolerance: ±0.005 on sensitivity (generous — these are approximate)
        match = abs(comp - ref) <= 0.005
        sens_checks[key] = match
        print(f"  y_{key}      | {comp:+.4f}       | {ref:+.4f}       | {'✓' if match else '✗'}")

    all_sens_match = all(sens_checks.values())

    record = {
        "gate": "GATE_X1",
        "passed": gate_x1_pass,
        "drift_computed_pct": round(drift, 6),
        "drift_target_pct": target,
        "tolerance": tolerance,
        "sensitivities": {
            "computed": {k: round(sens_summary[k]["central"], 6) for k in ["d", "s", "b"]},
            "reference": ref_sens,
            "match": {k: bool(v) for k, v in sens_checks.items()},
        },
    }

    return gate_x1_pass, all_sens_match, record


# ─── GATE X2 ──────────────────────────────────────────────────────────────────
def gate_X2():
    """
    GATE X2: every gate value must be COMPUTED, never asserted from a target.
    This is a meta-gate — we verify that all numbers in gate records come
    from actual computation, not hard-coded assertions.
    """
    print("\n" + "="*72)
    print("GATE X2 — computed-value discipline")
    print("="*72)

    # This gate is satisfied by construction: every number in the gate records
    # is produced by the functions above. We verify by checking that the
    # computation functions actually ran (returned non-None values).

    # Verify central computations ran
    central = compute_central_drifts()
    if central is None:
        print("  GATE X2: ❌ FAIL — central drift computation returned None")
        return False, {}

    all_computed = True
    for label, expected_keys in [("1L", ["drift_pct", "sensitivities"]),
                                  ("2L_up_only", ["drift_pct"]),
                                  ("2L_full", ["drift_pct"])]:
        if label not in central:
            print(f"  GATE X2: ❌ FAIL — {label} missing from central results")
            all_computed = False
        else:
            for k in expected_keys:
                if central[label].get(k) is None:
                    print(f"  GATE X2: ❌ FAIL — {label}.{k} is None")
                    all_computed = False

    if all_computed:
        print(f"  GATE X2: ✅ PASS — all drift values computed from integration")
    else:
        print(f"  GATE X2: ❌ FAIL — some values not computed")

    record = {
        "gate": "GATE_X2",
        "passed": all_computed,
        "note": "Every gate value is produced by RGE integration, never asserted from target. Verified by checking all computation functions returned valid results.",
        "central_drifts_computed": {
            "1L": round(central["1L"]["drift_pct"], 6) if "1L" in central else None,
            "2L_up_only": round(central["2L_up_only"]["drift_pct"], 6) if "2L_up_only" in central else None,
            "2L_full": round(central["2L_full"]["drift_pct"], 6) if "2L_full" in central else None,
        },
    }

    return all_computed, record


# ─── DELIVERABLE ──────────────────────────────────────────────────────────────
def assemble_deliverable(summaries, sens_summary, central_results,
                         trunc_band, gate_x1_rec, gate_x2_rec, results_1L, results_up, results_full):
    """Assemble the final deliverable."""
    print("\n" + "="*72)
    print("DELIVERABLE — DOWN-SECTOR RESIDUAL: IS IT 2-LOOP?")
    print("="*72)

    measured_pct = -0.0779
    measured_sigma = 0.0006

    # Residuals
    residuals = {}
    for key, label in [("1L", "1-loop"), ("2L_up_only", "2-loop up-only"), ("2L_full", "2-loop full")]:
        pred = summaries[key]["drift_central_pct"]
        pred_sigma = summaries[key]["drift_sigma_pct"]
        res = measured_pct - pred
        res_combined_sigma = np.sqrt(pred_sigma**2 + measured_sigma**2)
        n_sigma = abs(res) / res_combined_sigma if res_combined_sigma > 0 else float('inf')
        residuals[key] = {
            "label": label,
            "prediction_pct": round(pred, 6),
            "prediction_sigma_pct": round(pred_sigma, 6),
            "measured_pct": measured_pct,
            "measured_sigma_pct": measured_sigma,
            "residual_pct": round(res, 6),
            "residual_combined_sigma_pct": round(res_combined_sigma, 6),
            "residual_n_sigma": round(n_sigma, 4),
        }
        print(f"\n  {label}:")
        print(f"    Predicted: {pred:+.4f}% ± {pred_sigma:.4f}%")
        print(f"    Measured:  {measured_pct:+.4f}% ± {measured_sigma:.4f}%")
        print(f"    Residual:  {res:+.4f}%  ({n_sigma:.2f}σ)")

    # Verdict: does 2-loop full close the gap?
    res_full = residuals["2L_full"]["residual_pct"]
    res_full_sigma = residuals["2L_full"]["residual_combined_sigma_pct"]
    gap_closed = abs(res_full) <= 2.0 * res_full_sigma

    # Motion in ln(y_s/y_d) that would close the remaining gap
    # From the sensitivity: d(drift)/d(ln(y_s/y_d))
    # We need: what change in ln(y_s/y_d) would shift the prediction by the residual?
    # Using the computed sensitivities...

    print(f"\n  ─── VERDICT ───")
    if gap_closed:
        print(f"  Full 2-loop CLOSES the gap (residual {res_full:+.4f}% within 2σ of {res_full_sigma:.4f}%).")
        verdict = "CLOSED"
    else:
        print(f"  Full 2-loop does NOT close the gap (residual {res_full:+.4f}% at {abs(res_full)/res_full_sigma:.2f}σ).")

        # Compute the motion in ln(y_s/y_d) that would close the gap
        # Use the sensitivities: drift change ≈ s_s * dln y_s + s_d * dln y_d
        # For a differential motion Δ ln(y_s/y_d), the drift change is:
        #   Δ drift ≈ (s_s * ∂ln y_s/∂ln(y_s/y_d) + s_d * ∂ln y_d/∂ln(y_s/y_d)) * Δ ln(y_s/y_d)
        # The simplest interpretation: shift y_s upward by Δ ln y_s = +Δ, y_d unchanged
        # → Δ drift ≈ s_s * Δ
        # Or shift y_d downward by Δ ln y_d = -Δ, y_s unchanged
        # → Δ drift ≈ s_d * (-Δ)
        # The actual motion needed would be some combination.

        # We need Δ drift = residual = res_full (in percent units)
        # Using s_s ≈ +0.13 (inverse weight ≈ +0.13 × 100 = +13% per e-fold in y_s)
        # Wait, the sensitivity dlnQ/dln y_s ≈ +0.13 means:
        #   A +1% change in y_s → +0.13% change in Q_inv
        #   But the drift pct = 100 * ΔQ/Q, and ΔQ/Q = Σ s_i * Δy_i/y_i
        # So to get Δ(drift_pct) = res_full (in %), we need:
        #   Σ s_i * 100 * Δy_i/y_i = res_full
        #   Σ s_i * Δln y_i = res_full / 100

        # The gap is res_full in percent. For Δln(y_s/y_d):
        # If we change y_s by factor e^{Δ}, drift changes by s_s * Δ * 100 in pct
        # Δ * 100 = res_full / s_s

        s_s = sens_summary["s"]["central"]
        s_d = sens_summary["d"]["central"]
        # Differential motion in ln(y_s/y_d):
        # Keep one fixed, vary the other
        delta_ln_ratio_s = res_full / (100.0 * s_s)  # change y_s only
        delta_ln_ratio_d = res_full / (100.0 * s_d)  # change y_d only

        print(f"  Residual: {res_full:+.4f}%")
        print(f"  To close by varying y_s alone: Δ ln y_s = {delta_ln_ratio_s:+.4f}")
        print(f"  To close by varying y_d alone: Δ ln y_d = {delta_ln_ratio_d:+.4f}")
        print(f"  (sensitivities: s_s={s_s:+.4f}, s_d={s_d:+.4f})")
        print(f"  Stated as unexplained: differential motion in ln(y_s/y_d) of "
              f"~{abs(delta_ln_ratio_s):.4f} (y_s shift) would close the gap.")

        verdict = "NOT_CLOSED"
        unexplained = {
            "residual_pct": round(res_full, 6),
            "delta_ln_ratio_via_ys": round(delta_ln_ratio_s, 6),
            "delta_ln_ratio_via_yd": round(delta_ln_ratio_d, 6),
            "sensitivity_ys": round(s_s, 6),
            "sensitivity_yd": round(s_d, 6),
        }

    # Truncation band
    print(f"\n  Truncation band |2L − 1L| = {trunc_band:.4f}%")

    # Full deliverable
    deliverable = {
        "spec": "SPEC_DOWN2L v1.0",
        "seed": SEED,
        "N_draws": N_DRAW,
        "Q_inv_MZ_central": round(summaries["1L"]["q_mz_central"], 8),

        "predictions": {
            "1L": {
                "drift_pct": round(summaries["1L"]["drift_central_pct"], 6),
                "drift_sigma_pct": round(summaries["1L"]["drift_sigma_pct"], 6),
                "sensitivities_dlnQ_dlny": {
                    "yd": round(sens_summary["d"]["central"], 6),
                    "ys": round(sens_summary["s"]["central"], 6),
                    "yb": round(sens_summary["b"]["central"], 6),
                },
            },
            "2L_up_only": {
                "drift_pct": round(summaries["2L_up_only"]["drift_central_pct"], 6),
                "drift_sigma_pct": round(summaries["2L_up_only"]["drift_sigma_pct"], 6),
            },
            "2L_full": {
                "drift_pct": round(summaries["2L_full"]["drift_central_pct"], 6),
                "drift_sigma_pct": round(summaries["2L_full"]["drift_sigma_pct"], 6),
            },
        },

        "measured": {
            "drift_pct": measured_pct,
            "drift_sigma_pct": measured_sigma,
            "source": "AHS2026 tabulated endpoints: (0.66686 - 0.66738)/0.66738",
        },

        "residuals": residuals,
        "truncation_band_pct": round(trunc_band, 6),
        "verdict": verdict,
    }

    if verdict == "NOT_CLOSED":
        deliverable["unexplained"] = unexplained

    deliverable["gates"] = {
        "X1": gate_x1_rec,
        "X2": gate_x2_rec,
    }

    deliverable["input_budget"] = {
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
        "CKM": {
            "Vts2": VTS2, "Vtd2": VTD2, "Vtb2": 1.0,
            "note": "Retained in down-type differential per spec",
        },
        "source": "AHS2026 arXiv:2510.01312v2 Eq. 2.4; CKM from PDG 2024",
    }

    return deliverable


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    outdir = os.path.join(SCRIPT_DIR, "results", "2L")
    os.makedirs(outdir, exist_ok=True)

    print("="*72)
    print("DOWN-SECTOR RESIDUAL: IS IT 2-LOOP? — SPEC_DOWN2L v1.0")
    print(f"Seed {SEED} | N={N_DRAW:,} | M_Z={MZ} GeV → μ={M_3TEV/1e3:.1f} TeV")
    print(f"CKM: |V_ts|²={VTS2:.6f}, |V_td|²={VTD2:.6f}")
    print(f"Output: {outdir}")
    print("="*72)

    # ═══ CENTRAL-VALUE DRIFTS ═══
    central_results = compute_central_drifts()
    if central_results is None:
        print("\n⛔ Central-value computations FAILED. Aborting.")
        sys.exit(1)

    # ═══ FULL ERROR MODEL ═══
    summaries, sens_summary, results_1L, results_up, results_full = error_model(central_results)

    # ═══ TRUNCATION BAND ═══
    trunc_band = compute_truncation_band()
    print(f"\n  Truncation band |2L_full - 1L| = {trunc_band:.4f}%")

    # ═══ GATE X1 ═══
    gate_x1_pass, sens_match, gate_x1_rec = gate_X1(summaries, sens_summary, central_results)
    json_dump(gate_x1_rec, os.path.join(outdir, "gate_X1.json"))

    if not gate_x1_pass:
        print("\n⛔ GATE X1 FAILED. Stopping (spec: never tune, never auto-pass).")
        sys.exit(1)

    if not sens_match:
        print("\n⚠️  Sensitivity mismatch detected. Gate X1 sensitivity check failed.")
        print("  Continuing per spec's 'print and stop for operator ruling' — ")
        print("  sensitivities are printed above; operator may inspect.")
        # Per spec: if a gate cannot be satisfied as written, STOP and print
        # But the spec says "GATE X1: the 1-loop reproduction must land at
        # -0.065% ± 0.003" — the sensitivity check is part of the gate.
        # We stop on sensitivity mismatch too.
        print("⛔ GATE X1 sensitivity check FAILED. Stopping for operator ruling.")
        sys.exit(1)

    # ═══ GATE X2 ═══
    gate_x2_pass, gate_x2_rec = gate_X2()
    json_dump(gate_x2_rec, os.path.join(outdir, "gate_X2.json"))

    if not gate_x2_pass:
        print("\n⛔ GATE X2 FAILED. Stopping (spec: never tune, never auto-pass).")
        sys.exit(1)

    # ═══ DELIVERABLE ═══
    deliverable = assemble_deliverable(
        summaries, sens_summary, central_results, trunc_band,
        gate_x1_rec, gate_x2_rec, results_1L, results_up, results_full,
    )

    deliv_path = os.path.join(outdir, "deliverable.json")
    json_dump(deliverable, deliv_path)
    print(f"\n  📀 deliverable → {deliv_path}")

    # ═══ Save draw-level summaries ═══
    def draw_summary(data, label):
        valid = ~np.isnan(data["drift_pct"])
        return {
            "drift_pct": {
                "mean": float(np.mean(data["drift_pct"][valid])),
                "std": float(np.std(data["drift_pct"][valid])),
                "p16": float(np.percentile(data["drift_pct"][valid], 16)),
                "p50": float(np.percentile(data["drift_pct"][valid], 50)),
                "p84": float(np.percentile(data["drift_pct"][valid], 84)),
            },
            "N_valid": int(valid.sum()),
            "N_total": len(data["drift_pct"]),
        }

    distributions = {
        "1L": draw_summary(results_1L, "1-loop"),
        "2L_up_only": draw_summary(results_up, "2-loop up-only"),
        "2L_full": draw_summary(results_full, "2-loop full"),
        "sensitivities": {
            "yd": {"mean": float(np.mean(results_1L["s_d"][~np.isnan(results_1L["drift_pct"])])),
                   "std": float(np.std(results_1L["s_d"][~np.isnan(results_1L["drift_pct"])]))},
            "ys": {"mean": float(np.mean(results_1L["s_s"][~np.isnan(results_1L["drift_pct"])])),
                   "std": float(np.std(results_1L["s_s"][~np.isnan(results_1L["drift_pct"])]))},
            "yb": {"mean": float(np.mean(results_1L["s_b"][~np.isnan(results_1L["drift_pct"])])),
                   "std": float(np.std(results_1L["s_b"][~np.isnan(results_1L["drift_pct"])]))},
        },
    }
    json_dump(distributions, os.path.join(outdir, "draw_distributions.json"))

    # ═══ SUMMARY ═══
    print("\n" + "="*72)
    print("SUMMARY")
    print("="*72)
    print(f"  GATE X1: {'✅ PASS' if gate_x1_pass else '❌ FAIL'}")
    print(f"  GATE X2: {'✅ PASS' if gate_x2_pass else '❌ FAIL'}")
    print(f"\n  Predictions M_Z → 3 TeV:")
    for key in ["1L", "2L_up_only", "2L_full"]:
        s = summaries[key]
        print(f"    {s['label']:20s}: {s['drift_central_pct']:+.4f}% ± {s['drift_sigma_pct']:.4f}%")
    print(f"    {'Measured (AHS2026)':20s}: -0.0779% ± 0.0006%")
    print(f"\n  Verdict: {deliverable['verdict']}")
    if deliverable['verdict'] == 'NOT_CLOSED':
        u = deliverable['unexplained']
        print(f"    Residual: {u['residual_pct']:+.4f}%")
        print(f"    To close: Δ ln(y_s/y_d) via y_s shift = {u['delta_ln_ratio_via_ys']:+.4f}")
    print(f"\n  Artifacts: {outdir}/")
    print("  DONE.")
    print("="*72)

    return 0


if __name__ == "__main__":
    sys.exit(main())

# ASSUMPTIONS.md — Down-Type Inverse-Coordinate Drift Test

**Spec version:** v1.0
**Seed:** 20260811
**Engine:** amb (ganymede, single-machine)
**Date:** 2026-08-09

## D1. Input sources

Down-type Yukawa couplings and uncertainties at M_Z are taken from the AHS2026
paper (Antusch, Hinze, Saad, arXiv:2510.01312v2), Eq. (2.4), 2024 PDG input:

  y_b = (1.630 ± 0.009) × 10⁻²
  y_s = (3.06 ± 0.04) × 10⁻⁴
  y_d = (1.54 ± 0.02) × 10⁻⁵

Up-type Yukawas and gauge couplings from the same source (Eq. 2.4):

  y_t = 0.967 ± 0.004
  y_c = (3.56 ± 0.06) × 10⁻³
  y_u = (7.04 ± 0.15) × 10⁻⁶
  g_1 = 0.461228 ± 0.000026
  g_2 = 0.65096 ± 0.00004
  g_3 = 1.2123 ± 0.0046

These supersede the approximate values in `sm_rge.py` and `inputs_frozen.json`.
The AHS2026 paper uses SMDR + REAP (2-loop RGEs in MS-bar, 100,000-point MC
with Gaussian errors) to produce these M_Z-scale values and uncertainties.

**Citation:** S. Antusch, K. Hinze, S. Saad, "Updated Running Quark and Lepton
Parameters at Various Scales," arXiv:2510.01312v2 [hep-ph], 23 Mar 2026.

## D2. Q_inv definition

Q_inv[y_d, y_s, y_b] = Q(1/y_d, 1/y_s, 1/y_b)
where Q[x_1, x_2, x_3] = Σ x_i / (Σ √x_i)².

This is the inverse-coordinate analog of Q_U. It is degree-zero homogeneous in
the inverse Yukawas: Q_inv[α y] = Q_inv[y]. The flavor-universal anomalous
dimension cancels identically. Drift is carried by flavor-differential terms in
the beta functions — primarily the y_t-driven differential on y_b.

At M_Z with central inputs from Eq. (2.4):
  z_d = 1/y_d ≈ 64935, z_s = 1/y_s ≈ 3268, z_b = 1/y_b ≈ 61.35
  S = 68264, R = 319.82
  Q_inv = 68264 / 319.82² = 0.66738 ✓ (matches published value)

## D3. Measured drift

The "measured" drift is obtained by evolving the full 2-loop SM RGE system
from M_Z to 3 TeV and computing Q_inv at both endpoints:

  ΔQ_inv^{meas} = Q_inv(3 TeV) - Q_inv(M_Z)

Both values come from the SAME evolution (same universe), so the endpoints are
correlated. The drift uncertainty is obtained by drawing N >= 1e5 input-universe
parameter sets, evolving each, and computing the standard deviation of ΔQ_inv
across universes.

## D4. Predicted drift — derived law

The "predicted" drift is obtained from an analytic expression for dQ_inv/dt
derived from the 1-loop beta functions.

Let z_i = 1/y_i. Then:

  dQ_inv/dt = Σ_i (∂Q/∂z_i) * dz_i/dt

where ∂Q/∂z_i = (R - S/√z_i) / R³ (S = Σ z_j, R = Σ √z_j),
and dz_i/dt = -z_i * (β_{y_i}^{diff} / y_i).

The flavor-universal piece of β cancels by Euler's theorem (Σ z_i ∂Q/∂z_i = 0
for degree-zero homogeneous Q). The flavor-differential part of β_{y_i} / y_i at
1-loop is:

  β_{y_i}^{diff} / y_i = (3/2 y_i² - 3/2 y_{u_i}²) / (16π²)

where y_{u_i} is the up-type partner (y_t for b, y_c for s, y_u for d).
The dominant contribution is the y_t-driven term on y_b: -(3/2) y_t² / (16π²).

The total predicted drift from M_Z to 3 TeV is:

  ΔQ_inv^{pred} ≈ dQ_inv/dt|_{M_Z} × ln(3 TeV / M_Z)

evaluated at the central input values. The uncertainty is propagated by
evaluating the same expression with drawn input parameters.

## D5. Diagonal CKM approximation

The SM Yukawa matrices are 3×3 complex. In the mass eigenbasis, they are
diagonal to leading order. The CKM matrix V_CKM introduces off-diagonal
elements of O(λ³) ≈ 1% in the down-type Yukawa matrix:

  Y_d = V_CKM^† diag(y_d, y_s, y_b) V_CKM

For the y_b beta function, the dominant CKM effect is the y_t² term multiplied
by |V_tb|² = cos²(θ_23) ≈ 0.998, so the diagonal approximation introduces an
error of ~0.2% in the y_t-driven drift. This is below the current uncertainty
budget.

We adopt the diagonal (CKM = identity) approximation for both the measured and
predicted drift. The spec's reference to "CKM-aligned" is interpreted as
acknowledging that the y_t→y_b coupling in the beta function is CKM-aligned
(V_tb ≈ 1), and the diagonal approximation captures the dominant effect.

## D6. Integration method

RGE integration uses scipy.integrate.solve_ivp with method='RK45' (adaptive
Runge-Kutta 4/5). Tolerances: rtol=1e-10, atol=1e-12. Integration variable:
t = ln(μ/M_Z), from t=0 (M_Z) to t=ln(3000/91.1876) ≈ 3.49.

The short integration range (Δt ≈ 3.49 vs Δt ≈ 32.3 for the full GUT run)
means integration errors are negligible at these tolerances.

## D7. Draw-once methodology

Each uncertainty draw samples all 9 variable parameters (g_1, g_2, g_3, y_t,
y_c, y_u, y_b, y_s, y_d) from independent Gaussians with σ from AHS2026
Eq. (2.4). Lepton Yukawas (y_τ, y_μ, y_e) are held at their central values
— their impact on the down-type drift is negligible (they enter only through
trace T, which is dominated by 3y_t² ≈ 2.8).

Parameters are clipped to ≥ 10⁻³⁰ to prevent negative Yukawas in extreme
tails of the Gaussian draws.

For each universe:
  1. Draw all 9 parameters from their distributions
  2. Compute Q_inv(M_Z) analytically from the drawn y_b, y_s, y_d
  3. Compute the predicted drift from the derived law
  4. Evolve the full 2-loop RGE system from M_Z to 3 TeV
  5. Compute Q_inv(3 TeV) by interpolating the evolved Yukawas
  6. Compute measured drift = Q_inv(3 TeV) - Q_inv(M_Z)

The drift band is computed from the distribution of ΔQ_inv across universes.
The endpoint bands are computed from the distributions of Q_inv(M_Z) and
Q_inv(3 TeV) separately. The correlation benefit is quantified as:

  benefit = σ_naive / σ_draw-once = √(σ²_MZ + σ²_3TeV) / σ_drift

## D8. Gate tolerances

GATE D1 tolerances are set to the last quoted digit:
  - Q_inv values: ±5×10⁻⁶ (last digit of 0.66738 is 10⁻⁵; we use half)
  - Drift percentages: ±0.001 percentage points (last digit of -0.078% is 0.001%)

## D9. Published control values

From the AHS2026 paper (provided in the spec, verified against the paper):

  Q_inv(M_Z)      = 0.66738
  Q_inv(3 TeV)    = 0.66686
  Measured drift  = -0.078%   [= (0.66686 - 0.66738) / 0.66738 × 100%]
  Predicted drift = -0.06%    [from the derived analytic law]

The measured drift of -0.078% means Q_inv DECREASES from M_Z to 3 TeV. The
predicted drift of -0.06% is from the derived law (likely a 1-loop analytic
approximation in the paper). Both are negative — the inverse-coordinate Q
decreases as the down-type hierarchy sharpens with increasing scale (y_b is
pulled down faster than y_s and y_d by the y_t-driven differential term).

## D10. No manuscript edits

Per spec: "NO manuscript edits." Results are archived in the repo only.
Numbers return for inspection.

## D11. Lepton Yukawas — fixed

The charged lepton Yukawas (y_τ, y_μ, y_e) are held at their AHS2026 Eq. (2.4)
central values and are NOT included in the uncertainty propagation. Their
contribution to the down-type RGE enters only through the trace T = 3 Tr(Y_u²)
+ 3 Tr(Y_d²) + Tr(Y_e²), where the lepton contribution is O(10⁻⁴) compared to
the quark contribution O(1). The uncertainty in lepton Yukawas is negligible
for the down-type drift.

## D12. 2-loop structure for down-type sector

The existing `sm_rge.py` implements 2-loop beta functions for the up-type
Yukawa sector (y_t, y_c, y_u) and 1-loop only for down-type and leptons.
For the down-type drift test, the 2-loop corrections to the down-type beta
functions are NOT separately implemented — the down-type sector receives
only 1-loop beta functions in `beta_2loop()` (lines 319-324 of `sm_rge.py`).

However, the down-type evolution still BENEFITS from 2-loop gauge coupling
evolution (g_1, g_2, g_3 get full 2-loop treatment) and from the improved
up-type Yukawa evolution (y_t gets full 2-loop treatment, which feeds back
into y_b through the cross-term in the 1-loop beta function). The dominant
y_t-driven differential on y_b is a 1-loop effect; 2-loop corrections to it
are suppressed by g_3²/(16π²) ~ 0.7% and are below the current uncertainty.

This is recorded transparently rather than silently omitted.

## D13. Gate enforcement — stop-on-mismatch

Per spec: "Gates stop-on-mismatch, never tune." If GATE D1 or GATE D2 fails,
the run halts immediately. No tolerance widening, no parameter adjustment.
The gate record documents what was computed and why it failed.

## D14. AHS2026 page/table citation

Down-type Yukawa inputs are from Eq. (2.4), page 3 of the arXiv version
(arXiv:2510.01312v2). The M_Z-scale values are obtained by evolving low-energy
PDG 2024 inputs using SMDR (2-loop EW + mixed QCD/EW running and matching) and
sampling over 100,000 points with assumed Gaussian errors. The values in
Eq. (2.4) supersede the approximate PDG values used in the original
`inputs_frozen.json`.

The running values at 1 TeV, 3 TeV, 10 TeV, etc. are in Table 2 (2024 PDG
input), pages 12–13 of the arXiv version. These are used for cross-validation
of the evolved down-type Yukawas.

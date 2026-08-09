# APPROXIMATION-ERROR BUDGET FOR THE UV DIFFERENCING — SPEC v1.0

**Self-contained. Never ask; conservative reading + ASSUMPTIONS.md.**
**Purpose:** replace the bracketed truncation term with a COMPUTED budget.

The differencing result Delta = |9Q_U^{2L} - 9Q_U^{1L}| = 8.05e-4 at 1e16 GeV
was obtained under stated approximations. Each is now switched on and MEASURED,
so the truncation term becomes differencing (+) approximation error, not a bracket.

## THE APPROXIMATIONS TO PRICE (each measured as a delta on 9Q_U at 1e16 GeV)

### A1 — DIAGONAL CKM
Rerun the 2-loop evolution with the full CKM matrix (PDG Wolfenstein values, cite)
in the Yukawa RGEs vs the diagonal-dominant treatment. The CKM enters at 2-loop
through:
- Gauge × F_D†F_D terms: (F_D†F_D)_ii = Σ_k |V_ki|² y_d_k² (CKM-rotated down-type
  Yukawa matrix in the up-type mass eigenbasis)
- Pure Yukawa χ₄(S) term: Tr{H†H, F_D†F_D} = 2 Σ_i y_u_i² (F_D†F_D)_ii
- Flavor-specific cross terms: (H†H F_D†F_D)_ii = y_u_i² (F_D†F_D)_ii,
  (F_D†F_D H†H)_ii = (F_D†F_D)_ii y_u_i²

PDG 2024 Wolfenstein parameters:
  λ = 0.22500, A = 0.814, ρ̄ = 0.155, η̄ = 0.353

CKM matrix to O(λ⁵) in the standard parameterization:
  V_ud = 1 - λ²/2 - λ⁴/8
  V_us = λ
  V_ub = A λ³ (ρ̄ - i η̄)
  V_cd = -λ + A² λ⁵ (1/2 - ρ̄ - i η̄)
  V_cs = 1 - λ²/2 - λ⁴/8 (1 + 4 A²)
  V_cb = A λ²
  V_td = A λ³ (1 - ρ̄ - i η̄)
  V_ts = -A λ² + A λ⁴ (1/2 - ρ̄ - i η̄)
  V_tb = 1 - A² λ⁴/2

Report delta_A1 = |9Q_U(full CKM) - 9Q_U(diagonal)|.

### A2 — SPECTATOR YUKAWAS AT 2 LOOP
The current baseline gives explicit 2-loop terms to up-type and down-type Yukawas
(per SPEC_DRIFT.md extension). Charged-lepton Yukawas run at 1-loop only.

Rerun with charged-lepton Yukawas carrying their full 2-loop terms from Luo & Xiao.
The 2-loop charged-lepton anomalous dimension (from Luo & Xiao PRD 67, 065019,
general gauge theory, then specialized to SM) has the same structure as the
up-type 2-loop with the following replacements:
- No SU(3) gauge terms (d₃ᵉ = 0)
- Gauge × F_L†F_L self-coupling terms replace gauge × H†H
- Pure Yukawa: F_L†F_L terms replace H†H terms
- Higgs quartic λ: same structure

Report delta_A2 = |9Q_U(full 2L spectator) - 9Q_U(baseline)|.

### A3 — STATIC lambda
lambda is held at 0.126 (static) in the baseline. Rerun with lambda evolved at
1-loop from the same M_Z boundary value λ(M_Z) = 0.126.

1-loop λ beta function (SM, MS-bar, SU(5) normalization for g₁):
  β_λ^{(1)} = 1/(16π²) × [24λ² + 12λ T - 6 Tr_H4 - 6 Tr_FD4 - 2 Tr_FL4
                          - 3λ(3g₂² + (3/5)g₁²)
                          + (9/8)g₂⁴ + (9/20)g₁²g₂² + (27/200)g₁⁴]

Source: Ford, Jack, Jones NPB 387 (1992) 373; cross-checked against Luo & Xiao.

λ enters the 2-loop Yukawa beta functions through:
  +(3/2)λ² - 6λ (H†H)_ii   (up-type)
  +(3/2)λ² - 6λ (F_D†F_D)_ii   (down-type)
  +(3/2)λ² - 6λ (F_L†F_L)_ii   (charged lepton)

Report delta_A3 = |9Q_U(running λ) - 9Q_U(static λ)|.

### A4 — GAUGE SECTOR ORDER
Both gauge and Yukawa sectors run at 2-loop in the baseline. No mismatch to price.
delta_A4 = 0. Reported as N/A with reason.

## COMBINATION

approximation_error = quadrature sum of the measured deltas:
  approx_err = sqrt(delta_A1² + delta_A2² + delta_A3² + delta_A4²)

Also report the linear sum as a conservative bound.

truncation_total = sqrt(Delta² + approximation_error²)

Then recompute the discriminator:
  overshoot = 0.042 over sqrt(parametric² + truncation_total²)

for:
  parametric = 0.010 (independent) and 0.014 (most conservative correlation).

Report the ratio for each:
  R_ind = 0.042 / sqrt(0.010² + truncation_total²)
  R_con = 0.042 / sqrt(0.014² + truncation_total²)

## GATES (stop on mismatch; NEVER widen, NEVER auto-pass by asserting a target)

### GATE A0 — BASELINE REPRODUCTION
The baseline rerun must reproduce the archived
  9Q_U^{1L} = 8.041858 and 9Q_U^{2L} = 8.041054 at 1e16 GeV
to 1e-5 before any variant is run. If not, STOP and print the mismatch.

### GATE A5 — VARIANT ISOLATION
Each variant must differ from baseline ONLY in the named approximation;
print a diff summary of what changed in the integrator for each run.
If a variant cannot be isolated, STOP and print for ruling.

## DELIVERABLE

### Table
| Approximation | delta on 9Q_U(1e16) | fraction of Delta |
|--------------|---------------------|-------------------|
| A1: Diagonal CKM | delta_A1 | delta_A1 / Delta |
| A2: Spectator 2L | delta_A2 | delta_A2 / Delta |
| A3: Static lambda | delta_A3 | delta_A3 / Delta |
| A4: Gauge sector order | 0 (N/A) | 0 |

### Combined
- approximation_error (quadrature) = ...
- approximation_error (linear sum) = ...
- truncation_total = sqrt(Delta² + approx_err²) = ...
- R_ind = 0.042 / sqrt(0.010² + truncation_total²) = ...
- R_con = 0.042 / sqrt(0.014² + truncation_total²) = ...

### Judgment
State plainly whether truncation_total is dominated by omitted orders (Delta)
or by omitted approximations (approx_err).

## ARCHIVE
Results → trunc-differencing/results/approx/
Commit + push. Print all, DONE. NO manuscript edits.

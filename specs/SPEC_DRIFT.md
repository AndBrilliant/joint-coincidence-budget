# DOWN-TYPE CONTROL: UNCERTAINTY-QUANTIFIED DRIFT TEST — SPEC v1.0

**Self-contained. Ambiguity: conservative reading, ASSUMPTIONS.md, continue — never ask.**

**Purpose:** Replace "sign and magnitude at the precision of the diagnostic" with a
quantified test: predicted vs measured down-type inverse-coordinate drift,
each with propagated uncertainty, and their consistency in sigma.

## INPUTS

Down-type Yukawas (y_d, y_s, y_b) at M_Z with 1-sigma intervals: extracted from
the AHS tabulation (AHS2026, arXiv:2510.01312v2, Eq. 2.4, 2024 PDG input):

| Parameter | Central | ±1σ |
|-----------|---------|-----|
| y_b | 1.630e-2 | 0.009e-2 |
| y_s | 3.06e-4 | 0.04e-4 |
| y_d | 1.54e-5 | 0.02e-5 |

Up-type + gauge inputs: as in specs/SPEC_TRUNC.md (frozen). From AHS2026 Eq. 2.4
(2024 PDG input):

| Parameter | Central | ±1σ |
|-----------|---------|-----|
| y_t | 0.967 | 0.004 |
| y_c | 3.56e-3 | 0.06e-3 |
| y_u | 7.04e-6 | 0.15e-6 |
| g_1 | 0.461228 | 0.000026 |
| g_2 | 0.65096 | 0.00004 |
| g_3 | 1.2123 | 0.0046 |

Existing integrator: trunc-differencing work products in `trunc-differencing/sm_rge.py`
— EXTEND it to the down sector (y_t-driven CKM-aligned differential term on y_b;
same 2-loop structure), do not rewrite from scratch.

## Q_inv DEFINITION

Q_inv[y_d, y_s, y_b] = Q(1/y_d, 1/y_s, 1/y_b)
where Q[x_1, x_2, x_3] = Σx_i / (Σ√x_i)²

This is the inverse-coordinate analog of Q_U: degree-zero homogeneous in the
inverse Yukawas. The flavor-universal anomalous dimension cancels identically.
The drift is carried by flavor-differential (y_t-driven) terms on y_b.

## DERIVED LAW

The analytic drift rate for Q_inv follows from the 1-loop beta functions.
Let z_i = 1/y_i. Then:

  dQ_inv/dt = Σ_i (∂Q/∂z_i) * dz_i/dt

where ∂Q/∂z_i = (R - S/√z_i) / R³, S = Σz_j, R = Σ√z_j, and
dz_i/dt = -z_i * (β_{y_i} / y_i).

The flavor-universal piece of β cancels by Euler's theorem. The leading
flavor-differential contribution is the y_t-driven term on y_b:

  β_{y_b}^{diff} = -(3/2) y_t² y_b / (16π²)   [1-loop]

giving dz_b/dt = +(3/2) y_t² z_b / (16π²).

The total predicted drift M_Z → 3 TeV is:

  ΔQ_inv^{pred} = ∫ dQ_inv/dt dt ≈ dQ_inv/dt|_{M_Z} × ln(3 TeV / M_Z)

keeping only the y_t-driven differential term. Reported as a percentage of Q_inv(M_Z).

## GATES (stop on mismatch, print, never tune)

### GATE D1: Control-value reproduction

From central inputs, reproduce the paper's published control values:

| Quantity | Target | Tolerance |
|----------|--------|-----------|
| Q_inv(M_Z) | 0.66738 | last quoted digit |
| Q_inv(3 TeV) | 0.66686 | last quoted digit |
| Measured drift | -0.078% | last quoted digit |
| Predicted drift (derived law) | -0.06% | last quoted digit |

### GATE D2: Correlation discipline

Uncertainty on the DRIFT must be computed by drawing each input universe ONCE
and evolving it (draw-once, correlated endpoints), never by combining independent
endpoint bands. Report both the endpoint bands and the drift band so the
correlation benefit is explicit.

## DELIVERABLE

- **predicted drift:** central ± 1σ (propagating y_t, y_b, α_s and any input
  the law depends on; state the budget).
- **measured drift:** central ± 1σ (draw-once propagation of tabulated input
  errors through the evolution).
- **Consistency:** |pred - meas| in units of the combined sigma.
- N ≥ 1e5 draw-once universes, seed 20260811, archived under
  `downtype-drift/results/20260811/` with ASSUMPTIONS.md.
- Commit + push (gh token fallback as in SPEC_TRUNC).
- **NO manuscript edits** — numbers return for inspection.

## OUTPUT

Print: both drifts with bands, the sigma consistency, gate records, artifact
paths, DONE.

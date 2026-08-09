# SPEC: CO-AREA SURFACE MEASURE — T1 AS DERIVED CONDITIONAL — v1.0
# Never ask; conservative reading + ASSUMPTIONS.md. Gates stop-on-mismatch.
# Context: T1 is currently a SPECIFIED sheet measure (log-uniform in (m3, m1/m3)).
# An external audit noted this is a choice, not the conditional of T0 on Q=2/3.
# Purpose: derive the induced conditional and test whether the bound depends on it.

## §C1. DERIVATION — CO-AREA FORMULA IN LOG-COORDINATES

### C1.1 Setup

T0 null: three lepton masses drawn iid log-uniform on [a, b] where a=0.3 MeV,
b=2000 MeV. After sorting ascending: m₁ < m₂ < m₃.

In log-coordinates yᵢ = ln mᵢ, the T0 joint density on the sorted simplex is:
  ρ(y₁, y₂, y₃) = 6/L³  for ln a ≤ y₁ < y₂ < y₃ ≤ ln b,
where L = ln(b/a) ≈ 8.8049.

The Koide function:
  Q(m₁,m₂,m₃) = (m₁+m₂+m₃) / (√m₁+√m₂+√m₃)².

### C1.2 Gradient in log-coordinates

Let sᵢ = √mᵢ, S₁ = Σ sᵢ, S₂ = Σ sᵢ² = Σ mᵢ.

In mass coordinates:
  ∂Q/∂mᵢ = (sᵢ·S₁ − S₂) / (sᵢ·S₁³).

In log coordinates (the coordinates of the T0 measure):
  ∂Q/∂yᵢ = mᵢ · ∂Q/∂mᵢ = sᵢ · (sᵢ·S₁ − S₂) / S₁³.

Gradient magnitude squared:
  |∇_y Q|² = Σᵢ [sᵢ·(sᵢ·S₁ − S₂) / S₁³]²
           = (S₁²·S₄ − 2·S₁·S₂·S₃ + S₂³) / S₁⁶

where S₃ = Σ sᵢ³, S₄ = Σ sᵢ⁴.

This derivative is taken in LOG-COORDINATES — the same coordinates used for
the T0 measure. If one used Cartesian (mass) coordinates, the Jacobian from
log to Cartesian would differ, and the induced surface measure would NOT be
the T0 conditional. This is stated explicitly here and in ASSUMPTIONS.md.

### C1.3 Co-area formula

For a density ρ on ℝ³ and a smooth function H: ℝ³ → ℝ, the co-area formula:
  ∫ ρ(y) d³y = ∫ [∫_{H(y)=c} ρ(y)/|∇H(y)| dS_c(y)] dc,

where dS_c is the 2D surface-area element on the level set H = c.

Applying this with H = Q and c = 2/3:

The conditional distribution on the level set {Q = 2/3} induced by the T0
density ρ has surface density proportional to:
  ρ(y) / |∇_y Q(y)|  (with respect to dS).

Since ρ(y) = 6/L³ is constant on the simplex, the surface measure simplifies to:
  p_coarea(y) ∝ 1 / |∇_y Q(y)|  on {Q = 2/3} ∩ simplex,
  zero elsewhere.

### C1.4 Comparison with the specified T1 measure

The specified T1 sampler draws (m₃, r=m₁/m₃) log-uniform and solves for m₂
from Q(m₁,m₂,m₃) = 2/3. This parameterizes the SAME surface {Q = 2/3} but
with a DIFFERENT measure — the specified measure is uniform in (ln m₃, ln r),
not proportional to 1/|∇Q|.

The co-area conditional and the specified measure agree iff |∇_y Q| is
constant over the surface. We do NOT assume this — we compute it.

## §C2. EPSILON-SHELL VERIFICATION METHOD

### C2.1 Rationale

The co-area conditional can be directly approximated by sampling from T0
and restricting to a narrow band around the Koide surface:

1. Draw N_T0 samples from the T0 prior (log-uniform, sorted).
2. Keep those with |Q(m) − 2/3| < ε.
3. For each kept sample, compute surface coordinates (e.g., r = m₁/m₃,
   ln m₃).
4. As ε → 0, the distribution of these coordinates approaches the
   co-area conditional (proof: the shell volume element dV ≈ dS · 2ε/|∇Q|,
   cancelling the 1/|∇Q| factor to produce the correct surface measure).

### C2.2 Convergence test

Run at ε ∈ {1e-2, 3e-3, 1e-3, 3e-4, 1e-4} with T0 sample size chosen to
yield ≥ 1000 shell survivors at the smallest ε. Verify the scalar test
statistic (median log₁₀(r)) converges to a stable limit. Report the
spread across the three smallest ε as a systematic uncertainty.

### C2.3 Alternative: MCMC on the surface

As a cross-check, implement a Metropolis-Hastings sampler directly on the
surface {Q = 2/3} with target density 1/|∇Q|. The T1 specified sampler
serves as the proposal. Accept with probability:
  min(1, [|∇Q|_proposal / |∇Q|_current] × [ρ_specified(proposal) /
   ρ_specified(current)])

where the ratio of ρ_specified accounts for the Jacobian of the proposal
distribution. This MCMC chain produces samples distributed as the co-area
conditional without any ε-shell approximation.

### C2.4 Scalar test statistic

Primary: median of log₁₀(r) = median of log₁₀(m₁/m₃) on the sheet.
Secondary: median of log₁₀(m₃) on the sheet.
Both are one-number summaries sensitive to shifts in the surface measure.

## §C3. GATES

### GATE C1: Agreement gate

The epsilon-shell limit (ε → 0) and the MCMC co-area sampler must agree
with each other within Monte Carlo error on the scalar test statistic.

Report: median log₁₀(r) from both methods with MC uncertainties.
Pass if: |median_shell − median_mcmc| ≤ 2 × max(σ_shell, σ_mcmc).

The specified T1 sampler is then compared against the co-area result:
Report: median log₁₀(r)_specified and whether it falls within 2σ of the
co-area median. If it does NOT, the sheet measures differ (the expected
finding; this is the point of the exercise).

### GATE C2: Support gate

The co-area conditional must contain the observed lepton spectrum.
Verify: r_obs = m_e/m_τ = 2.876e-4 is within the sampled range of r on
the co-area sheet.
Pass if: the minimum sampled r ≤ r_obs ≤ maximum sampled r.
FAIL = VOID by construction.

## §C4. PRODUCTION

### C4.1 Co-area T1 Tier-1 cascade

Run the T1 headline cell (menu, logU, U1-menu variant) under the co-area
conditional at N_eff = 2e9 accepted worlds (or the largest N that fits
within available compute, with the bound stated for that N).

Co-area sampler implementation:
- Use the T1 specified sampler as a proposal distribution
- Apply Metropolis-Hastings acceptance to target the co-area distribution
  (1/|∇Q| on the surface)
- The autocorrelation length of the MCMC chain determines the effective
  sample size
- Target: N_eff ≥ 2e9 ACCEPTED MCMC steps (after burn-in and thinning)

If the MCMC chain is too inefficient, fall back to:
- Direct rejection with importance sampling: draw from T1 specified,
  weight by 1/|∇Q|, and resample (sampling importance resampling, SIR)

### C4.2 Comparison table

Compare three T1 constructions side by side:

| Construction | Measure description | Zero-hit CP95 bound | Median log₁₀(r) |
|-------------|---------------------|---------------------|------------------|
| Specified sheet | logU(m₃, m₁/m₃) on {Q=2/3} | from v0.5 results | from v0.5 |
| Alt sheet | logU(m₂, m₁/m₂) on {Q=2/3} | from v0.5 results | from v0.5 |
| Co-area derived | T0 conditional via 1/|∇Q| on {Q=2/3} | THIS WORK | THIS WORK |

State plainly: "The joint conclusion [is / is not] sheet-dependent at the
precision of this study."

## §C5. DELIVERABLES

1. specs/SPEC_COAREA.md — this file, committed.
2. scripts/coarea_engine.py — full implementation.
3. results/coarea/ — all output artifacts.
4. Updated ASSUMPTIONS.md with co-area derivation details.
5. Comparison table printed to stdout and saved as results/coarea/comparison.json.
6. All gates recorded (C1, C2).
7. Git commit and push (tag v1.0-coarea).
8. DONE. NO manuscript edits.

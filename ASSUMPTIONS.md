# ASSUMPTIONS.md — Joint Coincidence Budget (P3+P4)

**ENGINE_ID:** amb
**Latest spec version:** v0.3 (audit-response revision)
**Archived spec versions:** v0.5-freeze, v0.4, v0.3, v0.2
**Production seed:** SHA-256 derived per cell from spec text (see specs/SPEC_V03.md)

## v0.3 Changes — Audit-Response Revision + T1 Rerun (2026-08-09)

### A43. v0.3: Accepted-N semantics — root-cause fix

External audit identified that v0.2 T1 cells counted REJECTED sheet proposals
in N_eff. The T1 Koide-sheet sampler has an internal rejection rate (~20.8%):
draws where the Koide quadratic has no valid solution, where m2 falls outside
(m1, m3), or where the mass hierarchy m3/m1 > HIERARCHY_MIN fails. In v0.2,
these rejected proposals were included in N_eff, making L1@T1 read 0.792
(= acceptance rate) rather than 1.000 (= granted on sheet by construction).

**Fix:** The sampler now generates exactly N ACCEPTED worlds per cell.
Rejection is internal and tracked separately. N_eff = N accepted worlds,
unambiguous. L1@T1 = 1.000000 exactly (verified at G2). The acceptance
rate is reported per cell as a separate diagnostic.

Analytic acceptance rate: ln(0.0147/1e-5) / ln(1e4) = 0.79197.
The 0.0147 threshold comes from the hierarchy filter m3/m1 > (4+√18)² ≈ 67.9:
with m3/m1 = 1/r, this requires r < 1/67.9 ≈ 0.0147. Since r ~ logU[1e-5, 1e-1],
this rejects the upper ~20.8% of the r-range.

### A44. v0.3: T1 described as standalone Koide-sheet measure

T1 is now documented as "a separately specified Koide-sheet measure, log-uniform
in (m3, m1/m3)" — never as 'T0 conditioned on Koide'. The two priors (T0 and T1)
are distinct: T0 draws three independent log-uniform masses, T1 draws from a
2-parameter Koide-sheet subspace parameterized by (m3, m1/m3) with m2 determined
by the Koide equation.

Support behavior: m1 can fall below the T0 floor (0.3 MeV) because m1 = r×m3
with r as low as 1e-5 and m3 as low as 0.3, so m1 ∈ [3×10⁻⁶, 2000] MeV.
The T0 floor of 0.3 MeV is a property of the T0 prior, not of Nature, and the
Koide sheet naturally extends below it.

### A45. v0.3: 20-target U1 menu (1/3 included)

v0.2 used 19 irreducible fractions p/q with p,q ≤ 9 in (1/3, 1] — the
half-open interval excluded 1/3. v0.3 adds (1,3) for a 20-target closed
menu matching uptype-pinning v2.0. The full list: 1/3, 1/2, 2/3, 3/4,
2/5, 3/5, 4/5, 5/6, 3/7, 4/7, 5/7, 6/7, 3/8, 5/8, 7/8, 4/9, 5/9, 7/9, 8/9, 1/1.
Enforced by assertion in the engine.

### A46. v0.3: True truncated normal for logN prior

v0.2's 'logN' prior sampled log-masses from a normal distribution with
np.clip() truncation. This is a censored normal, not a truncated normal.
v0.3 uses scipy.stats.truncnorm for proper truncated normal in log space,
with bounds [log(lo), log(hi)], location at the log-midpoint, and scale
1.5×log(10). The old clip-based variant is retained under the name
'censored' for backward compatibility.

The difference is small in practice: with a scale of 3.45 log-units spanning
a range of 8.80 log-units, the truncation cuts ~0.5% of probability mass
at each tail. The truncnorm properly renormalizes; the old clip variant
does not.

### A47. v0.3: |log(r/r0)| <= eps for L2/L3 criterion

The frozen spec (§3) states the criterion as |log(r/r0)| ≤ ε. v0.2
implemented |r/r0 − 1| ≤ ε instead. For ε = 1e−5 and 2.1e−5, the two
forms differ by ∼ε²/2 ≈ 5e−11, which is negligible against the tolerances.
Nevertheless, v0.3 implements the exact spec form. The r-intersection
function for T1 lepton-block probability also uses the log form.

Recorded here so the change is explicit. The effect on singletons and
cascade rates is indistinguishable at the precision of this study.

### A48. v0.3: Alt-sheet T1 robustness variant

One alternative T1 sheet parameterization is included at reduced N=2e8:
log-uniform in (m2, m1/m2), solving the Koide equation for m3 (plus branch
of the quadratic, giving m3 > m2). This tests whether the parameterization
choice (m3 as the free mass vs m2) affects the joint budget. Reported
alongside the standard parameterization.

### A49. v0.3: SHA-256 seed derivation

All per-cell seeds are derived from SHA-256(spec_text + cell_name).
The spec text is the contents of specs/SPEC_V03.md. This makes seeds
deterministic, auditable, and tied to the exact spec version — any
change to the spec produces different seeds. Recorded in MANIFEST.json.

### A50. v0.3: T0 cells NOT rerun

Per the spec, T0 cells are unaffected by the accepted-N change (T0 draws
have no internal rejection; every draw is within the prior range so the
acceptance rate is identically 1.0). v0.2 T0 results are carried forward
as archived. Only T1 cells are rerun.

### A51. v0.3: v0.2-v0.5 results preserved

All v0.1-v0.5 result directories, tags, and data remain untouched per §10
void discipline. v0.3 results go into results/amb-v0.3/. v0.2 T1 cells
remain archived as the pre-audit reference state.

## v0.5 Changes — Anchor Averaging and Stage-Local Inflation

### A28. v0.5: Root cause of v0.4 +Q2 7.22σ fail IDENTIFIED and FIXED

The v0.4 +Q2 failure was caused by evaluating the quark-block probability
at the OBSERVED lepton anchors only, rather than averaging over the full
lepton-window anchor distribution. At factor 100, the lepton window is
wide enough that anchor variation (mu_star, twome varying across the window)
produces a 3.6× spread in P(quark | lepton). Using a single point (observed
leptons) underestimates by 3.6×, causing the 7.22σ discrepancy.

**Fix (a): Anchor averaging.** At every calibration stage, the analytic
side must average the quark-block probability over >= 1e4 lepton draws
from within the SAME inflated lepton window used by brute force, with
anchors (mu_star, twome) recomputed per draw. This eliminates the
observed-anchor shortcut.

### A29. v0.5: Stage-local inflation

v0.4 inflated ALL claims in a stage uniformly. v0.5 inflates ONLY the
claims NEW to the stage under test. The lepton window runs at the smallest
factor giving >= 100 lepton-stage survivors in brute force (found once at
the first stage and carried forward). New-claim inflation factors are then
tested independently.

This prevents the "window inflation race" where both lepton and quark
windows inflate together, producing vacuous lepton windows before quark
validation can complete.

### A30. v0.5: Compositional eligibility for +U1

If no direct inflation factor for +U1 reaches 100 hits non-vacuously,
the stage is validated compositionally via:
  (i) P(U1) singleton integral at factor 1 — brute rate ~3.4e-3,
      trivially sampled with ~34k hits per 1e7 draws.
  (ii) mu-coupled Q2^U1 pair at adaptive stage-local factor — find
      smallest factor in {1,3,10,30,100,300,1000} giving >=100 Q2^U1 hits
      from lepton-stage survivors.
Both must be within 2 Poisson sigma of the anchor-averaged analytic estimate.

If both pass: VALIDATED-BY-COMPOSITION, labeled as such.

### A31. v0.5: U1 menu targets confirmed (19)

Re-verified enumeration from v0.2: irreducible p/q with p,q <= 9 in (1/3, 1]
yields exactly 19 targets as stated in the spec.

### A32. v0.5: Tier-1 cascade — all cells expected BOUND

Based on singleton measurements at N=1e7:
- T0 L2 singleton rate: 8/10M = 8e-7
- T1 L2 singleton rate: 18/10M = 1.8e-6

These are the rarest claims. At N=2e9, expected L2 survivors:
- T0: 8e-7 × 2e9 = 1600
- T1: 1.8e-6 × 2e9 = 3600

After downstream filtering (L1, L3, Q1, Q2, U1), expected joint hits << 100
for all cells. All Tier-1 cells will be BOUND (CP95 upper bounds only, no
point estimates). This is expected and accepted per the spec.

### A33. v0.5: Anchor averaging implementation

For each calibration stage with lepton-stage factor f_lep and new-claim factor
f_new, the analytic estimate is built as:

1. Compute P(lepton stage at f_lep) analytically using the verified density
   (T0) or r-intersection (T1).
2. Draw >= 1e4 lepton samples from within the inflated lepton window at f_lep.
3. For each lepton sample, compute P(new claims at f_new | lepton) via MC
   with N_quark_per_lepton = 50000 draws.
4. Average the quark-block probabilities across all lepton samples.
5. Tier-2 analytic = P(lepton stage) × avg[P(quark | lepton)].

The brute force side mirrors this: draw from full prior, filter by lepton
stage at f_lep, then from survivors check new claims at f_new.

Both sides use the SAME lepton window factor (stage-local), ensuring the
anchor distribution matches.

### A34. v0.5: v0.1-v0.4 results preserved

Per spec §10, all v0.1-v0.4 result directories, tags, and data remain
untouched. v0.5 results go into `results/amb-20260811-v0.5/`.

### A35. v0.5: Informational cross-run regression

At production seed 20260811 with the §5 draw order, T0_fixed_logU is expected
to reproduce v0.1's L2 stage count. v0.1 T0 L2 = 2081/2e9 ≈ 1.041e-6.
We report match or mismatch per §10, do not gate on it.

## v0.4 Changes — Density Root-Cause and Partial Fix

### A23. v0.4: Root cause of v0.3 factorized-estimate failure IDENTIFIED

The v0.3 Tier-2 lepton-block integration for T1 (Koide-conditioned sheet)
used a 2M-point log-spaced r-grid over 9.21 decades. The L2∧L3 window in
ln(r) is only ~1.8e-5 wide, yielding only ~3 grid points in the window.
This caused a 24% discretization error in Δln(r), which propagated to a
22% underestimation of P(L2∧L3) (1.500e-6 analytic vs 1.925e-6 brute force).

**Fix:** Replaced the coarse grid with high-resolution bisection to find
the exact r-range satisfying both L2 and L3. The corrected P(L2∧L3) =
1.981e-6 matches brute force (204/1e8 = 2.040e-6) to within 0.41σ.

### A24. v0.4: Density derivation and unit test

The exact joint density of (u,v) = (ln m2/m1, ln m3/m2) for three iid
log-uniform draws sorted ascending on [a,b] with L = ln(b/a):

  f(u,v) = 6/L³ × max(0, L-u-v)  for u ≥ 0, v ≥ 0, u+v ≤ L.

Derivation: f(y1,y2,y3) = 6/L³ for sorted log-values. Transform to
(u,v,y1): Jacobian = 1. Integrate out y1 over [ln a, ln b - u - v]:
  f(u,v) = 6/L³ × (L-u-v) for u+v ≤ L.

Verified against brute force: 9/10 test rectangles passed within 2σ at
1e8 draws. Rectangle #1 failed at 2.30σ (seed 20260811) — confirmed as
statistical fluctuation by cross-check with seed 271828 (0.02σ).

Density GATE: PASSED.

### A25. v0.4: Per-stage validation results

| Stage | Best Factor | T2 Analytic | BF Rate | σ | Within 2σ? |
|-------|------------|-------------|---------|---|------------|
| L2∧L3 | 1 | 1.981e-6 | 1.925e-6 ± 0.191e-6 | 0.30σ | ✓ PASS |
| +Q1 | 30 | 1.841e-7 | 1.931e-7 ± 0.193e-7 | 0.46σ | ✓ PASS |
| +Q2 | 100 | 1.586e-7 | 5.714e-7 ± 0.571e-7 | 7.22σ | ✗ FAIL |
| +U1 | — | — | — | — | INELIGIBLE |

The density fix resolved the v0.3 failures in L2∧L3 and +Q1. The +Q2
failure at 7.22σ is a factorization breakdown of a different nature:
at factor 100, the lepton window is wide enough that the anchor
distribution (mu_star, twome varying across the window) significantly
affects the quark-block probability. Using observed leptons alone
underestimates P(quark | lepton) by 3.6× relative to the proper
lepton-window-averaged value.

Per spec §v0.4(d): "If any stage still fails at 2 sigma, report FAIL with
the stage numbers and stop — do not iterate the math past the freeze."

**Outcome:** +Q2 stage FAILED at 7.22σ, +U1 INELIGIBLE. Tier-2 point
estimates remain UNVALIDATED. Tier-1 bounds remain the headline.

### A26. v0.4: T1 lepton block — corrected P(L2∧L3)

v0.3 gave P(L2∧L3 | T1) = 1.60e-6 (coarse grid).
v0.4 corrected: P(L2∧L3 | T1) = 1.981e-6 (high-res bisection).
The 24% increase matches the direction and magnitude of the v0.3
calibration discrepancy.

### A27. v0.4: no further iteration

The spec freeze prohibits iterating the math past the freeze. The +Q2
failure is reported as-is. The correct fix would involve integrating
P(quark | lepton) over the full lepton-window anchor distribution,
not evaluating at observed leptons alone. This is NOT a density error
(the lepton density is now verified to ±0.02σ) — it is a factorization
approximation error in the quark block.

## v0.3 Changes (see VOID.md for v0.1 T1 void declaration)

### A0. v0.3 key changes
1. Tier-2 redesigned as exact block integration: P(lepton block) × P(quark block | lepton draw).
2. Lepton block: 2D quadrature over (u,v) shape space with analytic s-integration.
   L1 overlap with L2∧L3 window: 95.6% (T0).
3. Quark block: integrated over anchor distribution (m3 for T1, s for T0) with
   100-point grid × 200-300k MC draws per grid point.
4. Calibration factors: {1, 3, 10, 30, 100, 300, 1000} with non-vacuous gate
   (>50% prior mass excluded per claim per stage).
5. Validation: Tier-2 vs brute force within 2 Poisson sigma per stage.
   FAILED at all stages — factorized estimates overestimate by 1.3-2.6×.
   +U1 stage INELIGIBLE (vacuous before 100 hits reached).
6. v0.2 plateau resolved: v0.2 showed identical 140-hit counts across all
   stage-pairs at ×1000 (artifact of no per-stage variability). v0.3 shows
   distinct best factors per stage (1, 30, 100, INELIGIBLE).

## v0.2 Changes (see VOID.md for v0.1 T1 void declaration)

### A0. v0.2 key changes
1. T1 r-range [1e-5, 1e-1] (was [1e-3, 1e-1]) — fixes voided v0.1 T1 cells.
2. Support gate verified: r_obs = 2.876e-4 ∈ [1e-5, 1e-1], m3/m1 = 3477.4 > 67.9. PASSED.
3. Tier-2 calibration redesigned to adaptive per-stage inflation per v0.2 spec §8.
4. All v0.1 T0 cells remain valid. v0.1 T1 cells voided by support exclusion.

## Conservative Readings and Implementation Decisions

### A1. Single-machine execution
The spec (§10) declares "Single-machine assumption: gh is authenticated.
Nothing else assumed." All computation runs on ganymede (single machine).
No distributed execution.

### A2. N_eff definition
N_eff is the total number of draws generated (before any rejection).
For T0: N_eff = batch_size * num_batches.
For T1: N_eff counts draws from the sheet sampler before the mass-hierarchy
rejection (the mathematical inadmissibility filter); draws rejected by the
hierarchy condition do not count toward N_eff.

### A3. Cascade ordering
Per §8 Tier-1, singletons are measured first at N=1e7, then the cascade
is ordered by measured rarity. The rarest claim becomes the first filter.
For the T1 cells, L1 is granted (identically satisfied), so the cascade
begins with L2, L3, then Q1, Q2, U1.

### A4. Frame union implementation
Per §3b, frame freedom is priced by same-draw union. For each null world,
we evaluate the claim in BOTH frames (direct and inverse) and count a hit
if EITHER frame satisfies the claim. This applies to L1 (coordinate menu)
and U1-menu (direct/inverse union).

### A5. U1-menu: 19 targets
Per §4 U1 row, the VARIANT grants any irreducible p/q with p,q <= 9 in
(1/3, 1]. The irreducible fractions are:
1/2, 1/3, 2/3, 1/4, 3/4, 1/5, 2/5, 3/5, 4/5, 1/6, 5/6, 1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 1/8, 3/8, 5/8, 7/8, 1/9, 2/9, 4/9, 5/9, 7/9, 8/9
Wait — that's more than 19. Let me enumerate properly.

Irreducible p/q with p,q <= 9, in (1/3, 1]:
1/2, 1/3, 2/3, 1/4, 3/4, 1/5, 2/5, 3/5, 4/5, 1/6, 5/6, 1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 1/8, 3/8, 5/8, 7/8, 1/9, 2/9, 4/9, 5/9, 7/9, 8/9

That's 27 fractions. But the spec says "19 targets". Let me re-read...

"any irreducible p/q, p,q<=9, in (1/3,1] (19 targets, same tol on 9Q about each 9*(p/q))"

Hmm, 19 targets with p,q <= 9 in (1/3, 1]. Let me recount carefully.

Actually wait - the spec says "19 targets" explicitly. Let me trust the spec's count of 19 and enumerate to verify:

Irreducible fractions p/q with p,q in [1,9], p/q in (1/3, 1]:
- q=1: 1/1 = 1
- q=2: 1/2
- q=3: 1/3, 2/3
- q=4: 1/4, 3/4
- q=5: 1/5, 2/5, 3/5, 4/5
- q=6: 1/6, 5/6
- q=7: 1/7, 2/7, 3/7, 4/7, 5/7, 6/7
- q=8: 1/8, 3/8, 5/8, 7/8
- q=9: 1/9, 2/9, 4/9, 5/9, 7/9, 8/9

That's 1+1+2+2+4+2+6+4+6 = 28. 

But wait, the spec says "(1/3, 1]" exclusive of 1/3. So 1/3 is excluded.

Actually re-reading: "in (1/3,1]" — this is half-open, excludes 1/3, includes 1.

So removing 1/3: 28-1 = 27.

But spec says 19. Hmm. Let me re-read more carefully...

"any irreducible p/q, p,q<=9, in (1/3,1] (19 targets, same tol on 9Q about each 9*(p/q))"

Maybe the count of 19 includes only fractions where the same tolerance window applies correctly? Or maybe there's a different interpretation.

Actually wait - the spec says p,q <= 9 and in (1/3, 1]. Let me re-examine: maybe it means q <= 9 and p <= 9 but also p/q > 1/3 and p/q <= 1.

Let me just enumerate them programmatically and see if I get 19.

Actually, maybe "19 targets" is the spec's own count and I should verify it. Let me just enumerate computationally and trust what I find. If it's not 19, I'll record the discrepancy in ASSUMPTIONS.md and use the correct enumeration.

Let me actually count more carefully.

p,q ∈ {1,...,9}, gcd(p,q)=1, 1/3 < p/q ≤ 1:

q=1: p=1 → 1/1 ✓
q=2: p=1 → 1/2 ✓
q=3: p=1 → 1/3 (excluded, not > 1/3), p=2 → 2/3 ✓
q=4: p=1 → 1/4 (≤1/3, excluded), p=2 → 2/4=1/2 (already counted), p=3 → 3/4 ✓
q=5: p=1 → 1/5 (excluded), p=2 → 2/5 ✓, p=3 → 3/5 ✓, p=4 → 4/5 ✓
q=6: p=1 → 1/6 (excluded), p=2 → 1/3 (excluded, not > 1/3), p=3 → 1/2 (dup), p=4 → 2/3 (dup), p=5 → 5/6 ✓
q=7: p=1 → 1/7 (excluded), p=2 → 2/7 (excluded, ≤1/3), p=3 → 3/7 ✓, p=4 → 4/7 ✓, p=5 → 5/7 ✓, p=6 → 6/7 ✓
q=8: p=1 → 1/8 (excluded), p=2 → 1/4 (excluded), p=3 → 3/8 ✓, p=4 → 1/2 (dup), p=5 → 5/8 ✓, p=6 → 3/4 (dup), p=7 → 7/8 ✓
q=9: p=1 → 1/9 (excluded), p=2 → 2/9 (excluded, ≤1/3), p=3 → 1/3 (excluded), p=4 → 4/9 ✓, p=5 → 5/9 ✓, p=6 → 2/3 (dup), p=7 → 7/9 ✓, p=8 → 8/9 ✓

So: q=1:1, q=2:1, q=3:1, q=4:1, q=5:3, q=6:1, q=7:4, q=8:3, q=9:4 = 19.

Yes, 19 targets confirmed! I was miscounting earlier by not properly checking > vs ≥ for 1/3 and by not deduplicating.

Wait, let me double-check:
q=1: 1/1 → count 1
q=2: 1/2 → count 1
q=3: 2/3 → count 1
q=4: 3/4 → count 1
q=5: 2/5, 3/5, 4/5 → count 3
q=6: 5/6 → count 1
q=7: 3/7, 4/7, 5/7, 6/7 → count 4
q=8: 3/8, 5/8, 7/8 → count 3
q=9: 4/9, 5/9, 7/9, 8/9 → count 4

Total: 1+1+1+1+3+1+4+3+4 = 19. ✓

OK so the 19 targets are correct. I'll enumerate them in the code.

### A6. Priority of gates
The golden regression (§6) MUST pass before any production code runs. The
observed-side checks (§7) MUST pass before production code runs. If either
fails, STOP — do not tune, report the mismatch.

### A7. No hierarchy-null cell
Per §10: "Rank-ordering, if ever used, applies ONLY to a hierarchy-preserving
null, never to the log-uniform null (no hierarchy-null cell exists in this
spec; if you believe one is needed, record it in ASSUMPTIONS.md and proceed
without it)." No hierarchy-null cell is needed; proceeding without one.

### A8. Machine precision
Where the spec uses tolerances like 1e-5 or 2.10e-5, these are handled as
relative tolerances on ratios. Float64 precision is adequate throughout.

### A9. Batch memory
Batch size 1e6 with float64 arrays: ~8 MB per mass array, ~24 MB per triple.
Well within ganymede's RAM. Larger effective sample sizes are achieved by
iterating batches.

### A10. Runtime estimate
Primary cells: N_eff >= 2e9, batch size 1e6 → 2000 batches per cell.
4 primary cells (T0×U1-fixed×P-logU, T0×U1-menu×P-logU, T1×U1-fixed×P-logU,
T1×U1-menu×P-logU) → ~8000 batches. At ~0.5s/batch (vectorized NumPy),
roughly 1-2 hours per primary cell. Variant cells add ~20% overhead.
Total runtime estimate: 8-16 hours. This is a conservative reading — actual
runtime may be longer if cascade filtering is expensive.

### A11. CP95 implementation
Clopper-Pearson intervals computed via scipy.stats.beta:
lower = beta.ppf(alpha/2, k, n-k+1)
upper = beta.ppf(1-alpha/2, k+1, n-k)
with alpha=0.05 for 95% confidence.

### A12. T1 sheet sampler — minus branch
The Koide equation Q(m1,m2,m3)=2/3 with m1 and m3 known yields a quadratic
in sqrt(m2). The "minus branch" is the smaller root (the one that places m2
between m1 and m3 in mass). Verified against the standard Koide formula.

### A13. Q_U inverse coordinate
Per §3d, the inverse coordinate for U1 is Q(1/m), i.e., evaluate Q_U on
(1/mu, 1/mc, 1/mt). This is the frame union for U1-menu: a hit if either
direct Q_U or inverse Q_U satisfies the tolerance.

### A14. Running time constraint
Given the N_eff >= 2e9 requirement and available compute (ganymede CPU only,
GPU passed through to Windows VM), production runs will use vectorized NumPy
on CPU. Actual measured throughput is ~11.4M draws/s for T0 cells and
~5-6M draws/s for T1 cells (Koide sheet sampler overhead).

### A15. Calibration cell — Tier-2 validation infeasible
The calibration cell (all tolerances ×100, calibration seed 271828, N=1e8)
produced 0 joint hits. Stage breakdown: L2=10523 survivors, L3=3, L1=3,
Q1=0, Q2=0, U1=0. The claims are strongly anti-correlated in the null —
L2 survivors systematically fail L3, and the joint of L2∧L3∧L1 systematically
fails Q1. CP95 upper bound at calibration: 3.69e-8.

Since both brute-force Tier-1 and Tier-2 would give 0 hits at calibration,
the spec's validation criterion (within 2 Poisson sigma) cannot distinguish
them. Tier-2 is therefore not validated and not used. All cells use Tier-1
cascade only. This is the conservative reading required by the spec's
ambiguity rule.

### A16. Runtime update
Measured throughput is ~11.4M draws/s for T0, ~5-6M for T1. At these rates:
- Primary cells (N_eff=2e9): T0 ~175s, T1 ~360s
- Variant cells (N_eff=2e8): T0 ~18s, T1 ~36s
- Total: ~(2×175 + 2×360) + (4×18 + 4×36) ≈ 1070 + 216 ≈ 1286s ≈ 21 min
This is significantly faster than the initial A10 estimate (~8-16 hours).

### A17. v0.3: Tier-2 lepton block — L1 is NOT automatically satisfied
The v0.2 assumption that "within the L2∧L3 window, L1 is automatically satisfied
because the ratios determine kdist" was tested and found FALSE. At the corners of
the L2∧L3 tolerance rectangle (±1e-5 in u, ±2.1e-5 in v), kdist exceeds L1_TOL
(3.71e-6 > 3.30e-6 at two corners). The L1 overlap fraction is 95.6%, reducing
the T0 lepton-block probability from 4.81e-12 to 4.59e-12. This is a conservative
correction — the spec says "conservative reading" and the reduction is small (4.4%).

### A18. v0.3: Tier-2 quark block — anchor integration is essential
The v0.3 spec §8 says "Average over the lepton-block-conditioned anchor distribution
(sample >= 1e4 lepton draws from within the lepton windows; anchors vary negligibly
inside windows this narrow — verify and report the spread)." The "vary negligibly"
claim was verified and found FALSE for the T1 m3 dimension: mu_star varies by factor
~6,667× across m3 ∈ [0.3, 2000] MeV. The anchor spread is the dominant effect in
the quark-block integration, not negligible. The conservative reading is to perform
the full integration over the anchor distribution (100-point grid). The spec's
parenthetical "(anchors vary negligibly...)" is interpreted as a hypothesis to be
tested, not an assertion — it fails, and the full integration is performed instead.

### A19. v0.3: Tier-2 validation failure — accepted outcome
Per spec §8: "If no factor in the set reaches 100 hits for a stage-pair, Tier-2 is
ineligible for cells containing that pair and those cells are reported BOUND — an
accepted outcome, not a failure." The +U1 stage is INELIGIBLE — at factor 30,
expected hits ~8 (below 100 threshold); at factor ≥100, the U1/Q2 windows become
vacuous. The other three stages (L2∧L3, +Q1, +Q2) reached ≥100 hits but FAILED
the within-2σ validation. Per the spec, Tier-2 point estimates are reported but
labeled UNVALIDATED. The Tier-1 bounds remain the headline.

### A20. v0.3: v0.2 plateau artifact resolved
v0.2 calibration showed identical 140-hit counts across all four stage-pairs at
factor 1000, and the v0.2 spec identified this as a failure mode: "The v0.2 plateau
(identical counts across nested stages) is the failure mode this rule exists to
prevent." The v0.3 calibration with per-stage adaptive factors {1,3,10,30,100,300,1000}
and the non-vacuous gate correctly resolves this: each stage-pair finds a different
best factor (1, 30, 100, INELIGIBLE) with distinct hit counts (102, 100, 100, —).
The plateau is eliminated.

### A21. v0.3: Tier-1 results reused from v0.2
Per spec §10: "v0.2 cells remain VALID and their bounds are quoted alongside v0.3
point estimates." The Tier-1 cascade code paths are byte-identical between v0.2
and v0.3 (same draw functions, same claim checks, same cascade logic). The v0.3
singletons confirm identical rates. Full v0.3 Tier-1 cascade reruns are performed
for verification at N=1e7 (singletons) and the results match v0.2. The v0.2 Tier-1
bounds (all 12 cells BOUND at CP95 upper bound) are quoted as the headline.

### A22. v0.3: single-machine execution
All computation performed on ganymede (single machine). gh CLI authentication
via token at ~/.keys/github. No distributed execution.

## CRASH AND RECOVERY — 2026-08-08

### A36. Ganymede power loss during v0.5 Tier-2 validation

**Date:** 2026-08-08, approximately 14:33 UTC (between REPORT.md last write and
power loss).

**What happened:** Ganymede lost power mid-execution of the v0.5 spec. At the
time of the crash, the following work was complete:
- Tier-1 v0.5 cascade: ALL 12 cells computed, checkpointed, and summarized in
  `tier1_summary_v0.5.json` (commit d86349e).
- Singletons at N=1e7 for T0 and T1 logU: complete (`singletons.json`).
- REPORT.md: partially written — Tier-1 tables and singletons filled in,
  Tier-2 sections marked [TBD]/[TO BE FILLED].
- ASSUMPTIONS.md: v0.5 changes documented (A28-A35) but crash not yet recorded.
- Git: commit d86349e created, NOT pushed, NOT tagged.

**What was mid-run:** Tier-2 anchor-averaged validation had NOT yet produced
any output files. No `tier2_validation_*.json` files exist in
`results/amb-20260811-v0.5/`. The process was either initializing or
had not been launched yet when power was lost.

**What was NOT done at crash time:**
1. Tier-2 per-stage validation (anchor averaging, stage-local inflation,
   compositional +U1 eligibility)
2. Cross-run regression (T0_fixed_logU L2 stage count vs v0.1's 2081)
3. REPORT.md Tier-2 sections, headline result
4. Git tag v0.5-amb
5. Git push to origin

**Recovery actions (this session):**
1. Inventory complete: verified all Tier-1 checkpoints parse and counts are
   internally consistent. Tier-1 data preserved as-is — NO recomputation.
2. Tier-2 validation run from scratch for headline cell T1_menu_logU using
   `scripts/tier2_optimized_v0.5.py` (calibration seed 271828, N_max=1e9).
3. Cross-run regression computed.
4. REPORT.md completed with Tier-2 results.
5. Tag v0.5-amb applied and pushed.

**What was recomputed:** Tier-2 validation only. Tier-1 results, singletons,
golden regression, density test — all preserved from pre-crash state.

**Verification:** All checkpoint JSON files parse correctly. Tier-1 cell
counts match the summary. No file corruption detected.

### A37. Three defects in `tier2_optimized_v0.5.py` found during recovery

The pre-crash Tier-2 script could not have completed. The stall observed in
the crash log ("Drawing 10000 lepton samples for anchor averaging...") was
not slowness — it was an infinite-in-practice loop. Three defects:

1. **Hang (lines 275, 296).** `while len(lep_samples_list) < N_lep_samples`
   counts list CHUNKS, not samples. Each append adds ~1e6 draws but
   increments `len()` by 1, so the loop demanded 10,000 x 1e6 = 1e10 draws.
   The `np.vstack(...)[:N_lep_samples]` immediately after proves 10,000
   TOTAL was the intent. This is exactly where the pre-crash run was stuck.
2. **Unit mismatch (line 387).** `bf_rate = bf_hits / N_cached` is the
   CONDITIONAL rate P(new claim | lepton survivor) ~ 1e-1, while
   `t2_analytic = p_lep * p_quark_avg` is the JOINT ~ 1e-7. The comparison
   would fail by ~6 orders of magnitude independently of the physics.
   `joint_engine_v0.5.py` uses `bf_hits / bf_N` (total attempted draws),
   which is correctly the joint rate.
3. **Infeasible anchor loop.** `quark_block_anchored_batched` iterates
   10,000 leptons x 20 batches x 50,000 quarks x 38 U1-menu comparisons
   ~ 4e11 element-ops per factor.

Per void discipline the original script is PRESERVED unmodified. The
corrected implementation is `scripts/tier2_recovery_v0.5.py`.

### A38. Anchor averaging reformulated as a band statistic (exact, not approximate)

Defect (3) is fixed without weakening §v0.5(a). Both lepton-anchored claims
are one-sided bands in a single quark statistic:

    Q1: |ln(ms^2/(mu_star*md))| <= B1*f  <=>  |w1 - ln(mu_star)| <= B1*f,
        w1 = 2*ln(ms) - ln(md)
    Q2: |ln(mu^2/(md*twome))|  <= B2*f  <=>  |w2 - ln(twome)|   <= B2*f,
        w2 = 2*ln(mu) - ln(md)
    U1: depends only on the quark multiset {mu, mc, mt} — no lepton anchor.

So P(claim | anchor) is the fraction of a fixed quark pool whose statistic
lies in a window centred on that draw's OWN anchor. Sorting the statistic
once and running two binary searches per anchor returns the IDENTICAL
fraction in O(N log N). Anchors are still recomputed per lepton draw; the
observed-anchor shortcut that v0.5 exists to eliminate is not reintroduced.

Verified by unit test against the naive per-anchor MC over the same pool:
relative difference 0.000e+00 (exact agreement) for Q1 at f=1000, Q2 at
f=100, and U1_menu at f=1. The reformulation is the same computation
reorganized, not an approximation.

### A39. T0 anchor sampling must include the overall mass scale

The pre-crash script sampled the T0 lepton window with `m1 = 1` fixed. The
anchors mu_star = sum(lep) and twome = 2*min(lep) are extensive in the
overall scale, so fixing m1 collapses the very anchor distribution the
averaging exists to capture. The recovery script samples the two shape dofs
(u,v) inside the window and then draws ln(m1) uniformly over its admissible
range [ln a, ln b - u - v], which is the log-uniform prior conditioned on
the shape. (The headline cell is T1, where m3 was already sampled over the
full prior range, so this correction does not affect the reported result.)

### A40. v0.5 VERDICT — +Q2 stage FAILS at 2.39 sigma; Tier-2 remains UNVALIDATED

Headline cell T1 x U1-menu x P-logU, calibration seed 271828, N_max 1e9,
10,000 anchor draws, lepton-stage factor 1:

| Stage | Factor | T2 analytic | BF rate ± sigma | hits | dev | Verdict |
|-------|--------|-------------|-----------------|------|-----|---------|
| L2^L3 | 1      | 1.9813e-6   | 1.9510e-6 ± 4.417e-8 | 1951 | 0.68σ | PASS |
| +Q1   | 1000   | 2.0561e-7   | 1.9700e-7 ± 1.404e-8 | 197  | 0.61σ | PASS |
| +Q2   | 30     | 8.9368e-8   | 1.1500e-7 ± 1.072e-8 | 115  | 2.39σ | **FAIL** |
| +U1   | 1      | 2.6487e-7   | 2.8200e-7 ± 1.679e-8 | 282  | 1.02σ | PASS |

The chosen factor is the smallest non-vacuous one reaching >=100 brute-force
hits, per §8. For +Q2 that is factor 30 (115 hits), and it deviates by
2.39 sigma — outside the 2 sigma gate.

**Anchor averaging worked but was not sufficient.** The +Q2 discrepancy fell
from v0.4's 7.22 sigma to 2.39 sigma, a 3.0x reduction, closely matching the
3.6x anchor-spread effect diagnosed in A28. The measured anchor spread
confirms the mechanism: across the 10,000-draw window mu_star spans a factor
of 6659, and the per-anchor quark probability has relative standard deviation
0.855 (+Q1) and 0.4875 (+Q2). For +U1 the relative spread is 4.2e-16 —
numerically zero, exactly as expected since U1 carries no lepton anchor.

Residual direction: at factor 30 the analytic estimate UNDERSHOOTS brute
force by 29% (8.94e-8 vs 1.15e-7), while at factors 1, 3, 10 it overshoots
slightly. The sign flip with factor indicates remaining structure in the
md coupling between Q1 and Q2 that the single-band factorization does not
carry. Diagnosing it further would be iterating the math past the freeze.

Per §v0.5(d): "any residual fail => report FAIL with numbers and stop."
**Tier-2 point estimates are UNVALIDATED. Tier-1 bounds remain the headline.**

### A41. Conservative reading — "stop" halts Tier-2, not the deliverables

§v0.5(d) says to report FAIL and stop. The conservative reading taken here:
Tier-2 validation work stops at the headline cell — the remaining three
primary cells (T0 x fixed, T0 x menu, T1 x fixed) were NOT run, and no
Tier-2 point estimate is promoted for any cell. Running them could not
change the verdict, and continuing to search for a passing configuration
would be iterating past the freeze. The reporting deliverables (REPORT.md,
tag, push) are completed, since "stop" governs the estimator search and not
the artifact contract of §10.

### A42. +U1 validated directly — compositional route not needed

v0.4 recorded +U1 as INELIGIBLE. Under v0.5 stage-local inflation, +U1
reaches 282 brute-force hits at factor 1 with the window non-vacuous
(U1-menu covers 13.4% of prior mass, well under the 50% bar), and validates
at 1.02 sigma. The §v0.5(c) compositional path is implemented in the
recovery script but was not exercised, because the direct route qualified.
This is a genuine v0.5 improvement over v0.4, independent of the +Q2 failure.

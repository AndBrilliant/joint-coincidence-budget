# PREREGISTRATION — Joint Coincidence Budget (P3+P4)

**ENGINE_ID:** amb
**Production seed:** 20260811
**Prior-variant seed:** 314160
**Calibration seed:** 271828
**Execution spec version:** v0.5-freeze
**Timestamp (pre-first-batch v0.3):** 2026-08-08T18:00:00Z
**Timestamp (pre-first-batch v0.4):** 2026-08-08T21:00:00Z
**Timestamp (pre-first-batch v0.5):** 2026-08-08T23:00:00Z

## v0.5 Changes from v0.4

1. **Anchor averaging (§v0.5 a):** At every calibration stage the analytic side must
   average the quark-block probability over the SAME inflated lepton window brute
   force samples — >= 1e4 lepton draws inside that window, anchors recomputed per draw.
   The observed-anchor shortcut caused v0.4's +Q2 7.22σ fail (3.6× anchor-spread
   effect at factor 100).
2. **Stage-local inflation (§v0.5 b):** Inflate ONLY the claims new to the stage
   under test; lepton windows run at the smallest factor giving >= 100 lepton-stage
   survivors in brute force, with anchor averaging applied at that factor.
3. **Compositional eligibility for +U1 (§v0.5 c):** If no direct factor reaches 100
   hits non-vacuously, validate instead (i) P(U1) singleton integral at factor 1
   (brute rate ~3.4e-3, trivially sampled) and (ii) mu-coupled pair Q2^U1 at
   adaptive stage-local factor. Both within 2 Poisson sigma => VALIDATED-BY-COMPOSITION.
4. **Updated validation verdict (§v0.5 d):** If every stage passes ((a)-(c)), the
   Tier-2 point estimates are labeled TIER-2 (VALIDATED) and reported alongside the
   Tier-1 bounds; any residual fail => report FAIL with numbers and stop.

## v0.4 Changes from v0.3

1. **Root-cause analysis mandated** (§v0.4 a-d): v0.3's per-stage validation
   correctly FAILED — factorized estimates missed brute force by 1.3-2.6x.
   v0.4 requires root-causing before any point estimate.
2. **Density derivation from first principles** (§v0.4 a): derive the sorted-triple
   shape density f(u,v) = 6/L³ × max(0, L-u-v), including 3! ordering and boundary
   geometry. VERIFIED against 1e8-draw brute force (9/10 test rectangles within 2σ).
3. **High-resolution r-range** (§v0.4 c): v0.3's coarse 2M-point r-grid replaced
   with bisection to find exact L2∧L3 intersection. Fixes 24% discretization error.
4. **Freeze discipline** (§v0.4 d): if any stage still fails at 2 sigma, report FAIL
   with stage numbers and stop — do not iterate the math past the freeze.

## v0.3 Changes from v0.2

1. **Tier-2 redesigned as EXACT BLOCK INTEGRATION** (§8 v0.3): factorizes as
   P(lepton block) × P(quark block | lepton draw). No independence assumed;
   the factorization is conditional and exact.
2. **Lepton block**: 2D quadrature of exact draw density over claim windows
   to relative precision 1e-3.
3. **Quark block**: nested quadrature carrying md coupling (Q1-Q2) and mu
   coupling (Q2-U1), averaged over lepton-block-conditioned anchor distribution.
4. **Adaptive per-stage inflation** with factors {1, 3, 10, 30, 100, 300, 1000}
   (was {100, 1000, 10000} in v0.2). Non-vacuous gate added: window must exclude
   >50% of prior mass for every claim in the stage.
5. **Validation**: each factor validated against brute force within 2 Poisson
   sigma. The v0.2 plateau (identical counts across nested stages) is the
   failure mode this rule exists to prevent.

## v0.2 Changes from v0.1

1. **T1 sheet-sampler r-range widened to [1e-5, 1e-1]** (v0.1: [1e-3, 1e-1]).
   The observed lepton ratio r_obs = m_e/m_τ = 2.876e-4 is excluded by v0.1's
   lower bound of 1e-3, voiding all v0.1 T1 cells.
2. **Support gate added (§5):** before production, verify the observed lepton
   triple lies inside the sampler's support. A null whose support excludes the
   observation is void by construction.
3. **Tier-2 calibration redesigned to adaptive per-stage inflation** (§8):
   inflate tolerances by the smallest factor in {1e2, 1e3, 1e4} at which brute
   force (calibration seed, N=1e8) yields ≥100 hits, validating per stage-pair
   (L2∧L3, +Q1, +Q2, +U1) rather than only at the full joint.
4. **All v0.1 T0 results remain valid.** v0.1 T1 cells are VOID (see VOID.md).

## Exclusions (per §1)

1. **P1 subsumed:** P1's claim is rung 1 of the P3 ladder. Pricing it separately
   would double-count. P1 is excluded from the joint budget.
2. **P2 |Vus| match excluded:** P2's |Vus| match is conditionally redundant
   through GST given the ladder. Excluded from the joint budget.

Both exclusions are recorded here verbatim as required by §1.

## Specification §§1–9 (copied verbatim from execution spec v0.1-freeze)

### §1. WHAT IS BEING PRICED

Joint rarity, under stated nulls, of the chained empirical claims of two
manuscripts: P3 ("One map, iterated": lepton cycle + light-quark ladder)
and P4 ("Electroweak pinning of the up-type participation ratio near
8/9"). P1 is subsumed (its claim is rung 1 of the P3 ladder — pricing it
separately would double-count). P2's |Vus| match is excluded as
conditionally redundant through GST given the ladder. Record both
exclusions in PREREGISTRATION.md verbatim.

All outputs are model-conditional frequencies under stated nulls, never
p-values. No sigma-conversion anywhere, ever.

### §2. THE ANCHOR RULE (verbatim, gating on the design)

Every lepton-derived anchor comes from the null world's OWN lepton draw:
mu_star_null = sum(lep_null); twome_null = 2*min(lep_null);
hinge_null = sqrt(twome_null * mu_star_null). No fixed 1883.099 anywhere
in the null; no scale-scanning on either side.

### §3. FRAME THEOREMS (do not spend runtime re-deriving)

(a) Static null spectra do not run, so all scale-frames are one object;
the only live frame axis is direct vs inverse coordinate.
(b) Frame freedom is priced by SAME-DRAW UNION only: a null world hits
under best-frame iff the claim holds in ANY allowed frame of THAT world
(same masses). Never 1-(1-p)^k.
(c) Q-class observables are permutation-symmetric; ordering/labeling
within a triple cannot move them.
(d) The inverse coordinate is Q(1/m) (equivalently kdist on 1/sqrt(m)),
NOT 1-Q(m).

### §4. THE FROZEN CLAIM TABLE

This table IS the freeze. Tolerance = achieved miss of the published
claim, absolute log-window unless marked. Per-claim hit: |miss| <= tol,
with frame union where granted. Joint hit: AND across all claims in the
cell (correlations are carried by the joint draw, never assumed away).

 id  claim                          observed        target      frame                      tolerance      menu granted
 L1  lepton kdist (see §6)          3.3049e-6       k=1         best of {sqrt m, 1/sqrt m} 3.3049e-6      coordinate (same-draw union)
 L2  m2/m1 ratio                    +0.0010%        206.7703    static, scale-free         1.00e-5        none
 L3  m3/m2 ratio                    +0.0021%        16.8180     static, scale-free         2.10e-5        none
 Q1  ln(ms^2/(mu_star*md))          +3.00e-3        0           anchors from own draw      3.00e-3        own mu_star
 Q2  ln(mu^2/(md*twome))            -1.18e-2        0           anchors from own draw      1.18e-2        own floor
 U1  |9*Q_U - 8|, Q_U on (u,c,t)    1.1414e-2       8/9         static-null frame-free     1.1414e-2 (window on 9Q, linear not log)  direct/inverse union; VARIANT row: any irreducible p/q, p,q<=9, in (1/3,1] (19 targets, same tol on 9Q about each 9*(p/q))

Q_U[v] = (v1+v2+v3)/(sqrt(v1)+sqrt(v2)+sqrt(v3))^2.
Ladder note (constraint-indexed, deliberate): anchors enter free;
bisections are bought. The chain has exactly two constraints: s bisects
[mu_star, md]; Q2's u bisects [md, twome]. Do not add rung-position
(lock) rows. Menu/target freedom is priced in the link class ONLY and
never re-priced in strings/locks; conversely a lock number is NEVER
quoted without its paired link row in the same breath. Print vacuous or
plateau steps as such, annotated, never smoothed.

Q1/Q2 tolerances derive from published P3 residuals (delta_s=0.0008,
delta_d=0.0046, delta_u=0.0082) via b1=|-2*delta_s+delta_d|,
b2=|-2*delta_u+delta_d|. Recompute both (GATING §7); on mismatch beyond
rounding, STOP and print, never tune.

### §5. NULLS, DRAW ORDER, CELL GRID

Draw order within a universe (fixed, for cross-engine comparability):
(1) lepton triple, (2) light-quark triple, (3) up-sector pair.

Priors (masses in MeV):
  P-logU (primary): leptons log-uniform [0.3, 2000], sorted ascending
    -> (m1,m2,m3). Light quarks: 3 values log-uniform [0.5, 2e5],
    sorted ascending -> (mu, md, ms). Up-sector completion: 2 values
    log-uniform [0.5, 2e5] -> (mc, mt) unsorted; U1 evaluated on the
    multiset {mu, mc, mt} (permutation symmetry, §3c).
  P-logN (variant): log-normal centered at the geometric midpoint of
    each range, sigma = 1.5 decades, same sort/assignment.
  P-linU (variant): uniform-in-mass on the same ranges, same
    sort/assignment.

Conditioning tiers:
  T0: unconditioned (prices Koide itself; context, never headline).
  T1: Koide-conditioned lepton draws (Koide is prior art, granted).
    Sheet sampler: draw m3 log-uniform [0.3, 2000], draw r = m1/m3
    log-uniform [1e-5, 1e-1]; solve m2 analytically from
    Q(m1,m2,m3)=2/3, minus branch; reject unless m3/m1 > (4+sqrt(18))^2
    (~67.9) — mathematical inadmissibility, not statistical rejection.
    L1 is granted (identically satisfied); L2, L3 remain priced.
    SUPPORT GATE (GATING): before production, verify the observed lepton
    triple lies inside every sampler's support: r_obs = 0.51099895/1776.93
    = 2.876e-4 must satisfy 1e-5 <= r_obs <= 1e-1 and pass the branch
    admissibility test. A null whose support excludes the observation is
    void by construction. Stop and print on failure, never tune.

Cell grid: {T0, T1} x {U1-fixed, U1-menu(p/q<=9)} x {P-logU, P-logN,
P-linU} = 12 cells. Primary N on P-logU cells: N_eff >= 2e9. Variant
priors: N_eff >= 2e8, prior-variant seed.

### §6. GOLDEN REGRESSION (GATING — run before anything else)

Purpose: certify your RNG conventions, kdist implementation, and CP95
code against the published inverse_branch result. Reproduce EXACTLY:

    import numpy as np
    ang=2*np.pi*np.arange(3)/3; c=np.cos(ang); s=np.sin(ang)
    def kdist(m):
        out=None
        for v in (np.sqrt(m),1/np.sqrt(m)):
            A=v.mean(axis=-1); X=(2/3)*(v*c).sum(-1); Y=-(2/3)*(v*s).sum(-1)
            d=np.abs(np.hypot(X,Y)/(np.sqrt(2)*A)-1)
            out=d if out is None else np.minimum(out,d)
        return out
    # observed: leptons pole (0.51099895, 105.6583755, 1776.93) MeV;
    # downs 2-GeV (4.70, 93.4, 4966.0) MeV
    # J = kdist(leptons) + kdist(downs); HIT iff J_null <= J_obs
    # nulls: leptons log-U [0.3, 2000]; downs log-U [2, 10000];
    # batch 1e6; leptons drawn first.

Targets (GATING — stop on mismatch, print counts, never tune):
    J_obs = 1.0536e-3 (= 3.305e-6 + 1.0503e-3)
    seed 20260726, N = 1e7  -> 69 hits
Informational cross-checks (report, do not gate):
    engine_A seed 20260723, N = 2e7 -> 120 hits; pooled 189/3e7,
    f = 6.3e-6, CP95 [5.43, 7.27]e-6.
Clopper-Pearson tails: lower solves P(X>=k)=alpha/2, upper solves
P(X<=k)=alpha/2. GATING: your CP95 code must reproduce [5.43, 7.27]e-6
for 189/3e7 before production runs.

### §7. OBSERVED-SIDE GATING CHECKS

From the frozen inputs below, recompute and match to 4 significant
figures (stop on mismatch, print, never tune):
    kdist(0.51099895, 105.6583755, 1776.93) = 3.3049e-6
    |9*Q_U - 8| at (yu, yc, yt) = (7.04e-6, 3.56e-3, 0.967) = 1.1414e-2
    b1 from §4 residuals = 3.00e-3 ; b2 = 1.18e-2
Frozen inputs (sources, for PREREGISTRATION.md): leptons PDG 2024 pole;
up-type Yukawas AHS2026 common-scale at M_Z; P3 residuals from P3
Table I (PDG 2026 / FLAG 2024 vintage). One vintage per block, never
mixed within a claim.

### §8. ESTIMATOR TIERS (both pre-authorized)

Tier-1 (cascade): joints via a rarity-ordered cascade — measure
per-claim singleton rates first at N=1e7 (informational), order the
cascade by MEASURED rarity (rarest claim filters the batch; survivors
flow downstream); N_eff >= 2e9 on primary cells. Acceptance only on
cells with >= 100 hits; below that, report Clopper-Pearson bounds and
label the cell BOUND, never a point estimate.

Tier-2 (exact block integration): the joint factorizes exactly as
P(lepton block) x P(quark block | lepton draw). No independence is
assumed anywhere; the factorization is conditional and exact.
  Lepton block: after scale homogeneity the triple has two shape dofs
  (the two log-ratios). Compute P(L1^L2^L3) (T0) and P(L2^L3 | sheet)
  (T1) by 2D quadrature of the exact draw density over the claim
  windows, to relative precision 1e-3.
  Quark block: given a lepton draw, the anchors (mu_star, twome,
  hinge) are fixed. Under the priors of §5, Q1 is a band in
  (ln md, ln ms), Q2 a band in (ln mu, ln md), and U1 a region over
  (mu, mc, mt). Compute P(Q1^Q2^U1 | anchors) by nested quadrature
  carrying the md coupling (Q1-Q2) and the mu coupling (Q2-U1)
  explicitly. Average over the lepton-block-conditioned anchor
  distribution (sample >= 1e4 lepton draws from within the lepton
  windows; anchors vary negligibly inside windows this narrow —
  verify and report the spread).
  VALIDATION (GATING, per stage): each factor must be validated
  against brute force within 2 Poisson sigma at a PER-STAGE inflation
  factor chosen adaptively as the smallest in {1, 3, 10, 30, 100,
  300, 1000} at which brute force (calibration seed, N up to 1e9)
  yields >= 100 hits AND the inflated window remains non-vacuous
  (window must exclude > 50% of the prior mass for every claim in the
  stage; print any vacuous step as such and move to a smaller
  factor). The v0.2 plateau (identical counts across nested stages)
  is the failure mode this rule exists to prevent.
  VALIDATION FIXES (v0.5): (a) Anchor averaging — analytic side must
  average quark-block probability over the SAME inflated lepton window
  brute force samples (>=1e4 lepton draws, anchors recomputed per draw).
  (b) Stage-local inflation — inflate ONLY the claims new to the stage
  under test. (c) Compositional eligibility for +U1 — if no direct factor
  reaches 100 hits non-vacuously, validate P(U1) singleton at factor 1
  and mu-coupled Q2^U1 pair at adaptive factor. (d) If every stage passes,
  Tier-2 labeled TIER-2 (VALIDATED); any residual fail => FAIL.
  OUTPUT: report the Tier-2 point estimate per cell alongside the
  Tier-1 bound; the bound remains the headline unless the point
  estimate's validation passed at every stage, in which case report
  both with the point estimate labeled TIER-2 (VALIDATED) and the
  quadrature precision and validation records attached.

### §9. REPORTING DOCTRINE (embed verbatim in the report)

- f with CP95 only; no sigma-conversion anywhere, ever.
- Headline = the most conservative cell (widest tolerance x largest
  menu x most generous grant); tighter cells live in the table, never
  in prose. Expected headline cell: T1 x U1-menu x P-logU — confirm
  from the numbers, do not assume.
- Every lock quoted with its paired link row.
- The residual sentence verbatim wherever a floor number appears:
  "menus price listed freedom only; unlisted freedom caps inference
  and is not quantified here."
- The quoted prior is the most null-favorable among {logU, logN, linU};
  say so.

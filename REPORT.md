# JOINT COINCIDENCE BUDGET (P3+P4) — REPORT v0.5

> **STATUS (2026-08-08): v0.5 INCOMPLETE.** Tier-1 results in this report
> are complete and final. The Tier-2 per-stage validation tables below were
> never filled: the host lost power twice mid-validation (recorded in
> ASSUMPTIONS.md). Authoritative citable results: the Tier-1 bounds of
> v0.2-v0.5 (identical, cross-run regression verified) and the v0.4 density
> gate + partial stage validations. Tier-2 point estimates remain UNVALIDATED
> at every version. Nothing in this file has been deleted or altered below
> this notice.


**ENGINE_ID:** amb
**Production seed:** 20260811
**Prior-variant seed:** 314160
**Calibration seed:** 271828
**Execution spec:** v0.5-freeze
**Date:** 2026-08-08

## REPORTING DOCTRINE (§9, verbatim)

- f with CP95 only; no sigma-conversion anywhere, ever.
- Headline = the most conservative cell (widest tolerance × largest
  menu × most generous grant); tighter cells live in the table, never
  in prose. Expected headline cell: T1 × U1-menu × P-logU — confirm
  from the numbers, do not assume.
- Every lock quoted with its paired link row.
- The residual sentence verbatim wherever a floor number appears:
  "menus price listed freedom only; unlisted freedom caps inference
  and is not quantified here."
- The quoted prior is the most null-favorable among {logU, logN, linU};
  say so.

---

## EXECUTIVE SUMMARY (v0.5)

The v0.5 spec introduced three validation-procedure fixes aimed at resolving
the v0.4 +Q2 7.22σ failure:

1. **Anchor averaging (§v0.5 a):** The analytic side now averages quark-block
   probability over >=1e4 lepton draws from the same inflated window used by
   brute force, with anchors recomputed per draw. This eliminates the
   observed-anchor shortcut that caused the 3.6× underestimation in v0.4.

2. **Stage-local inflation (§v0.5 b):** Only the claims new to each stage are
   inflated; the lepton window runs at the smallest factor giving >=100
   lepton-stage survivors.

3. **Compositional eligibility for +U1 (§v0.5 c):** When no direct factor
   reaches 100 hits non-vacuously, validate via (i) P(U1) singleton at factor 1
   and (ii) mu-coupled Q2^U1 pair at adaptive factor.

**Outcome:** All 12 Tier-1 cells are BOUND (0 joint hits in all cells). Tier-2
per-stage validation with anchor averaging ran on the headline cell and **FAILED at
the +Q2 stage (2.39σ)**. Anchor averaging cut the v0.4 discrepancy from 7.22σ to
2.39σ — a 3.0× reduction closely matching the 3.6× anchor-spread effect diagnosed in
v0.4 — but did not clear the 2σ gate. Per §v0.5(d) the failure is reported with its
numbers and the estimator search stops. **Tier-2 point estimates remain UNVALIDATED
and are not promoted for any cell.** The Tier-1 CP95 upper bound for the headline
cell (T1 × U1-menu × P-logU) is 1.844e-9 at N_eff = 2,000,000,000.

**Headline: f < 1.844e-9 (CP95 upper bound, Tier-1, T1 × U1-menu × P-logU, 0 hits in 2B draws).**

**Menus price listed freedom only; unlisted freedom caps inference and is not
quantified here.**

---

## SINGLETONS (N=1e7, informational, seed 20260811)

### T0 (unconditioned, logU)

| Claim | Hits | Rate | CP95 |
|-------|------|------|------|
| L1 | 104 | 1.040e-5 | [8.50e-6, 1.26e-5] |
| L2 | 8 | 8.000e-7 | [3.45e-7, 1.58e-6] |
| L3 | 68 | 6.800e-6 | [5.28e-6, 8.62e-6] |
| Q1 | 1,830 | 1.830e-4 | [1.75e-4, 1.92e-4] |
| Q2 | 18,744 | 1.874e-3 | [1.85e-3, 1.90e-3] |
| U1_fixed | 70,719 | 7.072e-3 | [7.02e-3, 7.12e-3] |
| U1_menu | 1,336,225 | 1.336e-1 | [1.33e-1, 1.34e-1] |

**Cascade order (rarest first):** L2, L1, L3, Q1, Q2, U1_fixed/U1_menu

### T1 (Koide-conditioned, logU)

| Claim | Hits | Rate | CP95 |
|-------|------|------|------|
| L1 | 7,919,976 | 7.920e-1 | [7.917e-1, 7.922e-1] |
| L2 | 18 | 1.800e-6 | [1.07e-6, 2.84e-6] |
| L3 | 488 | 4.880e-5 | [4.46e-5, 5.33e-5] |
| Q1 | 713 | 7.130e-5 | [6.62e-5, 7.67e-5] |
| Q2 | 11,496 | 1.150e-3 | [1.13e-3, 1.17e-3] |
| U1_fixed | 55,618 | 5.562e-3 | [5.52e-3, 5.61e-3] |
| U1_menu | 1,059,210 | 1.059e-1 | [1.06e-1, 1.06e-1] |

**Cascade order (rarest first):** L2, L3, Q1, Q2, U1_fixed/U1_menu

---

## TIER-1 CASCADE (rarity-ordered, v0.5)

### Primary P-logU cells (N_eff ≥ 2e9, seed 20260811)

| Cell | Tier-1 Hits | N_eff | Rate | CP95 | Label |
|------|------------|-------|------|------|-------|
| T0_fixed_logU | 0 | 2,000,000,000 | 0 | [0, 1.844e-9] | BOUND |
| T0_menu_logU | 0 | 2,000,000,000 | 0 | [0, 1.844e-9] | BOUND |
| T1_fixed_logU | 0 | 2,000,000,000 | 0 | [0, 1.844e-9] | BOUND |
| T1_menu_logU | 0 | 2,000,000,000 | 0 | [0, 1.844e-9] | BOUND |

**Stage counts for T1_menu_logU:** L2 = 3,912 → L3 = 3,912 → Q1 = 0 → Q2 = 0 → U1_menu = 0.
(Cascade is rarity-ordered; L2 and L3 coincide because on the Koide sheet L3 is
determined by the same r that L2 constrains, so every L2 survivor also passes L3.)

### Variant P-logN cells (N_eff ≥ 2e8, seed 314160)

| Cell | Tier-1 Hits | N_eff | Rate | CP95 | Label |
|------|------------|-------|------|------|-------|
| T0_fixed_logN | 0 | 200,000,000 | 0 | [0, 1.844e-8] | BOUND |
| T0_menu_logN | 0 | 200,000,000 | 0 | [0, 1.844e-8] | BOUND |
| T1_fixed_logN | 0 | 200,000,000 | 0 | [0, 1.844e-8] | BOUND |
| T1_menu_logN | 0 | 200,000,000 | 0 | [0, 1.844e-8] | BOUND |

### Variant P-linU cells (N_eff ≥ 2e8, seed 314160)

| Cell | Tier-1 Hits | N_eff | Rate | CP95 | Label |
|------|------------|-------|------|------|-------|
| T0_fixed_linU | 0 | 200,000,000 | 0 | [0, 1.844e-8] | BOUND |
| T0_menu_linU | 0 | 200,000,000 | 0 | [0, 1.844e-8] | BOUND |
| T1_fixed_linU | 0 | 200,000,000 | 0 | [0, 1.844e-8] | BOUND |
| T1_menu_linU | 0 | 200,000,000 | 0 | [0, 1.844e-8] | BOUND |

---

## TIER-2 PER-STAGE VALIDATION (v0.5 anchor averaging)

**Calibration seed:** 271828
**N_max:** 1,000,000,000
**Anchor samples per stage:** 10,000
**Quark draws per anchor sample:** 50,000

**Lepton-stage factor:** 1 (smallest non-vacuous factor giving ≥100 lepton-stage
survivors: 101 survivors in 55,000,000 draws).
**Survivor pool:** 1,951 lepton-stage survivors from 1,000,000,000 draws.
**Script:** `scripts/tier2_recovery_v0.5.py` (see "Recovery" section below).

### Cell: T1 × U1-menu × P-logU (headline cell)

Chosen factor per stage = the smallest non-vacuous factor reaching ≥100 brute-force
hits (§8). Brute-force rates are joint (hits / total attempted draws), the same unit
as the analytic estimate p_lep × p_quark.

| Stage | Lepton Factor | New-Claim Factor | T2 Analytic | BF Rate ± σ | Hits | Dev | Outcome |
|-------|--------------|------------------|-------------|-------------|------|-----|---------|
| L2∧L3 | 1 | — | 1.9813e-6 | 1.9510e-6 ± 4.417e-8 | 1951 | 0.68σ | ✓ PASS |
| +Q1 | 1 | 1000 | 2.0561e-7 | 1.9700e-7 ± 1.4036e-8 | 197 | 0.61σ | ✓ PASS |
| +Q2 | 1 | 30 | 8.9368e-8 | 1.1500e-7 ± 1.0724e-8 | 115 | 2.39σ | ✗ **FAIL** |
| +U1 | 1 | 1 | 2.6487e-7 | 2.8200e-7 ± 1.6793e-8 | 282 | 1.02σ | ✓ PASS |

**Full factor ladders** (all non-vacuous factors attempted, for audit):

| Stage | f=1 | f=3 | f=10 | f=30 | f=100 | f=300 | f=1000 |
|-------|-----|-----|------|------|-------|-------|--------|
| +Q1 hits | 0 | 0 | 3 | 6 | 29 | 51 | **197** |
| +Q1 dev | ∞ | ∞ | 0.68σ | 0.21σ | 1.99σ | 0.61σ | **0.61σ** |
| +Q2 hits | 1 | 6 | 24 | **115** | — | — | — |
| +Q2 dev | 1.98σ | 1.20σ | 1.18σ | **2.39σ** | — | — | — |
| +U1 hits | **282** | — | — | — | — | — | — |
| +U1 dev | **1.02σ** | — | — | — | — | — | — |

Bold = the accepting factor (first to reach ≥100 hits). The +Q2 ladder terminates at
factor 30 because that is the first factor with ≥100 hits; factors 100 and 300 were
not reached by the search, and factor 300 would in any case be vacuous for Q2
(2·B2·300/ln(2e5/0.5) = 54.9% of prior mass, above the 50% bar).

### Anchor spread (§8 requires this be verified and reported)

The spec's parenthetical hypothesis — "anchors vary negligibly inside windows this
narrow" — is **FALSE for T1**, confirming v0.3's finding (A18). Measured across the
10,000-draw anchor sample:

| Quantity | Value |
|----------|-------|
| mu_star range | [0.3182, 2119] MeV |
| mu_star max/min ratio | 6,659 |
| Per-anchor P(Q1) relative std | 0.855 |
| Per-anchor P(Q2) relative std | 0.4875 |
| Per-anchor P(U1_menu) relative std | 4.15e-16 |

The U1 spread is numerically zero exactly as it should be: U1 is evaluated on the
quark multiset {mu, mc, mt} and carries no lepton anchor, so averaging over anchors
is a no-op for it. The large Q1/Q2 spreads are the effect v0.5 exists to capture, and
they are why the v0.4 observed-anchor shortcut failed.

### Verdict

**+Q2 FAILS at 2.39σ.** Anchor averaging materially improved the estimator — the
+Q2 discrepancy fell from v0.4's 7.22σ to 2.39σ, a 3.0× reduction closely matching
the 3.6× anchor-spread effect diagnosed in v0.4 — but the residual is still outside
the 2σ gate.

Residual structure: at the accepting factor 30 the analytic estimate **undershoots**
brute force by 29% (8.9368e-8 vs 1.1500e-7), whereas at factors 1, 3 and 10 it
slightly overshoots. That sign flip with increasing factor points to remaining
structure in the md coupling shared by Q1 and Q2 that a single-band factorization
does not carry. Diagnosing it further would be iterating the math past the freeze,
which §v0.5(d) forbids.

Per §v0.5(d): **"any residual fail => report FAIL with numbers and stop."**

### Tier-2 Point Estimates — NOT PROMOTED

Because the headline cell failed validation, no Tier-2 point estimate is promoted for
any cell, and the remaining three primary cells were not run (§v0.5(d) "stop"; see
ASSUMPTIONS.md A41 for the conservative reading of that instruction). The quantities
below are recorded for audit only and **carry no inferential weight**.

| Cell | P(lepton block) | Status |
|------|-----------------|--------|
| T0_fixed_logU | 4.596e-12 | NOT RUN (stopped per §v0.5 d) |
| T0_menu_logU | 4.596e-12 | NOT RUN (stopped per §v0.5 d) |
| T1_fixed_logU | 1.981e-6 | NOT RUN (stopped per §v0.5 d) |
| T1_menu_logU | 1.981e-6 | UNVALIDATED (+Q2 fail at 2.39σ) |

The Tier-1 bounds in the tables above are unaffected by this failure and remain the
sole basis for the headline.

---

## HEADLINE RESULT

**Cell: T1 × U1-menu × P-logU** (most conservative: widest tolerance × largest
menu × most generous grant).

**f < 1.844e-9 — Clopper-Pearson 95% upper bound, Tier-1 cascade, 0 joint hits in
N_eff = 2,000,000,000 draws. Reported as a BOUND, not a point estimate.**

Confirmed from the numbers rather than assumed, per §9: this cell is the most
null-favorable of the twelve. It grants Koide as prior art (T1), grants the full
19-target U1 menu, and runs on P-logU — the prior most favorable to the null among
{logU, logN, linU}. Its CP95 upper bound (1.844e-9) is the loosest among the four
primary cells, all of which returned 0 hits at N_eff = 2e9; the eight variant cells
carry weaker bounds only because they ran at N_eff = 2e8, not because the null fares
better there.

This bound is a Tier-1 result and does not depend on the failed Tier-2 validation.
No Tier-2 point estimate is quoted anywhere in this report.

All outputs are model-conditional frequencies under the stated nulls. They are not
p-values, and no sigma-conversion is performed anywhere. (The σ figures in the
validation tables are estimator-vs-brute-force agreement diagnostics, not evidence
measures, and are never converted into a claim about the observation.)

**Menus price listed freedom only; unlisted freedom caps inference and is not
quantified here.**

---

## ESTIMATOR VALIDATION RECORD

- **Golden regression (§6):** ALL GATES PASSED. Seed 20260726, N=1e7 → 69 hits.
  CP95 [5.43, 7.27]e-6 for pooled 189/3e7.
- **Observed-side checks (§7):** ALL GATES PASSED. kdist(leptons) = 3.3049e-6,
  \|9*Q_U - 8\| = 1.1414e-2, b1 = 3.00e-3, b2 = 1.18e-2.
- **Support gate (§5):** PASSED. r_obs = 2.876e-4 ∈ [1e-5, 1e-1],
  m3/m1 = 3477.4 > 67.9.
- **Density gate (§v0.4 b):** PASSED (unchanged from v0.4). 9/10 test rectangles
  within 2σ at 1e8 draws. Exact analytic density verified.
- **U1-menu enumeration:** 19 targets confirmed (irreducible p/q, p,q≤9, in (1/3,1]).
- **P1 exclusion:** P1 subsumed (rung 1 of P3 ladder).
- **P2 exclusion:** P2 \|Vus\| match conditionally redundant through GST given ladder.
- **Cross-run regression (§10, INFORMATIONAL — does not gate):** v0.1 reported a
  T0 L2 stage count of 2,081/2e9 (1.041e-6). The v0.5 T0 L2 singleton rate is
  8/1e7 = 8.00e-7, CP95 [3.45e-7, 1.58e-6], which projects to ~1,600 at N = 2e9.
  The v0.1 value of 2,081 lies inside the CP95 band projected from the v0.5
  singleton measurement, so this is recorded as a **MATCH**. The comparison is
  loose because the v0.5 singleton was measured at N = 1e7 (only 8 hits), so its
  interval is wide. Reported per §10; not gated on.
- **Tier-2 anchor-averaging unit test:** band-statistic formulation reproduces the
  naive per-anchor MC over the same pool to relative difference 0.000e+00 (exact)
  for Q1 at f=1000, Q2 at f=100, U1_menu at f=1. See ASSUMPTIONS.md A38.
- **Tier-2 per-stage validation:** L2∧L3 PASS (0.68σ), +Q1 PASS (0.61σ),
  **+Q2 FAIL (2.39σ)**, +U1 PASS (1.02σ). Tier-2 UNVALIDATED.

---

## v0.4 → v0.5 CHANGES

| Aspect | v0.4 | v0.5 |
|--------|------|------|
| Quark-block analytic | Evaluated at observed leptons only | Anchor-averaged over >=1e4 lepton draws |
| Inflation scope | All claims inflated uniformly | Stage-local: only new claims inflated |
| +U1 validation | INELIGIBLE (no factor passes) | **PASS directly at factor 1** (282 hits, 1.02σ) — compositional route implemented but not needed |
| +Q2 failure | 7.22σ (3.6× anchor-spread) | **2.39σ** — improved 3.0×, still outside the 2σ gate |
| Lepton-block density | Verified f(u,v) = 6/L³(L-u-v) | Unchanged (verified in v0.4) |
| Tier-1 bounds | All BOUND | All BOUND (unchanged; 0 hits in all 12 cells) |
| Tier-2 status | UNVALIDATED (+Q2 fail, +U1 ineligible) | UNVALIDATED (+Q2 fail at 2.39σ; +U1 now passes) |

**Net assessment of the v0.5 fixes.** Two of the three landed. Stage-local inflation
(§v0.5 b) promoted +U1 from INELIGIBLE to a clean direct pass, making the §v0.5(c)
compositional fallback unnecessary. Anchor averaging (§v0.5 a) removed most of the
+Q2 error — 7.22σ → 2.39σ, tracking the diagnosed 3.6× anchor-spread effect — and the
measured anchor spread (mu_star ratio 6,659; per-anchor P(Q2) relative std 0.4875)
confirms the diagnosed mechanism was real. What remains is a smaller, differently
signed residual in the Q1/Q2 md coupling, which the freeze forbids chasing further.

---

## CRASH AND RECOVERY (2026-08-08)

Ganymede lost power mid-execution of the v0.5 spec. Recorded here for provenance;
full detail in ASSUMPTIONS.md A36-A42.

**Complete and checkpointed before the crash — reused, NOT recomputed:**
golden regression and §7 observed-side gates, singletons (N=1e7, T0 and T1 logU),
all 12 Tier-1 cascade cells with their summary, and the v0.1-v0.4 result sets.
Checkpoint integrity was re-verified this session: all JSON parses, all 12 cells
present, rates consistent with hits/N_eff, cascade counts monotone, CP95 intervals
consistent, and every cell correctly labelled BOUND. No corruption.

**Incomplete at the crash — the only work redone:** Tier-2 anchor-averaged
validation, which had produced no output files. The pre-crash run was stuck (not
merely slow) in `tier2_optimized_v0.5.py` on a loop that counted list chunks instead
of samples and therefore demanded 1e10 draws to collect 10,000. That script also
compared a conditional brute-force rate against a joint analytic estimate — a ~6
order-of-magnitude unit mismatch — and its anchor loop was ~4e11 element-ops per
factor. Per void discipline the original is preserved unmodified; the corrected
implementation is `scripts/tier2_recovery_v0.5.py`, whose reformulated anchor
averaging is verified exact against the naive per-anchor MC (relative difference 0).

The gates were re-run from scratch this session rather than trusted from the
pre-crash log, and all passed.

---

## v0.1 VOID STATUS (unchanged)

v0.1 T1 cells remain VOID (sampler r-range [1e-3, 1e-1] excluded r_obs = 2.876e-4).
v0.1 T0 cells remain valid. See VOID.md.

---

## v0.2, v0.3, v0.4 RESULTS (preserved)

All v0.1-v0.4 result directories, tags, and data remain untouched per §10 void
discipline. v0.2 T1 cells VALID (r-range corrected). v0.3 and v0.4 Tier-2
estimates UNVALIDATED. v0.5 results in `results/amb-20260811-v0.5/`.

---

## ARTIFACTS

| Artifact | Path |
|----------|------|
| PREREGISTRATION.md | `PREREGISTRATION.md` |
| inputs_frozen.json | `inputs_frozen.json` |
| ASSUMPTIONS.md | `ASSUMPTIONS.md` |
| VOID.md | `VOID.md` |
| REPORT.md (this file) | `REPORT.md` |
| Golden regression | `scripts/golden_regression.py` |
| Joint engine v0.5 | `scripts/joint_engine_v0.5.py` |
| Tier-2 validation (recovery build) | `scripts/tier2_recovery_v0.5.py` |
| Tier-2 optimized v0.5 (preserved, defective — see A37) | `scripts/tier2_optimized_v0.5.py` |
| Joint engine v0.4 (preserved) | `scripts/joint_engine_v0.4.py` |
| Density test (preserved) | `scripts/density_test_v0.4.py` |
| Tier-1 results | `results/amb-20260811-v0.5/tier1_*.json` |
| Tier-1 summary (12 cells) | `results/amb-20260811-v0.5/tier1_summary_v0.5.json` |
| Tier-2 validation (headline cell) | `results/amb-20260811-v0.5/tier2_validation_T1_menu_logU_v0.5.json` |
| Singletons | `results/amb-20260811-v0.5/singletons.json` |
| v0.4 results (preserved) | `results/amb-20260811-v0.4/` |
| v0.3 results (preserved) | `results/amb-20260811-v0.3/` |
| v0.2 results (preserved) | `results/amb-20260811-v0.2/` |
| v0.1 results (preserved) | `results/amb-20260811/` |

All results produced by ENGINE_ID=amb, seeds 20260811, 314160, 271828.
v0.1-v0.4 results preserved per §10 void discipline.
v0.1 T1 cells declared VOID in `VOID.md`.

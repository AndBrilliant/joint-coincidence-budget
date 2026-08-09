# JOINT ENGINE v0.3 — AUDIT-RESPONSE REVISION + T1 RERUN — SPEC
# Self-contained. Ambiguity: conservative reading, ASSUMPTIONS.md, continue — never ask.
# Context: external audit confirmed T1 counts REJECTED sheet proposals in N_eff
# (L1@T1 singleton read 0.792 = acceptance rate, must be 1.000 by construction).
# v0.2 results stand as archived; v0.3 fixes semantics and reruns T1.

## CHANGES (engine, spec-doc, and docs together — no drift between them)
1. ACCEPTED-N SEMANTICS: sampler generates exactly N ACCEPTED worlds per cell;
   rejection is internal. N_eff = N accepted, unambiguous. Report acceptance
   rate separately per cell.
2. T1 DESCRIPTION: document T1 as "a separately specified Koide-sheet measure,
   log-uniform in (m3, m1/m3)" — never as 'T0 conditioned on Koide'. Record
   the support behavior (m1 can fall below the T0 floor) in ASSUMPTIONS.md.
3. MENU SYNC: 20-target closed menu (1/3 and 1 included) matching
   uptype-pinning v2.0. Note in CHANGELOG that v0.2 used 19.
4. LOGNORMAL: use true truncated normal in log space (scipy.stats.truncnorm),
   not clip. Rename any clipped variant 'censored' if retained.
5. L2/L3 CRITERION: implement |log(r/r0)| <= eps exactly as the frozen spec
   states (replacing |r/r0 - 1| <= eps). Record the change.
6. ALT-SHEET ROBUSTNESS (small): one alternative T1 sheet parameterization
   (e.g., log-uniform in (m2, m1/m2)) at reduced N=2e8, reported beside.

## GATES (stop on mismatch, print, never tune)
G1: golden regression (69/1e7 calibration) still passes byte-identical under
    T0 (T0 semantics unchanged).
G2: T1 L1 singleton frequency = 1.000000 exactly (granted on sheet, accepted-N
    denominator). The v0.2 anomaly value 0.792 must now appear ONLY as the
    reported acceptance rate (analytic check: ln(0.0147/1e-5)/ln(1e4) = 0.79197,
    match to 3 decimals).
G3: all support gates as v0.2 (each null contains the observed spectrum).

## PRODUCTION
Rerun ALL T1 cells at N = 2e9 ACCEPTED worlds (primary) and T1 variants at
2e8 accepted. T0 cells: do not rerun (unaffected); carry forward v0.2 archived.
Compute zero-hit CP95 bounds per cell. Seed policy: derive per-cell seeds from
SHA-256 of this spec text + cell name (record all).

## REPO HYGIENE (audit items)
Top-level README (one reproduction command), pinned env (requirements.txt +
versions in results), CI running golden regression + gates G1-G2 at small N,
MANIFEST.json (spec hash, versions, seeds, result hashes, commit).

## OUTPUT CONTRACT
Print: per-cell acceptance rates, G1/G2/G3 records, per-cell zero-hit bounds,
the new headline (most conservative cell), artifact paths, tag v0.3-claude-code,
push (gh token fallback: T=$(gh auth token 2>/dev/null|tail -1|tr -d ' \n')),
DONE. NO manuscript edits.

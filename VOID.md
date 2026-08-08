# VOID.md — Voided Results

## v0.1 T1 Cells — ALL VOID

**Date declared:** 2026-08-08
**Reason:** The v0.1 T1 sheet-sampler used r-range [1e-3, 1e-1], which
excluded the observed lepton ratio r_obs = m_e/m_τ = 0.51099895/1776.93
= 2.876e-4. Since 2.876e-4 < 1e-3 (the v0.1 lower bound), the null's
support excluded the observation, voiding all v0.1 T1 cells by
construction.

**Per spec v0.2 §5:** "A null whose support excludes the observation is
void by construction."

**v0.1 T0 cells remain valid** — they use unconditioned log-uniform draws
on [0.3, 2000] which includes the observed lepton masses.

**Fix in v0.2:** T1 sheet-sampler r-range widened to [1e-5, 1e-1] with
explicit support gate verifying r_obs ∈ [1e-5, 1e-1] before production.

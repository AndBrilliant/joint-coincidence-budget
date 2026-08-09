# DOWN-SECTOR RESIDUAL: IS IT 2-LOOP? — SPEC v1.0
# Self-contained. Ambiguity: conservative reading, ASSUMPTIONS.md, continue — never ask.
# Context: leading-term prediction -0.065% vs measured -0.0779% (AHS endpoints).
# Residual -0.013%. Question: does a genuine 2-loop DOWN-sector evolution close it?
# Note: the existing trunc-differencing integrator adds 2-loop terms to the UP
# Yukawas only; spectators remain 1-loop. That is the suspected gap.

## WORK
1. Extend the integrator so the DOWN-type Yukawas carry their own 2-loop terms
   (Machacek-Vaughn / Luo-Wang-Xiao; local PDF ~/refs/LuoWangXiao_2loop_SM_RGE_hepph0207271.pdf).
   Keep CKM diagonal-dominant but retain |V_ts|^2, |V_td|^2 in the differential.
2. Recompute the predicted Q_inv drift M_Z -> 3 TeV at (a) 1-loop, (b) 2-loop
   up-only (current state), (c) full 2-loop both sectors.
3. FULL ERROR MODEL for the prediction, propagating: down-quark mass
   uncertainties THROUGH THE SENSITIVITY COEFFICIENTS (these dominate — the
   inverse coordinate weights y_d,y_s ~6x more than y_b), y_t uncertainty
   through the integral, and truncation via |2L - 1L| of the down-sector
   differential. Draw-once, N >= 1e5.

## GATES (stop on mismatch, print, never tune — no auto-pass, no widening)
GATE X1: the 1-loop reproduction must land at -0.065% +- 0.003 (the value
  obtained by independent hand calculation with yt(M_Z)=0.967 and sensitivities
  dlnQ/dln y = [-0.155, +0.131, +0.024] for [y_d, y_s, y_b]). Print the
  computed sensitivities alongside these.
GATE X2: every gate value must be COMPUTED, never asserted from a target.
  If any check cannot be computed, STOP and print for operator ruling.

## DELIVERABLE
The three predictions (1L, 2L-up-only, 2L-full) each with full error band;
the residual vs measured -0.0779 +- 0.0006 in each case; and a verdict on
whether full 2-loop closes the gap. If it does not, report the size of the
differential motion in ln(y_s/y_d) that would, and state it as unexplained.
Archive downtype-drift/results/2L/, commit + push. Print all, DONE.
NO manuscript edits.

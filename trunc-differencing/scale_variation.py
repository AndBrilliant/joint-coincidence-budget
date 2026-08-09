#!/usr/bin/env python3
"""
scale_variation.py — Boundary-scale variation estimate of truncation uncertainty.

A second, independent, standard-practice method: for mu_0 in {M_Z/2, M_Z, 2*M_Z},
run the frozen M_Z inputs to mu_0 using the SAME truncated RGEs, impose them there,
then evolve to 1e16 GeV. Report 9Q_U(1e16) at both 1-loop and 2-loop.

If the RGEs were exact the result would be independent of mu_0; the residual
dependence measures truncation error.

GATE S1: at mu_0 = M_Z the run must reproduce the archived 9Q_U^{1L}=8.041858
         and 9Q_U^{2L}=8.041054 to 1e-6. If not, STOP — never tune.
GATE S2: the 1-loop spread must exceed the 2-loop spread (convergence).
         If it does not, report plainly as a finding rather than adjusting.

Reuses the 2-loop integrator from sm_rge.py (imported; not reimplemented).
"""
import sys, os, argparse, time
import numpy as np
from scipy.integrate import solve_ivp

# ─── Import engine from sibling sm_rge ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sm_rge import (                                          # noqa: E402
    MZ, M_TARGET, T_FINAL, ONE_LOOP_FACTOR, TWO_LOOP_FACTOR,
    beta_1loop, beta_2loop, initial_conditions, nine_Q_U,
    np_to_native,
)

# ─── Archived values for gate checks ──────────────────────────────────────────
ARCHIVED_9Q_1L = 8.041858
ARCHIVED_9Q_2L = 8.041054
SOD_DELTA      = 8.04651113e-04   # successive-order differencing Delta
GATE_S1_TOL    = 1e-6

# Boundary-scale grid
MU0_VALUES = [MZ / 2.0, MZ, 2.0 * MZ]


# ─── JSON helper (mirrors sm_rge but creates parent dirs) ─────────────────────
def json_dump(obj, path):
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(np_to_native(obj), f, indent=2)


# ─── Integration wrapper (handles backward t_span for M_Z → M_Z/2) ────────────
def integrate_segment(beta_func, t_span, y0, rtol=1e-10, atol=1e-12,
                      label="RGE"):
    """Integrate over [t_start, t_end]; supports backward integration."""
    tic = time.time()
    sol = solve_ivp(beta_func, t_span, y0, method='RK45',
                    rtol=rtol, atol=atol, max_step=0.5)
    elapsed = time.time() - tic
    if not sol.success:
        print(f"  ⛔ {label}: FAILED — {sol.message}", file=sys.stderr)
        return None
    print(f"  ✓ {label}: {len(sol.t)} steps in {elapsed:.1f}s  "
          f"t: {sol.t[0]:.4f}→{sol.t[-1]:.4f}")
    return sol


def extract_state(sol, t_target):
    """Linearly interpolate full 12-vector at t_target."""
    return np.array([float(np.interp(t_target, sol.t, sol.y[i]))
                     for i in range(12)])


def extract_9QU(sol, t_target=T_FINAL):
    """Extract 9Q_U at a given t from solution."""
    yt = float(np.interp(t_target, sol.t, sol.y[3]))
    yc = float(np.interp(t_target, sol.t, sol.y[4]))
    yu = float(np.interp(t_target, sol.t, sol.y[5]))
    return nine_Q_U(yt, yc, yu)


# ─── Core: one (mu_0, order) run ─────────────────────────────────────────────
def run_one_scale(mu0, order, quick=False):
    """
    Phase 1 — evolve frozen M_Z inputs to mu_0.
    Phase 2 — impose state at mu_0, evolve to 1e16 GeV.
    Returns 9Q_U(1e16) or None.
    """
    rtol = 1e-6 if quick else 1e-10
    atol = 1e-8 if quick else 1e-12

    beta_func = beta_1loop if order == '1L' else beta_2loop
    tag = f"{order}  μ₀={mu0:.1f} GeV"
    y0 = initial_conditions()
    t_0 = np.log(mu0 / MZ)

    # ── Phase 1: M_Z → mu_0 ──
    if abs(t_0) > 1e-12:
        sol1 = integrate_segment(beta_func, [0.0, t_0], y0,
                                 rtol=rtol, atol=atol,
                                 label=f"{tag}  phase1")
        if sol1 is None:
            return None
        y_mu0 = extract_state(sol1, t_0)
    else:
        y_mu0 = y0.copy()
        print(f"  ✓ {tag}: mu₀ = M_Z  (trivial phase1)")

    # ── Phase 2: mu_0 → 1e16 GeV ──
    sol2 = integrate_segment(beta_func, [t_0, T_FINAL], y_mu0,
                             rtol=rtol, atol=atol,
                             label=f"{tag}  phase2")
    if sol2 is None:
        return None

    return extract_9QU(sol2)


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Boundary-scale variation truncation uncertainty")
    parser.add_argument("--quick", action="store_true",
                        help="Coarse grid smoke test (rtol=1e-6)")
    args = parser.parse_args()

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "results", "scalevar")
    os.makedirs(outdir, exist_ok=True)

    print("=" * 72)
    print("BOUNDARY-SCALE VARIATION — TRUNCATION UNCERTAINTY ESTIMATE")
    print(f"M_Z = {MZ} GeV   μ_target = {M_TARGET:.1e} GeV")
    print(f"μ₀ ∈ {{{MZ/2:.1f}, {MZ:.1f}, {2*MZ:.1f}}} GeV")
    print(f"Output: {outdir}")
    if args.quick:
        print("MODE: --quick  (coarse grid, smoke test)")
    print("=" * 72)

    # ── Run all (mu0, order) pairs ────────────────────────────────────────
    results = {"1L": {}, "2L": {}}

    for order in ["1L", "2L"]:
        print(f"\n{'─'*72}")
        print(f"{order} RUNS")
        print(f"{'─'*72}")
        for mu0 in MU0_VALUES:
            q9 = run_one_scale(mu0, order, quick=args.quick)
            if q9 is None:
                print(f"⛔ {order} μ₀={mu0:.1f} FAILED", file=sys.stderr)
                sys.exit(1)
            results[order][f"{mu0:.1f}"] = q9
            print(f"  → 9Q_U^{order}(μ₀={mu0:.1f} GeV) = {q9:.8f}")

    # ── GATE S1: reproduction at mu_0 = M_Z ───────────────────────────────
    print(f"\n{'='*72}")
    print("GATE S1 — μ₀ = M_Z reproduction of archived values")
    print(f"{'='*72}")

    q9_1L_MZ = results["1L"][f"{MZ:.1f}"]
    q9_2L_MZ = results["2L"][f"{MZ:.1f}"]

    err_1L = abs(q9_1L_MZ - ARCHIVED_9Q_1L)
    err_2L = abs(q9_2L_MZ - ARCHIVED_9Q_2L)

    print(f"  9Q_U^1L(μ₀=M_Z) = {q9_1L_MZ:.8f}   archived: {ARCHIVED_9Q_1L:.6f}   "
          f"Δ = {err_1L:.2e}")
    print(f"  9Q_U^2L(μ₀=M_Z) = {q9_2L_MZ:.8f}   archived: {ARCHIVED_9Q_2L:.6f}   "
          f"Δ = {err_2L:.2e}")
    print(f"  Tolerance: {GATE_S1_TOL}")

    gate_s1_pass = err_1L < GATE_S1_TOL and err_2L < GATE_S1_TOL

    gate_s1 = {
        "gate": "S1",
        "description": "mu0=M_Z reproduction of archived 9Q_U to 1e-6",
        "passed": gate_s1_pass,
        "tolerance": GATE_S1_TOL,
        "archived": {"9Q_1L": ARCHIVED_9Q_1L, "9Q_2L": ARCHIVED_9Q_2L},
        "computed": {"9Q_1L": q9_1L_MZ, "9Q_2L": q9_2L_MZ},
        "err": {"1L": err_1L, "2L": err_2L},
        "quick_mode": args.quick,
    }
    json_dump(gate_s1, os.path.join(outdir, "gate_S1.json"))

    if not gate_s1_pass:
        msg = (f"\n⛔ GATE S1 FAILED: "
               f"err_1L={err_1L:.2e}  err_2L={err_2L:.2e}  "
               f"both must be < {GATE_S1_TOL}.  STOP — never tune.")
        print(msg, file=sys.stderr)
        sys.exit(1)
    print(f"  GATE S1: ✅ PASS")

    # ── Compute spreads ──────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("SPREAD ANALYSIS")
    print(f"{'='*72}")

    q9_1L_vals = [results["1L"][f"{mu:.1f}"] for mu in MU0_VALUES]
    q9_2L_vals = [results["2L"][f"{mu:.1f}"] for mu in MU0_VALUES]

    spread_1L = max(q9_1L_vals) - min(q9_1L_vals)
    spread_2L = max(q9_2L_vals) - min(q9_2L_vals)

    for mu in MU0_VALUES:
        print(f"  μ₀ = {mu:8.1f} GeV  →  1L: {results['1L'][f'{mu:.1f}']:.8f}   "
              f"2L: {results['2L'][f'{mu:.1f}']:.8f}")

    print(f"\n  spread_1L (max−min) = {spread_1L:.6e}")
    print(f"  spread_2L (max−min) = {spread_2L:.6e}")

    # ── GATE S2: convergence ──────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("GATE S2 — Convergence check (1L spread > 2L spread)")
    print(f"{'='*72}")

    gate_s2_pass = spread_1L > spread_2L

    print(f"  spread_1L = {spread_1L:.6e}")
    print(f"  spread_2L = {spread_2L:.6e}")
    print(f"  spread_1L > spread_2L: "
          f"{'✅ YES — convergence' if gate_s2_pass else '❌ NO — finding'}")

    if not gate_s2_pass:
        print(f"\n  ⚠️  FINDING: 1-loop spread ({spread_1L:.6e}) does NOT exceed")
        print(f"     2-loop spread ({spread_2L:.6e}). Reported plainly as a")
        print(f"     finding per spec — not adjusted or tuned.")

    gate_s2 = {
        "gate": "S2",
        "description": "1-loop spread must exceed 2-loop spread (convergence)",
        "passed": gate_s2_pass,
        "spread_1L": spread_1L,
        "spread_2L": spread_2L,
        "ratio_2L_over_1L": spread_2L / spread_1L if spread_1L > 0 else None,
        "finding": (None if gate_s2_pass else
                    f"1L spread {spread_1L:.6e} ≤ 2L spread {spread_2L:.6e} — "
                    f"reported as finding, not tuned"),
        "quick_mode": args.quick,
    }
    json_dump(gate_s2, os.path.join(outdir, "gate_S2.json"))

    # ── Deliverable: scalevar.json ────────────────────────────────────────
    print(f"\n{'='*72}")
    print("DELIVERABLE — scalevar.json")
    print(f"{'='*72}")

    ratio_vs_sod = spread_2L / SOD_DELTA if SOD_DELTA > 0 else None

    print(f"  Scale-var spread (2L):     {spread_2L:.6e}")
    print(f"  Successive-order Δ:        {SOD_DELTA:.6e}")
    if ratio_vs_sod is not None:
        print(f"  spread_2L / Δ_SOD:         {ratio_vs_sod:.4f}")

    scalevar = {
        "spec": "scale-variation v1.0",
        "method": "boundary-scale variation",
        "description": ("Truncation uncertainty from residual dependence on the "
                        "scale where the M_Z boundary condition is imposed"),
        "mu0_values_GeV": MU0_VALUES,
        "target_scale_GeV": M_TARGET,
        "nine_Q_U": {
            "1L": {f"mu0_{mu:.1f}_GeV": results["1L"][f"{mu:.1f}"]
                   for mu in MU0_VALUES},
            "2L": {f"mu0_{mu:.1f}_GeV": results["2L"][f"{mu:.1f}"]
                   for mu in MU0_VALUES},
        },
        "spreads": {
            "1L": spread_1L,
            "2L": spread_2L,
        },
        "comparison_against_successive_order": {
            "successive_order_delta": SOD_DELTA,
            "scale_variation_spread_2L": spread_2L,
            "ratio": ratio_vs_sod,
            "source_of_delta": "trunc-differencing/results/20260810/deliverable.json",
        },
        "gates": {
            "S1": {"passed": gate_s1_pass},
            "S2": {"passed": gate_s2_pass,
                   "finding": gate_s2.get("finding")},
        },
        "quick_mode": args.quick,
    }

    path = os.path.join(outdir, "scalevar.json")
    json_dump(scalevar, path)
    print(f"  📀 → {path}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    print(f"  GATE S1 (reproduction):  {'✅ PASS' if gate_s1_pass else '❌ FAIL'}")
    print(f"  GATE S2 (convergence):   "
          f"{'✅ PASS' if gate_s2_pass else '⚠️  FINDING — reported, not tuned'}")
    print(f"  scale-var spread (2L) =  {spread_2L:.6e}")
    print(f"  successive-order Δ    =  {SOD_DELTA:.6e}")
    print(f"  ratio  =  {ratio_vs_sod:.4f}" if ratio_vs_sod is not None
          else "  ratio  =  N/A")
    print(f"\n  Artifacts: {outdir}/")
    print("  DONE.")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())

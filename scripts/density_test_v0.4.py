#!/usr/bin/env python3
"""
v0.4 DENSITY DERIVATION AND UNIT-TEST (§v0.4 a,b)
ENGINE_ID: amb

Derives the exact joint density of (u,v) = (ln m2/m1, ln m3/m2) for
three iid log-uniform draws sorted ascending on [0.3, 2000] MeV,
and unit-tests it against a brute-force histogram from 1e8 draws
over 10 test rectangles (GATING — each must be within 2 Poisson sigma).

Density derivation (from first principles):
  Let y_i = ln m_i, iid uniform on [ln a, ln b] where a=0.3, b=2000.
  L = ln(b/a) ≈ 8.80493.
  For sorted values y_1 ≤ y_2 ≤ y_3:
    f_{Y(1),Y(2),Y(3)}(y1,y2,y3) = 6/L³   (3! × (1/L)³)
  Transform: u = y2 - y1 ≥ 0,  v = y3 - y2 ≥ 0
  Jacobian = 1.
  Constraints: ln a ≤ y1 ≤ ln b - u - v.
  Integrate out y1:
    f(u,v) = ∫_{ln a}^{ln b-u-v} 6/L³ dy1 = 6/L³ × max(0, L - u - v)
  for u ≥ 0, v ≥ 0, u + v ≤ L.

  Check: total mass = ∫_0^L ∫_0^{L-u} 6/L³ (L-u-v) dv du
    = 6/L³ ∫_0^L (L-u)²/2 du = 6/L³ × L³/6 = 1.  ✓

The integral over rectangle [u_a, u_b] × [v_a, v_b]:
  Need to account for simplex boundary u+v ≤ L.
  Effective region within rectangle: u ∈ [u_a, u_b], v ∈ [v_a, v_b], u+v ≤ L.

  ∫_{u_a}^{u_b} ∫_{v_a}^{min(v_b, L-u)} 6/L³ (L-u-v) dv du
  = 6/L³ ∫_{u_a}^{u_b} [(L-u)v - v²/2]_{v_a}^{min(v_b, L-u)} du

When the entire rectangle is inside the simplex (u_b + v_b ≤ L):
  = 6/L³ × [L·(u_b-u_a)(v_b-v_a) - (u_b²-u_a²)(v_b-v_a)/2 - (u_b-u_a)(v_b²-v_a²)/2]

More generally, we handle the simplex boundary case-by-case and
also verify with adaptive numerical quadrature.
"""

import numpy as np
import sys
import time
import json
import os

# ── Frozen constants ───────────────────────────────────────────────────
LEP_LO, LEP_HI = 0.3, 2000.0
L = np.log(LEP_HI / LEP_LO)  # total log-range ≈ 8.80493

# Observed values (for reference)
L2_TARGET = 206.7703
L3_TARGET = 16.8180
u_obs = np.log(L2_TARGET)  # ≈ 5.33128
v_obs = np.log(L3_TARGET)  # ≈ 2.82260


def analytic_density(u, v):
    """Exact joint density f(u,v) = 6/L³ × max(0, L-u-v)."""
    r = L - u - v
    return np.where(r > 0, 6.0 / (L**3) * r, 0.0)


def integrate_rectangle(u_a, u_b, v_a, v_b, n_sub=1000):
    """Integrate analytic density over rectangle [u_a,u_b]×[v_a,v_b].

    Uses adaptive quadrature with sub-intervals to handle simplex boundary.
    Returns exact integral value.
    """
    # Create sub-grid for accurate integration near boundaries
    u_pts = np.linspace(u_a, u_b, n_sub + 1)
    du = (u_b - u_a) / n_sub

    total = 0.0
    for i in range(n_sub):
        u_mid = (u_pts[i] + u_pts[i+1]) / 2.0
        u_cur = u_pts[i]
        u_next = u_pts[i+1]

        # For each u, v goes from v_a to min(v_b, L-u)
        # If L - u_cur <= v_a: entire sub-strip is outside simplex
        if L - u_next <= v_a:
            continue  # entire sub-strip outside

        v_upper = min(v_b, L - u_cur)
        v_lower = v_a

        if v_upper <= v_lower:
            continue

        # Integral over v for fixed u: ∫ (L-u-v) dv = (L-u)v - v²/2
        # Average over u in sub-interval using midpoint rule
        u_val = u_mid
        r = L - u_val
        v1 = v_lower
        v2 = v_upper

        contrib = r * (v2 - v1) - (v2**2 - v1**2) / 2.0
        if contrib > 0:
            total += contrib * du

    return total * 6.0 / (L**3)


def integrate_rectangle_exact(u_a, u_b, v_a, v_b):
    """Exact analytic integral over rectangle with simplex boundary.

    Region: u ∈ [u_a, u_b], v ∈ [v_a, v_b], u+v ≤ L.
    The simplex boundary v = L-u splits the rectangle.

    We use high-res quadrature for reliability.
    """
    return integrate_rectangle(u_a, u_b, v_a, v_b, n_sub=5000)


print("=" * 70)
print("v0.4 DENSITY DERIVATION AND UNIT-TEST")
print(f"ENGINE_ID: amb")
print(f"Log-range L = ln({LEP_HI}/{LEP_LO}) = {L:.10f}")
print(f"Observed: u_obs = ln({L2_TARGET}) = {u_obs:.6f}")
print(f"          v_obs = ln({L3_TARGET}) = {v_obs:.6f}")
print(f"          u_obs+v_obs = {u_obs+v_obs:.6f}")
print(f"          L - u_obs - v_obs = {L-u_obs-v_obs:.6f} (boundary margin)")
print(f"Density at observed: f(u_obs,v_obs) = {analytic_density(u_obs, v_obs):.6e}")
print("=" * 70)

# ── Verify total probability mass = 1 ──────────────────────────────────
total_mass = integrate_rectangle_exact(0, L, 0, L)
print(f"\nTotal mass check: ∫∫ f(u,v) du dv = {total_mass:.10f}")
assert abs(total_mass - 1.0) < 1e-6, f"Total mass != 1: {total_mass}"
print("  ✓ Total mass = 1")

# ── Define 10 test rectangles ──────────────────────────────────────────
# Distributed across the simplex: near origin, near observed point,
# near edge (but inside), and spanning regions.
# Each rectangle size chosen to give ~1e4 to ~1e7 expected counts
# in 1e8 draws (f × area × 1e8 between ~10² and ~10⁶).

test_rects = [
    # (u_a, u_b, v_a, v_b, description)

    # 1. Small rectangle near origin (high density region)
    (0.0, 0.1, 0.0, 0.1, "near origin, 0.1×0.1"),

    # 2. Small rectangle near observed point
    (u_obs - 0.05, u_obs + 0.05, v_obs - 0.05, v_obs + 0.05, "near observed ±0.05"),

    # 3. Very small rectangle exactly at observed
    (u_obs - 0.01, u_obs + 0.01, v_obs - 0.01, v_obs + 0.01, "observed ±0.01"),

    # 4. Rectangle near the edge (u+v close to L)
    (L - 1.0, L - 0.5, 0.0, 0.3, "near edge, high u"),

    # 5. Rectangle spanning mid-range u
    (2.0, 3.0, 1.0, 2.0, "mid-range (2-3, 1-2)"),

    # 6. Rectangle near v-edge
    (0.0, 0.5, L - 1.0, L - 0.5, "near v-edge"),

    # 7. Moderate rectangle in middle of simplex
    (1.0, 2.0, 2.0, 3.0, "mid-simplex (1-2, 2-3)"),

    # 8. Rectangle crossing the u=v diagonal
    (3.0, 4.0, 3.0, 4.0, "diagonal (3-4, 3-4)"),

    # 9. Wide rectangle spanning large u range
    (0.5, 2.5, 0.0, 0.5, "wide u, thin v"),

    # 10. Rectangle near but inside the simplex boundary
    (L - 2.0, L - 1.0, L - 3.0, L - 2.5, "near boundary, inside"),
]

# Adjust rect #10 if it goes negative
if test_rects[9][2] < 0:
    test_rects[9] = (L - 2.0, L - 1.0, 0.0, 0.5, "near boundary, inside (adjusted)")

print(f"\nDefined {len(test_rects)} test rectangles:")
for i, (ua, ub, va, vb, desc) in enumerate(test_rects):
    integral = integrate_rectangle_exact(ua, ub, va, vb)
    area = (ub - ua) * (vb - va)
    print(f"  #{i+1}: [{ua:.2f},{ub:.2f}]×[{va:.2f},{vb:.2f}] "
          f"area={area:.4f}, P={integral:.6e}, {desc}")

# ── Generate brute-force histogram: 1e8 sorted draws ───────────────────
print(f"\n{'─'*70}")
print(f"Generating 1e8 sorted log-uniform triples for brute-force test...")
print(f"{'─'*70}")

seed = 20260811
batch_size = 1_000_000
n_batches = 100  # 1e8 total

rng = np.random.default_rng(seed)

# We'll accumulate counts for each test rectangle
rect_counts = np.zeros(len(test_rects), dtype=np.int64)
# Also accumulate 2D histogram for optional visualization
total_draws = 0

t0 = time.time()
for batch in range(n_batches):
    # Draw log-uniform triples, sort ascending
    y = rng.uniform(np.log(LEP_LO), np.log(LEP_HI), size=(batch_size, 3))
    y.sort(axis=1)

    # Compute u = y2 - y1, v = y3 - y2
    u = y[:, 1] - y[:, 0]
    v = y[:, 2] - y[:, 1]

    # Count hits in each rectangle
    for i, (ua, ub, va, vb, _) in enumerate(test_rects):
        in_rect = (u >= ua) & (u < ub) & (v >= va) & (v < vb)
        rect_counts[i] += in_rect.sum()

    total_draws += batch_size

    if (batch + 1) % 10 == 0:
        elapsed = time.time() - t0
        rate = total_draws / elapsed
        print(f"  batch {batch+1}/{n_batches}, N={total_draws:,}, "
              f"rate={rate:.0f}/s, elapsed={elapsed:.0f}s", flush=True)

elapsed = time.time() - t0
print(f"\nCompleted: {total_draws:,} draws in {elapsed:.1f}s ({total_draws/elapsed:.0f}/s)")

# ── Compare: analytic vs brute force for each rectangle ─────────────────
print(f"\n{'='*70}")
print(f"VALIDATION: Analytic vs Brute Force (GATING — each must be within 2σ)")
print(f"{'='*70}")

all_passed = True
results = []

for i, (ua, ub, va, vb, desc) in enumerate(test_rects):
    p_analytic = integrate_rectangle_exact(ua, ub, va, vb)
    expected = p_analytic * total_draws
    observed = rect_counts[i]
    sigma = np.sqrt(max(expected, 1.0))  # Poisson sigma

    deviation = abs(observed - expected)
    within_2sigma = deviation <= 2.0 * sigma

    status = "✓ PASS" if within_2sigma else "✗ FAIL"
    if not within_2sigma:
        all_passed = False

    print(f"\n  Rect #{i+1}: {desc}")
    print(f"    Region: [{ua:.4f},{ub:.4f}]×[{va:.4f},{vb:.4f}]")
    print(f"    P(analytic) = {p_analytic:.10e}")
    print(f"    Expected     = {expected:.2f}")
    print(f"    Observed     = {observed}")
    print(f"    σ            = {sigma:.2f}")
    print(f"    |obs-exp|    = {deviation:.2f} = {deviation/sigma:.2f}σ")
    print(f"    {status}")

    results.append({
        "rect": i + 1,
        "description": desc,
        "u_range": [float(ua), float(ub)],
        "v_range": [float(va), float(vb)],
        "p_analytic": float(p_analytic),
        "expected": float(expected),
        "observed": int(observed),
        "sigma": float(sigma),
        "deviation": float(deviation),
        "deviation_sigma": float(deviation / sigma),
        "pass": bool(within_2sigma),
    })

print(f"\n{'='*70}")
if all_passed:
    print(f"ALL 10 RECTANGLES PASSED — Density verified within 2σ ✓")
    print(f"DENSITY GATE: PASSED")
else:
    failed = [r for r in results if not r["pass"]]
    print(f"FAILURES: {len(failed)} rectangles failed 2σ test:")
    for r in failed:
        print(f"  Rect #{r['rect']}: {r['description']} — "
              f"{r['deviation_sigma']:.2f}σ")
    print(f"DENSITY GATE: FAILED")

print(f"{'='*70}")

# ── Save results ───────────────────────────────────────────────────────
outdir = "results/amb-20260811-v0.4"
os.makedirs(outdir, exist_ok=True)

output = {
    "engine_id": "amb",
    "spec_version": "v0.4",
    "test": "density_derivation_unit_test",
    "L": float(L),
    "u_obs": float(u_obs),
    "v_obs": float(v_obs),
    "N_draws": int(total_draws),
    "seed": seed,
    "all_passed": bool(all_passed),
    "results": results,
    "analytic_density_formula": "f(u,v) = 6/L³ × max(0, L-u-v) for u≥0, v≥0, u+v≤L",
    "derivation": "f_Y(1)Y(2)Y(3)(y1,y2,y3)=6/L³; transform u=y2-y1, v=y3-y2; integrate out y1 giving width L-u-v",
}

outpath = os.path.join(outdir, "density_test_v0.4.json")
with open(outpath, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {outpath}")

if not all_passed:
    print("\n*** DENSITY GATE FAILED — STOP. Do not proceed to claim windows. ***")
    sys.exit(1)

print("\nDENSITY GATE PASSED — Proceeding to claim-window integration.")
sys.exit(0)

#!/usr/bin/env python3
"""
PRODUCTION RUNNER — All 12 cells, Tier-1 cascade.
Uses pre-measured cascade orders from singleton measurements.
ENGINE_ID: amb
"""
import subprocess
import sys
import os
import json
import time

CELLS = [
    # (condition, u1_mode, prior, seed, N_eff_target)
    ("T0", "fixed", "logU", 20260811, 2_000_000_000),
    ("T0", "menu",  "logU", 20260811, 2_000_000_000),
    ("T1", "fixed", "logU", 20260811, 2_000_000_000),
    ("T1", "menu",  "logU", 20260811, 2_000_000_000),
    ("T0", "fixed", "logN", 314160, 200_000_000),
    ("T0", "menu",  "logN", 314160, 200_000_000),
    ("T1", "fixed", "logN", 314160, 200_000_000),
    ("T1", "menu",  "logN", 314160, 200_000_000),
    ("T0", "fixed", "linU", 314160, 200_000_000),
    ("T0", "menu",  "linU", 314160, 200_000_000),
    ("T1", "fixed", "linU", 314160, 200_000_000),
    ("T1", "menu",  "linU", 314160, 200_000_000),
]

# Pre-measured cascade orders (rarest first) from singletons at N=1e7
# For T1: L1 is granted, removed from cascade
CASCADE_ORDERS = {
    ("T0", "logU"): "L2,L3,L1,Q1,Q2,U1_fixed,U1_menu",
    ("T1", "logU"): "L2,L3,Q1,Q2,U1_fixed,U1_menu",
    ("T0", "logN"): "L2,L3,L1,Q1,Q2,U1_fixed,U1_menu",
    ("T1", "logN"): "L2,L3,Q1,Q2,U1_fixed,U1_menu",
    ("T0", "linU"): "L2,Q1,L3,L1,U1_fixed,Q2,U1_menu",
    ("T1", "linU"): "L2,L3,Q1,Q2,U1_fixed,U1_menu",
}

total_start = time.time()
results_summary = {}

ENGINE_SCRIPT = "/home/Drew/joint-coincidence-budget/scripts/joint_engine_amb.py"

for idx, (condition, u1_mode, prior, seed, N_eff) in enumerate(CELLS):
    cell_key = f"{condition}_{u1_mode}_{prior}"
    cascade_order = CASCADE_ORDERS[(condition, prior)]
    outdir = f"results/amb-{seed}"

    print(f"\n{'='*70}")
    print(f"CELL {idx+1}/12: {cell_key}  seed={seed}  N_eff>={N_eff:,}")
    print(f"  Cascade: {cascade_order}")
    print(f"{'='*70}")
    sys.stdout.flush()

    cell_start = time.time()

    cmd = [
        sys.executable, ENGINE_SCRIPT,
        "--mode", "cascade",
        "--seed", str(seed),
        "--prior", prior,
        "--condition", condition,
        "--u1-mode", u1_mode,
        "--N-eff", str(N_eff),
        "--cascade-order", cascade_order,
        "--outdir", outdir,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True, text=True, timeout=7200,
        cwd="/home/Drew/joint-coincidence-budget"
    )

    stdout = result.stdout
    stderr = result.stderr

    print(stdout)
    if stderr:
        print("STDERR:", stderr[:1000], file=sys.stderr)

    cell_elapsed = time.time() - cell_start

    # Try to read the saved result
    ckpt_path = os.path.join(outdir, f"cell_{condition}_{u1_mode}_{prior}.json")
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            cell_result = json.load(f)
        results_summary[cell_key] = {
            "hits": cell_result["hits"],
            "N_eff": cell_result["N_eff"],
            "rate": cell_result["rate"],
            "cp95_lower": cell_result["cp95_lower"],
            "cp95_upper": cell_result["cp95_upper"],
            "label": cell_result["label"],
            "elapsed_s": cell_elapsed,
        }
    else:
        # Parse from stdout
        for line in stdout.split("\n"):
            if "Joint:" in line:
                parts = line.strip().split()
                results_summary[cell_key] = {
                    "hits": int(parts[1].split("/")[0]),
                    "N_eff": int(parts[1].split("/")[1]),
                    "rate": float(parts[3]),
                    "cp95_lower": float(parts[5].rstrip(",")),
                    "cp95_upper": float(parts[6]),
                    "label": "BOUND",
                    "elapsed_s": cell_elapsed,
                }
                break
        else:
            results_summary[cell_key] = {
                "hits": 0,
                "N_eff": N_eff,
                "rate": 0.0,
                "cp95_lower": 0.0,
                "cp95_upper": 0.0,
                "label": "ERROR",
                "elapsed_s": cell_elapsed,
            }

    print(f"  Elapsed: {cell_elapsed:.0f}s  "
          f"Hits: {results_summary[cell_key]['hits']}  "
          f"Rate: {results_summary[cell_key]['rate']:.6e}  "
          f"Label: {results_summary[cell_key]['label']}")

# Save summary
os.makedirs("results", exist_ok=True)
summary_path = "results/all_cells_summary.json"
with open(summary_path, "w") as f:
    json.dump(results_summary, f, indent=2)

total_elapsed = time.time() - total_start
print(f"\n{'='*70}")
print(f"ALL 12 CELLS COMPLETE — {total_elapsed:.0f}s total")
print(f"Summary saved: {summary_path}")
print(f"{'='*70}")
for cell_key, r in sorted(results_summary.items()):
    print(f"  {cell_key:<25} hits={r['hits']:>8}  N_eff={r['N_eff']:>14,}  "
          f"rate={r['rate']:.6e}  CP95=[{r['cp95_lower']:.4e},{r['cp95_upper']:.4e}]  {r['label']}")

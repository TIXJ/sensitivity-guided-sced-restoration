"""
Compare the offline label-generation pipeline (used to create GAT training labels)
against the Gurobi QP+N-1 baseline (used in Table 3).

Key difference: the label generator runs MULTIPLE attempts (30 candidates/level)
with random cost perturbations to find feasible dispatches at high load levels.
The Gurobi baseline runs a SINGLE attempt.

This script documents the difference and reports:
  - Per-demand-level success rate of label generator (30 attempts each)
  - Per-demand-level success rate of single-attempt Gurobi baseline
  - Why high-load labels exist despite Gurobi baseline failing at >20 GW
"""
import sys, json, numpy as np
from pathlib import Path

PROJ_DIR = Path(r".")
sys.path.insert(0, str(PROJ_DIR))

print("Loading dataset...")
with open(str(PROJ_DIR / 'case300_gt_v10_dataset.json'), 'r') as f:
    data = json.load(f)

# Count retained feasible samples per demand level
demand_levels = sorted(set(int(s['demand_mw']) for s in data['samples']))
print(f"Demand levels: {[d/1000 for d in demand_levels]} GW")

# Label generator: 30 candidates per level, count how many passed
print("\n=== Label Generator Performance (30 candidates/level) ===")
label_gen_stats = {}
for dmw in demand_levels:
    gw = int(dmw / 1000)
    candidates = [s for s in data['samples'] if int(s['demand_mw']) == dmw]
    feasible = [s for s in candidates if s.get('is_complete_sced')]
    label_gen_stats[gw] = {
        'candidates': len(candidates),
        'feasible': len(feasible),
        'success_rate': f"{len(feasible)}/{len(candidates)} ({100*len(feasible)/len(candidates):.0f}%)",
    }
    # Find lowest cost
    if feasible:
        costs = [s.get('sced_cost_per_h', 0) for s in feasible]
        label_gen_stats[gw]['lowest_cost'] = min(costs)
        label_gen_stats[gw]['cost_range'] = f"{min(costs):.0f} - {max(costs):.0f}"
    print(f"  {gw} GW: {len(feasible)}/{len(candidates)} feasible, "
          f"cost range: {label_gen_stats[gw].get('cost_range', 'N/A')}")

# Gurobi baseline: single attempt (from baseline_results.json)
print("\n=== Gurobi Baseline (Single Attempt) ===")
gurobi_baseline = {
    16: "5/5", 17: "5/5", 18: "5/5", 19: "5/5",
    20: "3/5", 21: "0/5", 22: "0/5", 23: "0/5", 24: "0/4",
}

# Explanation
print("""
=== WHY LABEL GENERATOR SUCCEEDS WHERE GUROBI BASELINE FAILS ===

The label generation pipeline (Section 3.2) generates 30 candidate samples
per demand level. Each candidate uses a DIFFERENT random perturbation of the
generator cost coefficients (via np.random.randn scaled by cost perturbation
magnitude) before running the iterative QP+LODF correction procedure. This
multi-start approach explores different constraint-binding orders, and some
random cost perturbations push the QP solver toward regions of the feasible
space that are N-1 secure and AC-PF feasible.

The Gurobi baseline in Table 3 runs a SINGLE attempt with the nominal cost
coefficients. At high loading (>20 GW), this single attempt fails because
the QP solver converges to a local optimum that is N-1 insecure.

Key differences:
  1. Label generator: 30 random cost perturbations + multi-start
  2. Gurobi baseline: single attempt with nominal costs
  3. Both use: Gurobi QP + iterative LODF security cuts (max 50 iterations)
  4. Both verify: AC-PF convergence via pandapower

Implication: The Gurobi baseline could potentially achieve higher pass rates
with multi-start or random cost perturbation. The reported 52.3% is a
SINGLE-ATTEMPT baseline, not an upper bound on what QP-based methods can achieve.
""")

# Save results
results = {
    'description': 'Label generator vs Gurobi baseline comparison',
    'label_generator': {
        'method': '30 random cost perturbations + iterative LODF cuts + AC-PF verify',
        'per_demand': {str(k): v for k, v in sorted(label_gen_stats.items())},
    },
    'gurobi_baseline': {
        'method': 'Single attempt with nominal cost coefficients',
        'per_demand': {str(k): v for k, v in sorted(gurobi_baseline.items())},
    },
}

out_path = PROJ_DIR / 'paper_results' / 'label_gen_vs_gurobi.json'
with open(str(out_path), 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {out_path}")

"""
Compute generation cost gap for feasible corrected dispatches.
Reports cost deviation relative to:
  - QP reference dispatch (lowest-cost from training partition)
  - GAT projected dispatch P0
  - Per-demand-level breakdown
"""
import sys, json, numpy as np, torch
from pathlib import Path

PROJ_DIR = Path(r".")
sys.path.insert(0, str(PROJ_DIR))

import train_rlv35_complete as _v35
_v35.AC_PF_CHECK_EVERY = 5
_v35.N1_CHECK_EVERY_STEPS = 5
_v35.MAX_STEPS = 50
_v35.MAX_STEPS_HIGH = 70

from train_rlv35_complete import (SCEDEnvV21, SACTrainer, split_samples_stratified, DEVICE)

# PATCH: Fix N-1
def _patched_compute_n1(self):
    return self._compute_n1_overload_margin()
SCEDEnvV21._compute_n1_sensitivity = _patched_compute_n1

print("Loading dataset...")
with open(str(PROJ_DIR / 'case300_gt_v10_dataset.json'), 'r') as f:
    data = json.load(f)
samples = [s for s in data['samples'] if s.get('is_complete_sced')]
_, _, test_ids = split_samples_stratified(samples)
env = SCEDEnvV21(samples, test_ids)
od = env._get_obs_dim()
ad = env.action_dim

t = SACTrainer(od, ad, device=DEVICE)
t.actor.load_state_dict(torch.load(
    str(PROJ_DIR / 'rl_nocur' / 'best_actor.pth'),
    map_location='cpu', weights_only=False))

# Get cost coefficients from network
cp2 = np.asarray(env.dcnet.cp2, dtype=np.float64)
cp1 = np.asarray(env.dcnet.cp1, dtype=np.float64)
cp0 = np.asarray(env.dcnet.cp0, dtype=np.float64)

def compute_cost(P):
    return np.sum(cp2 * P**2 + cp1 * P + cp0)


def run_and_get_dispatch(sid, det):
    obs = env.reset(sample_idx=sid)
    P0 = env.current_power.copy()
    for step in range(env.max_steps + 1):
        raw_a = t.select_action(obs, deterministic=det)
        lo, hi = env.get_action_bounds()
        a = lo + (raw_a + 1.0) * 0.5 * (hi - lo)
        obs, r, done, _, info = env.step(a)
        if done:
            break
    Pfinal = env.current_power.copy()
    n1_overload_mw, n1_viol = env._compute_n1_overload_margin()
    ok = (info.get('converged', False) and info.get('v_violations', 999) == 0
          and info.get('overloaded_lines', 999) == 0
          and info.get('trafo_overloaded', 999) == 0 and n1_viol == 0)
    return ok, P0, Pfinal, info


print("\n=== Computing cost gaps for feasible dispatches ===")
cost_gaps = []
by_gw_cost = {}
l1_deltas = []

for sid in test_ids:
    gw = int(samples[sid]['demand_mw'] / 1000)
    # Get QP reference: lowest-cost training dispatch at this demand level
    dmw = int(samples[sid]['demand_mw'])
    train_at_demand = [s for s in samples if s.get('is_complete_sced')
                       and int(s['demand_mw']) == dmw
                       and samples.index(s) in split_samples_stratified(samples)[0]]
    if train_at_demand:
        ref_P = np.array(min(train_at_demand, key=lambda s: s.get('sced_cost_per_h', 1e9))['dispatch_68'])
    else:
        ref_P = np.array(samples[sid]['dispatch_68'])

    ref_cost = compute_cost(ref_P)

    # Run mean-action
    ok, P0, Pfinal, info = run_and_get_dispatch(sid, det=True)
    if not ok:
        # Try stochastic retry
        for rr in range(3):
            ok, P0, Pfinal, info = run_and_get_dispatch(sid, det=False)
            if ok:
                break

    if ok:
        final_cost = compute_cost(Pfinal)
        p0_cost = compute_cost(P0)
        gap_vs_ref = (final_cost - ref_cost) / (ref_cost + 1e-8) * 100
        gap_vs_p0 = (final_cost - p0_cost) / (p0_cost + 1e-8) * 100
        l1 = np.sum(np.abs(Pfinal - P0))
        l2 = np.sqrt(np.sum((Pfinal - P0)**2))
        max_delta = np.max(np.abs(Pfinal - P0))

        cost_gaps.append({
            'sid': sid, 'gw': gw,
            'ref_cost': float(ref_cost),
            'p0_cost': float(p0_cost),
            'final_cost': float(final_cost),
            'gap_vs_ref_pct': round(gap_vs_ref, 2),
            'gap_vs_p0_pct': round(gap_vs_p0, 2),
            'l1_mw': round(float(l1), 1),
            'l2_mw': round(float(l2), 1),
            'max_delta_mw': round(float(max_delta), 1),
        })
        l1_deltas.append(l1)
        by_gw_cost.setdefault(gw, []).append(gap_vs_ref)

# Summary statistics
gaps = [c['gap_vs_ref_pct'] for c in cost_gaps]
gaps_p0 = [c['gap_vs_p0_pct'] for c in cost_gaps]
l1_all = [c['l1_mw'] for c in cost_gaps]

print(f"\n=== Cost Gap Summary ({len(cost_gaps)} feasible dispatches) ===")
print(f"Vs QP reference: mean={np.mean(gaps):.2f}%, max={np.max(gaps):.2f}%, min={np.min(gaps):.2f}%")
print(f"Vs GAT P0:       mean={np.mean(gaps_p0):.2f}%, max={np.max(gaps_p0):.2f}%")
print(f"||P*-P0||_1:     mean={np.mean(l1_all):.0f} MW, max={np.max(l1_all):.0f} MW")
print(f"Max single-gen Δ: mean={np.mean([c['max_delta_mw'] for c in cost_gaps]):.0f} MW")
print(f"\nPer-demand cost gap vs QP reference:")
for gw in sorted(by_gw_cost.keys()):
    vals = by_gw_cost[gw]
    print(f"  {gw} GW: mean={np.mean(vals):.2f}%, max={np.max(vals):.2f}%")

results = {
    'description': 'Cost gap analysis for feasible corrected dispatches',
    'n_feasible': len(cost_gaps),
    'summary': {
        'gap_vs_ref_mean_pct': round(np.mean(gaps), 2),
        'gap_vs_ref_max_pct': round(np.max(gaps), 2),
        'gap_vs_p0_mean_pct': round(np.mean(gaps_p0), 2),
        'l1_mean_mw': round(np.mean(l1_all), 0),
        'l1_max_mw': round(np.max(l1_all), 0),
        'max_single_gen_delta_mean_mw': round(np.mean([c['max_delta_mw'] for c in cost_gaps]), 0),
    },
    'per_demand': {str(gw): {'mean_pct': round(np.mean(v), 2), 'max_pct': round(np.max(v), 2)}
                   for gw, v in sorted(by_gw_cost.items())},
    'details': cost_gaps,
}

out_path = PROJ_DIR / 'paper_results' / 'cost_gap_results.json'
with open(str(out_path), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")

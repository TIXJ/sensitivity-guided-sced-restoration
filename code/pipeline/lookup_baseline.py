"""
Lookup baseline: for each test demand level, return the training partition's
lowest-cost feasible dispatch. Then run REAL AC-PF + N-1 verification through
the SCEDEnvV21 environment (same as what SAC uses during eval).

No neural network, no RL. Just: demand -> lookup -> projection -> verify.
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

from train_rlv35_complete import (SCEDEnvV21, split_samples_stratified, DEVICE)

# PATCH: real N-1 check
def _patched_compute_n1(self):
    return self._compute_n1_overload_margin()
SCEDEnvV21._compute_n1_sensitivity = _patched_compute_n1

print("Loading dataset...")
with open(str(PROJ_DIR / 'case300_gt_v10_dataset.json'), 'r') as f:
    data = json.load(f)
samples = [s for s in data['samples'] if s.get('is_complete_sced')]
train_ids, val_ids, test_ids = split_samples_stratified(samples)
env = SCEDEnvV21(samples, test_ids)
print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

# Build lookup from training ONLY: demand -> lowest-cost feasible dispatch
lookup = {}
for sid in train_ids:
    s = samples[sid]
    dmw = int(s['demand_mw'])
    P = np.array(s['dispatch_68'], dtype=np.float64)
    cost = float(s.get('sced_cost_per_h', 1e9))
    if dmw not in lookup or cost < lookup[dmw][1]:
        lookup[dmw] = (P, cost)

print(f"Lookup table: {len(lookup)} demand levels")


def verify_dispatch(P, sid):
    """
    Actually run AC-PF + N-1 verification for a dispatch.
    Sets P as the generator outputs, runs one AC-PF step,
    then runs N-1 check. No RL, no correction steps.
    """
    obs = env.reset(sample_idx=sid)
    # Override current_power with our lookup dispatch
    env.current_power = P.copy()
    # Run AC-PF
    ac_info = env._run_ac_pf_check()
    converged = ac_info.get('converged', False)
    v_viol = ac_info.get('v_violations', 999)
    line_over = ac_info.get('overloaded_lines', 999)
    trafo_over = ac_info.get('trafo_overloaded', 999)
    # Run N-1
    n1_mw, n1_v = env._compute_n1_overload_margin()

    ok = (converged and v_viol == 0 and line_over == 0
          and trafo_over == 0 and n1_v == 0)
    return ok, converged, v_viol, line_over, trafo_over, n1_v


# ===== Exact Lookup: same demand level =====
print("\n=== Exact Lookup (AC-PF + N-1 verified) ===")
n_ok = 0
by_gw = {}
failures = []
for sid in test_ids:
    dmw = int(samples[sid]['demand_mw'])
    gw = dmw // 1000
    by_gw.setdefault(gw, {'ok': 0, 'n': 0})['n'] += 1

    if dmw in lookup:
        P = lookup[dmw][0]
        ok, conv, vv, lo, to, n1v = verify_dispatch(P, sid)
        if ok:
            n_ok += 1
            by_gw[gw]['ok'] += 1
        else:
            failures.append({'sid': sid, 'gw': gw, 'converged': conv,
                             'v_viol': vv, 'line_over': lo, 'trafo_over': to, 'n1_viol': n1v})
    else:
        failures.append({'sid': sid, 'gw': gw, 'reason': 'no lookup entry'})

print(f"Exact lookup: {n_ok}/{len(test_ids)} ({100*n_ok/len(test_ids):.1f}%)")
for gw in sorted(by_gw.keys()):
    d = by_gw[gw]
    print(f"  {gw} GW: {d['ok']}/{d['n']}")

if failures:
    print(f"\n  Failures ({len(failures)}):")
    for f in failures[:10]:
        print(f"    sid={f.get('sid')} GW={f.get('gw')} "
              f"conv={f.get('converged')} vv={f.get('v_viol')} "
              f"line={f.get('line_over')} trafo={f.get('trafo_over')} n1={f.get('n1_viol')}")

results = {
    'description': 'Lookup baseline: demand->training label, with REAL AC-PF + N-1 verification',
    'exact_lookup_verified': f'{n_ok}/{len(test_ids)} ({100*n_ok/len(test_ids):.1f}%)',
    'by_gw': {str(k): v for k, v in sorted(by_gw.items())},
    'failures': failures,
}

out_path = PROJ_DIR / 'paper_results' / 'lookup_baseline_results.json'
with open(str(out_path), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")

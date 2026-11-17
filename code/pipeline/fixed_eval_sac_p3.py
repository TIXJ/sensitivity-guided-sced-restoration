"""
Fixed SAC from-scratch evaluation with PROPER N-1 verification.
Fixes the V35 bug where _compute_n1_sensitivity() was empty,
causing n1_violations to always be 0.

Now calls _compute_n1_overload_margin() (which IS implemented correctly
using S @ P + LODF-based post-contingency flow check) during the pass check.
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

from train_rlv35_complete import (SCEDEnvV21, SACTrainer, split_samples_stratified,
    DEVICE, N_SHIFT_DIRECTIONS, N_KEY_GENERATORS)

# ===== PATCH: Fix N-1 verification =====
def _patched_compute_n1(self):
    """Replace empty _compute_n1_sensitivity with real N-1 check."""
    return self._compute_n1_overload_margin()

SCEDEnvV21._compute_n1_sensitivity = _patched_compute_n1

print("Loading dataset...")
with open(str(PROJ_DIR / 'case300_gt_v10_dataset.json'), 'r') as f:
    data = json.load(f)
samples = [s for s in data['samples'] if s.get('is_complete_sced')]
train_ids, val_ids, test_ids = split_samples_stratified(samples)
env = SCEDEnvV21(samples, test_ids)
od = env._get_obs_dim()
ad = env.action_dim
print(f"Test: {len(test_ids)} samples, ObsDim={od}, ActDim={ad}")

t = SACTrainer(od, ad, device=DEVICE)
ckpt_path = str(PROJ_DIR / 'rl_scratch_sac' / 'best_actor.pth')
t.actor.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=False))
print(f"Loaded: {ckpt_path}")


def run(sid, det):
    obs = env.reset(sample_idx=sid)
    for step in range(env.max_steps + 1):
        raw_a = t.select_action(obs, deterministic=det)
        lo, hi = env.get_action_bounds()
        a = lo + (raw_a + 1.0) * 0.5 * (hi - lo)
        obs, r, done, _, info = env.step(a)
        if done:
            break

    # PROPER pass check with real N-1 verification
    converged = info.get('converged', False)
    v_viol = info.get('v_violations', 999)
    line_over = info.get('overloaded_lines', 999)
    trafo_over = info.get('trafo_overloaded', 999)

    # ★ FIX: Call the real N-1 check
    n1_overload_mw, n1_viol = env._compute_n1_overload_margin()

    ok = (converged and v_viol == 0 and line_over == 0
          and trafo_over == 0 and n1_viol == 0)
    return ok, converged, v_viol, line_over, trafo_over, n1_viol


# ===== P2: Mean-action =====
print("\n=== P2: Mean-action evaluation ===")
n_p2 = 0
by_gw_p2 = {}
for sid in test_ids:
    gw = int(samples[sid]['demand_mw'] / 1000)
    if gw not in by_gw_p2:
        by_gw_p2[gw] = {'ok': 0, 'n': 0}
    by_gw_p2[gw]['n'] += 1
    ok, _, _, _, _, n1 = run(sid, det=True)
    if ok:
        n_p2 += 1
        by_gw_p2[gw]['ok'] += 1
    else:
        pass  # will try P3

print(f"P2 mean-action: {n_p2}/{len(test_ids)} ({100*n_p2/len(test_ids):.1f}%)")

# ===== P3: Stochastic retry (K=3) =====
print("\n=== P3: Stochastic retry (K=3) ===")
n_p3 = 0
by_gw_p3 = {}
for sid in test_ids:
    gw = int(samples[sid]['demand_mw'] / 1000)
    if gw not in by_gw_p3:
        by_gw_p3[gw] = {'p2': 0, 'p3': 0, 'n': 0}
    by_gw_p3[gw]['n'] += 1

    # Try mean-action first
    ok, _, _, _, _, _ = run(sid, det=True)
    if ok:
        n_p3 += 1
        by_gw_p3[gw]['p2'] += 1
        by_gw_p3[gw]['p3'] += 1
        continue

    # Try K=3 stochastic retries
    rescued = False
    for rr in range(3):
        ok, _, _, _, _, _ = run(sid, det=False)
        if ok:
            n_p3 += 1
            by_gw_p3[gw]['p3'] += 1
            rescued = True
            break
    if not rescued:
        by_gw_p3[gw]['p3'] += 0

print(f"P2+P3 stochastic retry: {n_p3}/{len(test_ids)} ({100*n_p3/len(test_ids):.1f}%)")

# ===== Results =====
results = {
    'description': 'Fixed SAC from-scratch eval with PROPER N-1 verification',
    'checkpoint': str(ckpt_path),
    'total': len(test_ids),
    'p2_pass': n_p2,
    'p2_rate': round(100 * n_p2 / len(test_ids), 1),
    'p3_pass': n_p3,
    'p3_rate': round(100 * n_p3 / len(test_ids), 1),
    'p3_rescued': n_p3 - n_p2,
    'by_gw_p2': {str(k): v for k, v in sorted(by_gw_p2.items())},
    'by_gw_p3': {str(k): v for k, v in sorted(by_gw_p3.items())},
}

print(f"\n=== SUMMARY ===")
print(f"P2 mean-action : {n_p2}/{len(test_ids)} ({100*n_p2/len(test_ids):.1f}%)")
print(f"P2+P3 retry    : {n_p3}/{len(test_ids)} ({100*n_p3/len(test_ids):.1f}%)")
print(f"P3 rescued     : {n_p3 - n_p2}")
print(f"\nPer-GW breakdown (P3):")
for gw in sorted(by_gw_p3.keys()):
    d = by_gw_p3[gw]
    print(f"  {gw}GW: P2={d.get('p2',0)}/{d['n']}, P3={d.get('p3',0)}/{d['n']}")

out_path = PROJ_DIR / 'paper_results' / 'fixed_n1_eval_results.json'
with open(str(out_path), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")

"""
Cost-gap analysis for the final method (rl_nocur, hybrid no-curriculum, seeded, strict).
For each FEASIBLE corrected dispatch P* (passes strict AC-PF + N-1 + power balance),
report the generation-cost gap relative to:
  - the lowest-cost training-partition label at that demand level (QP reference)
  - the GAT-projected warm-start P0
and the dispatch deviation ||P*-P0||_1 and max single-generator adjustment.
"""
import sys, json, numpy as np, torch, random
from pathlib import Path
from collections import defaultdict
PROJ = Path(r".")
sys.path.insert(0, str(PROJ))
import train_rlv35_complete as v
v.AC_PF_CHECK_EVERY = 5; v.N1_CHECK_EVERY_STEPS = 5
v.MAX_STEPS = 50; v.MAX_STEPS_HIGH = 70
v.N_SHIFT_DIRECTIONS = 12; v.N_KEY_GENERATORS = 20
from train_rlv35_complete import SCEDEnvV21, SACTrainer, split_samples_stratified, DEVICE

SEED = 42
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

print("Loading dataset + model...", flush=True)
data = json.load(open(str(PROJ / 'case300_gt_v10_dataset.json')))
samples = [s for s in data['samples'] if s.get('is_complete_sced')]
train_ids, _, test_ids = split_samples_stratified(samples)

env = SCEDEnvV21(samples, test_ids)
od, ad = env._get_obs_dim(), env.action_dim
t = SACTrainer(od, ad, device=DEVICE)
t.actor.load_state_dict(torch.load(str(PROJ / 'rl_nocur' / 'best_actor.pth'),
                                    map_location='cpu', weights_only=False))

# Cost coefficients straight from the env's network (same object the labels used)
cp2 = np.asarray(env.dcnet.cp2, dtype=np.float64)
cp1 = np.asarray(env.dcnet.cp1, dtype=np.float64)
cp0 = np.asarray(env.dcnet.cp0, dtype=np.float64)
def cost(P): return float(np.sum(cp2 * P**2 + cp1 * P + cp0))

# lowest-ORIGINAL-cost training label per demand level (unperturbed cost)
ref = {}
for sid in train_ids:
    s = samples[sid]; d = int(s['demand_mw'])
    P = np.array(s['dispatch_68'], dtype=np.float64)
    c = cost(P)
    if d not in ref or c < ref[d][1]:
        ref[d] = (P, c)

def check(env):
    if abs(env.current_power.sum() - env.target_demand) > 0.01 * env.target_demand:
        return False
    ac = env._run_ac_pf_check()
    _, n1 = env._compute_n1_overload_margin()
    return (ac.get('converged', False) and ac.get('v_violations', 999) == 0
            and ac.get('overloaded_lines', 999) == 0
            and ac.get('trafo_overloaded', 999) == 0 and n1 == 0)

def rollout(sid, det):
    obs = env.reset(sample_idx=sid)
    P0 = env.current_power.copy()
    for _ in range(env.max_steps + 1):
        lo, hi = env.get_action_bounds()
        a = lo + (t.select_action(obs, deterministic=det) + 1.0) * 0.5 * (hi - lo)
        obs, _, done, _, info = env.step(a)
        if done: break
    return P0, env.current_power.copy()

gaps_ref = []; gaps_p0 = []; l1s = []; maxd = []; by_gw = defaultdict(list)
n_feasible = 0
print("Rolling out test set...", flush=True)
for sid in test_ids:
    gw = int(samples[sid]['demand_mw'] / 1000)
    P0, Pf = rollout(sid, True)
    ok = check(env)
    if not ok:
        for _ in range(3):
            P0, Pf = rollout(sid, False)
            if check(env): ok = True; break
    if not ok:
        continue
    d = int(samples[sid]['demand_mw'])
    if d not in ref:
        continue
    n_feasible += 1
    rc, fc, pc = ref[d][1], cost(Pf), cost(P0)
    g_ref = (fc - rc) / rc * 100
    g_p0 = (fc - pc) / pc * 100
    gaps_ref.append(g_ref); gaps_p0.append(g_p0)
    l1s.append(float(np.sum(np.abs(Pf - P0)))); maxd.append(float(np.max(np.abs(Pf - P0))))
    by_gw[gw].append(g_ref)
    print(f"  GW={gw} sid={sid:3d}  gap_vs_ref={g_ref:+.2f}%  gap_vs_P0={g_p0:+.2f}%  L1={l1s[-1]:.0f}MW", flush=True)

res = {
    'n_feasible': n_feasible,
    'gap_vs_ref_mean_pct': round(float(np.mean(gaps_ref)), 2),
    'gap_vs_ref_max_pct': round(float(np.max(gaps_ref)), 2),
    'gap_vs_ref_median_pct': round(float(np.median(gaps_ref)), 2),
    'gap_vs_p0_mean_pct': round(float(np.mean(gaps_p0)), 2),
    'gap_vs_p0_max_pct': round(float(np.max(gaps_p0)), 2),
    'l1_mean_mw': round(float(np.mean(l1s)), 1),
    'l1_max_mw': round(float(np.max(l1s)), 1),
    'max_gen_delta_mean_mw': round(float(np.mean(maxd)), 1),
    'max_gen_delta_max_mw': round(float(np.max(maxd)), 1),
    'by_gw_gap_vs_ref': {str(k): round(float(np.mean(vs)), 2) for k, vs in sorted(by_gw.items())},
}
print("\n=== COST GAP SUMMARY ===", flush=True)
print(json.dumps(res, indent=2), flush=True)
json.dump(res, open(str(PROJ / 'paper_results' / 'cost_gap_final.json'), 'w'), indent=2)
print("Saved paper_results/cost_gap_final.json", flush=True)

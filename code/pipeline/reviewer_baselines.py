"""
Two reviewer-requested baselines, evaluated under the SAME strict protocol
(enforce_q_lims AC-PF + fresh N-1 + power-balance guard) and fixed seed:

  #4  Lookup + SAC : SAC correction starting from the training-label lookup
                     dispatch instead of the GAT prediction. Answers "is the GAT
                     warm-start necessary, or can we just look up the label and
                     correct it?"
  #8  Hybrid (no curriculum) + stochastic retry : does the retry gain depend on
                     curriculum? Compares against Proposed (72.7%).
"""
import sys, json, numpy as np, torch, random, time
from pathlib import Path
from collections import defaultdict
PROJ = Path(r".")
sys.path.insert(0, str(PROJ))

import train_rlv35_complete as v
v.AC_PF_CHECK_EVERY = 5; v.N1_CHECK_EVERY_STEPS = 5
v.MAX_STEPS = 50; v.MAX_STEPS_HIGH = 70
from train_rlv35_complete import SCEDEnvV21, SACTrainer, split_samples_stratified, DEVICE

SEED = 42
def seed_all():
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

data = json.load(open(str(PROJ / 'case300_gt_v10_dataset.json')))
samples = [s for s in data['samples'] if s.get('is_complete_sced')]
train_ids, _, test_ids = split_samples_stratified(samples)

# lowest-cost training label per demand level
lookup = {}
for sid in train_ids:
    s = samples[sid]; dmw = int(s['demand_mw'])
    P = np.array(s['dispatch_68'], dtype=np.float64); cost = float(s.get('sced_cost_per_h', 1e9))
    if dmw not in lookup or cost < lookup[dmw][1]:
        lookup[dmw] = (P, cost)

def check_pass(info, env):
    if abs(env.current_power.sum() - env.target_demand) > 0.01 * env.target_demand: return False
    n1_v = getattr(env, '_n1_violations', 0)
    return (info.get('converged', False) and info.get('v_violations', 999) == 0
            and info.get('overloaded_lines', 999) == 0
            and info.get('trafo_overloaded', 999) == 0 and n1_v == 0)

def final_check(info, env):
    ac = env._run_ac_pf_check(); info = dict(info); info.update(ac)
    _, n1 = env._compute_n1_overload_margin(); env._n1_violations = n1
    return check_pass(info, env)

def macro(by_gw): return round(100*np.mean([x['ok']/x['n'] for x in by_gw.values()]), 1)

def run_eval(ckpt, n_shift, n_key, lookup_start=False, retry=False):
    seed_all()
    v.N_SHIFT_DIRECTIONS = n_shift; v.N_KEY_GENERATORS = n_key
    env = SCEDEnvV21(samples, test_ids)
    od, ad = env._get_obs_dim(), env.action_dim
    tr = SACTrainer(od, ad, device=DEVICE)
    tr.actor.load_state_dict(torch.load(str(ckpt), map_location='cpu', weights_only=False))
    det = lambda o: tr.select_action(o, deterministic=True)
    sto = lambda o: tr.select_action(o, deterministic=False)

    def set_lookup():
        dmw = int(env.target_demand)
        if dmw in lookup:
            env.current_power = lookup[dmw][0].copy()
            env.gat_power = env.current_power.copy()
        return env._get_obs()

    n_ok = 0; failed = []; by = defaultdict(lambda: {'ok':0,'n':0})
    for sid in test_ids:
        gw = int(samples[sid]['demand_mw']/1000); by[gw]['n'] += 1
        obs = env.reset(sample_idx=sid)
        if lookup_start: obs = set_lookup()
        for _ in range(env.max_steps+1):
            lo, hi = env.get_action_bounds(); a = lo + (det(obs)+1.0)*0.5*(hi-lo)
            obs, _, done, _, info = env.step(a)
            if done: break
        if final_check(info, env): n_ok += 1; by[gw]['ok'] += 1
        else: failed.append(sid)
    res = {'p2': n_ok, 'p2_rate': round(100*n_ok/len(test_ids),1), 'p2_macro': macro(by)}
    if retry:
        by3 = {k: dict(x) for k,x in by.items()}; n3 = n_ok; resc = 0
        for sid in failed:
            gw = int(samples[sid]['demand_mw']/1000); ok = False
            for _ in range(3):
                obs = env.reset(sample_idx=sid)
                if lookup_start: obs = set_lookup()
                for _ in range(env.max_steps+1):
                    lo, hi = env.get_action_bounds(); a = lo + (sto(obs)+1.0)*0.5*(hi-lo)
                    obs, _, done, _, info = env.step(a)
                    if done: break
                if final_check(info, env): ok = True; break
            if ok: n3 += 1; resc += 1; by3[gw]['ok'] += 1
        res.update({'p3': n3, 'p3_rate': round(100*n3/len(test_ids),1),
                    'p3_macro': round(100*np.mean([x['ok']/x['n'] for x in by3.values()]),1), 'rescued': resc})
    return res

out = {}
print("== #4 Lookup + SAC correction (start from training label, not GAT) ==")
out['Lookup+SAC mean'] = run_eval(PROJ/'rl_scratch_sac'/'best_actor.pth', 12, 20, lookup_start=True, retry=True)
print(f"   P2={out['Lookup+SAC mean']['p2_rate']}%  P3={out['Lookup+SAC mean']['p3_rate']}%")

print("== #8 Hybrid (no curriculum) + stochastic retry ==")
out['Hybrid-nocur+retry'] = run_eval(PROJ/'rl_nocur'/'best_actor.pth', 12, 20, lookup_start=False, retry=True)
print(f"   P2={out['Hybrid-nocur+retry']['p2_rate']}%  P3={out['Hybrid-nocur+retry']['p3_rate']}%")

json.dump(out, open(str(PROJ/'paper_results'/'reviewer_baselines.json'),'w'), indent=2)
print("\nSaved: paper_results/reviewer_baselines.json")

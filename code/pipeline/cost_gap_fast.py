"""Fast cost-gap analysis for the final method (rl_nocur, seeded, strict).
For each feasible corrected dispatch, report cost gap vs the lowest-cost training
label at that demand level, and the dispatch deviation from the GAT warm-start."""
import sys, json, numpy as np, torch, random
from pathlib import Path
from collections import defaultdict
PROJ = Path(r".")
sys.path.insert(0, str(PROJ))
import train_rlv35_complete as v
v.AC_PF_CHECK_EVERY=5; v.N1_CHECK_EVERY_STEPS=5; v.MAX_STEPS=50; v.MAX_STEPS_HIGH=70
v.N_SHIFT_DIRECTIONS=12; v.N_KEY_GENERATORS=20
from train_rlv35_complete import SCEDEnvV21, SACTrainer, split_samples_stratified, DEVICE
from generate_case300_gt_v10 import Case300Network
SEED=42
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

data=json.load(open(str(PROJ/'case300_gt_v10_dataset.json')))
samples=[s for s in data['samples'] if s.get('is_complete_sced')]
train_ids,_,test_ids=split_samples_stratified(samples)
dc=Case300Network(); cp2=np.asarray(dc.cp2); cp1=np.asarray(dc.cp1); cp0=np.asarray(dc.cp0)
cost=lambda P: float(np.sum(cp2*P**2+cp1*P+cp0))

# lowest-cost training label per demand level (precomputed once)
ref={}
for sid in train_ids:
    s=samples[sid]; d=int(s['demand_mw']); c=float(s.get('sced_cost_per_h',1e18))
    if d not in ref or c<ref[d][1]: ref[d]=(np.array(s['dispatch_68'],dtype=np.float64),c)

env=SCEDEnvV21(samples,test_ids); od,ad=env._get_obs_dim(),env.action_dim
t=SACTrainer(od,ad,device=DEVICE)
t.actor.load_state_dict(torch.load(str(PROJ/'rl_nocur'/'best_actor.pth'),map_location='cpu',weights_only=False))

def check(info):
    if abs(env.current_power.sum()-env.target_demand)>0.01*env.target_demand: return False
    _,n1=env._compute_n1_overload_margin()
    ac=env._run_ac_pf_check()
    return (ac.get('converged',False) and ac.get('v_violations',999)==0
            and ac.get('overloaded_lines',999)==0 and ac.get('trafo_overloaded',999)==0 and n1==0)

def rollout(sid,det):
    obs=env.reset(sample_idx=sid); P0=env.current_power.copy()
    for _ in range(env.max_steps+1):
        lo,hi=env.get_action_bounds(); a=lo+(t.select_action(obs,deterministic=det)+1.0)*0.5*(hi-lo)
        obs,_,d,_,info=env.step(a)
        if d: break
    return P0, env.current_power.copy()

gaps_ref=[]; gaps_p0=[]; l1s=[]; maxd=[]
for sid in test_ids:
    d=int(samples[sid]['demand_mw'])
    P0,Pf=rollout(sid,True)
    okf = check({})
    if not okf:
        for _ in range(3):
            P0,Pf=rollout(sid,False)
            if check({}): okf=True; break
    if not okf: continue
    if d not in ref: continue
    rc=cost(ref[d][0]); fc=cost(Pf); pc=cost(P0)
    gaps_ref.append((fc-rc)/(rc+1e-9)*100); gaps_p0.append((fc-pc)/(pc+1e-9)*100)
    l1s.append(np.sum(np.abs(Pf-P0))); maxd.append(np.max(np.abs(Pf-P0)))

print(f"\n=== Cost gap ({len(gaps_ref)} feasible dispatches) ===")
print(f"vs QP reference: mean={np.mean(gaps_ref):.2f}%  max={np.max(gaps_ref):.2f}%")
print(f"vs GAT P0:       mean={np.mean(gaps_p0):.2f}%  max={np.max(gaps_p0):.2f}%")
print(f"||P*-P0||_1:     mean={np.mean(l1s):.0f} MW  max={np.max(l1s):.0f} MW")
print(f"max single-gen delta: mean={np.mean(maxd):.0f} MW  max={np.max(maxd):.0f} MW")
json.dump({'n':len(gaps_ref),'gap_vs_ref_mean':round(float(np.mean(gaps_ref)),2),
           'gap_vs_ref_max':round(float(np.max(gaps_ref)),2),
           'gap_vs_p0_mean':round(float(np.mean(gaps_p0)),2),
           'l1_mean_mw':round(float(np.mean(l1s)),0),'l1_max_mw':round(float(np.max(l1s)),0),
           'max_gen_delta_mean_mw':round(float(np.mean(maxd)),0)},
          open(str(PROJ/'paper_results'/'cost_gap_final.json'),'w'),indent=2)
print("Saved cost_gap_final.json")

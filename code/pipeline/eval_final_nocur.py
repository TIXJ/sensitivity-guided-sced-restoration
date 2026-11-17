"""Evaluate the NEW final method: Hybrid (no curriculum) + stochastic retry,
with full by-GW breakdown for P2 (mean-action) and P3 (retry). Seeded, strict."""
import sys, json, numpy as np, torch, random
from pathlib import Path
from collections import defaultdict
PROJ = Path(r".")
sys.path.insert(0, str(PROJ))
import train_rlv35_complete as v
v.AC_PF_CHECK_EVERY = 5; v.N1_CHECK_EVERY_STEPS = 5
v.MAX_STEPS = 50; v.MAX_STEPS_HIGH = 70
from train_rlv35_complete import SCEDEnvV21, SACTrainer, split_samples_stratified, DEVICE
SEED=42
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

data = json.load(open(str(PROJ/'case300_gt_v10_dataset.json')))
samples=[s for s in data['samples'] if s.get('is_complete_sced')]
_,_,test_ids = split_samples_stratified(samples)
v.N_SHIFT_DIRECTIONS=12; v.N_KEY_GENERATORS=20
env=SCEDEnvV21(samples,test_ids); od,ad=env._get_obs_dim(),env.action_dim
t=SACTrainer(od,ad,device=DEVICE)
t.actor.load_state_dict(torch.load(str(PROJ/'rl_nocur'/'best_actor.pth'),map_location='cpu',weights_only=False))

def check(info,env):
    if abs(env.current_power.sum()-env.target_demand)>0.01*env.target_demand: return False
    n1=getattr(env,'_n1_violations',0)
    return (info.get('converged',False) and info.get('v_violations',999)==0
            and info.get('overloaded_lines',999)==0 and info.get('trafo_overloaded',999)==0 and n1==0)
def final(info,env):
    ac=env._run_ac_pf_check(); info=dict(info); info.update(ac)
    _,n1=env._compute_n1_overload_margin(); env._n1_violations=n1
    return check(info,env)

by2=defaultdict(lambda:{'ok':0,'n':0}); failed=[]; n2=0
for sid in test_ids:
    gw=int(samples[sid]['demand_mw']/1000); by2[gw]['n']+=1
    obs=env.reset(sample_idx=sid)
    for _ in range(env.max_steps+1):
        lo,hi=env.get_action_bounds(); a=lo+(t.select_action(obs,deterministic=True)+1.0)*0.5*(hi-lo)
        obs,_,d,_,info=env.step(a)
        if d: break
    if final(info,env): n2+=1; by2[gw]['ok']+=1
    else: failed.append(sid)
by3={k:dict(x) for k,x in by2.items()}; n3=n2; resc=0
for sid in failed:
    gw=int(samples[sid]['demand_mw']/1000); ok=False
    for _ in range(3):
        obs=env.reset(sample_idx=sid)
        for _ in range(env.max_steps+1):
            lo,hi=env.get_action_bounds(); a=lo+(t.select_action(obs,deterministic=False)+1.0)*0.5*(hi-lo)
            obs,_,d,_,info=env.step(a)
            if d: break
        if final(info,env): ok=True; break
    if ok: n3+=1; resc+=1; by3[gw]['ok']+=1
mac=lambda b: round(100*np.mean([x['ok']/x['n'] for x in b.values()]),1)
res={'p2':n2,'p2_rate':round(100*n2/44,1),'p2_macro':mac(by2),
     'p3':n3,'p3_rate':round(100*n3/44,1),'p3_macro':mac(by3),'rescued':resc,
     'by_gw':{str(k):v for k,v in sorted(by2.items())},
     'by_gw_p3':{str(k):v for k,v in sorted(by3.items())}}
print(json.dumps(res,indent=2))
# by-GW one-liners
for k in sorted(by2,key=int):
    print(f"  {k}GW: P2 {by2[k]['ok']}/{by2[k]['n']}  P3 {by3[k]['ok']}/{by3[k]['n']}")
json.dump(res, open(str(PROJ/'paper_results'/'final_nocur_retry.json'),'w'), indent=2)
print("Saved final_nocur_retry.json")

"""Cost analysis on ALL 34 dispatches accepted by the FINAL stochastic-retry method
(mirrors eval_final_nocur.py's exact seeding and P2-then-P3 protocol). Records the
accepted dispatch P* for every passing trial (mean-action OR retry-rescued) and
reports relative cost increase vs the lowest-cost retained QP reference and vs P0."""
import sys, json, numpy as np, torch, random
from pathlib import Path
from collections import defaultdict
PROJ = Path(r".")
sys.path.insert(0, str(PROJ))
import train_rlv35_complete as v
v.AC_PF_CHECK_EVERY=5; v.N1_CHECK_EVERY_STEPS=5; v.MAX_STEPS=50; v.MAX_STEPS_HIGH=70
v.N_SHIFT_DIRECTIONS=12; v.N_KEY_GENERATORS=20
from train_rlv35_complete import SCEDEnvV21, SACTrainer, split_samples_stratified, DEVICE
SEED=42
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

data=json.load(open(str(PROJ/'case300_gt_v10_dataset.json')))
samples=[s for s in data['samples'] if s.get('is_complete_sced')]
train_ids,_,test_ids=split_samples_stratified(samples)
env=SCEDEnvV21(samples,test_ids); od,ad=env._get_obs_dim(),env.action_dim
t=SACTrainer(od,ad,device=DEVICE)
t.actor.load_state_dict(torch.load(str(PROJ/'rl_nocur'/'best_actor.pth'),map_location='cpu',weights_only=False))

cp2=np.asarray(env.dcnet.cp2); cp1=np.asarray(env.dcnet.cp1); cp0=np.asarray(env.dcnet.cp0)
cost=lambda P: float(np.sum(cp2*P**2+cp1*P+cp0))
ref={}
for sid in train_ids:
    s=samples[sid]; d=int(s['demand_mw']); P=np.array(s['dispatch_68'],dtype=np.float64); c=cost(P)
    if d not in ref or c<ref[d][1]: ref[d]=(P,c)

def check(env):
    if abs(env.current_power.sum()-env.target_demand)>0.01*env.target_demand: return False
    ac=env._run_ac_pf_check(); _,n1=env._compute_n1_overload_margin()
    return (ac.get('converged',False) and ac.get('v_violations',999)==0
            and ac.get('overloaded_lines',999)==0 and ac.get('trafo_overloaded',999)==0 and n1==0)

def rollout(sid,det):
    obs=env.reset(sample_idx=sid); P0=env.current_power.copy()
    for _ in range(env.max_steps+1):
        lo,hi=env.get_action_bounds(); a=lo+(t.select_action(obs,deterministic=det)+1.0)*0.5*(hi-lo)
        obs,_,d,_,info=env.step(a)
        if d: break
    return P0, env.current_power.copy()

accepted=[]  # (sid, gw, P0, Pstar, rescued_bool)
# P2 pass (mean-action) — exact same order as eval_final_nocur
failed=[]
for sid in test_ids:
    P0,Pf=rollout(sid,True)
    if check(env): accepted.append((sid,int(samples[sid]['demand_mw']/1000),P0.copy(),Pf.copy(),False))
    else: failed.append(sid)
# P3 retry (stochastic) on failed
for sid in failed:
    for _ in range(3):
        P0,Pf=rollout(sid,False)
        if check(env): accepted.append((sid,int(samples[sid]['demand_mw']/1000),P0.copy(),Pf.copy(),True)); break

print(f"Accepted by final method: {len(accepted)} (mean-action {sum(1 for a in accepted if not a[4])}, retry {sum(1 for a in accepted if a[4])})", flush=True)
grp=[]; grr=[]; l1=[]; l1n=[]; md=[]
for sid,gw,P0,Pf,resc in accepted:
    d=int(samples[sid]['demand_mw'])
    if d not in ref: continue
    rc=ref[d][1]; g=(cost(Pf)-rc)/rc*100; gp=(cost(Pf)-cost(P0))/cost(P0)*100
    grp.append(g); grr.append(gp); L=float(np.abs(Pf-P0).sum())
    l1.append(L); l1n.append(L/d*100); md.append(float(np.abs(Pf-P0).max()))
    print(f"  GW={gw} sid={sid:3d} {'RETRY' if resc else 'mean '} cost+{g:.2f}% vsP0+{gp:.2f}% L1={L:.0f}MW({L/d*100:.1f}%)", flush=True)

f=lambda x: (round(float(np.mean(x)),2), round(float(np.median(x)),2), round(float(np.max(x)),2))
res={'n_accepted':len(accepted),
     'cost_vs_ref_pct':dict(zip(['mean','median','max'],f(grp))),
     'cost_vs_p0_pct':dict(zip(['mean','median','max'],f(grr))),
     'l1_mw':dict(zip(['mean','median','max'],f(l1))),
     'l1_norm_pct':dict(zip(['mean','median','max'],f(l1n))),
     'max_gen_delta_mw':dict(zip(['mean','median','max'],f(md)))}
print("\n=== COST GAP (ALL 34 ACCEPTED) ===", flush=True)
print(json.dumps(res,indent=2), flush=True)
json.dump(res, open(str(PROJ/'paper_results'/'cost_gap_34.json'),'w'), indent=2)
print("Saved cost_gap_34.json", flush=True)

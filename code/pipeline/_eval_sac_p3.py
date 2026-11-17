"""Eval best SAC scratch (70.5%) with P3 stochastic retry (using learned distribution)."""
import sys,json,numpy as np,torch; from pathlib import Path
PROJ_DIR=Path(r"."); sys.path.insert(0,str(PROJ_DIR))
import train_rlv35_complete as _v35
_v35.AC_PF_CHECK_EVERY=5; _v35.N1_CHECK_EVERY_STEPS=5; _v35.MAX_STEPS=50; _v35.MAX_STEPS_HIGH=70
from train_rlv35_complete import *
from train_rlv35_complete import SACTrainer

with open(str(PROJ_DIR/'case300_gt_v10_dataset.json')) as f: data=json.load(f)
samples=[s for s in data['samples'] if s.get('is_complete_sced')]
_,_,test_ids=split_samples_stratified(samples)
env=SCEDEnvV21(samples,test_ids); od=env._get_obs_dim(); ad=env.action_dim
t=SACTrainer(od,ad,device=DEVICE)
t.actor.load_state_dict(torch.load(str(PROJ_DIR/'rl_scratch_sac/best_actor.pth'),map_location='cpu',weights_only=False))

def run(sid,det,n1_check=True):
    obs=env.reset(sample_idx=sid)
    for step in range(env.max_steps+1):
        raw_a=t.select_action(obs,deterministic=det)
        lo,hi=env.get_action_bounds(); a=lo+(raw_a+1.0)*0.5*(hi-lo)
        obs,r,done,_,info=env.step(a)
        if done: break
    return info.get('converged',False) and info.get('v_violations',999)==0 and info.get('n1_violations',999)==0 and info.get('overloaded_lines',999)==0

n_p2=0; n_p3=0; by_gw={}
for sid in test_ids:
    gw=int(samples[sid]['demand_mw']/1000)
    if gw not in by_gw: by_gw[gw]={'p2':0,'p3':0,'n':0}
    by_gw[gw]['n']+=1
    if run(sid,det=True): n_p2+=1; by_gw[gw]['p2']+=1
for sid in test_ids:
    gw=int(samples[sid]['demand_mw']/1000)
    if run(sid,det=True): n_p3+=1; by_gw[gw]['p3']+=1; continue
    for rr in range(3):
        if run(sid,det=False): n_p3+=1; by_gw[gw]['p3']+=1; break
n=len(test_ids)
print(f"SAC scratch: P2={n_p2}/{n} ({100*n_p2/n:.1f}%)")
print(f"SAC scratch: P2+P3={n_p3}/{n} ({100*n_p3/n:.1f}%) rescued={n_p3-n_p2}")
for gw in sorted(by_gw.keys()):
    d=by_gw[gw]; print(f"  {gw}GW: P2={d['p2']}/{d['n']}, P3={d['p3']}/{d['n']}")

"""Tier 1b: SAC without feasibility-preserving projection.
The per-generator capacity projection step is disabled (pass-through).
Train 100 updates, save to rl_noproj/."""
import sys, os, json, time, numpy as np, torch
from pathlib import Path
PROJ_DIR = Path(r".")
sys.path.insert(0, str(PROJ_DIR))

import train_rlv35_complete as _v35
_v35.AC_PF_CHECK_EVERY = 5; _v35.N1_CHECK_EVERY_STEPS = 5
_v35.N_GRADIENT_STEPS = 30; _v35.N_EPISODES_PER_UPDATE = 10
_v35.MAX_STEPS = 50; _v35.MAX_STEPS_HIGH = 70

from train_rlv35_complete import (SCEDEnvV21, SACTrainer, split_samples_stratified,
    curriculum_sample, DEVICE)

# Monkey-patch: disable feasibility-preserving projection (pass-through, only basic clip)
def _no_projection(self, new_power):
    """No projection — just clip to physical limits (bare minimum for AC-PF solver)."""
    return np.clip(new_power, self.PMIN, self.PMAX * 0.98)
SCEDEnvV21._project_action_per_gen_cap = _no_projection

print("╔══════════════════════════════════════════════════╗")
print("║  Tier 1b: SAC without projection                ║")
print("╚══════════════════════════════════════════════════╝")
with open(str(PROJ_DIR/'case300_gt_v10_dataset.json'),'r') as f: data=json.load(f)
samples=[s for s in data['samples'] if s.get('is_complete_sced')]
train_ids,_,_ = split_samples_stratified(samples)
env=SCEDEnvV21(samples,train_ids)
od=env._get_obs_dim(); ad=env.action_dim
print(f"ObsDim={od}, ActDim={ad}")

t=SACTrainer(od,ad,device=DEVICE)
rewards=[]; best_r=-1e9; t0=time.time()
save_dir=PROJ_DIR/'rl_noproj'; save_dir.mkdir(exist_ok=True)

for u in range(1,101):
    us=time.time(); eps=curriculum_sample(samples,train_ids,u,10); ep_rs=[]
    for sid in eps:
        obs=env.reset(sample_idx=sid); er=0.0
        for step in range(env.max_steps+1):
            raw_a=t.select_action(obs,deterministic=False)
            lo,hi=env.get_action_bounds(); a=lo+(raw_a+1.0)*0.5*(hi-lo)
            no,r,done,_,info=env.step(a); er+=r
            t.replay_buffer.push(obs,raw_a,r,no,float(done))
            obs=no
            if done: break
        ep_rs.append(er)
    t.update()
    avg_r=np.mean(ep_rs) if ep_rs else 0; rewards.append(avg_r)
    if avg_r>best_r: best_r=avg_r; torch.save(t.actor.state_dict(),str(save_dir/'best_actor.pth'))
    if u%10==0 or u==1: print(f"U{u:3d} R={avg_r:+.0f} {time.time()-us:.0f}s")
    with open(str(save_dir/'rewards.json'),'w') as f: json.dump({"algo":"SAC-NoProj","rewards":rewards},f)

torch.save(t.actor.state_dict(),str(save_dir/'final_actor.pth'))
print(f"SAC-NoProj done: {(time.time()-t0)/3600:.1f}h, best R={best_r:+.0f}")

"""Tier 2c: HIGH-focused SAC — Phase 2 only samples from 23-24 GW (not 80/20 split).
Train 100 updates, save to rl_highfocus/."""
import sys, os, json, time, numpy as np, torch
from pathlib import Path
from collections import defaultdict
import random
PROJ_DIR = Path(r".")
sys.path.insert(0, str(PROJ_DIR))

import train_rlv35_complete as _v35
_v35.AC_PF_CHECK_EVERY = 5; _v35.N1_CHECK_EVERY_STEPS = 5
_v35.N_GRADIENT_STEPS = 30; _v35.N_EPISODES_PER_UPDATE = 10
_v35.MAX_STEPS = 50; _v35.MAX_STEPS_HIGH = 70
_v35.PHASE_1_END = 30; _v35.PHASE_2_END = 100

from train_rlv35_complete import (SCEDEnvV21, SACTrainer, split_samples_stratified,
    DEVICE)

# ── Custom curriculum: Phase 2 = 100% from 23-24 GW ──
def _highfocus_curriculum(samples, pool_ids, update_num, n_total=10):
    by_demand = defaultdict(list)
    for sid in pool_ids:
        gw = int(samples[sid]['demand_mw'] / 1000)
        by_demand[gw].append(sid)
    if update_num <= 30:
        # Phase 1: uniform (same as standard)
        allowed_gw = sorted(by_demand.keys())
        n_per_seg = max(1, n_total // len(allowed_gw))
        selected = []
        for gw in allowed_gw:
            pool = list(by_demand[gw])
            n = min(n_per_seg, len(pool))
            selected.extend(random.sample(pool, n))
    else:
        # Phase 2: 100% from 23-24 GW only
        high_pool = by_demand.get(23, []) + by_demand.get(24, [])
        selected = []
        if high_pool:
            n = min(n_total, len(high_pool))
            selected = random.sample(high_pool, n * 2) if len(high_pool) > n else high_pool[:]
            random.shuffle(selected)
            selected = selected[:n]
    # Pad if needed
    while len(selected) < n_total:
        all_pool = []
        for gw in by_demand: all_pool.extend(by_demand[gw])
        if all_pool: selected.append(random.choice(all_pool))
        else: break
    random.shuffle(selected)
    return selected[:n_total]

print("╔══════════════════════════════════════════════════╗")
print("║  Tier 2c: HIGH-focused SAC (Phase 2: 23-24GW)   ║")
print("╚══════════════════════════════════════════════════╝")
with open(str(PROJ_DIR/'case300_gt_v10_dataset.json'),'r') as f: data=json.load(f)
samples=[s for s in data['samples'] if s.get('is_complete_sced')]
train_ids,_,_ = split_samples_stratified(samples)
env=SCEDEnvV21(samples,train_ids)
od=env._get_obs_dim(); ad=env.action_dim
print(f"ObsDim={od}, ActDim={ad}")

t=SACTrainer(od,ad,device=DEVICE)
rewards=[]; best_r=-1e9; t0=time.time()
save_dir=PROJ_DIR/'rl_highfocus'; save_dir.mkdir(exist_ok=True)

for u in range(1,101):
    us=time.time(); eps=_highfocus_curriculum(samples,train_ids,u,10); ep_rs=[]
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
    with open(str(save_dir/'rewards.json'),'w') as f: json.dump({"algo":"SAC-HighFocus","rewards":rewards},f)

torch.save(t.actor.state_dict(),str(save_dir/'final_actor.pth'))
print(f"SAC-HighFocus done: {(time.time()-t0)/3600:.1f}h, best R={best_r:+.0f}")

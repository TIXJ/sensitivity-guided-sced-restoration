"""Train one RL algorithm from scratch, save rewards + checkpoints."""
import sys, os, json, time, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from collections import deque; import random
from pathlib import Path

algo_name = sys.argv[1]
PROJ_DIR = Path(r".")
sys.path.insert(0, str(PROJ_DIR))

import train_rlv35_complete as _v35
_v35.AC_PF_CHECK_EVERY = 5; _v35.N1_CHECK_EVERY_STEPS = 5
_v35.N_GRADIENT_STEPS = 30; _v35.N_EPISODES_PER_UPDATE = 10
_v35.MAX_STEPS = 50; _v35.MAX_STEPS_HIGH = 70

from train_rlv35_complete import (SCEDEnvV21, split_samples_stratified, curriculum_sample,
    DEVICE, N_SHIFT_DIRECTIONS, N_KEY_GENERATORS, BATCH_SIZE, BUFFER_SIZE)
TOTAL_UPDATES=100; BATCH=256; DEV=DEVICE

class Actor(nn.Module):
    def __init__(self,od,ad,h=256): super().__init__(); self.net=nn.Sequential(nn.Linear(od,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,ad),nn.Tanh())
    def forward(self,x): return self.net(x)
class Critic(nn.Module):
    def __init__(self,od,ad,h=256): super().__init__(); self.net=nn.Sequential(nn.Linear(od+ad,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))
    def forward(self,x,a): return self.net(torch.cat([x,a],-1))
class TwinCritic(nn.Module):
    def __init__(self,od,ad,h=256): super().__init__(); self.q1=Critic(od,ad,h); self.q2=Critic(od,ad,h)
    def forward(self,x,a): return self.q1(x,a),self.q2(x,a)

class DDPG:
    def __init__(self,od,ad):
        self.actor=Actor(od,ad).to(DEV); self.at=Actor(od,ad).to(DEV); self.at.load_state_dict(self.actor.state_dict())
        self.critic=Critic(od,ad).to(DEV); self.ct=Critic(od,ad).to(DEV); self.ct.load_state_dict(self.critic.state_dict())
        self.ao=torch.optim.Adam(self.actor.parameters(),lr=1e-3); self.co=torch.optim.Adam(self.critic.parameters(),lr=1e-3)
        self.buf=deque(maxlen=BUFFER_SIZE); self.gs=0
    def act(self,obs,det=True):
        with torch.no_grad(): o=torch.FloatTensor(obs).unsqueeze(0).to(DEV); a=self.actor(o)
        if not det: a=(a+torch.randn_like(a)*0.3).clamp(-1,1)
        return a.cpu().numpy()[0]
    def update(self):
        if len(self.buf)<BATCH: return
        for _ in range(30):
            idx=np.random.choice(len(self.buf),min(BATCH,len(self.buf)),replace=False)
            batch=[self.buf[i] for i in idx]
            s=torch.FloatTensor(np.array([b[0] for b in batch])).to(DEV); a=torch.FloatTensor(np.array([b[1] for b in batch])).to(DEV)
            r=torch.FloatTensor(np.array([b[2] for b in batch])).unsqueeze(-1).to(DEV)
            ns=torch.FloatTensor(np.array([b[3] for b in batch])).to(DEV); d=torch.FloatTensor(np.array([b[4] for b in batch])).unsqueeze(-1).to(DEV)
            with torch.no_grad(): tq=r+0.99*(1-d)*self.ct(ns,self.at(ns))
            cl=F.mse_loss(self.critic(s,a),tq); self.co.zero_grad(); cl.backward(); nn.utils.clip_grad_norm_(self.critic.parameters(),1.0); self.co.step()
            if self.gs%2==0:
                al=-self.critic(s,self.actor(s)).mean(); self.ao.zero_grad(); al.backward(); nn.utils.clip_grad_norm_(self.actor.parameters(),1.0); self.ao.step()
                for p,tp in zip(self.actor.parameters(),self.at.parameters()): tp.data.copy_(0.005*p+0.995*tp)
                for p,tp in zip(self.critic.parameters(),self.ct.parameters()): tp.data.copy_(0.005*p+0.995*tp)
            self.gs+=1

class TD3:
    def __init__(self,od,ad):
        self.actor=Actor(od,ad).to(DEV); self.at=Actor(od,ad).to(DEV); self.at.load_state_dict(self.actor.state_dict())
        self.critic=TwinCritic(od,ad).to(DEV); self.ct=TwinCritic(od,ad).to(DEV); self.ct.load_state_dict(self.critic.state_dict())
        self.ao=torch.optim.Adam(self.actor.parameters(),lr=1e-3); self.co=torch.optim.Adam(self.critic.parameters(),lr=1e-3)
        self.buf=deque(maxlen=BUFFER_SIZE); self.gs=0
    def act(self,obs,det=True):
        with torch.no_grad(): o=torch.FloatTensor(obs).unsqueeze(0).to(DEV); a=self.actor(o)
        if not det: a=(a+torch.randn_like(a)*0.3).clamp(-1,1)
        return a.cpu().numpy()[0]
    def update(self):
        if len(self.buf)<BATCH: return
        for _ in range(30):
            idx=np.random.choice(len(self.buf),min(BATCH,len(self.buf)),replace=False)
            batch=[self.buf[i] for i in idx]
            s=torch.FloatTensor(np.array([b[0] for b in batch])).to(DEV); a=torch.FloatTensor(np.array([b[1] for b in batch])).to(DEV)
            r=torch.FloatTensor(np.array([b[2] for b in batch])).unsqueeze(-1).to(DEV)
            ns=torch.FloatTensor(np.array([b[3] for b in batch])).to(DEV); d=torch.FloatTensor(np.array([b[4] for b in batch])).unsqueeze(-1).to(DEV)
            with torch.no_grad():
                noise=(torch.randn_like(a)*0.2).clamp(-0.5,0.5); na=(self.at(ns)+noise).clamp(-1,1); q1t,q2t=self.ct(ns,na)
                tq=r+0.99*(1-d)*torch.min(q1t,q2t)
            q1,q2=self.critic(s,a); cl=F.mse_loss(q1,tq.detach())+F.mse_loss(q2,tq.detach())
            self.co.zero_grad(); cl.backward(); nn.utils.clip_grad_norm_(self.critic.parameters(),1.0); self.co.step()
            if self.gs%2==0:
                al=-self.critic.q1(s,self.actor(s)).mean(); self.ao.zero_grad(); al.backward(); nn.utils.clip_grad_norm_(self.actor.parameters(),1.0); self.ao.step()
                for p,tp in zip(self.actor.parameters(),self.at.parameters()): tp.data.copy_(0.005*p+0.995*tp)
                for p,tp in zip(self.critic.parameters(),self.ct.parameters()): tp.data.copy_(0.005*p+0.995*tp)
            self.gs+=1

# Use exact V3 PPOTrainer (proven working)
from train_rl_comparison_v3 import PPOTrainer as PPO

# SAC scratch using V35 SACTrainer
from train_rlv35_complete import SACTrainer as SAC_Scratch

print(f"Training {algo_name} for {TOTAL_UPDATES} updates on {DEV}")
with open(str(PROJ_DIR/'case300_gt_v10_dataset.json'),'r') as f: data=json.load(f)
samples=[s for s in data['samples'] if s.get('is_complete_sced')]
train_ids,_,_ = split_samples_stratified(samples)
env=SCEDEnvV21(samples,train_ids); od=env._get_obs_dim(); ad=env.action_dim
print(f"ObsDim={od}, ActDim={ad}")

if algo_name=='DDPG': t=DDPG(od,ad); is_ppo=False; is_sac=False
elif algo_name=='TD3': t=TD3(od,ad); is_ppo=False; is_sac=False
elif algo_name=='PPO': t=PPO(od,ad,lr=1e-4); is_ppo=True; is_sac=False
else: t=SAC_Scratch(od,ad,device=DEV); is_ppo=False; is_sac=True

rewards=[]; best_r=-1e9; t0=time.time()
save_dir=PROJ_DIR/f'rl_scratch_{algo_name.lower()}'; save_dir.mkdir(exist_ok=True)

for u in range(1,TOTAL_UPDATES+1):
    us=time.time(); eps=curriculum_sample(samples,train_ids,u,10); ep_rs=[]
    for sid in eps:
        obs=env.reset(sample_idx=sid); er=0.0
        for step in range(env.max_steps+1):
            if is_sac:
                raw_a=t.select_action(obs,deterministic=False)
                lo,hi=env.get_action_bounds(); a=lo+(raw_a+1.0)*0.5*(hi-lo)
                no,r,done,_,info=env.step(a); er+=r
                t.replay_buffer.push(obs,raw_a,r,no,float(done))
            elif is_ppo:
                ret=t.select_action(obs,deterministic=False)
                raw_a,lp,v=ret[0],ret[1],ret[2]
                lo,hi=env.get_action_bounds(); a=lo+(raw_a+1.0)*0.5*(hi-lo)
                no,r,done,_,info=env.step(a); er+=r
                t.store(obs,raw_a,r,no,float(done),lp,v)
            else:
                raw_a=t.act(obs,det=False)
                lo,hi=env.get_action_bounds(); a=lo+(raw_a+1.0)*0.5*(hi-lo)
                no,r,done,_,info=env.step(a); er+=r
                t.buf.append((obs,raw_a,r,no,float(done)))
            obs=no
            if done: break
        ep_rs.append(er)
    if is_sac: t.update()
    elif is_ppo: t.update(batch_size=256,n_steps=30)
    else: t.update()
    avg_r=np.mean(ep_rs) if ep_rs else 0; rewards.append(avg_r)

    if avg_r>best_r:
        best_r=avg_r
        torch.save(t.actor.state_dict(),str(save_dir/'best_actor.pth'))

    if u%10==0 or u==1: print(f"U{u:3d} R={avg_r:+.0f} {time.time()-us:.0f}s")
    with open(str(save_dir/'rewards.json'),'w') as f: json.dump({"algo":algo_name,"rewards":rewards},f)

torch.save(t.actor.state_dict(),str(save_dir/'final_actor.pth'))
elapsed=(time.time()-t0)/3600
print(f"{algo_name} done: {elapsed:.1f}h, best R={best_r:+.0f}, final R={rewards[-1]:+.0f}")

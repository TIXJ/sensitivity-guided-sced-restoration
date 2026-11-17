"""Tier 1a: SAC No-GAT — starts from uniform dispatch instead of GAT warm-start.
Train 100 updates, save to rl_nogat/."""
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

# Monkey-patch: replace GAT warm-start with uniform dispatch
_original_reset = SCEDEnvV21.reset
def _uniform_reset(self, sample_idx=None):
    """Override reset: initialize from uniform dispatch, not GAT."""
    import numpy as np
    sample = self.all_samples[sample_idx]
    self.target_demand = float(sample['demand_mw'])
    self._gt_hhi_baseline = 343.5
    # Uniform init: equal share of demand, clipped to limits
    uniform = np.full(self.N, self.target_demand / self.N, dtype=np.float64)
    uniform = np.clip(uniform, self.PMIN, self.PMAX * 0.99)
    self.gat_power = uniform.copy()  # keep name for compatibility
    self.current_power = uniform.copy()
    self.cumulative_delta = np.zeros(self.N, dtype=np.float64)
    self.corr_step = 0
    # Adaptive action scale (same as original)
    if self.target_demand >= 21000:
        self.current_residual_scale = 120.0; self.max_steps = _v35.MAX_STEPS_HIGH
    elif self.target_demand >= 19000:
        self.current_residual_scale = 50.0; self.max_steps = _v35.MAX_STEPS
    else:
        self.current_residual_scale = 20.0; self.max_steps = _v35.MAX_STEPS
    self.episode_history = []
    self._current_vmin = 0.95; self._prev_vmin = 0.95
    self._n_v_viol = 300; self._prev_n_v_viol = 300
    self._n_overloaded_lines = 68; self._max_line_load_pct = 0.0; self._trafo_overloaded = 0
    self.prev_ac_result = np.array([0., 0.95, 1.05, 0.], dtype=np.float32)
    self._prev_n1_violations = 999; self._current_n1_violations = 999
    self._cached_n1_sens = np.zeros(self.N, dtype=np.float64)
    self._cached_n1_overload_mw = 0.0; self._cached_n1_viol = 0; self._last_n1_step = -999
    self._last_acpf_power = self.current_power.copy(); self._last_acpf_step = -999
    return self._get_obs()
SCEDEnvV21.reset = _uniform_reset

print("╔══════════════════════════════════════════════════╗")
print("║  Tier 1a: SAC No-GAT (uniform init)             ║")
print("╚══════════════════════════════════════════════════╝")
with open(str(PROJ_DIR/'case300_gt_v10_dataset.json'),'r') as f: data=json.load(f)
samples=[s for s in data['samples'] if s.get('is_complete_sced')]
train_ids,_,_ = split_samples_stratified(samples)
env=SCEDEnvV21(samples,train_ids)
od=env._get_obs_dim(); ad=env.action_dim
print(f"ObsDim={od}, ActDim={ad}")

t=SACTrainer(od,ad,device=DEVICE)
rewards=[]; best_r=-1e9; t0=time.time()
save_dir=PROJ_DIR/'rl_nogat'; save_dir.mkdir(exist_ok=True)

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
    with open(str(save_dir/'rewards.json'),'w') as f: json.dump({"algo":"SAC-NoGAT","rewards":rewards},f)

torch.save(t.actor.state_dict(),str(save_dir/'final_actor.pth'))
print(f"SAC-NoGAT done: {(time.time()-t0)/3600:.1f}h, best R={best_r:+.0f}")

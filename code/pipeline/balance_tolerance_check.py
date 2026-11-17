"""Measure the actual power-balance deviation |sum(P)-D|/D of the FINAL method's
accepted (feasible) dispatches, and report pass counts under progressively tighter
balance tolerances (1%, 0.5%, 0.2%, 0.1%). Answers whether the 1% guard is doing
heavy lifting or whether accepted dispatches are near-exact balanced."""
import sys, json, numpy as np, torch, random
from pathlib import Path
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
_,_,test_ids=split_samples_stratified(samples)
env=SCEDEnvV21(samples,test_ids); od,ad=env._get_obs_dim(),env.action_dim
t=SACTrainer(od,ad,device=DEVICE)
t.actor.load_state_dict(torch.load(str(PROJ/'rl_nocur'/'best_actor.pth'),map_location='cpu',weights_only=False))

def feasible_ignoring_balance(env):
    ac=env._run_ac_pf_check(); _,n1=env._compute_n1_overload_margin()
    return (ac.get('converged',False) and ac.get('v_violations',999)==0
            and ac.get('overloaded_lines',999)==0 and ac.get('trafo_overloaded',999)==0 and n1==0)

def rollout(sid,det):
    obs=env.reset(sample_idx=sid)
    for _ in range(env.max_steps+1):
        lo,hi=env.get_action_bounds(); a=lo+(t.select_action(obs,deterministic=det)+1.0)*0.5*(hi-lo)
        obs,_,d,_,info=env.step(a)
        if d: break
    return env.current_power.copy()

# collect AC/N-1-feasible dispatches (mean-action, then retry) and their balance deviation
devs=[]  # relative balance deviation of AC+N1-feasible dispatches
for sid in test_ids:
    Pf=rollout(sid,True); ok=feasible_ignoring_balance(env)
    if not ok:
        for _ in range(3):
            Pf=rollout(sid,False)
            if feasible_ignoring_balance(env): ok=True; break
    if ok:
        dev=abs(env.current_power.sum()-env.target_demand)/env.target_demand
        devs.append(dev)
devs=np.array(devs)
print(f"AC+N-1-feasible dispatches (ignoring balance): {len(devs)}")
print(f"balance deviation |sumP-D|/D:  mean={100*devs.mean():.3f}%  max={100*devs.max():.3f}%  median={100*np.median(devs):.3f}%")
for tol in [0.01,0.005,0.002,0.001]:
    print(f"  pass @ tol {100*tol:.1f}%: {(devs<=tol).sum()}/44")

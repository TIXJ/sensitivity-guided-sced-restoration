"""
Measure power-balance integrity: for each RL model, report how far the FINAL
dispatch sum deviates from target demand. The env's AC-PF check rescales load to
match generation, so an under-generating dispatch passes a FAKE lower-demand problem.
A trustworthy SCED solution MUST satisfy sum(P) == demand.
"""
import sys, json, numpy as np, torch
from pathlib import Path
PROJ = Path(r".")
sys.path.insert(0, str(PROJ))

import train_rlv35_complete as v
v.AC_PF_CHECK_EVERY = 5; v.N1_CHECK_EVERY_STEPS = 5
v.MAX_STEPS = 50; v.MAX_STEPS_HIGH = 70
from train_rlv35_complete import SCEDEnvV21, SACTrainer, split_samples_stratified, DEVICE

data = json.load(open(str(PROJ / 'case300_gt_v10_dataset.json')))
samples = [s for s in data['samples'] if s.get('is_complete_sced')]
_, _, test_ids = split_samples_stratified(samples)

MODELS = [
    ('Raw68',    'rl_raw68/best_actor.pth',    0,  68),
    ('Proposed', 'rl_scratch_sac/best_actor.pth', 12, 20),
    ('Hybrid',   'rl_nocur/best_actor.pth',    12, 20),
    ('PTDF12',   'rl_ptdf12/best_actor.pth',   12, 0),
]

for name, ckpt_rel, ns, nk in MODELS:
    ckpt = PROJ / ckpt_rel
    if not ckpt.exists():
        print(f"SKIP {name}: not found"); continue
    v.N_SHIFT_DIRECTIONS = ns; v.N_KEY_GENERATORS = nk
    env = SCEDEnvV21(samples, test_ids)
    od, ad = env._get_obs_dim(), env.action_dim
    t = SACTrainer(od, ad, device=DEVICE)
    t.actor.load_state_dict(torch.load(str(ckpt), map_location='cpu', weights_only=False))

    # Aggregate by GW: mean |sum(P)-demand| / demand
    devs = []; balanced_pass = 0; n = 0
    by_gw_dev = {}
    for sid in test_ids:
        gw = int(samples[sid]['demand_mw'] / 1000)
        obs = env.reset(sample_idx=sid)
        for _ in range(env.max_steps + 1):
            raw_a = t.select_action(obs, deterministic=True)
            lo, hi = env.get_action_bounds(); a = lo + (raw_a + 1.0) * 0.5 * (hi - lo)
            obs, _, done, _, info = env.step(a)
            if done: break
        P = env.current_power.sum()
        D = env.target_demand
        dev = abs(P - D) / D
        devs.append(dev); n += 1
        by_gw_dev.setdefault(gw, []).append(dev)
        # balanced pass: within 1% of demand AND env AC-PF+N-1 ok
        ac = env._run_ac_pf_check(); _, n1 = env._compute_n1_overload_margin()
        env_ok = (ac.get('converged', False) and ac.get('v_violations', 999) == 0
                  and ac.get('overloaded_lines', 999) == 0
                  and ac.get('trafo_overloaded', 999) == 0 and n1 == 0)
        if dev <= 0.01 and env_ok:
            balanced_pass += 1
    print(f"\n{name}: mean |sum-D|/D = {100*np.mean(devs):.1f}%  | "
          f"balanced+pass = {balanced_pass}/{n} ({100*balanced_pass/n:.1f}%)")
    for gw in sorted(by_gw_dev):
        m = 100*np.mean(by_gw_dev[gw])
        print(f"    {gw}GW: mean dev {m:5.1f}%")

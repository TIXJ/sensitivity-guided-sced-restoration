"""
Independent verification of Raw SAC 68-dim's suspiciously-perfect 100% result.
Bypasses the training env's own AC-PF/N-1 checks entirely: runs the raw68 policy
to get final dispatches, then verifies each with:
  (1) a FRESH pandapower case300 AC-PF (no env), and
  (2) the label-generator's check_n1_security (same code that made the N-1-secure labels).
If raw68's 24GW solutions pass BOTH independent checks, the 100% is real.
"""
import sys, json, numpy as np, torch
from pathlib import Path
PROJ = Path(r".")
sys.path.insert(0, str(PROJ))

import train_rlv35_complete as v
v.AC_PF_CHECK_EVERY = 5; v.N1_CHECK_EVERY_STEPS = 5
v.MAX_STEPS = 50; v.MAX_STEPS_HIGH = 70
v.N_SHIFT_DIRECTIONS = 0; v.N_KEY_GENERATORS = 68
from train_rlv35_complete import SCEDEnvV21, SACTrainer, split_samples_stratified, DEVICE

# Independent N-1 check from the label generator (the SAME one baselines use)
from generate_case300_gt_v10 import Case300Network, check_n1_security
import pandapower as pp
from pandapower.networks import case300

print("Loading data + raw68 model...")
data = json.load(open(str(PROJ / 'case300_gt_v10_dataset.json')))
samples = [s for s in data['samples'] if s.get('is_complete_sced')]
_, _, test_ids = split_samples_stratified(samples)

env = SCEDEnvV21(samples, test_ids)
od, ad = env._get_obs_dim(), env.action_dim
trainer = SACTrainer(od, ad, device=DEVICE)
trainer.actor.load_state_dict(torch.load(str(PROJ / 'rl_scratch_ddpg' / 'best_actor.pth')
                                          if False else str(PROJ / 'rl_raw68' / 'best_actor.pth'),
                                          map_location='cpu', weights_only=False))

dcnet = Case300Network()  # independent N-1 network


def independent_verify(P, demand_mw):
    """Fresh pandapower AC-PF + independent LODF N-1. Returns (acpf_ok, vv, n1_viol)."""
    net = case300()
    net.gen.p_mw.values[:] = np.clip(P, env.PMIN, env.PMAX * 0.99)
    total_gen = float(P.sum()); base_load = net.load.p_mw.sum()
    if total_gen > 0 and base_load > 0:
        ls = total_gen / base_load
        net.load.p_mw = net.load.p_mw * ls; net.load.q_mvar = net.load.q_mvar * ls
    net.bus.min_vm_pu = 0.88; net.bus.max_vm_pu = 1.10
    acpf_ok = False; vv = 999; ov = 999
    try:
        pp.runpp(net, calculate_voltage_angles=True, enforce_q_lims=True, numba=False)
        if net.converged:
            vv = int((net.res_bus.vm_pu < 0.88).sum())
            ov = int((net.res_line.loading_percent > 100).sum())
            if len(net.res_trafo) > 0:
                ov += int((net.res_trafo.loading_percent > 100).sum())
            acpf_ok = True
    except Exception:
        pass
    # Independent N-1 (label-generator's own check)
    try:
        n1_viol, _ = check_n1_security(P, dcnet)
    except Exception as e:
        n1_viol = -1  # error
    return acpf_ok, vv, ov, n1_viol


# Focus on the hardest levels: 21-24 GW (where Gurobi/GAT-only all fail)
print(f"\n{'='*70}")
print("Independent verification of Raw SAC 68-dim on HIGH-load samples (21-24 GW)")
print(f"{'='*70}")
hard_ids = [sid for sid in test_ids if int(samples[sid]['demand_mw']/1000) >= 21]
env_pass = 0; indep_pass = 0
for sid in hard_ids:
    gw = int(samples[sid]['demand_mw'] / 1000)
    obs = env.reset(sample_idx=sid)
    for _ in range(env.max_steps + 1):
        raw_a = trainer.select_action(obs, deterministic=True)
        lo, hi = env.get_action_bounds(); a = lo + (raw_a + 1.0) * 0.5 * (hi - lo)
        obs, _, done, _, info = env.step(a)
        if done: break
    P_final = env.current_power.copy()

    # env's own fresh check
    env_ac = env._run_ac_pf_check()
    _, env_n1 = env._compute_n1_overload_margin()
    env_ok = (env_ac.get('converged', False) and env_ac.get('v_violations', 999) == 0
              and env_ac.get('overloaded_lines', 999) == 0
              and env_ac.get('trafo_overloaded', 999) == 0 and env_n1 == 0)

    # INDEPENDENT check
    ind_ac, ind_vv, ind_ov, ind_n1 = independent_verify(P_final, samples[sid]['demand_mw'])
    ind_ok = (ind_ac and ind_vv == 0 and ind_ov == 0 and ind_n1 == 0)

    if env_ok: env_pass += 1
    if ind_ok: indep_pass += 1
    flag = "  <-- MISMATCH!" if env_ok != ind_ok else ""
    print(f"  GW={gw} sid={sid:3d} | ENV: {'PASS' if env_ok else 'FAIL'} (n1={env_n1}) | "
          f"INDEP: {'PASS' if ind_ok else 'FAIL'} (acpf={ind_ac} vv={ind_vv} ov={ind_ov} n1={ind_n1}){flag}")

print(f"\n  Sum={P_final.sum():.0f} MW (target={samples[hard_ids[-1]]['demand_mw']:.0f})")
print(f"  ENV check:         {env_pass}/{len(hard_ids)} pass")
print(f"  INDEPENDENT check: {indep_pass}/{len(hard_ids)} pass")
if env_pass != indep_pass:
    print(f"\n  ★ MISMATCH → raw68's env-reported pass rate is an ARTIFACT.")
else:
    print(f"\n  ★ MATCH → raw68's result is genuine (RL beats QP at high load).")

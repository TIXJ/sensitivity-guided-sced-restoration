"""Matched GAT+optimization versus GAT+SAC+same-optimization evaluation."""
import argparse
import copy
import os
from pathlib import Path
import time

import numpy as np
import torch
from revision_common import CANDIDATES, CANDIDATE_SPLIT, CHECKPOINTS, CODE, DATASET, SPLIT, GAT, HERE, RESULTS, load_aligned, read, sha, write
from constraint_diagnostics import operation_flags
from evaluate_revision import cost, verify
from shared_joint_optimizer import solve, voltage_box

DEST = RESULTS/'joint_workflow_comparison_v1'
ARMS = ('gat_optimizer', 'gat_sac_optimizer')


def hashes():
    files = [Path(__file__), HERE/'shared_joint_optimizer.py', HERE/'pilot_joint_ac_lodf.py',
             HERE/'pilot_validation_feasibility.py', HERE/'evaluate_revision.py', HERE/'constraint_diagnostics.py',
             CODE/'train_sac_n1_active_aligned.py', CODE/'train_rlv35_complete.py',
             HERE/'revision_common.py', CANDIDATES, CANDIDATE_SPLIT, DATASET, SPLIT, GAT, CHECKPOINTS['sac']]
    caches = sorted((RESULTS/'jacobian_cache/none').glob('*.npz'))
    assert len(caches) == 5, 'Use the five previously recorded fixed-demand Jacobian caches.'
    files += caches
    return {str(path): sha(path) for path in files}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def check(env, mod, low, high):
    passed, info = verify(env, mod)
    controls = np.r_[env.current_gen_vm_pu, env.current_ext_grid_vm_pu]
    within = bool(np.all(controls >= low-1e-6) and np.all(controls <= high+1e-6))
    finite = bool(np.isfinite(info.get('bus_vm_pu', [np.nan])).all())
    info['controls_within_original_voltage_box'] = within
    return bool(passed and within and finite), info


def evaluate(env, mod, actor, sample, sid, model, arm):
    sync()
    started = time.perf_counter()
    env.current_sample = sample
    demand = float(sample['demand_mw'])
    raw_prediction = env._predict_gat(demand)
    reference = mod.core.project_box_sum(raw_prediction, env.PMIN, env.PMAX*.99, demand)
    nominal, low, high = voltage_box(env, mod)
    env.target_demand = demand
    env.current_power = reference.copy()
    env.current_gen_vm_pu = nominal[:env.N].copy()
    env.current_ext_grid_vm_pu = nominal[env.N:].copy()
    env.corr_step = 0
    env._aligned_n1_cache_key = None
    sync()
    after_prediction = time.perf_counter()
    initial_passed, initial_info = check(env, mod, low, high)
    after_initial = time.perf_counter()
    steps = 0
    gate_passed = initial_passed
    if arm == 'gat_sac_optimizer' and not initial_passed:
        predictor = env._predict_gat
        env._predict_gat = lambda _: raw_prediction.copy()
        try:
            obs = env.reset(sid)
        finally:
            env._predict_gat = predictor
        assert np.allclose(env.gat_power, reference, atol=1e-8, rtol=0)
        for steps in range(1, env.max_steps+1):
            tensor = torch.as_tensor(obs, dtype=torch.float32, device=mod.core.DEVICE).unsqueeze(0)
            with torch.no_grad():
                raw = actor.select_action(tensor, deterministic=True).cpu().numpy()[0]
            lower, upper = env.get_action_bounds()
            obs, _, done, truncated, _ = env.step(lower+(raw+1)*.5*(upper-lower))
            if done or truncated:
                break
        gate_passed, gate_info = check(env, mod, low, high)
    else:
        gate_info = initial_info
    sync()
    after_policy = time.perf_counter()
    invoked = not gate_passed
    optimizer = None
    if invoked:
        # Identical objective, starting point and cuts in both arms. This tests
        # avoided optimizer calls, not a benefit from an SAC optimizer warm start.
        env.current_power = reference.copy()
        env.current_gen_vm_pu = nominal[:env.N].copy()
        env.current_ext_grid_vm_pu = nominal[env.N:].copy()
        optimizer = solve(env, mod, model, reference, demand)
        if optimizer['returned_candidate']:
            env.current_power = np.asarray(optimizer['power'])
            env.current_gen_vm_pu = np.asarray(optimizer['gen_vm'])
            env.current_ext_grid_vm_pu = np.asarray(optimizer['ext_vm'])
    after_optimizer = time.perf_counter()
    env._aligned_n1_cache_key = None
    passed, info = check(env, mod, low, high)
    sync()
    ended = time.perf_counter()
    return {'sample_id': sample['sample_id'], 'arm': arm, 'demand_mw': demand,
            'passed': passed, 'initial_passed': initial_passed, 'policy_gate_passed': gate_passed,
            'policy_steps': steps, 'optimizer_called': invoked, 'optimizer': optimizer,
            'runtime_sec': ended-started, 'prediction_projection_sec': after_prediction-started,
            'initial_verification_sec': after_initial-after_prediction, 'policy_and_gate_sec': after_policy-after_initial,
            'optimizer_stage_sec': after_optimizer-after_policy, 'final_verification_sec': ended-after_optimizer,
            'scheduled_cost': cost(sample, env.current_power),
            'gat_relative_l2_mw': float(np.linalg.norm(env.current_power-reference)),
            'final_dispatch_mw': env.current_power.tolist(), 'final_generator_vm_pu': env.current_gen_vm_pu.tolist(),
            'final_reference_vm_pu': env.current_ext_grid_vm_pu.tolist(),
            'reference_dispatch_mw': reference.tolist(), 'policy_gate_info': gate_info,
            'constraint_flags': operation_flags(info), 'final_verification': info}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=('val', 'test'), default='val')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('NUMBA_CACHE_DIR', str(RESULTS/'numba_cache'))
    provenance = hashes()
    protocol_path = DEST/'protocol.json'
    if args.freeze:
        if protocol_path.exists():
            parser.error('Protocol already frozen')
        validation = read(DEST/'val.json')
        assert validation['complete'] and validation['hashes'] == provenance
        write(protocol_path, {'hashes': provenance, 'validation_sha256': sha(DEST/'val.json'),
                              'arms': ARMS, 'objective': '1e-4 times squared displacement from common projected GAT dispatch',
                              'budget': '8 cutting-plane rounds, 100 interior-point iterations/round, batch 256, no restarts',
                              'scope': 'RL invocation ablation with GAT shared; not a no-GAT comparison or AC contingency certification'})
        print('Protocol frozen before test.')
        return
    if args.scope == 'test':
        assert not args.pilot
        protocol = read(protocol_path)
        assert protocol['hashes'] == provenance and protocol['validation_sha256'] == sha(DEST/'val.json')
    path = DEST/('pilot.json' if args.pilot else args.scope+'.json')
    if path.exists():
        old = read(path)
        if old.get('complete') and old['hashes']==provenance:
            print('Completed output preserved:', path)
            return
        parser.error('Partial or different-source output exists; preserve it before a new run')
    data = read(CANDIDATES)
    samples = [s for s in data['samples'] if s.get('is_complete_sced', True)]
    ids = read(CANDIDATE_SPLIT)[args.scope+'_ids']
    if args.pilot:
        chosen = {}
        for sid in sorted(ids, key=lambda i: samples[i]['sample_id']):
            chosen.setdefault(samples[sid]['demand_mw'], sid)
        ids = list(chosen.values())
    mod = load_aligned()
    envs = {arm: mod.CleanN1Env(samples, ids, data) for arm in ARMS}
    for prepared_env in envs.values():
        prepared_env._voltage_jacobian_dir = RESULTS/'jacobian_cache/none'
        prepared_env._voltage_jacobian_dir.mkdir(parents=True, exist_ok=True)
    env = envs[ARMS[1]]
    obs = env.reset(ids[0])
    pdim = env._n_shift_dirs+env._n_key_gens
    mod.core.SAC_LOG_STD_MAX_VECTOR = np.r_[np.full(pdim, mod.ARGS.active_log_std_max),
                                           np.full(env.action_dim-pdim, mod.ARGS.voltage_log_std_max)].astype(np.float32)
    actor = mod.BranchedGaussianActor(len(obs), pdim, env.action_dim-pdim).to(mod.core.DEVICE)
    actor.load_state_dict(torch.load(CHECKPOINTS['sac'], map_location=mod.core.DEVICE, weights_only=True))
    actor.eval()
    # Prewarm fixed basis/Jacobian features for every demand represented in this
    # split, without using acceptance results to select any model or setting.
    warmed = set()
    for sid in ids:
        demand = samples[sid]['demand_mw']
        if demand not in warmed:
            env.reset(sid)
            warmed.add(demand)
    with torch.no_grad():
        actor.select_action(torch.as_tensor(obs, device=mod.core.DEVICE).unsqueeze(0), deterministic=True)
    payload = {'complete': False, 'scope': args.scope, 'pilot': args.pilot, 'hashes': provenance,
               'torch_version': torch.__version__, 'numpy_version': np.__version__,
               'torch_threads': torch.get_num_threads(),
               'device': str(mod.core.DEVICE),
               'protocol_sha256': sha(protocol_path) if args.scope=='test' else None,
               'sample_ids': [samples[i]['sample_id'] for i in ids],
               'timing_scope': 'Resident models/network and fixed-feature prewarming excluded. Per-case prediction, '
                               'initial checks, policy setup/rollout/gating when used, solver setup/solve when invoked, '
                               'and fresh final verification included. Failed cases included. Arms alternate order.',
               'rows': []}
    write(path, payload)
    for i, sid in enumerate(ids):
        for arm in (ARMS if i%2==0 else ARMS[::-1]):
            row = evaluate(envs[arm], mod, actor, samples[sid], sid, data['linear_security_model'], arm)
            payload['rows'].append(row)
            write(path, payload)
            print(f"{i+1}/{len(ids)} {samples[sid]['sample_id']} {arm} pass={row['passed']} optimizer={row['optimizer_called']} time={row['runtime_sec']:.3f}s", flush=True)
    payload['complete'] = True
    write(path, payload)


if __name__ == '__main__':
    main()

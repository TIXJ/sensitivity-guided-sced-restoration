"""Remeasure the frozen 75-case dual-start comparison without duplicate checks.

The old runner, weights, gate, constraints, and raw results remain untouched.
Only unchanged, independently accepted optimizer candidates reuse a receipt.
"""
import argparse
import copy
import os
import time
from revision_common import RESULTS, CANDIDATES, read, write, sha
os.environ.setdefault('NUMBA_CACHE_DIR', str(RESULTS/'numba_cache'))
import numpy as np
import torch
import reviewer_followup as old
import evaluate_short_policy as short
import compare_joint_workflows as common
import shared_voltage_runtime as shared
from verified_candidate import AcceptanceReceipt
from evaluate_revision import cost
from constraint_diagnostics import operation_flags

DEST = RESULTS/'dual_verified_once_v2'


def solve_verified(env, mod, model, reference, demand):
    import pandapower as pp
    from shared_joint_optimizer import solve, voltage_box
    original = pp.runopp
    attempts = []
    candidate = receipt = None
    started = time.perf_counter()
    nominal, low, high = voltage_box(env, mod)
    try:
        for init in ('pf', 'flat'):
            calls = []
            def capped(net, **kwargs):
                if len(calls) >= 4:
                    raise pp.OPFNotConverged('fixed four-call initializer budget exhausted')
                kwargs['numba'] = True
                if kwargs.get('init') == 'flat':
                    kwargs['init'] = init
                kwargs['PDIPM_MAX_IT'] = 100
                calls.append({'init': kwargs.get('init'), 'max_ip_iterations': 100})
                return original(net, **kwargs)
            pp.runopp = capped
            env.current_power = reference.copy()
            env.current_gen_vm_pu = nominal[:env.N].copy()
            env.current_ext_grid_vm_pu = nominal[env.N:].copy()
            env._aligned_n1_cache_key = None
            result = solve(env, mod, model, reference, demand)
            accepted, info = False, None
            if result['returned_candidate']:
                candidate = result
                env.current_power = np.asarray(result['power'])
                env.current_gen_vm_pu = np.asarray(result['gen_vm'])
                env.current_ext_grid_vm_pu = np.asarray(result['ext_vm'])
                env._aligned_n1_cache_key = None
                env._verification_fresh = True
                try:
                    accepted, info = common.check(env, mod, low, high)
                finally:
                    env._verification_fresh = False
                if accepted:
                    receipt = AcceptanceReceipt(env, mod, model, low, high, accepted, info)
            attempts.append({'initializer': init, 'actual_opf_calls': len(calls),
                             'calls': calls, 'accepted': accepted, 'verification': info,
                             'solver': result})
            if accepted:
                break
    finally:
        pp.runopp = original
    result = dict(candidate or {'returned_candidate': False, 'status': 'both_initializers_failed'})
    result.update(initializer_attempts=attempts,
                  actual_opf_calls=sum(a['actual_opf_calls'] for a in attempts),
                  runtime_sec=time.perf_counter()-started)
    assert result['actual_opf_calls'] <= 8
    return result, receipt


def evaluate(env, mod, actor, sample, sid, model, rule):
    common.sync()
    start = time.perf_counter()
    reference = mod.core.project_box_sum(env._predict_gat(sample['demand_mw']), env.PMIN,
                                       env.PMAX*.99, sample['demand_mw'])
    nominal, low, high = common.voltage_box(env, mod)
    env.target_demand = sample['demand_mw']
    env.current_power = reference.copy()
    env.current_gen_vm_pu = nominal[:env.N].copy()
    env.current_ext_grid_vm_pu = nominal[env.N:].copy()
    env.corr_step = 0
    env._aligned_n1_cache_key = None
    initial_passed, info = common.check(env, mod, low, high)
    features = short.feature(info, sample, mod)
    use = bool(rule and not initial_passed and short.selected(features, rule))
    after_initial = time.perf_counter()
    steps, policy_passed = 0, initial_passed
    if use:
        obs = env.reset(sid)
        for steps in range(1, rule['cap']+1):
            with torch.no_grad():
                raw = actor.select_action(torch.as_tensor(obs, dtype=torch.float32,
                    device=mod.core.DEVICE).unsqueeze(0), deterministic=True).cpu().numpy()[0]
            lo, hi = env.get_action_bounds()
            obs, _, done, truncated, _ = env.step(lo+(raw+1)*.5*(hi-lo))
            if done or truncated:
                break
        policy_passed, info = common.check(env, mod, low, high)
    after_policy = time.perf_counter()
    optimizer = receipt = None
    if not policy_passed:
        optimizer, receipt = solve_verified(env, mod, model, reference, sample['demand_mw'])
        if optimizer['returned_candidate']:
            env.current_power = np.asarray(optimizer['power'])
            env.current_gen_vm_pu = np.asarray(optimizer['gen_vm'])
            env.current_ext_grid_vm_pu = np.asarray(optimizer['ext_vm'])
    after_solver = time.perf_counter()
    env._aligned_n1_cache_key = None
    verified = receipt.consume(env, mod, model, low, high) if receipt else None
    reused = verified is not None
    if verified is None:
        env._verification_fresh = True
        try:
            verified = common.check(env, mod, low, high)
        finally:
            env._verification_fresh = False
    passed, info = verified
    common.sync()
    ended = time.perf_counter()
    return {'sample_id': sample['sample_id'], 'passed': passed, 'policy_attempted': use,
        'policy_gate_passed': policy_passed, 'initial_passed': initial_passed,
        'policy_steps': steps, 'optimizer_called': optimizer is not None, 'optimizer': optimizer,
        'runtime_sec': ended-start, 'initial_features': features,
        'initial_stage_sec': after_initial-start, 'policy_sec': after_policy-after_initial,
        'optimizer_sec': after_solver-after_policy, 'final_check_sec': ended-after_solver,
        'acceptance_receipt_reused': reused, 'receipt_context_unchanged': reused,
        'fresh_candidate_checks': sum(a['verification'] is not None for a in
             (optimizer or {}).get('initializer_attempts', [])) + int(not reused),
        'scheduled_cost': cost(sample, env.current_power), 'constraint_flags': operation_flags(info),
        'final_verification': info, 'reference_dispatch_mw': reference.tolist(),
        'gat_relative_l2_mw': float(np.linalg.norm(env.current_power-reference)),
        'final_dispatch_mw': env.current_power.tolist(),
        'final_generator_vm_pu': env.current_gen_vm_pu.tolist(),
        'final_reference_vm_pu': env.current_ext_grid_vm_pu.tolist()}


def prepare():
    if DEST.exists():
        raise RuntimeError('Preserve the existing measurement')
    protocol = copy.deepcopy(read(old.DEST/'protocol.json'))
    assert protocol['source_hashes'] == old.source_hashes()
    protocol.update(parent_protocol_sha256=sha(old.DEST/'protocol.json'),
        source_hashes_v2={str(p): sha(p) for p in (__file__, __import__('verified_candidate').__file__)},
        verification_contract='Fresh candidate verification inside optimizer; single-use receipt at outer acceptance only when all inputs match; otherwise fresh check. All fingerprint costs timed.')
    write(DEST/'protocol.json', protocol)


def run():
    protocol = read(DEST/'protocol.json')
    for path, digest in {**protocol['input_hashes'], **protocol['source_hashes'],
                         **protocol['source_hashes_v2']}.items():
        assert sha(path) == digest, path
    path = DEST/'dual.json'
    if path.exists():
        raise RuntimeError('Preserve the existing measurement')
    data = read(CANDIDATES)
    samples, ids = data['samples'], protocol['dual']['ids']
    mod, envs, actor = old.setup(samples, data)
    payload = {'complete': False, 'protocol_sha256': sha(DEST/'protocol.json'), 'rows': []}
    for repeat in range(3):
        for j, sid in enumerate(ids):
            arms = ('direct', 'selected') if (j+repeat)%2 == 0 else ('selected', 'direct')
            for arm in arms:
                env = envs[arm]
                env.current_sample = samples[sid]
                shared.clear_case(env)
                before = len(env.implicit_feature_calls)
                row = evaluate(env, mod, actor, samples[sid], sid, data['linear_security_model'],
                               protocol['gate'] if arm == 'selected' else None)
                row.update(arm=arm, repeat=repeat,
                           voltage_backend_calls=env.implicit_feature_calls[before:])
                payload['rows'].append(row)
            write(path, payload)
            print('verified dual', repeat+1, j+1, '/75', flush=True)
    payload['complete'] = True
    write(path, payload)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'run'))
    args = parser.parse_args()
    globals()[args.stage]()

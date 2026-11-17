"""Preselected 15-case tighter-voltage optimizer diagnostic; no policy retraining."""
import os
import argparse
from pathlib import Path
import numpy as np
from revision_common import CANDIDATES, CANDIDATE_SPLIT, RESULTS, load_aligned, read, write, sha
from compare_joint_workflows import evaluate, check, hashes
from shared_joint_optimizer import voltage_box


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zero-margin', action='store_true', help='Separate boundary diagnostic without the optimizer-only inward voltage margin')
    args = parser.parse_args()
    os.environ.setdefault('NUMBA_CACHE_DIR', str(RESULTS/'numba_cache'))
    destination = RESULTS/('tight_voltage_boundary_v2' if args.zero_margin else 'tight_voltage_diagnostic_v1')
    destination.mkdir(exist_ok=True)
    output = destination/'results.json'
    data = read(CANDIDATES)
    samples = [s for s in data['samples'] if s.get('is_complete_sced', True)]
    selected = {}
    for sid in sorted(read(CANDIDATE_SPLIT)['test_ids'], key=lambda i: samples[i]['sample_id']):
        group = selected.setdefault(samples[sid]['demand_mw'], [])
        if len(group) < 3:
            group.append(sid)
    ids = [sid for demand in sorted(selected) for sid in selected[demand]]
    assert len(ids) == 15
    sources = hashes()
    sources[str(Path(__file__))] = sha(Path(__file__))
    policy_path = RESULTS/'full_sac.json'
    sources[str(policy_path)] = sha(policy_path)
    protocol = {'sample_ids': [samples[i]['sample_id'] for i in ids],
                'selection': 'First three sorted upstream test IDs per demand, before solving tight bands',
                'bands': [[.94, 1.08], [.95, 1.05]], 'hashes': sources,
                'optimizer_voltage_margin': 0. if args.zero_margin else .0001,
                'scope': 'Original stored policy terminals reverified, versus full-P joint optimization under tighter bus limits; no retraining or new independent test set.'}
    protocol_path = destination/'protocol.json'
    if protocol_path.exists():
        assert read(protocol_path) == protocol
    else:
        write(protocol_path, protocol)
    payload = read(output) if output.exists() else {'complete': False, 'protocol': protocol, 'rows': []}
    assert payload['protocol'] == protocol
    if payload['complete']:
        print('Completed results retained')
        return
    mod = load_aligned()
    if args.zero_margin:
        import compare_joint_workflows as workflow
        original_solve = workflow.solve
        def solve_without_inset(env, module, model, power, demand):
            lower, upper = module.core.RL_VMIN_PU, module.core.RL_VMAX_PU
            try:
                # Cancel only the fixed solver-side inset; verification retains
                # the requested bounds after the solve returns.
                module.core.RL_VMIN_PU = lower-.0001
                module.core.RL_VMAX_PU = upper+.0001
                return original_solve(env, module, model, power, demand)
            finally:
                module.core.RL_VMIN_PU, module.core.RL_VMAX_PU = lower, upper
        workflow.solve = solve_without_inset
    env = mod.CleanN1Env(samples, ids, data)
    env._voltage_jacobian_dir = RESULTS/'jacobian_cache/none'
    env.reset(ids[0])
    points = {r['sample_id']: r for r in read(policy_path)['rows']}
    for vmin, vmax in protocol['bands']:
        mod.core.RL_VMIN_PU, mod.core.RL_VMAX_PU = vmin, vmax
        for sid in ids:
            sample = samples[sid]
            if any(r['sample_id'] == sample['sample_id'] and r['band'] == [vmin, vmax] for r in payload['rows']):
                continue
            point = points[sample['sample_id']]
            env.current_sample = sample
            env.target_demand = float(sample['demand_mw'])
            env.current_power = np.asarray(point['final_dispatch_mw'])
            env.current_gen_vm_pu = np.asarray(point['final_generator_vm_pu'])
            env.current_ext_grid_vm_pu = np.asarray(point['final_reference_vm_pu'])
            env._aligned_n1_cache_key = None
            _, low, high = voltage_box(env, mod)
            fixed_pass, fixed_info = check(env, mod, low, high)
            # Detect incompatible device/cumulative and bus bounds explicitly.
            margin = protocol['optimizer_voltage_margin']
            conflicts = np.flatnonzero(np.maximum(low, vmin+margin) > np.minimum(high, vmax-margin)+1e-12)
            row = {'sample_id': sample['sample_id'], 'band': [vmin, vmax],
                   'demand_mw': sample['demand_mw'], 'fixed_terminal_passed': fixed_pass,
                   'fixed_terminal_verification': fixed_info, 'incompatible_control_indices': conflicts.tolist(),
                   'control_lower': low.tolist(), 'control_upper': high.tolist()}
            if len(conflicts):
                row.update(joint_passed=False, status='incompatible_voltage_bounds', joint=None)
            else:
                result = evaluate(env, mod, None, sample, sid, data['linear_security_model'], 'gat_optimizer')
                row.update(joint_passed=result['passed'], status=result['optimizer']['status'] if result['optimizer'] else 'initial_passed', joint=result)
            payload['rows'].append(row)
            write(output, payload)
            print(sample['sample_id'], [vmin, vmax], fixed_pass, row['joint_passed'], row['status'], flush=True)
    payload['complete'] = True
    write(output, payload)


if __name__ == '__main__':
    main()

"""Matched exhaustive line-outage audit at fixed SAC/optimization operating points."""
import argparse
import copy
import os
from pathlib import Path
import numpy as np
from revision_common import CANDIDATES, RESULTS, load_aligned, read, sha, write
from pilot_preventive_voltage_margin import outage_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error('Output exists')
    os.environ.setdefault('NUMBA_CACHE_DIR', str(RESULTS/'numba_cache'))
    import pandapower as pp
    from pandapower.networks import case300
    data = read(CANDIDATES)
    samples = {s['sample_id']: s for s in data['samples']}
    paths = {m: RESULTS/f'full_{m}.json' for m in ('sac', 'scopf')}
    accepted = {m: {r['sample_id']: r for r in read(path)['rows'] if r['passed']} for m,path in paths.items()}
    common = sorted(set(accepted['sac']) & set(accepted['scopf']))
    selected = {}
    for sid in common:
        selected.setdefault(samples[sid]['demand_mw'], sid)
    model = data['linear_security_model']
    valid = np.asarray(model['valid_contingency_indices'], dtype=int)
    limits = np.asarray(model['effective_limit'])
    core = load_aligned().core
    payload = {'complete': False, 'selection': 'First sample ID at each demand among cases accepted by both methods, before outage results.',
               'common_accepted_cases': len(common), 'selected_sample_ids': list(selected.values()),
               'valid_outages': valid.tolist(), 'input_hashes': {str(p): sha(p) for p in paths.values()},
               'checks_hash': sha(Path(__file__).with_name('pilot_preventive_voltage_margin.py')),
               'scope': 'Paired diagnostic, not population reliability or full network N-1 certification. '
                        'Stored controls retained without clipping or post-outage corrective changes.', 'rows': []}
    write(args.output, payload)
    for demand, sid in selected.items():
        for method in ('sac', 'scopf'):
            point = accepted[method][sid]
            net = case300()
            factor = demand/net.load.p_mw.sum()
            net.load.p_mw *= factor
            net.load.q_mvar *= factor
            if method == 'sac':
                net.gen.p_mw = point['final_dispatch_mw']
                net.gen.vm_pu = point['final_generator_vm_pu']
                net.ext_grid.vm_pu = point['final_reference_vm_pu']
            else:
                net.gen.p_mw = point['security_dispatch_mw']
                net.gen.vm_pu = point['gen_vm_pu']
                net.ext_grid.vm_pu = point['ext_grid_vm_pu']
            # Both methods use the same physical bounds; do not clip the optimizer's
            # setpoints to the narrower policy action box during this audit.
            pp.runpp(net, enforce_q_lims=True, calculate_voltage_angles=True, numba=True, max_iteration=100)
            vm = net.res_bus.vm_pu.to_numpy()
            records = outage_checks(net, valid, limits, core)
            row = {'sample_id': sid, 'demand_mw': demand, 'method': method,
                   'base_converged': bool(net.converged),
                   'base_voltage_violations': int(((vm<core.RL_VMIN_PU)|(vm>core.RL_VMAX_PU)).sum()),
                   'base_min_vm_pu': float(vm.min()), 'base_max_vm_pu': float(vm.max()),
                   'passing_outages': sum(r['passed'] for r in records),
                   'total_outages': len(valid), 'all_outages_pass': all(r['passed'] for r in records),
                   'outages': records}
            payload['rows'].append(row)
            write(args.output, payload)
            print(sid, method, row['passing_outages'], '/', len(valid), flush=True)
    payload['complete'] = True
    write(args.output, payload)


if __name__ == '__main__':
    main()

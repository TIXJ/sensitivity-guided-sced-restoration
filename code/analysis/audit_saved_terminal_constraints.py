"""Recheck stored terminal controls without retraining or changing their values."""
import argparse
import os
from collections import Counter
import numpy as np
from revision_common import CANDIDATES, RESULTS, load_aligned, read, sha, write
from constraint_diagnostics import operation_flags


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if os.path.exists(args.output):
        p.error('Output exists')
    os.environ.setdefault('NUMBA_CACHE_DIR', str(RESULTS/'numba_cache'))
    source = RESULTS/'full_sac.json'
    original = read(source)
    assert original['complete']
    data = read(CANDIDATES)
    samples = [s for s in data['samples'] if s.get('is_complete_sced', True)]
    by_id = {s['sample_id']: (i, s) for i, s in enumerate(samples)}
    mod = load_aligned()
    env = mod.CleanN1Env(samples, [0], data)
    payload = {'complete': False, 'source_sha256': sha(source), 'source': str(source),
               'scope': 'Fresh verification of unchanged stored terminal controls; no new policy rollout.', 'rows': []}
    for row in original['rows']:
        sid, sample = by_id[row['sample_id']]
        env.current_sample = sample
        env.target_demand = float(row['demand_mw'])
        env.current_power = np.asarray(row['final_dispatch_mw'])
        env.current_gen_vm_pu = np.asarray(row['final_generator_vm_pu'])
        env.current_ext_grid_vm_pu = np.atleast_1d(row['final_reference_vm_pu'])
        info = dict(env._run_ac_pf_check())
        n1 = env._vectorized_n1_state()
        info.update(n1_violations=int(n1['violation_count']), n1_overload_mw=float(n1['total_overload_mw']))
        passed = bool(mod.core.adopted_operation_pass(info))
        payload['rows'].append({'sample_id': row['sample_id'], 'original_primary_reason': row['reason'],
                                'original_passed': row['passed'], 'passed': passed,
                                'constraint_flags': operation_flags(info), 'final_verification': info})
        write(args.output, payload)
    payload['summary'] = {'n': len(payload['rows']), 'passed': sum(r['passed'] for r in payload['rows']),
                          'acceptance_changed': [r['sample_id'] for r in payload['rows'] if r['passed'] != r['original_passed']],
                          'overlapping_violations': dict(Counter(k for r in payload['rows'] for k, v in r['constraint_flags'].items() if v is True))}
    payload['complete'] = True
    write(args.output, payload)
    print(payload['summary'], flush=True)


if __name__ == '__main__':
    main()

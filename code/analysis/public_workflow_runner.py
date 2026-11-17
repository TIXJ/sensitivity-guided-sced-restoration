"""Fresh physical workflow measurements with relocated frozen inputs."""
import argparse
import copy
from pathlib import Path
import time
import numpy as np
import pandapower as pp

from revision_common import REV, RESULTS, CANDIDATES, CANDIDATE_SPLIT, read, write, sha
import remeasure_verified_dual as final
import compare_joint_workflows as common
import shared_voltage_runtime as shared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('population', choices=('historical', 'spatial'))
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.max_samples < 0:
        parser.error('repeats must be positive and max-samples nonnegative')
    output = Path(args.output)
    if output.exists():
        parser.error('Preserve existing measurements; choose a new --name')
    source_protocol = REV / 'inputs/final_deployment.json'
    source_conditions = REV / 'inputs/spatial_conditions.json'
    protocol = read(source_protocol)
    assert protocol['gate'] == {'cap': 8, 'max_demand_mw': 16000, 'max_gap_pu': .04, 'sum_gap_pu': 10.}
    assert protocol['dual']['initializers'] == ['pf', 'flat']
    assert protocol['dual']['max_opf_calls_per_initializer'] == 4
    assert protocol['dual']['max_ip_iterations_per_call'] == 100
    data = read(CANDIDATES)
    if args.population == 'historical':
        samples = data['samples']
        ids = protocol['dual']['ids']
        assert ids == read(CANDIDATE_SPLIT)['test_ids']
    else:
        samples = read(source_conditions)['samples']
        assert len(samples) == 100
        ids = list(range(len(samples)))
    if args.max_samples:
        ids = ids[:args.max_samples]
    mod, envs, actor = final.old.setup(samples, data)
    template = copy.deepcopy(envs['direct']._ac_net_template)
    payload = {'complete': False, 'population': args.population, 'rows': [],
        'measurement_scope': 'Fresh physical rerun using the frozen gate, weights, inputs, optimizer and checks; timings are new measurements.',
        'execution_manifest': {str(p.relative_to(REV)): sha(p) for p in (source_protocol, source_conditions, Path(__file__))},
        'repeats': args.repeats, 'samples': len(ids)}
    for repeat in range(args.repeats):
        for j, sid in enumerate(ids):
            sample = samples[sid]
            arms = ('direct', 'selected') if (j + repeat) % 2 == 0 else ('selected', 'direct')
            for arm in arms:
                common.sync()
                started = time.perf_counter()
                env = envs[arm]
                env.current_sample = sample
                env.target_demand = sample['demand_mw']
                shared.clear_case(env)
                before = len(env.implicit_feature_calls)
                model = data['linear_security_model']
                prep = 0.
                predictor = env._predict_gat
                if args.population == 'spatial':
                    net = copy.deepcopy(template)
                    net.load.p_mw *= sample['p_weights']
                    net.load.q_mvar *= sample['q_weights']
                    scale = sample['demand_mw'] / net.load.p_mw.sum()
                    net.load.p_mw *= scale
                    net.load.q_mvar *= scale
                    env._ac_net_template = net
                    raw = predictor(sample['demand_mw'])
                    reference = mod.core.project_box_sum(raw, env.PMIN, env.PMAX*.99, sample['demand_mw'])
                    anchor = copy.deepcopy(net)
                    anchor.gen.p_mw = reference
                    try:
                        pp.runpp(anchor, enforce_q_lims=True, calculate_voltage_angles=True, numba=True, max_iteration=100)
                    except pp.LoadflowNotConverged:
                        common.sync()
                        row = {'sample_id': sample['sample_id'], 'passed': False, 'preparation_failed': True,
                            'optimizer_called': False, 'policy_attempted': False, 'runtime_sec': time.perf_counter()-started,
                            'voltage_backend_calls': []}
                    else:
                        model = copy.deepcopy(model)
                        intercept = anchor.res_line.p_from_mw.to_numpy() - np.asarray(model['S']) @ reference
                        model['flow_intercept'] = intercept.tolist()
                        env.dcnet.flow_intercept = intercept
                        env._aligned_n1_cache_key = None
                        env._predict_gat = lambda _: raw.copy()
                        prep = time.perf_counter() - started
                        try:
                            row = final.evaluate(env, mod, actor, sample, sid, model, protocol['gate'] if arm == 'selected' else None)
                        finally:
                            env._predict_gat = predictor
                        row.update(preparation_failed=False, preparation_sec=prep)
                        row['runtime_sec'] += prep
                else:
                    row = final.evaluate(env, mod, actor, sample, sid, model, protocol['gate'] if arm == 'selected' else None)
                row.update(arm=arm, repeat=repeat, demand_mw=sample['demand_mw'],
                           voltage_backend_calls=env.implicit_feature_calls[before:])
                payload['rows'].append(row)
            write(output, payload)
            print(args.population, repeat+1, j+1, '/', len(ids), [(r['arm'], r['passed']) for r in payload['rows'][-2:]], flush=True)
    assert len(payload['rows']) == 2 * len(ids) * args.repeats
    keys = [(row['repeat'], row['sample_id'], row['arm']) for row in payload['rows']]
    assert len(set(keys)) == len(keys)
    payload['complete'] = True
    write(output, payload)


if __name__ == '__main__':
    main()

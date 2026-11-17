"""Fixed terminal QP fallback; freeze on validation before a held-out evaluation."""
import argparse
import os
import time
from pathlib import Path

import numpy as np

from evaluate_revision import verify
from pilot_validation_feasibility import minimum_linear_dispatch
from revision_common import CANDIDATES, CANDIDATE_SPLIT, GAT, RESULTS, load_aligned, read, sha, write

PROTOCOL=RESULTS/'optimization_diagnostics/terminal_repair_protocol.json'


def freeze_protocol():
    if PROTOCOL.exists():
        raise ValueError('Protocol already frozen; preserve it')
    validation=RESULTS/'optimization_diagnostics/validation_terminal_repair.json'
    result=read(validation)
    assert result['complete'] and result['scope']=='validation only'
    write(PROTOCOL,{'rule_version':1,'selected_using':'validation only',
        'trigger':'base AC accepted, linear N-1 rejected at policy terminal',
        'radius_mw':40.,'voltage_controls':'unchanged from policy terminal',
        'basis':'same original orthogonal physics32','timeout_sec':10.,
        'cut_batch':512,'max_cut_rounds':30,'linear_tolerance_mw':1e-5,
        'validation_result_sha256':sha(validation),
        'data_sha256':sha(CANDIDATES),'split_sha256':sha(CANDIDATE_SPLIT),
        'source_sha256':sha(__file__),
        'solver_source_sha256':sha(Path(__file__).with_name('pilot_validation_feasibility.py')),
        'reporting':'Separate augmented method; retain all original pure-policy results. '
            'The test partition is the previously reported holdout, not a newly collected untouched dataset.'})
    print('Protocol frozen before test evaluation:',PROTOCOL,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',choices=('val','test'),default='val')
    parser.add_argument('--freeze-protocol',action='store_true')
    parser.add_argument('--input')
    parser.add_argument('--output')
    args=parser.parse_args()
    if args.freeze_protocol:
        freeze_protocol()
        return
    if args.input is None:
        args.input=str(RESULTS/('optimization_diagnostics/validation_full_reference.json' if args.scope=='val' else 'full_sac.json'))
    if args.output is None:
        args.output=str(RESULTS/f'optimization_diagnostics/{"validation" if args.scope=="val" else "test"}_terminal_repair.json')
    if Path(args.output).exists():
        parser.error('Output exists; preserve it and select a new path')
    os.environ.setdefault('NUMBA_CACHE_DIR',str(RESULTS/'numba_cache'))
    source=read(args.input)
    assert source['complete'] and source['settings']['scope']==args.scope
    assert source['settings']['method']=='sac' and source['settings']['radius']==40
    data=read(CANDIDATES)
    samples=[s for s in data['samples'] if s.get('is_complete_sced',True)]
    split=read(CANDIDATE_SPLIT)
    ids=split[args.scope+'_ids']
    other=split['test_ids'] if args.scope=='val' else split['val_ids']
    assert not set(ids).intersection(split['train_ids']+other)
    assert [r['sample_id'] for r in source['rows']]==[samples[i]['sample_id'] for i in ids]
    mod=load_aligned()
    env=mod.CleanN1Env(samples,ids,data)
    basis=env._build_orthogonal_physics_action_basis()
    if args.scope=='test':
        protocol=read(PROTOCOL)
        assert protocol['source_sha256']==sha(__file__)
        assert protocol['solver_source_sha256']==sha(Path(__file__).with_name('pilot_validation_feasibility.py'))
        assert protocol['validation_result_sha256']==sha(RESULTS/'optimization_diagnostics/validation_terminal_repair.json')
        assert protocol['data_sha256']==sha(CANDIDATES) and protocol['split_sha256']==sha(CANDIDATE_SPLIT)
    payload={'complete':False,'scope':'validation only' if args.scope=='val' else 'previously reported test partition',
             'protocol_sha256':sha(PROTOCOL) if args.scope=='test' else None,
             'method':'frozen_SAC_plus_terminal_QP',
             'protocol':'Keep accepted points. Repair only AC-accepted, linear-N1-rejected terminal points; '
                 'keep voltage setpoints fixed, preserve original 40-MW radial budget, then rerun original checks.',
             'input_sha256':sha(args.input),'checkpoint_sha256':source['checkpoint_sha256'],
             'data_sha256':sha(CANDIDATES),'split_sha256':sha(CANDIDATE_SPLIT),'gat_sha256':sha(GAT),
             'timing_note':'Additional repair runtime only; baseline and repair were measured in separate runs.',
             'rows':[]}
    for sid,baseline in zip(ids,source['rows']):
        row={'sample_id':baseline['sample_id'],'candidate_index':sid,'demand_mw':baseline['demand_mw'],
             'baseline_passed':baseline['passed'],'passed':baseline['passed'],
             'attempted':False,'baseline_reason':baseline['reason'],'additional_runtime_sec':0.}
        if not baseline['passed'] and baseline['reason']=='n1_violation':
            started=time.perf_counter()
            sample=samples[sid]
            env.current_sample=sample
            env.target_demand=float(sample['demand_mw'])
            env.current_power=np.asarray(baseline['final_dispatch_mw'])
            env.current_gen_vm_pu=np.asarray(baseline['final_generator_vm_pu'])
            env.current_ext_grid_vm_pu=np.array([baseline['final_reference_vm_pu']])
            _,before=verify(env,mod)
            ac_only=dict(before,n1_violations=0,n1_overload_mw=0.)
            if mod.core.adopted_operation_pass(ac_only) and before['n1_violations']>0:
                row['attempted']=True
                initial=mod.core.project_box_sum(env._predict_gat(env.target_demand),env.PMIN,env.PMAX*.99,env.target_demand)
                repaired,solver=minimum_linear_dispatch(env.current_power,env.PMIN,env.PMAX*.99,basis,
                    data['linear_security_model'],timeout=10.,radius_center=initial,radius=40.)
                row['solver']=solver
                if repaired is not None:
                    env.current_power=repaired
                    passed,info=verify(env,mod)
                    radial=float(np.linalg.norm(repaired-initial))
                    assert radial<=40.+1e-6
                    row.update(passed=passed,final_verification=info,final_dispatch_mw=repaired.tolist(),
                               final_generator_vm_pu=env.current_gen_vm_pu.tolist(),
                               final_reference_vm_pu=env.current_ext_grid_vm_pu.tolist(),
                               gat_relative_l2_mw=radial)
            else:
                row['not_attempted']='fresh_checks_not_eligible'
            row['additional_runtime_sec']=time.perf_counter()-started
            print(f"{row['sample_id']}: attempted={row['attempted']} accepted={row['passed']}",flush=True)
        payload['rows'].append(row)
        write(args.output,payload)
    payload['complete']=True
    payload['summary']={'n':len(ids),
        'baseline_passed':sum(r['baseline_passed'] for r in payload['rows']),
        'combined_passed':sum(r['passed'] for r in payload['rows']),
        'attempted':sum(r['attempted'] for r in payload['rows']),
        'newly_accepted_ids':[r['sample_id'] for r in payload['rows'] if r['passed'] and not r['baseline_passed']],
        'mean_additional_runtime_sec':float(np.mean([r['additional_runtime_sec'] for r in payload['rows']]))}
    write(args.output,payload)
    print(payload['summary'],flush=True)


if __name__=='__main__':
    main()

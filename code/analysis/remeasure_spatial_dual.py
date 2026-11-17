"""Same 100 previously examined profiles, final frozen dual-start deployment.

No new population, training, gate selection or optimizer-budget adjustment.
Profile preparation, re-anchoring and all acceptance work remain timed.
"""
import argparse
import copy
import os
import subprocess
import time
from revision_common import REV, RESULTS, CANDIDATES, read, write, sha
os.environ.setdefault('NUMBA_CACHE_DIR', str(RESULTS/'numba_cache'))
import numpy as np
import remeasure_verified_dual as final
import compare_joint_workflows as common
import shared_voltage_runtime as shared

DEST=RESULTS/'spatial_dual_final_v1'
CONDITIONS=RESULTS/'shared_voltage_speed_v3/test_conditions.json'


def prepare():
    if DEST.exists():
        raise RuntimeError('Preserve the existing frozen protocol/results')
    protocol=read(final.DEST/'protocol.json')
    protocol.update(parent_protocol_sha256=sha(final.DEST/'protocol.json'),
        spatial_runner_sha256=sha(__file__), conditions_file=str(CONDITIONS),
        conditions_sha256=sha(CONDITIONS), repetitions=3,
        backup_commit=subprocess.check_output(['git','rev-parse','HEAD'],
            cwd=REV/'original_git_snapshot',text=True).strip(),
        scope='Final-configuration remeasurement on the same previously examined 100 profiles (seed 2026101802), not a newly untouched confirmation set. No retuning.',
        timing='Resident models excluded equally; per-profile network preparation, load normalization, AC anchoring, analytic feature, policy attempt, dual-start fallback and verification/receipt costs included. Rotating serial order.')
    assert len(read(CONDITIONS)['samples'])==100
    write(DEST/'protocol.json',protocol)


def run():
    import pandapower as pp
    protocol=read(DEST/'protocol.json')
    assert sha(__file__)==protocol['spatial_runner_sha256']
    assert sha(CONDITIONS)==protocol['conditions_sha256']
    for path,digest in {**protocol['input_hashes'],**protocol['source_hashes'],
                        **protocol['source_hashes_v2']}.items():
        assert sha(path)==digest,path
    path=DEST/'results.json'
    if path.exists():
        raise RuntimeError('Preserve previous measurement')
    data=read(CANDIDATES);samples=read(CONDITIONS)['samples']
    mod,envs,actor=final.old.setup(samples,data)
    template=copy.deepcopy(envs['direct']._ac_net_template)
    payload={'complete':False,'protocol_sha256':sha(DEST/'protocol.json'),
             'conditions_sha256':sha(CONDITIONS),'scope':protocol['scope'],'rows':[]}
    for repeat in range(protocol['repetitions']):
        for sid,sample in enumerate(samples):
            arms=('direct','selected') if (sid+repeat)%2==0 else ('selected','direct')
            for arm in arms:
                common.sync();started=time.perf_counter()
                env=envs[arm];env.current_sample=sample;env.target_demand=sample['demand_mw']
                shared.clear_case(env);before=len(env.implicit_feature_calls)
                net=copy.deepcopy(template)
                net.load.p_mw*=sample['p_weights'];net.load.q_mvar*=sample['q_weights']
                scale=sample['demand_mw']/net.load.p_mw.sum()
                net.load.p_mw*=scale;net.load.q_mvar*=scale;env._ac_net_template=net
                raw=env._predict_gat(sample['demand_mw'])
                reference=mod.core.project_box_sum(raw,env.PMIN,env.PMAX*.99,sample['demand_mw'])
                anchor=copy.deepcopy(net);anchor.gen.p_mw=reference
                try:
                    pp.runpp(anchor,enforce_q_lims=True,calculate_voltage_angles=True,numba=True,max_iteration=100)
                except pp.LoadflowNotConverged:
                    common.sync()
                    row={'sample_id':sample['sample_id'],'passed':False,'preparation_failed':True,
                         'optimizer_called':False,'policy_attempted':False,
                         'runtime_sec':time.perf_counter()-started,'voltage_backend_calls':[]}
                else:
                    model=copy.deepcopy(data['linear_security_model'])
                    intercept=anchor.res_line.p_from_mw.to_numpy()-np.asarray(model['S'])@reference
                    model['flow_intercept']=intercept.tolist();env.dcnet.flow_intercept=intercept
                    env._aligned_n1_cache_key=None
                    predictor=env._predict_gat;env._predict_gat=lambda _:raw.copy()
                    preparation=time.perf_counter()-started
                    try:
                        row=final.evaluate(env,mod,actor,sample,sid,model,
                            protocol['gate'] if arm=='selected' else None)
                    finally:
                        env._predict_gat=predictor
                    row.update(preparation_failed=False,preparation_sec=preparation,
                               voltage_backend_calls=env.implicit_feature_calls[before:])
                    row['runtime_sec']+=preparation
                row.update(arm=arm,repeat=repeat,demand_mw=sample['demand_mw'])
                payload['rows'].append(row)
            write(path,payload)
            print('final spatial',repeat+1,sid+1,'/100',flush=True)
    payload['complete']=True;write(path,payload)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','run'))
    globals()[parser.parse_args().stage]()

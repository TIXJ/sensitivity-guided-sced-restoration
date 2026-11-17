"""Resume an exact paired prefix after a Windows atomic-save interruption."""
import copy
from datetime import datetime
import time
import numpy as np
import remeasure_spatial_dual as study
from revision_common import read,write,sha,CANDIDATES


def safe_write(path,payload):
    for attempt in range(50):
        try:
            write(path,payload)
            return
        except PermissionError:
            if attempt==49:raise
            time.sleep(.1)


def main():
    import pandapower as pp
    protocol=read(study.DEST/'protocol.json')
    assert sha(study.__file__)==protocol['spatial_runner_sha256']
    assert sha(study.CONDITIONS)==protocol['conditions_sha256']
    for path,digest in {**protocol['input_hashes'],**protocol['source_hashes'],**protocol['source_hashes_v2']}.items():
        assert sha(path)==digest,path
    path=study.DEST/'results.json';temporary=path.with_suffix('.json.tmp')
    saved=read(path);payload=saved
    recovery=study.DEST/'recovery'/datetime.now().strftime('%Y%m%d_%H%M%S')
    recovery.mkdir(parents=True)
    for source in (path,temporary):
        if source.exists():(recovery/source.name).write_bytes(source.read_bytes())
    if temporary.exists():
        pending=read(temporary)
        assert pending['protocol_sha256']==saved['protocol_sha256']==sha(study.DEST/'protocol.json')
        assert pending['rows'][:len(saved['rows'])]==saved['rows']
        assert len(pending['rows'])>=len(saved['rows'])
        payload=pending
    assert not payload['complete'] and len(payload['rows'])%2==0
    data=read(CANDIDATES);samples=read(study.CONDITIONS)['samples']
    expected=[(r,s['sample_id'],arm) for r in range(3) for sid,s in enumerate(samples)
              for arm in (('direct','selected') if (sid+r)%2==0 else ('selected','direct'))]
    actual=[(r['repeat'],r['sample_id'],r['arm']) for r in payload['rows']]
    assert actual==expected[:len(actual)] and len(set(actual))==len(actual)
    completed_pairs=len(actual)//2
    payload.setdefault('resume_segments',[]).append({'retained_pairs':completed_pairs,
        'resume_runner_sha256':sha(__file__),'recovery_folder':str(recovery),
        'reason':'Windows atomic replace denied; complete ordered records recovered without changing physical configuration or measurement order.'})
    safe_write(path,payload)
    mod,envs,actor=study.final.old.setup(samples,data)
    template=copy.deepcopy(envs['direct']._ac_net_template)
    for repeat in range(3):
        for sid,sample in enumerate(samples):
            if repeat*len(samples)+sid<completed_pairs:continue
            arms=('direct','selected') if (sid+repeat)%2==0 else ('selected','direct')
            for arm in arms:
                study.common.sync();started=time.perf_counter()
                env=envs[arm];env.current_sample=sample;env.target_demand=sample['demand_mw']
                study.shared.clear_case(env);before=len(env.implicit_feature_calls)
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
                    study.common.sync()
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
                        row=study.final.evaluate(env,mod,actor,sample,sid,model,
                            protocol['gate'] if arm=='selected' else None)
                    finally:env._predict_gat=predictor
                    row.update(preparation_failed=False,preparation_sec=preparation,
                               voltage_backend_calls=env.implicit_feature_calls[before:])
                    row['runtime_sec']+=preparation
                row.update(arm=arm,repeat=repeat,demand_mw=sample['demand_mw'],timing_segment=1)
                payload['rows'].append(row)
            safe_write(path,payload)
            print('resumed final spatial',repeat+1,sid+1,'/100',flush=True)
    payload['complete']=True;safe_write(path,payload)


if __name__=='__main__':
    main()

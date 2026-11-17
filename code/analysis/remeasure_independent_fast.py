"""Remeasure the fixed 100 cases with the frozen accelerated voltage feature.

Three timed workflows retain rotating serial order and first-use context costs.
A separate cached-legacy replay checks trajectories, outside workflow timings.
"""
import copy
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from revision_common import RESULTS,CANDIDATES,CHECKPOINTS,CODE,read,write,sha,load_aligned
import compare_joint_workflows as common
from compare_joint_gate_workflows import training_range
from evaluate_independent_conditions import pure_policy
from reset_voltage_jacobian_fast import VERSION,install_fast_voltage_jacobian

SOURCE=RESULTS/'independent_conditions_v1'
DEST=RESULTS/'independent_fast_remeasurement_20261018'
ARMS=('sac_only','direct_pf','gated_pf')
CONTROL_FIELDS=('final_dispatch_mw','final_generator_vm_pu','final_reference_vm_pu')


def compare_controls(a,b):
    result={}
    for key in CONTROL_FIELDS:
        if key not in a or key not in b:
            result[key]=None
        else:
            result[key]=float(np.max(abs(np.asarray(a[key]).ravel()-np.asarray(b[key]).ravel()),initial=0))
    return result


def capture(actor,callback):
    original=actor.select_action
    tensors=[]
    def action(*args,**kwargs):
        value=original(*args,**kwargs)
        tensors.append(value.detach())
        return value
    actor.select_action=action
    try: row=callback()
    finally: actor.select_action=original
    # Trace transfer/storage is diagnostic work, outside measured inference.
    trace=torch.cat(tensors,dim=0).cpu().numpy() if tensors else np.empty((0,101),dtype=np.float32)
    return row,trace


def configure(env,sample,template,raw,intercept):
    env.current_sample=sample
    env.target_demand=sample['demand_mw']
    env._ac_net_template=copy.deepcopy(template)
    env.dcnet.flow_intercept=intercept.copy()
    env._aligned_n1_cache_key=None
    env._voltage_jacobian_memory={}
    env._predict_gat=lambda demand:raw.copy()


def main():
    os.environ.setdefault('NUMBA_CACHE_DIR',str(RESULTS/'numba_cache'))
    if DEST.exists(): raise RuntimeError('Preserve previous output; do not rerun into a populated first-use cache')
    original=read(SOURCE/'results.json'); assert original['complete'] and len(original['rows'])==300
    samples=read(SOURCE/'conditions.json')['samples']; assert len(samples)==100
    old={(r['sample_id'],r['arm']):r for r in original['rows']}
    hashes={**common.hashes(),str(CODE/'reset_voltage_jacobian_fast.py'):sha(CODE/'reset_voltage_jacobian_fast.py'),
        str(Path(__file__)):sha(__file__),str(SOURCE/'results.json'):sha(SOURCE/'results.json'),
        str(SOURCE/'conditions.json'):sha(SOURCE/'conditions.json')}
    protocol={'backend':VERSION,'hashes':hashes,'training_range':list(training_range()),
        'sample_ids':[s['sample_id'] for s in samples],'arms':list(ARMS),
        'timing':'Rotating serial arm order, one repetition per existing case. All profile/anchor preparation, first-use feature construction, correction, optimization and final checks included. Resident setup excluded; remaining first-use compilation included. No cross-arm feature sharing.',
        'interpretation':'Implementation remeasurement of the same already evaluated 100 cases, not a new independent test set. No model, gate, action, reward or solver-budget tuning.',
        'audit':'Separate historical-feature replay uses copied original per-case feature caches to compare action sequences and terminal controls. Its time is not a workflow timing.'}
    DEST.mkdir(); write(DEST/'protocol.json',protocol)
    payload={'complete':False,'protocol':protocol,'rows':[],'audits':[]}
    write(DEST/'results.json',payload)
    import pandapower as pp
    data=read(CANDIDATES); mod=load_aligned(); ids=list(range(100))
    envs={arm:mod.CleanN1Env(samples,ids,data) for arm in ARMS}
    for arm in ('sac_only','gated_pf'):
        install_fast_voltage_jacobian(envs[arm],mod,DEST/'fast_cache'/arm)
    legacy=mod.CleanN1Env(samples,ids,data)
    template=copy.deepcopy(envs['sac_only']._ac_net_template)
    setup=mod.CleanN1Env(samples,ids,data)
    setup._reset_voltage_jacobian=lambda:np.zeros((setup._n_bus,setup.N+1))
    obs=setup.reset(0); dim=setup._n_shift_dirs+setup._n_key_gens
    mod.core.SAC_LOG_STD_MAX_VECTOR=np.r_[np.full(dim,mod.ARGS.active_log_std_max),
        np.full(setup.action_dim-dim,mod.ARGS.voltage_log_std_max)].astype(np.float32)
    actor=mod.BranchedGaussianActor(len(obs),dim,setup.action_dim-dim).to(mod.core.DEVICE)
    actor.load_state_dict(torch.load(CHECKPOINTS['sac'],map_location=mod.core.DEVICE,weights_only=True)); actor.eval()
    original_runopp=pp.runopp
    def pf_init(net,**kwargs):
        if kwargs.get('init')=='flat': kwargs['init']='pf'
        return original_runopp(net,**kwargs)
    pp.runopp=pf_init
    try:
        for sid,sample in enumerate(samples):
            case_rows={}; action_traces={}; matrices={}; prepared=None
            for arm in ARMS[sid%3:]+ARMS[:sid%3]:
                common.sync(); started=time.perf_counter(); env=envs[arm]
                env.current_sample=sample; env.target_demand=sample['demand_mw']
                env._ac_net_template=copy.deepcopy(template); net=env._ac_net_template
                net.load.p_mw*=sample['p_weights']; net.load.q_mvar*=sample['q_weights']
                factor=sample['demand_mw']/net.load.p_mw.sum(); net.load.p_mw*=factor; net.load.q_mvar*=factor
                raw=env._predict_gat(sample['demand_mw'])
                reference=mod.core.project_box_sum(raw,env.PMIN,env.PMAX*.99,sample['demand_mw'])
                anchor=copy.deepcopy(net); anchor.gen.p_mw=reference
                try:
                    pp.runpp(anchor,enforce_q_lims=True,calculate_voltage_angles=True,numba=True,max_iteration=100)
                except pp.LoadflowNotConverged as exc:
                    row={'sample_id':sample['sample_id'],'demand_mw':sample['demand_mw'],'arm':arm,
                        'passed':False,'preparation_failed':True,'reason':'anchor_nonconvergence','error':str(exc),
                        'runtime_sec':time.perf_counter()-started,'optimizer_called':False}
                    payload['rows'].append(row); case_rows[arm]=row
                    write(DEST/'results.json',payload); continue
                model=copy.deepcopy(data['linear_security_model'])
                intercept=anchor.res_line.p_from_mw.to_numpy()-np.asarray(model['S'])@reference
                model['flow_intercept']=intercept.tolist(); env.dcnet.flow_intercept=intercept
                env._aligned_n1_cache_key=None; env._voltage_jacobian_memory={}
                predictor=env._predict_gat; env._predict_gat=lambda demand:raw.copy()
                before_stats=len(getattr(env,'fast_voltage_stats',[]))
                prep=time.perf_counter()-started
                try:
                    if arm=='sac_only':
                        row,trace=capture(actor,lambda:pure_policy(env,mod,actor,sample,sid))
                    else:
                        inside=protocol['training_range'][0]<=sample['demand_mw']<=protocol['training_range'][1]
                        effective='gat_sac_optimizer' if arm=='gated_pf' and inside else 'gat_optimizer'
                        row,trace=capture(actor,lambda:common.evaluate(env,mod,actor,sample,sid,model,effective))
                finally: env._predict_gat=predictor
                row.update(arm=arm,demand_mw=sample['demand_mw'],preparation_failed=False,preparation_sec=prep)
                row['runtime_sec']+=prep
                if arm!='direct_pf':
                    row['voltage_backend_calls']=env.fast_voltage_stats[before_stats:]
                    assert all(r['source'] in ('computed','legacy_fallback','unavailable_zero') for r in row['voltage_backend_calls']), 'Unexpected cross-case cache hit'
                    if row['voltage_backend_calls']: matrices[arm]=env._reset_voltage_jacobian().copy()
                trace_dir=DEST/'traces'/sample['sample_id']; trace_dir.mkdir(parents=True,exist_ok=True)
                trace_path=trace_dir/(arm+'.npz')
                np.savez_compressed(trace_path,actions=trace)
                row['action_trace_sha256']=sha(trace_path)
                row['original_control_max_abs_differences']=compare_controls(row,old[sample['sample_id'],arm])
                payload['rows'].append(row); case_rows[arm]=row; action_traces[arm]=trace
                if arm=='sac_only': prepared=(copy.deepcopy(net),raw.copy(),intercept.copy())
                write(DEST/'results.json',payload)
            if prepared is not None:
                net,raw,intercept=prepared
                configure(legacy,sample,net,raw,intercept)
                legacy._voltage_jacobian_dir=DEST/'legacy_audit_cache'/sample['sample_id']
                legacy._voltage_jacobian_dir.mkdir(parents=True)
                source_cache=SOURCE/'jacobians/sac_only'/sample['sample_id']
                for path in source_cache.glob('*.npz'):
                    shutil.copy2(path,legacy._voltage_jacobian_dir/path.name)
                baseline,trace=capture(actor,lambda:pure_policy(legacy,mod,actor,sample,sid))
                jacobian=legacy._reset_voltage_jacobian()
                fast=case_rows['sac_only']; fast_trace=action_traces['sac_only']
                same_shape=trace.shape==fast_trace.shape
                audit={'sample_id':sample['sample_id'],'legacy_audit_runtime_sec':baseline['runtime_sec'],
                    'legacy_matches_original_controls':compare_controls(baseline,old[sample['sample_id'],'sac_only']),
                    'acceptance_same':baseline['passed']==fast['passed'],
                    'steps_same':baseline['policy_steps']==fast['policy_steps'],
                    'action_shape_same':same_shape,
                    'action_max_abs_error':float(np.max(abs(trace-fast_trace),initial=0)) if same_shape else None,
                    'jacobian_max_abs_error':float(np.max(abs(jacobian-matrices['sac_only']))),
                    'control_max_abs_errors':compare_controls(baseline,fast)}
                trace_path=DEST/'traces'/sample['sample_id']/'legacy_audit.npz'
                np.savez_compressed(trace_path,actions=trace,jacobian=jacobian)
                audit['legacy_trace_sha256']=sha(trace_path)
                payload['audits'].append(audit)
            write(DEST/'results.json',payload)
            print('CASE',sid+1,'/100',sample['sample_id'],
                {a:(case_rows[a]['passed'],round(case_rows[a]['runtime_sec'],3)) for a in ARMS},flush=True)
    finally: pp.runopp=original_runopp
    assert len(payload['rows'])==300
    for path,expected in hashes.items(): assert sha(path)==expected,path
    payload['complete']=True; write(DEST/'results.json',payload)


if __name__=='__main__': main()

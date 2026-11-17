"""Targeted reviewer diagnostics under protocols frozen before evaluation.

Historical and new records are never overwritten. No policy or gate is tuned.
"""
import argparse
import copy
import os
import subprocess
import time
from collections import Counter
import numpy as np
from revision_common import RESULTS,REV,CANDIDATES,CANDIDATE_SPLIT,read,write,sha
os.environ.setdefault('NUMBA_CACHE_DIR',str(RESULTS/'numba_cache'))
import compare_joint_workflows as common
import evaluate_short_policy as short
import evaluate_implicit_short_policy as previous
import shared_voltage_runtime as shared
from pilot_validation_feasibility import minimum_linear_dispatch,bounded_voltage_repair

DEST=RESULTS/'reviewer_followup_v1'


def source_hashes():
    return {str(p):sha(p) for p in (__file__,common.__file__,short.__file__,previous.__file__,
        shared.__file__,__import__('shared_joint_optimizer').__file__,
        __import__('pilot_validation_feasibility').__file__)}


def prepare():
    if DEST.exists():raise RuntimeError('Preserve the existing study')
    DEST.mkdir()
    data=read(CANDIDATES);samples=data['samples'];split=read(CANDIDATE_SPLIT)
    # Selection is by ID and demand, before any diagnostic outcome is known.
    ids=[]
    for demand in (16000.,16200.,16400.,16600.,16800.):
        ids.extend(sorted((i for i in split['val_ids'] if samples[i]['demand_mw']==demand),
                          key=lambda i:samples[i]['sample_id'])[:3])
    assert len(ids)==15 and not set(ids).intersection(split['test_ids']+split['train_ids'])
    write(DEST/'protocol.json',{'source_hashes':source_hashes(),'input_hashes':common.hashes(),
        'backup_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REV/'original_git_snapshot',text=True).strip(),
        'gate':read(RESULTS/'implicit_short_policy_v2/protocol.json')['rule'],
        'dual':{'ids':split['test_ids'],'repetitions':3,'initializers':['pf','flat'],
            'max_opf_calls_per_initializer':4,'max_ip_iterations_per_call':100,
            'scope':'Same maximum 8 OPF calls / 800 IP iterations in both workflows, including intermediate acceptance checks and final fresh verification. This is an iteration cap, not a hard wall-time deadline.'},
        'warm':{'seed':2026101804,'patterns':10,'periods':10,
            'demand_cycle_mw':[16000,16200,16400,16600,16800]*2,
            'scope':'Fixed spatial weights within each sequence; demand and costs vary. Only static network preparation is reused across periods in both arms; no cross-period voltage matrix or solved-point cache.'},
        'tight':{'validation_ids':ids,'band_pu':[.94,1.08],'radii_mw':[40,80],
            'linear_seconds':5,'voltage_seconds':10,'voltage_pf_calls':250,
            'scope':'Sequential bounded QP plus voltage repair gives feasible witnesses when successful, not joint infeasibility certificates when unsuccessful. Full generator-wise optimization is a separate diagnostic.'}})


def two_initializers(env,mod,model,reference,demand):
    """Same two starts and hard OPF-call/iteration caps for both workflows."""
    import pandapower as pp
    from shared_joint_optimizer import solve,voltage_box
    original=pp.runopp;attempts=[];candidate=None;started=time.perf_counter()
    nominal,low,high=voltage_box(env,mod)
    try:
        for init in ('pf','flat'):
            calls=[]
            def capped(net,**kwargs):
                if len(calls)>=4:raise pp.OPFNotConverged('fixed four-call initializer budget exhausted')
                kwargs['numba']=True
                if kwargs.get('init')=='flat':kwargs['init']=init
                kwargs['PDIPM_MAX_IT']=100
                calls.append({'init':kwargs.get('init'),'max_ip_iterations':100})
                return original(net,**kwargs)
            pp.runopp=capped
            env.current_power=reference.copy();env.current_gen_vm_pu=nominal[:env.N].copy()
            env.current_ext_grid_vm_pu=nominal[env.N:].copy();env._aligned_n1_cache_key=None
            result=solve(env,mod,model,reference,demand)
            accepted=False;info=None
            if result['returned_candidate']:
                candidate=result
                env.current_power=np.asarray(result['power']);env.current_gen_vm_pu=np.asarray(result['gen_vm'])
                env.current_ext_grid_vm_pu=np.asarray(result['ext_vm']);env._aligned_n1_cache_key=None
                env._verification_fresh=True
                try:accepted,info=common.check(env,mod,low,high)
                finally:env._verification_fresh=False
            attempts.append({'initializer':init,'actual_opf_calls':len(calls),'calls':calls,
                             'accepted':accepted,'verification':info,'solver':result})
            if accepted:break
    finally:pp.runopp=original
    result=dict(candidate or {'returned_candidate':False,'status':'both_initializers_failed'})
    result.update(initializer_attempts=attempts,actual_opf_calls=sum(a['actual_opf_calls'] for a in attempts),
                  runtime_sec=time.perf_counter()-started)
    assert result['actual_opf_calls']<=8
    return result


def setup(samples,data):
    mod,envs,actor=previous.setup(samples,data,{})
    for key in ('direct','selected'):shared.install(envs[key],mod)
    return mod,envs,actor


def execute(env,mod,actor,sample,sid,model,rule):
    shared.clear_case(env);before=len(env.implicit_feature_calls)
    row=short.evaluate(env,mod,actor,sample,sid,model,rule['cap'] if rule else 0,rule)
    row['voltage_backend_calls']=env.implicit_feature_calls[before:]
    return row


def dual(protocol):
    path=DEST/'dual.json'
    if path.exists():raise RuntimeError('Preserve existing measurement')
    data=read(CANDIDATES);samples=data['samples'];ids=protocol['dual']['ids']
    mod,envs,actor=setup(samples,data);original=common.solve;common.solve=two_initializers
    payload={'complete':False,'protocol_sha256':sha(DEST/'protocol.json'),'rows':[]}
    try:
        for repeat in range(3):
            for j,sid in enumerate(ids):
                arms=('direct','selected') if (j+repeat)%2==0 else ('selected','direct')
                for arm in arms:
                    env=envs[arm];env.current_sample=samples[sid]
                    row=execute(env,mod,actor,samples[sid],sid,data['linear_security_model'],protocol['gate'] if arm=='selected' else None)
                    row.update(arm=arm,repeat=repeat);payload['rows'].append(row)
                write(path,payload)
                print('dual',repeat+1,j+1,'/75',flush=True)
    finally:common.solve=original
    payload['complete']=True;write(path,payload)


def warm(protocol):
    path=DEST/'warm.json'
    if path.exists():raise RuntimeError('Preserve existing measurement')
    data=read(CANDIDATES)
    probe=__import__('revision_common').load_aligned().CleanN1Env(data['samples'],[0],data)
    patterns=short.profiles(data,probe._ac_net_template,protocol['warm']['seed'],2)
    train=[data['samples'][i] for i in read(CANDIDATE_SPLIT)['train_ids']]
    means={k:np.mean([s[k] for s in train],axis=0) for k in ('sample_cp0','sample_cp1','sample_cp2')}
    rng=np.random.default_rng(protocol['warm']['seed']+1);samples=[]
    for pattern,p in enumerate(patterns):
        for period,demand in enumerate(protocol['warm']['demand_cycle_mw']):
            row=copy.deepcopy(p);row.update(sample_id=f'seq{pattern:02d}_t{period:02d}',demand_mw=float(demand),pattern=pattern,period=period)
            sigma=float(rng.uniform(.02,.03))
            row.update({k:(v*(1+rng.normal(0,sigma,len(v)))).tolist() for k,v in means.items()})
            samples.append(row)
    write(DEST/'warm_conditions.json',{'samples':samples,'protocol_sha256':sha(DEST/'protocol.json')})
    mod,envs,actor=setup(samples,data);template=copy.deepcopy(envs['direct']._ac_net_template)
    import pandapower as pp
    original=pp.runopp
    def initialized(net,**kwargs):
        if kwargs.get('init')=='flat':kwargs['init']='pf'
        return original(net,**kwargs)
    pp.runopp=initialized
    payload={'complete':False,'protocol_sha256':sha(DEST/'protocol.json'),'rows':[]};network={}
    try:
        for sid,sample in enumerate(samples):
            arms=('direct','selected') if (sample['pattern']+sample['period'])%2==0 else ('selected','direct')
            for arm in arms:
                common.sync();started=time.perf_counter();env=envs[arm];env.current_sample=sample
                key=(sample['pattern'],arm)
                if key not in network:
                    net=copy.deepcopy(template);net.load.p_mw*=sample['p_weights'];net.load.q_mvar*=sample['q_weights']
                    network[key]=net
                net=copy.deepcopy(network[key]);scale=sample['demand_mw']/net.load.p_mw.sum()
                net.load.p_mw*=scale;net.load.q_mvar*=scale;env._ac_net_template=net;env.target_demand=sample['demand_mw']
                raw=env._predict_gat(sample['demand_mw']);reference=mod.core.project_box_sum(raw,env.PMIN,env.PMAX*.99,sample['demand_mw'])
                anchor=copy.deepcopy(net);anchor.gen.p_mw=reference
                try:pp.runpp(anchor,enforce_q_lims=True,calculate_voltage_angles=True,numba=True,max_iteration=100)
                except pp.LoadflowNotConverged:
                    row={'sample_id':sample['sample_id'],'passed':False,'preparation_failed':True,'runtime_sec':time.perf_counter()-started}
                else:
                    model=copy.deepcopy(data['linear_security_model'])
                    intercept=anchor.res_line.p_from_mw.to_numpy()-np.asarray(model['S'])@reference
                    model['flow_intercept']=intercept.tolist();env.dcnet.flow_intercept=intercept
                    predictor=env._predict_gat;env._predict_gat=lambda _:raw.copy()
                    preparation=time.perf_counter()-started
                    try:row=execute(env,mod,actor,sample,sid,model,protocol['gate'] if arm=='selected' else None)
                    finally:env._predict_gat=predictor
                    row['runtime_sec']+=preparation;row.update(preparation_failed=False,preparation_sec=preparation)
                row.update(arm=arm,pattern=sample['pattern'],period=sample['period'])
                payload['rows'].append(row)
            write(path,payload);print('sequence',sid+1,'/100',flush=True)
    finally:pp.runopp=original
    payload['complete']=True;write(path,payload)


def tight(protocol):
    path=DEST/'tight.json'
    if path.exists():raise RuntimeError('Preserve existing diagnostic')
    data=read(CANDIDATES);samples=data['samples'];ids=protocol['tight']['validation_ids']
    mod=__import__('revision_common').load_aligned();env=mod.CleanN1Env(samples,ids,data)
    mod.core.RL_VMIN_PU,mod.core.RL_VMAX_PU=protocol['tight']['band_pu']
    basis=env._build_orthogonal_physics_action_basis()
    payload={'complete':False,'protocol_sha256':sha(DEST/'protocol.json'),'rows':[]}
    import pandapower as pp
    original=pp.runopp
    def initialized(net,**kwargs):
        if kwargs.get('init')=='flat':kwargs['init']='pf'
        return original(net,**kwargs)
    pp.runopp=initialized
    try:
        for j,sid in enumerate(ids):
            sample=samples[sid];env.current_sample=sample;demand=sample['demand_mw']
            p0=mod.core.project_box_sum(env._predict_gat(demand),env.PMIN,env.PMAX*.99,demand)
            for radius in protocol['tight']['radii_mw']:
                started=time.perf_counter()
                power,linear=minimum_linear_dispatch(p0,env.PMIN,env.PMAX*.99,basis,data['linear_security_model'],
                    timeout=5,radius_center=p0,radius=radius)
                repair=None
                if power is not None:repair=bounded_voltage_repair(env,mod,power,demand,seconds=10,calls=250)
                row={'sample_id':sample['sample_id'],'arm':f'basis_{radius}','passed':bool(repair and repair['passed']),
                    'linear':linear,'voltage_repair':repair,'runtime_sec':time.perf_counter()-started,
                    'radius_mw':radius,'final_dispatch_mw':None if power is None else power.tolist()}
                if power is not None:
                    delta=power-p0
                    row['actual_l2_mw']=float(np.linalg.norm(delta));row['span_residual_mw']=float(np.linalg.norm(delta-basis@(basis.T@delta)))
                    assert row['actual_l2_mw']<=radius+1e-5 and row['span_residual_mw']<1e-6
                payload['rows'].append(row)
            env.current_power=p0.copy();env.target_demand=demand
            nominal,lo,hi=common.voltage_box(env,mod)
            env.current_gen_vm_pu=nominal[:env.N].copy();env.current_ext_grid_vm_pu=nominal[env.N:].copy()
            result=__import__('shared_joint_optimizer').solve(env,mod,data['linear_security_model'],p0,demand)
            if result['returned_candidate']:
                env.current_power=np.asarray(result['power']);env.current_gen_vm_pu=np.asarray(result['gen_vm']);env.current_ext_grid_vm_pu=np.asarray(result['ext_vm'])
            env._aligned_n1_cache_key=None;passed,info=common.check(env,mod,lo,hi)
            payload['rows'].append({'sample_id':sample['sample_id'],'arm':'full_joint','passed':passed,'optimizer':result,'final_verification':info})
            write(path,payload);print('tight diagnostic',j+1,'/15',flush=True)
    finally:pp.runopp=original
    payload['complete']=True;write(path,payload)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','dual','warm','tight'))
    args=parser.parse_args()
    if args.stage=='prepare':return prepare()
    protocol=read(DEST/'protocol.json');assert protocol['source_hashes']==source_hashes()
    for path,digest in protocol['input_hashes'].items():assert sha(path)==digest,path
    globals()[args.stage](protocol)


if __name__=='__main__':main()

"""Validation-selected short SAC attempts with training-reference features.

This is a new deployment variant, not a numerically equivalent implementation.
Original checkpoints, measurements, constraints and optimizer remain unchanged.
"""
import argparse
import copy
import time
import numpy as np
import torch
from revision_common import (RESULTS, CANDIDATES, CANDIDATE_SPLIT, CHECKPOINTS,
                             read, write, sha, load_aligned)
import compare_joint_workflows as common
from probe_voltage_kernel_v4 import solver_finite
from constraint_diagnostics import operation_flags
from evaluate_revision import cost

DEST = RESULTS/'short_policy_deployment_v1'


def profiles(data, net, seed, count):
    import networkx as nx
    import pandapower.topology as top
    rng = np.random.default_rng(seed)
    train = [data['samples'][i] for i in read(CANDIDATE_SPLIT)['train_ids']]
    costs = {k:np.mean([s[k] for s in train],axis=0) for k in ('sample_cp0','sample_cp1','sample_cp2')}
    graph = top.create_nxgraph(net,include_trafos=True)
    buses = net.load.bus.to_numpy()
    result=[]
    for demand in (16000.,16200.,16400.,16600.,16800.):
        for i in range(count):
            centers=rng.choice(np.unique(buses),3,replace=False)
            distances=[nx.single_source_shortest_path_length(graph,int(c)) for c in centers]
            regions=np.argmin([[d.get(int(b),10000) for b in buses] for d in distances],axis=0)
            pw=rng.uniform(.9,1.1,3)[regions]
            qw=rng.uniform(.9,1.1,3)[regions]*rng.uniform(.97,1.03,len(buses))
            sigma=float(rng.uniform(.02,.03))
            row={'sample_id':f's{seed}_{int(demand)}_{i:02d}', 'demand_mw':demand,
                 'p_weights':pw.tolist(),'q_weights':qw.tolist(),'region_centers':centers.tolist()}
            row.update({k:(v*(1+rng.normal(0,sigma,len(v)))).tolist() for k,v in costs.items()})
            result.append(row)
    return result


def feature(info, sample, mod):
    vm=np.asarray(info.get('bus_vm_pu',[]),dtype=float)
    if not vm.size or not np.isfinite(vm).all():
        return {'max_gap_pu':10.,'sum_gap_pu':10.,'valid':False,
                'demand_mw':float(sample['demand_mw'])}
    gaps=np.maximum(mod.core.RL_VMIN_PU-vm,0)+np.maximum(vm-mod.core.RL_VMAX_PU,0)
    return {'max_gap_pu':float(np.max(gaps)), 'sum_gap_pu':float(np.sum(gaps)),
            'demand_mw':float(sample['demand_mw'])}


def selected(features, rule):
    return (features.get('valid',True) and 16000<=features['demand_mw']<=rule.get('max_demand_mw',16400)
            and features['max_gap_pu']<=rule['max_gap_pu']
            and features['sum_gap_pu']<=rule['sum_gap_pu'])


def evaluate(env, mod, actor, sample, sid, model, cap, rule=None):
    common.sync(); start=time.perf_counter()
    reference=mod.core.project_box_sum(env._predict_gat(sample['demand_mw']), env.PMIN,
                                     env.PMAX*.99,sample['demand_mw'])
    nominal,low,high=common.voltage_box(env,mod)
    env.target_demand=sample['demand_mw']; env.current_power=reference.copy()
    env.current_gen_vm_pu=nominal[:env.N].copy(); env.current_ext_grid_vm_pu=nominal[env.N:].copy()
    env.corr_step=0; env._aligned_n1_cache_key=None
    initial_passed,info=common.check(env,mod,low,high)
    features=feature(info,sample,mod)
    use=bool(cap and not initial_passed and (rule is None or selected(features,rule)))
    after_initial=time.perf_counter(); steps=0; policy_passed=initial_passed
    if use:
        obs=env.reset(sid)
        for steps in range(1,cap+1):
            with torch.no_grad():
                raw=actor.select_action(torch.as_tensor(obs,dtype=torch.float32,device=mod.core.DEVICE).unsqueeze(0),deterministic=True).cpu().numpy()[0]
            lo,hi=env.get_action_bounds()
            obs,_,done,truncated,_=env.step(lo+(raw+1)*.5*(hi-lo))
            if done or truncated: break
        policy_passed,info=common.check(env,mod,low,high)
    after_policy=time.perf_counter()
    optimizer=None
    if not policy_passed:
        env.current_power=reference.copy(); env.current_gen_vm_pu=nominal[:env.N].copy()
        env.current_ext_grid_vm_pu=nominal[env.N:].copy()
        optimizer=common.solve(env,mod,model,reference,sample['demand_mw'])
        if optimizer['returned_candidate']:
            env.current_power=np.asarray(optimizer['power'])
            env.current_gen_vm_pu=np.asarray(optimizer['gen_vm'])
            env.current_ext_grid_vm_pu=np.asarray(optimizer['ext_vm'])
    after_solver=time.perf_counter()
    env._aligned_n1_cache_key=None
    env._verification_fresh=True
    try: passed,info=common.check(env,mod,low,high)
    finally: env._verification_fresh=False
    common.sync(); ended=time.perf_counter()
    return {'sample_id':sample['sample_id'],'passed':passed,'policy_attempted':use,
            'policy_gate_passed':policy_passed,'initial_passed':initial_passed,
            'policy_steps':steps,'optimizer_called':optimizer is not None,
            'optimizer':optimizer,'runtime_sec':ended-start,'initial_features':features,
            'initial_stage_sec':after_initial-start,'policy_sec':after_policy-after_initial,
            'optimizer_sec':after_solver-after_policy,'final_check_sec':ended-after_solver,
            'scheduled_cost':cost(sample,env.current_power),'constraint_flags':operation_flags(info),
            'final_verification':info,'final_dispatch_mw':env.current_power.tolist(),
            'final_generator_vm_pu':env.current_gen_vm_pu.tolist(),
            'final_reference_vm_pu':env.current_ext_grid_vm_pu.tolist()}


def setup(samples, data, bank):
    mod=load_aligned()
    envs={key:mod.CleanN1Env(samples,list(range(len(samples))),data) for key in ('direct','short4','short8','selected')}
    for env in envs.values():
        env._reset_voltage_jacobian=lambda env=env:bank[min(bank,key=lambda d:abs(d-env.target_demand))]
    env=envs['direct']; obs=env.reset(0); pdim=env._n_shift_dirs+env._n_key_gens
    mod.core.SAC_LOG_STD_MAX_VECTOR=np.r_[np.full(pdim,mod.ARGS.active_log_std_max),
        np.full(env.action_dim-pdim,mod.ARGS.voltage_log_std_max)].astype(np.float32)
    actor=mod.BranchedGaussianActor(len(obs),pdim,env.action_dim-pdim).to(mod.core.DEVICE)
    actor.load_state_dict(torch.load(CHECKPOINTS['sac'],map_location=mod.core.DEVICE,weights_only=True)); actor.eval()
    with torch.no_grad(): actor.select_action(torch.as_tensor(obs,device=mod.core.DEVICE).unsqueeze(0),deterministic=True)
    return mod,envs,actor


def make_bank():
    if (DEST/'bank.json').exists(): raise RuntimeError('Preserve frozen bank')
    DEST.mkdir(parents=True,exist_ok=True)
    data=read(CANDIDATES); samples=data['samples']; split=read(CANDIDATE_SPLIT)
    mod=load_aligned(); env=mod.CleanN1Env(samples,split['train_ids'],data)
    env._reset_voltage_jacobian=lambda:np.zeros((env._n_bus,env.N+1))
    rows=[]
    for demand in (16000.,16200.,16400.):
        sid=min((i for i in split['train_ids'] if samples[i]['demand_mw']==demand),key=lambda i:samples[i]['sample_id'])
        env.reset(sid); start=time.perf_counter()
        value=solver_finite(env._configure_voltage_jacobian_net(),mod.ARGS.voltage_step_pu,mod.core.GEN_VM_MAX,mod.core.EXT_VM_MAX)
        elapsed=time.perf_counter()-start
        path=DEST/f'bank_{int(demand)}.npz'; np.savez_compressed(path,jacobian=value)
        rows.append({'demand_mw':demand,'source_sample_id':samples[sid]['sample_id'],'source_partition':'train',
                     'source_index':sid,'construction_sec':elapsed,'file':str(path),'sha256':sha(path)})
    write(DEST/'bank.json',{'complete':True,'rows':rows,'scope':'Training-reference approximation, not new-profile exact reset Jacobians'})
    for scope,seed,count in (('val',2026101801,12),('test',2026101802,20)):
        write(DEST/(scope+'_conditions.json'),{'seed':seed,'scope':scope,'samples':profiles(data,env._ac_net_template,seed,count)})
    print('Training-only bank and unseen validation/test conditions saved.',flush=True)


def run(scope):
    path=DEST/(scope+'.json')
    if path.exists(): raise RuntimeError('Preserve prior result')
    rule=None; protocol=None
    if scope=='test':
        protocol=read(DEST/'protocol.json'); rule=protocol['rule']
        assert sha(__file__)==protocol['script_sha256']
        assert sha(DEST/'test_conditions.json')==protocol['test_conditions_sha256']
        assert sha(DEST/'val.json')==protocol['validation_sha256']
        assert sha(DEST/'bank.json')==protocol['bank_sha256']
        for p,h in protocol['input_hashes'].items(): assert sha(p)==h,p
    bank_data=read(DEST/'bank.json'); bank={}
    for row in bank_data['rows']:
        assert sha(row['file'])==row['sha256']
        with np.load(row['file']) as saved: bank[row['demand_mw']]=saved['jacobian'].copy()
    data=read(CANDIDATES); samples=read(DEST/(scope+'_conditions.json'))['samples']
    mod,envs,actor=setup(samples,data,bank)
    template=copy.deepcopy(envs['direct']._ac_net_template)
    import pandapower as pp
    original_runopp=pp.runopp
    def pf_init(net,**kwargs):
        if kwargs.get('init')=='flat': kwargs['init']='pf'
        return original_runopp(net,**kwargs)
    pp.runopp=pf_init
    arms=('direct','short4','short8') if scope=='val' else ('direct','selected')
    payload={'complete':False,'scope':scope,'rows':[],'script_sha256':sha(__file__),
             'conditions_sha256':sha(DEST/(scope+'_conditions.json')),
             'timing':'Fresh profile/anchor preparation, prediction, initial checks, short-policy attempt when selected, shared PF-initialized optimizer and final checks. Resident setup/bank excluded; bank construction reported separately.'}
    try:
        repetitions=protocol['repetitions'] if protocol else 1
        for repeat,sid,sample in ((r,i,s) for r in range(repetitions) for i,s in enumerate(samples)):
            case={}
            offset=(sid+repeat)%len(arms)
            for arm in arms[offset:]+arms[:offset]:
                common.sync(); started=time.perf_counter(); env=envs[arm]
                env.current_sample=sample; env.target_demand=sample['demand_mw']
                net=copy.deepcopy(template); net.load.p_mw*=sample['p_weights']; net.load.q_mvar*=sample['q_weights']
                scale=sample['demand_mw']/net.load.p_mw.sum(); net.load.p_mw*=scale; net.load.q_mvar*=scale
                env._ac_net_template=net
                raw=env._predict_gat(sample['demand_mw'])
                reference=mod.core.project_box_sum(raw,env.PMIN,env.PMAX*.99,sample['demand_mw'])
                anchor=copy.deepcopy(net); anchor.gen.p_mw=reference
                try: pp.runpp(anchor,enforce_q_lims=True,calculate_voltage_angles=True,numba=True,max_iteration=100)
                except pp.LoadflowNotConverged:
                    row={'sample_id':sample['sample_id'],'arm':arm,'passed':False,'preparation_failed':True,
                         'optimizer_called':False,'policy_attempted':False,'repeat':repeat,
                         'runtime_sec':time.perf_counter()-started}
                    payload['rows'].append(row); case[arm]=row; continue
                model=copy.deepcopy(data['linear_security_model'])
                intercept=anchor.res_line.p_from_mw.to_numpy()-np.asarray(model['S'])@reference
                model['flow_intercept']=intercept.tolist(); env.dcnet.flow_intercept=intercept
                env._aligned_n1_cache_key=None; predictor=env._predict_gat; env._predict_gat=lambda _:raw.copy()
                preparation=time.perf_counter()-started
                try: row=evaluate(env,mod,actor,sample,sid,model,
                    0 if arm=='direct' else (rule['cap'] if arm=='selected' else int(arm[-1])),
                    rule if arm=='selected' else None)
                finally: env._predict_gat=predictor
                row.update(arm=arm,preparation_failed=False,preparation_sec=preparation,repeat=repeat)
                row['runtime_sec']+=preparation; payload['rows'].append(row); case[arm]=row
            write(path,payload)
            print(scope,'repeat',repeat+1,sid+1,'/',len(samples),{a:(r['passed'],round(r['runtime_sec'],3)) for a,r in case.items()},flush=True)
        payload['complete']=True; write(path,payload)
    finally: pp.runopp=original_runopp


def freeze():
    if (DEST/'protocol.json').exists(): raise RuntimeError('Protocol already frozen')
    data=read(DEST/'val.json'); assert data['complete']
    grouped={a:{r['sample_id']:r for r in data['rows'] if r['arm']==a} for a in ('direct','short4','short8')}
    candidates=[]
    for cap,maxdemand in ((c,d) for c in (4,8) for d in (16000,16200,16400)):
        for maxgap in (.002,.005,.01,.02,.04,1.):
            for total in (.01,.025,.05,.1,.2,10.):
                rule={'cap':cap,'max_demand_mw':maxdemand,'max_gap_pu':maxgap,'sum_gap_pu':total}
                chosen=[]; attempted=0; saved=0.
                for sid,direct in grouped['direct'].items():
                    use=not direct['preparation_failed'] and selected(direct['initial_features'],rule)
                    alternative=grouped['short'+str(cap)][sid]
                    chosen.append(alternative if use else direct)
                    attempted+=int(use and not direct.get('initial_passed',False))
                    if use and not direct.get('initial_passed',False):
                        # Identical fallback solves can differ in wall time due
                        # to compilation/noise. Do not learn a gate from that.
                        saved+=(direct['optimizer_sec'] if alternative['policy_gate_passed'] else 0.)-alternative['policy_sec']
                if not attempted: continue
                lost=[r['sample_id'] for r in chosen if not r['passed'] and grouped['direct'][r['sample_id']]['passed']]
                candidates.append({'rule':rule,'mean_sec':sum(r['runtime_sec'] for r in chosen)/len(chosen),
                    'passed':sum(r['passed'] for r in chosen),'attempted':attempted,'lost_vs_direct':lost,
                    'estimated_saved_mean_sec':saved/len(chosen)})
    valid=[r for r in candidates if not r['lost_vs_direct']]
    best=max(valid,key=lambda r:r['estimated_saved_mean_sec'])
    base=np.mean([r['runtime_sec'] for r in grouped['direct'].values()])
    write(DEST/'validation_selection.json',{'complete':True,'candidates':candidates,'best':best,'direct_mean_sec':float(base),
        'interpretation':'Select by avoided direct optimizer time minus policy-attempt time, not noisy differences between identical fallback solves. Independent deployment must remeasure the gate and all preparation.'})
    if best['estimated_saved_mean_sec']<=0:
        raise RuntimeError('No positive validation component-time saving; holdout remains unopened')
    protocol={'rule':best['rule'],'script_sha256':sha(__file__),'validation_sha256':sha(DEST/'val.json'),
        'validation_runner_sha256':data['script_sha256'],'repetitions':3,
        'test_conditions_sha256':sha(DEST/'test_conditions.json'),'input_hashes':common.hashes(),
        'bank_sha256':sha(DEST/'bank.json'),'interpretation':'New short-policy/offline-reference-feature variant. No claim of exact old-policy trajectories or zero offline feature cost.'}
    write(DEST/'protocol.json',protocol)
    print({'best':best,'direct_mean_sec':float(base)},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('bank','val','freeze','test'))
    args=parser.parse_args()
    if args.stage=='bank': make_bank()
    elif args.stage=='freeze': freeze()
    else: run(args.stage)

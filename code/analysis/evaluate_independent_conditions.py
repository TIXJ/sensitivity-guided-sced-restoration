"""Frozen 100-case spatial/reactive/cost perturbation evaluation, no span screening."""
import copy
import os
import time
from pathlib import Path
import numpy as np
import torch
from revision_common import RESULTS, CANDIDATES, CANDIDATE_SPLIT, CHECKPOINTS, load_aligned, read, write, sha
import compare_joint_workflows as common
from compare_joint_gate_workflows import training_range
from evaluate_revision import verify, cost
from constraint_diagnostics import operation_flags

DEST=RESULTS/'independent_conditions_v1'
ARMS=('sac_only','direct_pf','gated_pf')


def make_samples(data,net):
    import networkx as nx
    import pandapower.topology as top
    rng=np.random.default_rng(2026101601)
    train=[data['samples'][i] for i in read(CANDIDATE_SPLIT)['train_ids']]
    costs={k:np.mean([s[k] for s in train],axis=0) for k in ('sample_cp0','sample_cp1','sample_cp2')}
    graph=top.create_nxgraph(net,include_trafos=True)
    buses=net.load.bus.to_numpy()
    rows=[]
    for d in (16000.,16200.,16400.,16600.,16800.):
        for j in range(20):
            centers=rng.choice(np.unique(buses),3,replace=False)
            distances=[nx.single_source_shortest_path_length(graph,int(c)) for c in centers]
            regions=np.argmin([[dist.get(int(b),10000) for b in buses] for dist in distances],axis=0)
            pw=rng.uniform(.9,1.1,3)[regions]
            qw=rng.uniform(.9,1.1,3)[regions]*rng.uniform(.97,1.03,len(buses))
            sigma=float(rng.uniform(.02,.03))
            row={'sample_id':f'new_{int(d)}_{j:02d}','demand_mw':d,'cost_noise_scale':sigma,
                 'region_centers':centers.tolist(),'p_weights':pw.tolist(),'q_weights':qw.tolist()}
            row.update({k:(v*(1+rng.normal(0,sigma,len(v)))).tolist() for k,v in costs.items()})
            rows.append(row)
    return rows


def pure_policy(env,mod,actor,sample,sid):
    common.sync(); started=time.perf_counter()
    obs=env.reset(sid); initial=env.gat_power.copy(); passed,_=verify(env,mod); steps=0
    if not passed:
        for steps in range(1,env.max_steps+1):
            with torch.no_grad():
                raw=actor.select_action(torch.as_tensor(obs,dtype=torch.float32,device=mod.core.DEVICE).unsqueeze(0),deterministic=True).cpu().numpy()[0]
            lo,hi=env.get_action_bounds(); obs,_,done,truncated,_=env.step(lo+(raw+1)*.5*(hi-lo))
            if done or truncated: break
    _,low,high=common.voltage_box(env,mod)
    passed,info=common.check(env,mod,low,high)
    common.sync()
    return {'sample_id':sample['sample_id'],'passed':passed,'runtime_sec':time.perf_counter()-started,
            'policy_steps':steps,'optimizer_called':False,'scheduled_cost':cost(sample,env.current_power),
            'gat_relative_l2_mw':float(np.linalg.norm(env.current_power-initial)),
            'constraint_flags':operation_flags(info),'final_verification':info,
            'final_dispatch_mw':env.current_power.tolist(),'final_generator_vm_pu':env.current_gen_vm_pu.tolist(),
            'final_reference_vm_pu':env.current_ext_grid_vm_pu.tolist()}


def main():
    os.environ.setdefault('NUMBA_CACHE_DIR',str(RESULTS/'numba_cache'))
    DEST.mkdir(exist_ok=True)
    protocol={'seed':2026101601,'n':100,'arms':list(ARMS),'training_range':list(training_range()),
              'generation':'20 per demand; three random graph-Voronoi load regions; regional P/Q factors U[.9,1.1], additional per-load Q factor U[.97,1.03]; normalize total active demand; independent Gaussian cost perturbations of training-mean coefficients',
              'screening':'None; retain every generated case including preparation failures',
              'timing':'Include per-case profile setup, GAT-anchored affine intercept, first-use sample-specific voltage Jacobian if policy invoked, rollout, PF-initialized optimization and final checks. Resident setup excluded. No prewarming new-case Jacobians.',
              'hashes':{**common.hashes(),str(Path(__file__)):sha(__file__)}}
    path=DEST/'protocol.json'
    if path.exists(): assert read(path)==protocol
    else: write(path,protocol)
    import pandapower as pp
    from pandapower.networks import case300
    data=read(CANDIDATES)
    dataset=DEST/'conditions.json'
    if not dataset.exists(): write(dataset,{'protocol_hash':sha(path),'samples':make_samples(data,case300())})
    assert read(dataset)['protocol_hash']==sha(path)
    samples=read(dataset)['samples']; assert len(samples)==100
    output=DEST/'results.json'
    payload=read(output) if output.exists() else {'complete':False,'protocol_hash':sha(path),'conditions_hash':sha(dataset),'rows':[]}
    assert payload['protocol_hash']==sha(path) and payload['conditions_hash']==sha(dataset)
    if payload['complete']: return
    mod=load_aligned(); ids=list(range(100))
    envs={arm:mod.CleanN1Env(samples,ids,data) for arm in ARMS}
    template=copy.deepcopy(envs['sac_only']._ac_net_template)
    env=envs['sac_only']; env._voltage_jacobian_dir=RESULTS/'jacobian_cache/none'; obs=env.reset(0)
    pdim=env._n_shift_dirs+env._n_key_gens
    mod.core.SAC_LOG_STD_MAX_VECTOR=np.r_[np.full(pdim,mod.ARGS.active_log_std_max),np.full(env.action_dim-pdim,mod.ARGS.voltage_log_std_max)].astype(np.float32)
    actor=mod.BranchedGaussianActor(len(obs),pdim,env.action_dim-pdim).to(mod.core.DEVICE)
    actor.load_state_dict(torch.load(CHECKPOINTS['sac'],map_location=mod.core.DEVICE,weights_only=True)); actor.eval()
    original_runopp=pp.runopp
    def pf_init(net,**kwargs):
        if kwargs.get('init')=='flat': kwargs['init']='pf'
        return original_runopp(net,**kwargs)
    pp.runopp=pf_init
    done={(r['sample_id'],r['arm']) for r in payload['rows']}
    try:
        for sid,sample in enumerate(samples):
            for arm in ARMS[sid%3:]+ARMS[:sid%3]:
                if (sample['sample_id'],arm) in done: continue
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
                    payload['rows'].append({'sample_id':sample['sample_id'],'demand_mw':sample['demand_mw'],'arm':arm,'passed':False,'preparation_failed':True,'reason':'anchor_nonconvergence','error':str(exc),'runtime_sec':time.perf_counter()-started,'optimizer_called':False})
                    write(output,payload); continue
                model=copy.deepcopy(data['linear_security_model'])
                intercept=anchor.res_line.p_from_mw.to_numpy()-np.asarray(model['S'])@reference
                model['flow_intercept']=intercept.tolist(); env.dcnet.flow_intercept=intercept
                env._aligned_n1_cache_key=None; env._voltage_jacobian_memory={}
                # Per-arm, per-sample directories prevent first-use cache sharing.
                env._voltage_jacobian_dir=DEST/'jacobians'/arm/sample['sample_id']
                env._voltage_jacobian_dir.mkdir(parents=True,exist_ok=True)
                predictor=env._predict_gat; env._predict_gat=lambda demand:raw.copy()
                prep=time.perf_counter()-started
                try:
                    if arm=='sac_only': row=pure_policy(env,mod,actor,sample,sid)
                    else:
                        inside=protocol['training_range'][0]<=sample['demand_mw']<=protocol['training_range'][1]
                        effective='gat_sac_optimizer' if arm=='gated_pf' and inside else 'gat_optimizer'
                        row=common.evaluate(env,mod,actor,sample,sid,model,effective)
                finally:
                    env._predict_gat=predictor
                row.update(arm=arm,demand_mw=sample['demand_mw'],preparation_failed=False,preparation_sec=prep)
                row['runtime_sec']+=prep
                payload['rows'].append(row); write(output,payload)
            print(f'new condition {sid+1}/100 complete',flush=True)
    finally: pp.runopp=original_runopp
    assert len(payload['rows'])==300
    payload['complete']=True; write(output,payload)


if __name__=='__main__': main()

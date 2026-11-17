"""Joint AC/linear-LODF optimization fallback pilot on fixed validation IDs.

Uses pandapower/PYPOWER's OPF formulation callback; does not edit the library.
This is full generator-wise optimization, not a new result for the SAC policy.
"""
import argparse
import copy
import importlib
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix

from pilot_validation_feasibility import linear_excess
from revision_common import CANDIDATES,CANDIDATE_SPLIT,GAT,RESULTS,load_aligned,read,sha,write


def constraint_rows(cuts, model, lookup, ng, base_mva, demand):
    """Scale MW constraints onto PYPOWER's per-unit generator variables."""
    s=np.asarray(model['S']); c=np.asarray(model['flow_intercept'])
    lodf=np.asarray(model['lodf']); limits=np.asarray(model['effective_limit'])
    a=np.zeros((len(cuts)+1,ng)); lower=np.full(len(cuts)+1,-np.inf); upper=np.zeros(len(cuts)+1)
    a[0,lookup]=1
    lower[0]=upper[0]=demand/base_mva
    for row,(outage,line,sign) in enumerate(cuts,1):
        gradient=s[line].copy(); constant=c[line]
        if outage>=0:
            gradient+=lodf[line,outage]*s[outage]
            constant+=lodf[line,outage]*c[outage]
        scale=max(limits[line],1.)
        a[row,lookup]=sign*gradient*base_mva/scale
        upper[row]=(limits[line]-sign*constant)/scale
    return csr_matrix(a),lower,upper


def add_violated_cuts(power, model, cuts, maximum=256):
    base,post,bad_base,bad_n1=linear_excess(power,model)
    valid=np.asarray(model['valid_contingency_indices'],dtype=int)
    pending=[]
    for line in np.flatnonzero(bad_base>1e-5):
        pending.append((bad_base[line],(-1,int(line),1 if base[line]>0 else -1)))
    ii,jj=np.where(bad_n1>1e-5)
    for line,col in zip(ii,jj):
        pending.append((bad_n1[line,col],(int(valid[col]),int(line),1 if post[line,col]>0 else -1)))
    seen=set(cuts)
    fresh=[key for _,key in sorted(pending,reverse=True) if key not in seen][:maximum]
    cuts.extend(fresh)
    return max(bad_base.max(initial=0),bad_n1.max(initial=0)),len(fresh)


def run_case(sample_id, voltage_range, output, voltage_margin=0., scope='val'):
    import pandapower as pp
    from pandapower.pypower.add_userfcn import add_userfcn
    from pandapower.pypower.idx_gen import GEN_BUS
    data=read(CANDIDATES)
    samples=[s for s in data['samples'] if s.get('is_complete_sced',True)]
    split=read(CANDIDATE_SPLIT)
    sid=next(i for i,s in enumerate(samples) if s['sample_id']==sample_id)
    other='test' if scope=='val' else 'val'
    assert sid in split[scope+'_ids'] and sid not in split[other+'_ids']+split['train_ids']
    mod=load_aligned(); core=mod.core
    env=mod.CleanN1Env(samples,[sid],data)
    sample=samples[sid]; env.current_sample=sample
    demand=float(sample['demand_mw'])
    p0=core.project_box_sum(env._predict_gat(demand),env.PMIN,env.PMAX*.99,demand)
    net=copy.deepcopy(env._ac_net_template)
    scale=demand/float(net.load.p_mw.sum())
    net.load.p_mw*=scale; net.load.q_mvar*=scale
    net.gen.p_mw=p0
    net.gen.min_p_mw=env.PMIN; net.gen.max_p_mw=env.PMAX*.99
    net.gen['controllable']=True
    net.ext_grid['controllable']=True
    net.ext_grid['min_p_mw']=core.REF_SOURCE_P_MIN_MW
    net.ext_grid['max_p_mw']=core.REF_SOURCE_P_MAX_MW
    net.ext_grid['min_q_mvar']=core.REF_SOURCE_Q_MIN_MVAR
    net.ext_grid['max_q_mvar']=core.REF_SOURCE_Q_MAX_MVAR
    assert 0<=voltage_margin<0.01
    net.bus.min_vm_pu=core.RL_VMIN_PU+voltage_margin
    net.bus.max_vm_pu=core.RL_VMAX_PU-voltage_margin
    net.line['max_loading_percent']=100.; net.trafo['max_loading_percent']=100.
    controls=np.r_[np.clip(net.gen.vm_pu,core.GEN_VM_MIN,core.GEN_VM_MAX),
                   np.clip(net.ext_grid.vm_pu,core.EXT_VM_MIN,core.EXT_VM_MAX)]
    low=np.full(len(controls),core.GEN_VM_MIN); high=np.full(len(controls),core.GEN_VM_MAX)
    if voltage_range=='policy':
        low=np.maximum(low,controls-mod.ARGS.voltage_target_range_pu)
        high=np.minimum(high,controls+mod.ARGS.voltage_target_range_pu)
    buses=np.r_[net.gen.bus,net.ext_grid.bus]
    for bus,lo,hi in zip(buses,low,high):
        net.bus.at[bus,'min_vm_pu']=max(net.bus.at[bus,'min_vm_pu'],lo)
        net.bus.at[bus,'max_vm_pu']=min(net.bus.at[bus,'max_vm_pu'],hi)
    assert (net.bus.min_vm_pu<=net.bus.max_vm_pu).all()
    net.gen['min_vm_pu']=net.bus.loc[net.gen.bus,'min_vm_pu'].to_numpy()
    net.gen['max_vm_pu']=net.bus.loc[net.gen.bus,'max_vm_pu'].to_numpy()
    net.gen.vm_pu=np.clip(net.gen.vm_pu,net.gen.min_vm_pu,net.gen.max_vm_pu)
    net.ext_grid.vm_pu=np.clip(net.ext_grid.vm_pu,
        net.bus.loc[net.ext_grid.bus,'min_vm_pu'].to_numpy(),net.bus.loc[net.ext_grid.bus,'max_vm_pu'].to_numpy())
    net.poly_cost.drop(net.poly_cost.index,inplace=True)
    for i,index in enumerate(net.gen.index):
        pp.create_poly_cost(net,int(index),'gen',cp2_eur_per_mw2=1e-4,
                            cp1_eur_per_mw=-2e-4*p0[i],cp0_eur=1e-4*p0[i]**2)
    pp.create_poly_cost(net,int(net.ext_grid.index[0]),'ext_grid',cp1_eur_per_mw=0.)
    model=data['linear_security_model']; cuts=[]
    add_violated_cuts(p0,model,cuts)
    opf_module=importlib.import_module('pandapower.optimal_powerflow')
    original_opf=opf_module.opf
    callback_calls=[]
    def wrapped_opf(ppci,options):
        lookup=np.asarray(net._pd2ppc_lookups['gen'])[net.gen.index.to_numpy()].astype(int)
        expected=np.asarray(net._pd2ppc_lookups['bus'])[net.gen.bus.to_numpy()]
        assert np.array_equal(ppci['gen'][lookup,GEN_BUS],expected)
        def formulation(om,_args):
            a,lo,hi=constraint_rows(cuts,model,lookup,len(ppci['gen']),ppci['baseMVA'],demand)
            om.add_constraints('scheduled_balance_and_lodf',a,lo,hi,['Pg'])
            callback_calls.append({'constraints':len(cuts)+1,'base_mva':float(ppci['baseMVA'])})
            return om
        return original_opf(add_userfcn(ppci,'formulation',formulation),options)
    result={'sample_id':sample_id,'scope':scope,'demand_mw':demand,'voltage_range':voltage_range,
            'optimization_voltage_margin_pu':voltage_margin,'verification_limits_unchanged':True,
            'method':'joint_full_generator_AC_linear_LODF_optimization','passed':False,
            'objective':'squared redispatch from projected GAT, not economic-cost optimality',
            'action_contract':'full generator-wise P without the policy 40 MW radius',
            'input_sha256':{p.name:sha(p) for p in (CANDIDATES,CANDIDATE_SPLIT,GAT)},'rounds':[]}
    started=time.perf_counter()
    opf_module.opf=wrapped_opf
    try:
        for iteration in range(8):
            t0=time.perf_counter()
            try:
                pp.runopp(net,calculate_voltage_angles=True,numba=False,init='flat' if iteration==0 else 'results',verbose=False,
                          OPF_FLOW_LIM=2,PDIPM_MAX_IT=100,OPF_VIOLATION=1e-6,
                          PDIPM_FEASTOL=1e-7,PDIPM_GRADTOL=1e-7,PDIPM_COMPTOL=1e-7)
            except pp.OPFNotConverged as exc:
                result['rounds'].append({'iteration':iteration+1,'opf_converged':False,
                                         'error':str(exc),'runtime_sec':time.perf_counter()-t0})
                break
            power=net.res_gen.p_mw.to_numpy().copy()
            excess,new=add_violated_cuts(power,model,cuts)
            result['rounds'].append({'iteration':iteration+1,'opf_converged':True,'max_linear_excess_mw':float(excess),
                                     'new_cuts':new,'runtime_sec':time.perf_counter()-t0})
            if excess>1e-5 and new:
                net.gen.p_mw=power
                continue
            gen_vm=net.res_bus.loc[net.gen.bus,'vm_pu'].to_numpy()
            ext_vm=net.res_bus.loc[net.ext_grid.bus,'vm_pu'].to_numpy()
            control=np.r_[gen_vm,ext_vm]
            within=bool(np.all(control>=low-1e-6) and np.all(control<=high+1e-6))
            env.current_power=power; env.target_demand=demand
            env.current_gen_vm_pu=gen_vm; env.current_ext_grid_vm_pu=ext_vm
            info=dict(env._run_ac_pf_check())
            state=env._vectorized_n1_state()
            info.update(n1_violations=int(state['violation_count']),n1_overload_mw=float(state['total_overload_mw']))
            result.update(passed=bool(within and core.adopted_operation_pass(info)),final_verification=info,
                          controls_within_bounds=within,final_dispatch_mw=power.tolist(),
                          final_generator_vm_pu=gen_vm.tolist(),final_reference_vm_pu=ext_vm.tolist(),
                          gat_relative_l2_mw=float(np.linalg.norm(power-p0)),
                          scheduled_balance_error_mw=float(abs(power.sum()-demand)))
            break
    finally:
        opf_module.opf=original_opf
    result.update(runtime_sec=time.perf_counter()-started,callback_calls=callback_calls,complete=True)
    write(output,result)
    print(sample_id,voltage_range,'PASS' if result['passed'] else 'FAIL',result['rounds'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-id')
    parser.add_argument('--voltage-range',choices=('policy','hardware'),default='policy')
    parser.add_argument('--suite',action='store_true')
    parser.add_argument('--voltage-margin',type=float,default=0.)
    parser.add_argument('--scope',choices=('val','test'),default='val')
    parser.add_argument('--output')
    args=parser.parse_args()
    os.environ.setdefault('NUMBA_CACHE_DIR',str(RESULTS/'numba_cache'))
    root=RESULTS/('optimization_diagnostics/joint_ac_lodf' + (f'_margin_{args.voltage_margin:g}' if args.voltage_margin else ''))
    root.mkdir(parents=True,exist_ok=True)
    if args.suite:
        plan=read(RESULTS/'optimization_diagnostics/validation_feasibility_pilot.json')['sample_ids']
        for sid in plan:
            for mode in ('policy','hardware'):
                output=root/f'{sid}_{mode}.json'
                if output.exists():
                    print('Preserved',output,flush=True)
                    continue
                cmd=[sys.executable,'-X','utf8',__file__,'--sample-id',sid,'--voltage-range',mode,
                     '--voltage-margin',str(args.voltage_margin),'--output',str(output)]
                with (root/f'{sid}_{mode}.log').open('a',encoding='utf-8') as log:
                    try:
                        process=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=90)
                        if process.returncode:
                            raise RuntimeError(f'Worker failed; inspect {log.name}')
                    except subprocess.TimeoutExpired:
                        write(output,{'sample_id':sid,'voltage_range':mode,'passed':False,'complete':True,
                                      'status':'worker_time_limit_90s'})
                row=read(output)
                print(sid,mode,'PASS' if row['passed'] else 'FAIL',row.get('runtime_sec',90),flush=True)
        return
    if args.sample_id is None or args.output is None:
        parser.error('Provide --suite or --sample-id with --output')
    if Path(args.output).exists():
        parser.error('Output exists; preserve previous results')
    run_case(args.sample_id,args.voltage_range,args.output,args.voltage_margin,args.scope)


if __name__=='__main__':
    main()

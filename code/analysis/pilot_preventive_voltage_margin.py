"""Validation-only preventive voltage candidates with exhaustive line-outage AC checks.

One voltage schedule is shared by all outages. No per-outage corrective action,
limit relaxation, skipped failed outage, or automatic replacement of paper data.
"""
import argparse
import copy
import os
from pathlib import Path
import time

import numpy as np

from evaluate_revision import verify
from revision_common import CANDIDATES,CANDIDATE_SPLIT,RESULTS,load_aligned,read,sha,write


def outage_checks(net, valid, limits, core):
    import pandapower as pp
    rows=[]
    for outage in valid:
        solved=None; errors=[]
        for initialization in ('results','flat'):
            trial=copy.deepcopy(net)
            trial.line.loc[trial.line.index[outage],'in_service']=False
            try:
                pp.runpp(trial,enforce_q_lims=True,calculate_voltage_angles=True,
                         numba=True,max_iteration=100,init=initialization)
                solved=trial; break
            except pp.LoadflowNotConverged as exc:
                errors.append(str(exc))
        row={'outage_position':int(outage),'passed':False,'converged':solved is not None}
        if solved is None:
            row['errors']=errors; rows.append(row); continue
        vm=solved.res_bus.vm_pu.to_numpy()
        q=solved.res_gen.q_mvar.to_numpy()
        p_ref,q_ref=solved.res_ext_grid[['p_mw','q_mvar']].sum().to_numpy()
        mask=np.arange(len(limits))!=outage
        flow=solved.res_line.p_from_mw.to_numpy()[mask]
        loading=np.r_[solved.res_line.loading_percent.to_numpy()[mask],solved.res_trafo.loading_percent.to_numpy()]
        finite=bool(np.isfinite(np.r_[vm,q,p_ref,q_ref,flow,loading]).all())
        voltage=int(((vm<core.RL_VMIN_PU)|(vm>core.RL_VMAX_PU)).sum())
        q_viol=int(((q<solved.gen.min_q_mvar.to_numpy()-1e-5)|(q>solved.gen.max_q_mvar.to_numpy()+1e-5)).sum())
        thermal=int((loading>100).sum())
        ref_bad=not(core.REF_SOURCE_P_MIN_MW<=p_ref<=core.REF_SOURCE_P_MAX_MW
                    and core.REF_SOURCE_Q_MIN_MVAR<=q_ref<=core.REF_SOURCE_Q_MAX_MVAR
                    and np.hypot(p_ref,q_ref)<=core.REF_SOURCE_S_MAX_MVA)
        overload=float(np.maximum(abs(flow)-limits[mask],0).sum())
        row.update(passed=bool(finite and voltage==0 and q_viol==0 and thermal==0 and not ref_bad and overload<.01),
                   finite=finite,voltage_violations=voltage,generator_q_violations=q_viol,
                   thermal_violations=thermal,reference_violation=ref_bad,active_overload_mw=overload,
                   min_bus_vm_pu=float(vm.min()),max_bus_vm_pu=float(vm.max()))
        rows.append(row)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demand',type=float,default=16000.)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if Path(args.output).exists():
        parser.error('Preserve the existing pilot; use a new output path')
    os.environ.setdefault('NUMBA_CACHE_DIR',str(RESULTS/'numba_cache'))
    import pandapower as pp
    reference_path=RESULTS/'optimization_diagnostics/validation_full_reference.json'
    reference=read(reference_path)
    assert reference['complete'] and reference['settings']['scope']=='val'
    point=sorted((r for r in reference['rows'] if r['passed'] and r['demand_mw']==args.demand),
                 key=lambda r:r['sample_id'])[0]
    data=read(CANDIDATES); samples=[s for s in data['samples'] if s.get('is_complete_sced',True)]
    sid=point['candidate_index']; split=read(CANDIDATE_SPLIT)
    assert sid in split['val_ids'] and sid not in split['test_ids']+split['train_ids']
    mod=load_aligned(); core=mod.core
    env=mod.CleanN1Env(samples,[sid],data)
    env.current_sample=samples[sid]
    env.current_power=np.asarray(point['final_dispatch_mw'])
    env.target_demand=float(point['demand_mw'])
    template=copy.deepcopy(env._ac_net_template)
    nominal=np.r_[np.clip(template.gen.vm_pu,core.GEN_VM_MIN,core.GEN_VM_MAX),
                  np.clip(template.ext_grid.vm_pu,core.EXT_VM_MIN,core.EXT_VM_MAX)]
    lower=np.maximum(nominal-mod.ARGS.voltage_target_range_pu,core.GEN_VM_MIN)
    upper=np.minimum(nominal+mod.ARGS.voltage_target_range_pu,core.GEN_VM_MAX)
    original=np.r_[point['final_generator_vm_pu'],point['final_reference_vm_pu']]
    valid=np.asarray(data['linear_security_model']['valid_contingency_indices'],dtype=int)
    limits=np.asarray(data['linear_security_model']['effective_limit'])
    candidates=[('unchanged',np.zeros(len(original)))]
    for step in (.001,.002,.004):
        candidates.append((f'all_plus_{step:g}',np.full(len(original),step)))
    for step in (.002,.004,.008):
        candidates.append((f'low_setpoints_plus_{step:g}',step*(original<1.0)))
    payload={'complete':False,'scope':'validation','sample_id':point['sample_id'],'demand_mw':args.demand,
             'selection':'first accepted validation sample ID at the requested demand',
             'reference_sha256':sha(reference_path),'valid_outages':valid.tolist(),
             'candidate_sha256':sha(CANDIDATES),'split_sha256':sha(CANDIDATE_SPLIT),'source_sha256':sha(__file__),
             'rule':'Shared preventive voltage schedule; original cumulative and device bounds. '
                    'All 253 valid outages checked for each base-case-accepted candidate.',
             'candidates':[]}
    started=time.perf_counter()
    for name,delta in candidates:
        control=np.clip(original+delta,lower,upper)
        env.current_gen_vm_pu=control[:env.N]; env.current_ext_grid_vm_pu=control[env.N:]
        base_pass,base_info=verify(env,mod)
        row={'name':name,'base_passed':base_pass,'base_verification':base_info,'control':control.tolist(),
             'post_contingency_passed':None,'all_valid_line_ac_checks_passed':False}
        if base_pass:
            net=copy.deepcopy(template)
            factor=args.demand/float(net.load.p_mw.sum())
            net.load.p_mw*=factor; net.load.q_mvar*=factor
            net.gen.p_mw=env.current_power; net.gen.vm_pu=control[:env.N]; net.ext_grid.vm_pu=control[env.N:]
            pp.runpp(net,enforce_q_lims=True,calculate_voltage_angles=True,numba=True,max_iteration=100)
            records=outage_checks(net,valid,limits,core)
            row['outages']=records
            row['post_contingency_passed']=sum(r['passed'] for r in records)
            row['all_valid_line_ac_checks_passed']=row['post_contingency_passed']==len(valid)
        payload['candidates'].append(row)
        write(args.output,payload)
        print(point['sample_id'],name,'base=',base_pass,'outages=',row['post_contingency_passed'], '/',len(valid),flush=True)
    payload['complete']=True; payload['runtime_sec']=time.perf_counter()-started
    feasible=[r for r in payload['candidates'] if r['base_passed']]
    best=max(feasible,key=lambda r:r['post_contingency_passed']) if feasible else None
    payload['best_candidate']=None if best is None else best['name']
    payload['best_all_valid_line_ac_checks_passed']=False if best is None else best['all_valid_line_ac_checks_passed']
    write(args.output,payload)


if __name__=='__main__':
    main()

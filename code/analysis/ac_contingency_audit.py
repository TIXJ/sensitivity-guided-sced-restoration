"""Explicit post-outage AC checks and hybrid sensitivity/LODF error audit."""
import argparse
import copy
import time
import numpy as np
import pandapower as pp
import pandapower.topology as topology
import networkx as nx
from pandapower.networks import case300
from revision_common import *

def exclusions(net, valid):
    from pandapower.converter.pypower.to_ppc import to_ppc
    from pandapower.pypower.makePTDF import makePTDF
    from pandapower.pypower.idx_brch import F_BUS, T_BUS
    scratch = copy.deepcopy(net)
    ppc = to_ppc(scratch, calculate_voltage_angles=True, check_connectivity=True, init='flat', mode='pf')
    ptdf = makePTDF(ppc['baseMVA'], ppc['bus'], ppc['branch'])
    b = ppc['branch']
    transfer = ptdf[:, b[:,F_BUS].astype(int)] - ptdf[:, b[:,T_BUS].astype(int)]
    denom = 1-np.diag(transfer)
    start, end = scratch._pd2ppc_lookups['branch']['line']
    result = []
    for k in range(len(net.line)):
        if k in valid: continue
        trial = copy.deepcopy(net)
        trial.line.loc[trial.line.index[k], 'in_service'] = False
        graph = topology.create_nxgraph(trial, include_trafos=True)
        components = list(nx.connected_components(graph))
        sources = set(trial.ext_grid.loc[trial.ext_grid.in_service, 'bus'])
        islands = [c for c in components if not c.intersection(sources)]
        island_load = float(trial.load.loc[trial.load.bus.isin(set().union(*islands) if islands else set()), 'p_mw'].sum())
        result.append({'line_position':k,'line_index':int(net.line.index[k]),'from_bus':int(net.line.iloc[k].from_bus),'to_bus':int(net.line.iloc[k].to_bus),'denominator':float(denom[start+k]),'near_singular':bool(abs(denom[start+k])<=1e-8),'islanding':bool(islands),'unsupplied_base_load_mw':island_load})
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default=str(ROOT/'paper_results/case300_fixed_slack_v1_full_test_audit.json'))
    p.add_argument('--top-k',type=int,default=3)
    p.add_argument('--full-per-demand',type=int,default=1)
    p.add_argument('--max-samples',type=int,default=0)
    p.add_argument('--output',default=str(RESULTS/'ac_contingencies.json'))
    a=p.parse_args()
    if Path(a.output).exists(): p.error('Output exists; choose a new path')
    data=read(CANDIDATES)
    samples={s['sample_id']:s for s in data['samples']}
    payload=read(a.input)
    points=payload['rows']
    points=[r for r in points if r.get('passed',False)]
    if a.max_samples: points=points[:a.max_samples]
    model=data['linear_security_model']
    valid=np.asarray(model['valid_contingency_indices'],dtype=int)
    lodf=np.asarray(model['lodf'])
    limits=np.asarray(model['effective_limit'])
    base=case300()
    excluded=exclusions(base,set(valid))
    write(RESULTS/'excluded_contingencies.json',{'network':'pandapower case300 as used by the original environment','valid_indices':valid.tolist(),'excluded_outages':excluded})
    chosen={}
    # Stratification and ordering are fixed before looking at AC audit outcomes.
    for row in sorted(points,key=lambda r:r['sample_id']):
        d=float(samples[row['sample_id']]['demand_mw'])
        chosen.setdefault(d,[])
        if len(chosen[d])<a.full_per_demand: chosen[d].append(row['sample_id'])
    exhaustive={s for v in chosen.values() for s in v}
    records=[]
    for point in points:
        sid=point['sample_id']; sample=samples[sid]
        net=copy.deepcopy(base)
        demand=float(point.get('demand_mw',sample['demand_mw']))
        scale=demand/float(net.load.p_mw.sum())
        net.load.p_mw*=scale; net.load.q_mvar*=scale
        power=np.asarray(point['final_dispatch_mw'])
        net.gen.p_mw=power
        net.gen.vm_pu=point['final_generator_vm_pu']
        net.ext_grid.vm_pu=float(point['final_reference_vm_pu'])
        pp.runpp(net,enforce_q_lims=True,calculate_voltage_angles=True,numba=False,max_iteration=50)
        actual_base=net.res_line.p_from_mw.to_numpy().copy()
        linear=np.asarray(model['flow_intercept'])+np.asarray(model['S'])@power
        predicted=linear[:,None]+lodf[:,valid]*linear[valid][None,:]
        predicted[valid,np.arange(len(valid))]=0
        score=np.max(abs(predicted)/np.maximum(limits[:,None],1e-9),axis=0)
        outages=valid if sid in exhaustive else valid[np.argsort(-score)[:a.top_k]]
        for k in outages:
            trial=copy.deepcopy(net)
            trial.line.loc[trial.line.index[k],'in_service']=False
            started=time.perf_counter()
            row={'sample_id':sid,'demand_mw':demand,'outage_position':int(k),'exhaustive_case':sid in exhaustive,'converged':False,'passed':False}
            try:
                errors=[]
                for initialization in ('results','flat'):
                    trial=copy.deepcopy(net)
                    trial.line.loc[trial.line.index[k],'in_service']=False
                    try:
                        pp.runpp(trial,enforce_q_lims=True,calculate_voltage_angles=True,numba=False,init=initialization,max_iteration=100)
                        row['solver_initialization']=initialization
                        break
                    except pp.LoadflowNotConverged as exc:
                        errors.append(str(exc))
                else:
                    raise pp.LoadflowNotConverged('; '.join(errors))
                row['solver_failed_attempts']=errors
                vm=trial.res_bus.vm_pu.to_numpy()
                finite=bool(np.isfinite(vm).all())
                ref=trial.res_ext_grid[['p_mw','q_mvar']].sum().to_numpy()
                mask=np.arange(len(limits))!=k
                flow=trial.res_line.p_from_mw.to_numpy()
                hybrid=linear+lodf[:,k]*linear[k]
                ac_anchor=actual_base+lodf[:,k]*actual_base[k]
                vbad=int(np.count_nonzero((vm<.933)|(vm>1.09)|~np.isfinite(vm)))
                thermal=int((trial.res_line.loading_percent>100).sum()+(trial.res_trafo.loading_percent>100).sum())
                q=trial.res_gen.q_mvar.to_numpy()
                qbad=int(np.count_nonzero((q<trial.gen.min_q_mvar.to_numpy()-1e-5)|(q>trial.gen.max_q_mvar.to_numpy()+1e-5)))
                refbad=bool(abs(ref[0])>2500 or abs(ref[1])>250 or np.linalg.norm(ref)>2500)
                row.update(converged=bool(trial.converged and finite),voltage_violations=vbad,thermal_violations=thermal,generator_q_violations=qbad,reference_violation=refbad,hybrid_mae_mw=float(np.mean(abs(hybrid[mask]-flow[mask]))),hybrid_max_error_mw=float(np.max(abs(hybrid[mask]-flow[mask]))),ac_anchored_lodf_mae_mw=float(np.mean(abs(ac_anchor[mask]-flow[mask]))),base_model_mae_mw=float(np.mean(abs(linear-actual_base))),ac_active_overload_mw=float(np.maximum(abs(flow[mask])-limits[mask],0).sum()),passed=bool(trial.converged and finite and not(vbad or thermal or qbad or refbad)))
                row['passed'] = row['passed'] and row['ac_active_overload_mw'] < .01
            except Exception as exc:
                row['error']=str(exc)
            row['runtime_sec']=time.perf_counter()-started
            records.append(row)
        write(a.output,{'complete':False,'settings':vars(a),'input_sha256':sha(a.input),'excluded_outages':excluded,'exhaustive_sample_ids':sorted(exhaustive),'rows':records})
        print(f"{sid}: {len(outages)} explicit AC contingencies",flush=True)
    write(a.output,{'complete':True,'settings':vars(a),'input_sha256':sha(a.input),'excluded_outages':excluded,'exhaustive_sample_ids':sorted(exhaustive),'rows':records})

if __name__=='__main__': main()

"""Re-anchor all accepted original SAC points without changing their controls."""
import copy
import os
from pathlib import Path
import numpy as np
from revision_common import RESULTS, CANDIDATES, read, write, sha


def metrics(flows,lodf,valid,limits):
    post=flows[:,None]+lodf[:,valid]*flows[valid][None,:]
    post[valid,np.arange(len(valid))]=0
    excess=np.maximum(abs(post)-limits[:,None],0)
    base=np.maximum(abs(flows)-limits,0)
    return {'passed':bool(np.isfinite(post).all() and not (excess>.01).any() and not (base>.01).any()),
            'max_post_excess_mw':float(excess.max()),'cumulative_post_excess_mw':float(excess.sum()),
            'violating_pairs':int((excess>.01).sum()),'max_base_excess_mw':float(base.max())}


def main():
    os.environ.setdefault('NUMBA_CACHE_DIR',str(RESULTS/'numba_cache'))
    import pandapower as pp
    from pandapower.networks import case300
    source=RESULTS/'full_sac.json'; output=RESULTS/'final_ac_anchor_v1.json'
    provenance={'policy_results':sha(source),'candidates':sha(CANDIDATES),'script':sha(__file__)}
    payload=read(output) if output.exists() else {'complete':False,'hashes':provenance,'rows':[]}
    assert payload['hashes']==provenance
    if payload['complete']: return
    data=read(CANDIDATES); samples={s['sample_id']:s for s in data['samples']}
    m=data['linear_security_model']; lodf=np.asarray(m['lodf']); valid=np.asarray(m['valid_contingency_indices'],int); limits=np.asarray(m['effective_limit'])
    base=case300(); points=[r for r in read(source)['rows'] if r['passed']]
    assert len(points)==44
    for point in points[len(payload['rows']):]:
        sid=point['sample_id']; sample=samples[sid]; net=copy.deepcopy(base)
        scale=sample['demand_mw']/net.load.p_mw.sum(); net.load.p_mw*=scale; net.load.q_mvar*=scale
        power=np.asarray(point['final_dispatch_mw']); net.gen.p_mw=power
        net.gen.vm_pu=point['final_generator_vm_pu']; net.ext_grid.vm_pu=point['final_reference_vm_pu']
        hybrid=np.asarray(m['flow_intercept'])+np.asarray(m['S'])@power
        row={'sample_id':sid,'original_linear':metrics(hybrid,lodf,valid,limits),'ac_anchor':None,'base_ac_converged':False}
        errors=[]
        for init in ('flat','dc'):
            try:
                pp.runpp(net,enforce_q_lims=True,calculate_voltage_angles=True,numba=True,max_iteration=100,init=init)
                row['base_ac_converged']=bool(net.converged)
                row['ac_anchor']=metrics(net.res_line.p_from_mw.to_numpy(),lodf,valid,limits)
                row['base_voltage_extrema']=[float(net.res_bus.vm_pu.min()),float(net.res_bus.vm_pu.max())]
                break
            except pp.LoadflowNotConverged as exc:
                errors.append(str(exc))
        row['errors']=errors
        payload['rows'].append(row); write(output,payload)
        print(sid,row['original_linear']['passed'],row['ac_anchor'],flush=True)
    assert all(r['original_linear']['passed'] for r in payload['rows']), 'Original linear contract did not reproduce'
    payload['complete']=True
    write(output,payload)


if __name__=='__main__': main()

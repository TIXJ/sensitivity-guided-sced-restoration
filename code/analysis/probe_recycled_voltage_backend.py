"""Test pandapower's public recycle path against saved validation matrices."""
import copy
import time
import numpy as np
from revision_common import RESULTS,load_aligned,write,sha
from diagnose_runtime_and_tight_validation import select


def recycled_jacobian(net,epsilon,gen_max,ext_max):
    import pandapower as pp
    nominal=copy.deepcopy(net)
    pp.runpp(nominal,calculate_voltage_angles=True,enforce_q_lims=True,numba=True)
    vm0=nominal.res_bus.vm_pu.to_numpy().copy()
    result=np.zeros((len(nominal.bus),len(nominal.gen)+1))
    for index in range(result.shape[1]):
        trial=copy.deepcopy(nominal)
        if index<len(trial.gen):
            trial.gen.vm_pu.values[index]=min(float(trial.gen.vm_pu.values[index]+epsilon),gen_max)
        else:
            trial.ext_grid.vm_pu.values[0]=min(float(trial.ext_grid.vm_pu.values[0]+epsilon),ext_max)
        assert trial._options['enforce_q_lims'] is True
        pp.runpp(trial,recycle={'bus_pq':False,'trafo':False,'gen':True})
        if trial.converged: result[:,index]=(trial.res_bus.vm_pu.to_numpy()-vm0)/epsilon
    return result


def main():
    out=RESULTS/'fast_voltage_validation_20261018/recycle_probe.json'
    if out.exists(): raise RuntimeError('Preserve previous probe')
    data,samples,ids=select(); mod=load_aligned()
    env=mod.CleanN1Env(samples,ids,data)
    env._reset_voltage_jacobian=lambda:np.zeros((env._n_bus,env.N+1))
    rows=[]
    for sid in ids:
        env.reset(sid)
        net=env._configure_voltage_jacobian_net()
        started=time.perf_counter()
        candidate=recycled_jacobian(net,mod.ARGS.voltage_step_pu,mod.core.GEN_VM_MAX,mod.core.EXT_VM_MAX)
        elapsed=time.perf_counter()-started
        path=next((out.parent/'numba_v1/cache/0'/samples[sid]['sample_id']/'legacy').glob('*.npz'))
        with np.load(path) as saved: reference=saved['jacobian']
        row={'sample_id':samples[sid]['sample_id'],'runtime_sec':elapsed,
            'max_abs_error':float(np.max(abs(reference-candidate))),
            'within_existing_tolerance':bool(np.allclose(reference,candidate,atol=1e-8,rtol=1e-7))}
        rows.append(row); print(row,flush=True)
    write(out,{'complete':True,'script_sha256':sha(__file__),'rows':rows,
        'scope':'Public recycle retains reactive-limit enforcement but changes initialization. Numerical equivalence must pass before any deployment.'})


if __name__=='__main__': main()

"""Validation-only implicit AC voltage derivative from Pandapower matrices."""
import copy
import time
import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import splu
from revision_common import RESULTS,load_aligned,write,sha
from diagnose_runtime_and_tight_validation import select


def implicit_voltage(net):
    import pandapower as pp
    from pandapower.pypower.dSbus_dV import dSbus_dV
    from pandapower.pypower.idx_bus import BUS_TYPE,PV,PQ,REF
    nominal=copy.deepcopy(net)
    pp.runpp(nominal,calculate_voltage_angles=True,enforce_q_lims=True,numba=True)
    if not nominal.converged: raise RuntimeError('Nominal AC-PF did not converge')
    internal=nominal['_ppc']['internal']
    bus=internal['bus']; voltage=internal['V']
    # The public result bus labels can be restored after Q-limit enforcement.
    # Newton's final index sets retain the actual constrained active set.
    pq=np.asarray(internal['pq'],dtype=int); pv=np.asarray(internal['pv'],dtype=int)
    ref=np.asarray(internal['ref'],dtype=int)
    pvpq=np.r_[pv,pq]
    dvm,dva=dSbus_dV(internal['Ybus'],voltage)
    jac=bmat([[dva[pvpq,:][:,pvpq].real,dvm[pvpq,:][:,pq].real],
              [dva[pq,:][:,pvpq].imag,dvm[pq,:][:,pq].imag]],format='csc')
    controls=np.r_[nominal.gen.bus.to_numpy(),nominal.ext_grid.bus.to_numpy()]
    lookup=nominal._pd2ppc_lookups['bus']
    buses=lookup[controls]
    if len(bus)!=len(nominal['_ppc']['bus']):
        raise ValueError('Unsupported out-of-service bus mapping')
    live=np.array([i for i,b in enumerate(buses) if b in np.r_[pv,ref]],dtype=int)
    result=np.zeros((len(bus),len(buses)))
    if live.size:
        cols=buses[live]
        rhs=-np.vstack([dvm[pvpq,:][:,cols].real.toarray(),dvm[pq,:][:,cols].imag.toarray()])
        solved=splu(jac).solve(rhs)
        result[np.ix_(pq,live)]=solved[len(pvpq):,:]
        result[buses[live],live]=1.
    return result[lookup[nominal.bus.index.to_numpy()],:]


def main():
    out=RESULTS/'voltage_kernel_v5_qfixed_validation/probe.json'
    if out.exists(): raise RuntimeError('Preserve prior probe')
    data,samples,ids=select(); mod=load_aligned()
    env=mod.CleanN1Env(samples,ids,data)
    env._reset_voltage_jacobian=lambda:np.zeros((env._n_bus,env.N+1))
    rows=[]
    for sid in ids:
        env.reset(sid); start=time.perf_counter()
        value=implicit_voltage(env._configure_voltage_jacobian_net())
        elapsed=time.perf_counter()-start
        path=next((RESULTS/'fast_voltage_validation_20261018/numba_v1/cache/0'/samples[sid]['sample_id']/'legacy').glob('*.npz'))
        with np.load(path) as saved: baseline=saved['jacobian']
        row={'sample_id':samples[sid]['sample_id'],'runtime_sec':elapsed,
             'max_abs_error':float(np.max(abs(value-baseline))),
             'relative_frobenius_error':float(np.linalg.norm(value-baseline)/np.linalg.norm(baseline)),
             'finite':bool(np.isfinite(value).all())}
        rows.append(row); print(row,flush=True)
        write(out,{'complete':False,'script_sha256':sha(__file__),'rows':rows})
    write(out,{'complete':True,'script_sha256':sha(__file__),'rows':rows,
        'scope':'Analytic local derivative is a new approximation to the trained one-sided finite-difference feature; never claim bitwise equivalence.'})


if __name__=='__main__': main()

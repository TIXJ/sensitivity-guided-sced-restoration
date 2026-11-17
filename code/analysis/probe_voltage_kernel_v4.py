"""Validation-only direct use of Pandapower's Q-limited Newton solver.

No power-flow equations are reimplemented. Avoid repeatedly extracting branch
and equipment result tables when only bus-voltage finite differences are used.
"""
import copy
import time
import numpy as np
from revision_common import RESULTS, load_aligned, write, sha
from diagnose_runtime_and_tight_validation import select


def solver_finite(net, epsilon, gen_max, ext_max):
    import pandapower as pp
    from pandapower.pd2ppc import _pd2ppc
    from pandapower.pf.run_newton_raphson_pf import _run_newton_raphson_pf
    from pandapower.pypower.idx_bus import VM
    from pandapower.pypower.idx_gen import VG
    nominal = copy.deepcopy(net)
    pp.runpp(nominal, calculate_voltage_angles=True, enforce_q_lims=True, numba=True)
    vm0 = nominal.res_bus.vm_pu.to_numpy().copy()
    admittance = {k: nominal['_ppc']['internal'][k].copy() for k in ('Ybus','Yf','Yt')}
    nominal._options.update(init_vm_pu='results', init_va_degree='results', init_results=True,
                            recycle={'bus_pq':False,'trafo':False,'gen':True})
    ppc, base = _pd2ppc(nominal)
    if len(ppc['bus']) != len(base['bus']):
        raise ValueError('Kernel probe requires all buses in service')
    lookup = nominal._pd2ppc_lookups['bus'][nominal.bus.index.to_numpy()]
    if np.any(lookup < 0) or np.any(lookup >= len(base['bus'])):
        raise ValueError('Unsupported bus mapping')
    base['internal'].update(admittance)
    positions = np.r_[nominal._pd2ppc_lookups['gen'][nominal.gen.index.to_numpy()],
                      nominal._pd2ppc_lookups['ext_grid'][nominal.ext_grid.index.to_numpy()]]
    assert len(positions)==len(nominal.gen)+1
    options = copy.deepcopy(nominal._options)
    assert options['enforce_q_lims'] is True and not options['only_v_results']
    result = np.zeros((len(net.bus),len(positions)))
    for i, pos in enumerate(positions):
        trial = copy.deepcopy(base)
        trial['gen'][pos, VG] = min(trial['gen'][pos, VG]+epsilon,
                                   gen_max if i < len(net.gen) else ext_max)
        solved = _run_newton_raphson_pf(trial, options)
        if solved['success'] != 1:
            raise RuntimeError('Perturbed Newton power flow did not converge')
        result[:,i] = (solved['bus'][lookup,VM]-vm0)/epsilon
    return result


def main():
    out = RESULTS/'voltage_kernel_v4_validation/probe.json'
    if out.exists(): raise RuntimeError('Preserve completed validation probe')
    data,samples,ids = select()
    mod = load_aligned()
    env = mod.CleanN1Env(samples,ids,data)
    env._reset_voltage_jacobian = lambda: np.zeros((env._n_bus,env.N+1))
    rows=[]
    for sid in ids:
        env.reset(sid)
        start=time.perf_counter()
        value=solver_finite(env._configure_voltage_jacobian_net(),mod.ARGS.voltage_step_pu,mod.core.GEN_VM_MAX,mod.core.EXT_VM_MAX)
        elapsed=time.perf_counter()-start
        path=next((RESULTS/'fast_voltage_validation_20261018/numba_v1/cache/0'/samples[sid]['sample_id']/'legacy').glob('*.npz'))
        with np.load(path) as saved: baseline=saved['jacobian']
        row={'sample_id':samples[sid]['sample_id'],'runtime_sec':elapsed,
             'max_abs_error':float(np.max(abs(value-baseline))),
             'allclose':bool(np.allclose(value,baseline,atol=1e-8,rtol=1e-7))}
        rows.append(row); print(row,flush=True)
        write(out,{'complete':False,'script_sha256':sha(__file__),'rows':rows})
    write(out,{'complete':True,'script_sha256':sha(__file__),'rows':rows})


if __name__=='__main__': main()

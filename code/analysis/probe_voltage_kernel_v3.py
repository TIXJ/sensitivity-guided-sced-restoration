"""Validation-only probe of network-copy overhead in finite differences."""
import copy
import time
import numpy as np
from revision_common import RESULTS, load_aligned, write, sha
from diagnose_runtime_and_tight_validation import select


def state_recycled(net, epsilon, gen_max, ext_max):
    import pandapower as pp
    nominal = copy.deepcopy(net)
    pp.runpp(nominal, calculate_voltage_angles=True, enforce_q_lims=True, numba=True)
    vm0 = nominal.res_bus.vm_pu.to_numpy().copy()
    gen0 = nominal.gen.vm_pu.to_numpy().copy()
    ext0 = nominal.ext_grid.vm_pu.to_numpy().copy()
    trial = copy.deepcopy(nominal)
    result = np.zeros((len(net.bus), len(net.gen)+1))
    for i in range(result.shape[1]):
        # Restore the complete solver state, including Q-limit bus switching.
        trial['_ppc'] = copy.deepcopy(nominal['_ppc'])
        trial.gen.loc[:, 'vm_pu'] = gen0
        trial.ext_grid.loc[:, 'vm_pu'] = ext0
        if i < len(gen0):
            trial.gen.loc[trial.gen.index[i], 'vm_pu'] = min(gen0[i]+epsilon, gen_max)
        else:
            trial.ext_grid.loc[trial.ext_grid.index[0], 'vm_pu'] = min(ext0[0]+epsilon, ext_max)
        assert trial._options['enforce_q_lims'] is True
        pp.runpp(trial, recycle={'bus_pq':False, 'trafo':False, 'gen':True})
        if not trial.converged:
            raise RuntimeError('Perturbed power flow did not converge')
        result[:, i] = (trial.res_bus.vm_pu.to_numpy()-vm0)/epsilon
    return result


def main():
    out = RESULTS/'voltage_kernel_v3_validation/probe.json'
    if out.exists(): raise RuntimeError('Preserve completed validation probe')
    data, samples, ids = select()
    mod = load_aligned()
    env = mod.CleanN1Env(samples, ids, data)
    env._reset_voltage_jacobian = lambda: np.zeros((env._n_bus, env.N+1))
    rows = []
    for sid in ids:
        env.reset(sid)
        net = env._configure_voltage_jacobian_net()
        start = time.perf_counter()
        value = state_recycled(net, mod.ARGS.voltage_step_pu, mod.core.GEN_VM_MAX, mod.core.EXT_VM_MAX)
        elapsed = time.perf_counter()-start
        path = next((RESULTS/'fast_voltage_validation_20261018/numba_v1/cache/0'/samples[sid]['sample_id']/'legacy').glob('*.npz'))
        with np.load(path) as saved: baseline = saved['jacobian']
        row = {'sample_id':samples[sid]['sample_id'], 'runtime_sec':elapsed,
               'max_abs_error':float(np.max(abs(value-baseline))),
               'allclose':bool(np.allclose(value, baseline, atol=1e-8, rtol=1e-7))}
        rows.append(row)
        print(row, flush=True)
        write(out, {'complete':False, 'script_sha256':sha(__file__), 'rows':rows})
    write(out, {'complete':True, 'script_sha256':sha(__file__), 'rows':rows})


if __name__ == '__main__': main()

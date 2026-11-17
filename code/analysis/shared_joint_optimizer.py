"""Shared solver for matched no-SAC/SAC-first workflows; original pilot is frozen."""
import copy
import importlib
import time
import numpy as np

from pilot_joint_ac_lodf import add_violated_cuts, constraint_rows


def voltage_box(env, mod):
    core = mod.core
    net = env._ac_net_template
    nominal = np.r_[np.clip(net.gen.vm_pu, core.GEN_VM_MIN, core.GEN_VM_MAX),
                    np.clip(net.ext_grid.vm_pu, core.EXT_VM_MIN, core.EXT_VM_MAX)]
    low = np.r_[np.full(env.N, core.GEN_VM_MIN), np.full(len(net.ext_grid), core.EXT_VM_MIN)]
    high = np.r_[np.full(env.N, core.GEN_VM_MAX), np.full(len(net.ext_grid), core.EXT_VM_MAX)]
    return nominal, np.maximum(low, nominal-mod.ARGS.voltage_target_range_pu), np.minimum(high, nominal+mod.ARGS.voltage_target_range_pu)


def solve(env, mod, model, reference_power, demand):
    import pandapower as pp
    from pandapower.pypower.add_userfcn import add_userfcn
    from pandapower.pypower.idx_gen import GEN_BUS
    core = mod.core
    started = time.perf_counter()
    net = copy.deepcopy(env._ac_net_template)
    factor = demand/net.load.p_mw.sum()
    net.load.p_mw *= factor
    net.load.q_mvar *= factor
    net.gen.p_mw = reference_power
    net.gen.min_p_mw = env.PMIN
    net.gen.max_p_mw = env.PMAX*.99
    net.gen['controllable'] = True
    net.ext_grid['controllable'] = True
    net.ext_grid['min_p_mw'] = core.REF_SOURCE_P_MIN_MW
    net.ext_grid['max_p_mw'] = core.REF_SOURCE_P_MAX_MW
    net.ext_grid['min_q_mvar'] = core.REF_SOURCE_Q_MIN_MVAR
    net.ext_grid['max_q_mvar'] = core.REF_SOURCE_Q_MAX_MVAR
    net.bus.min_vm_pu = core.RL_VMIN_PU+.0001
    net.bus.max_vm_pu = core.RL_VMAX_PU-.0001
    net.line['max_loading_percent'] = 100.
    net.trafo['max_loading_percent'] = 100.
    _, low, high = voltage_box(env, mod)
    for bus, lower, upper in zip(np.r_[net.gen.bus, net.ext_grid.bus], low, high):
        net.bus.at[bus, 'min_vm_pu'] = max(net.bus.at[bus, 'min_vm_pu'], lower)
        net.bus.at[bus, 'max_vm_pu'] = min(net.bus.at[bus, 'max_vm_pu'], upper)
    assert (net.bus.min_vm_pu <= net.bus.max_vm_pu).all()
    for devices in (net.gen, net.ext_grid):
        devices['min_vm_pu'] = net.bus.loc[devices.bus, 'min_vm_pu'].to_numpy()
        devices['max_vm_pu'] = net.bus.loc[devices.bus, 'max_vm_pu'].to_numpy()
        devices.vm_pu = np.clip(devices.vm_pu, devices.min_vm_pu, devices.max_vm_pu)
    net.poly_cost.drop(net.poly_cost.index, inplace=True)
    for i, index in enumerate(net.gen.index):
        pp.create_poly_cost(net, int(index), 'gen', cp2_eur_per_mw2=1e-4,
                            cp1_eur_per_mw=-2e-4*reference_power[i], cp0_eur=1e-4*reference_power[i]**2)
    pp.create_poly_cost(net, int(net.ext_grid.index[0]), 'ext_grid', cp1_eur_per_mw=0.)
    cuts = []
    add_violated_cuts(reference_power, model, cuts)
    opf_module = importlib.import_module('pandapower.optimal_powerflow')
    original = opf_module.opf
    def wrapped(ppci, options):
        lookup = np.asarray(net._pd2ppc_lookups['gen'])[net.gen.index.to_numpy()].astype(int)
        expected = np.asarray(net._pd2ppc_lookups['bus'])[net.gen.bus.to_numpy()]
        assert np.array_equal(ppci['gen'][lookup, GEN_BUS], expected)
        def formulation(om, _args):
            matrix, lo, hi = constraint_rows(cuts, model, lookup, len(ppci['gen']), ppci['baseMVA'], demand)
            om.add_constraints('scheduled_balance_and_lodf', matrix, lo, hi, ['Pg'])
            return om
        return original(add_userfcn(ppci, 'formulation', formulation), options)
    result = {'rounds': [], 'returned_candidate': False, 'status': 'iteration_budget_exhausted'}
    opf_module.opf = wrapped
    try:
        for iteration in range(8):
            t0 = time.perf_counter()
            try:
                pp.runopp(net, calculate_voltage_angles=True, numba=False,
                          init='flat' if iteration==0 else 'results', verbose=False,
                          OPF_FLOW_LIM=2, PDIPM_MAX_IT=100, OPF_VIOLATION=1e-6,
                          PDIPM_FEASTOL=1e-7, PDIPM_GRADTOL=1e-7, PDIPM_COMPTOL=1e-7)
            except pp.OPFNotConverged as exc:
                result['rounds'].append({'iteration': iteration+1, 'opf_converged': False,
                                         'error': str(exc), 'runtime_sec': time.perf_counter()-t0})
                result['status'] = 'opf_nonconvergence'
                break
            power = net.res_gen.p_mw.to_numpy().copy()
            excess, added = add_violated_cuts(power, model, cuts)
            result['rounds'].append({'iteration': iteration+1, 'opf_converged': True,
                                     'max_linear_excess_mw': float(excess), 'new_cuts': added,
                                     'runtime_sec': time.perf_counter()-t0})
            if excess > 1e-5 and added:
                net.gen.p_mw = power
                continue
            gen_vm = net.res_bus.loc[net.gen.bus, 'vm_pu'].to_numpy()
            ext_vm = net.res_bus.loc[net.ext_grid.bus, 'vm_pu'].to_numpy()
            result.update(returned_candidate=True, status='candidate_returned',
                          power=power.tolist(), gen_vm=gen_vm.tolist(), ext_vm=ext_vm.tolist())
            break
    finally:
        opf_module.opf = original
    result['runtime_sec'] = time.perf_counter()-started
    return result

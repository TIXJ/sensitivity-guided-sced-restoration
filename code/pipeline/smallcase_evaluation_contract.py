"""Explicit physical-check semantics shared by small-system evaluations."""
import numpy as np


def constraint_flags(info):
    ac = bool(info.get('ac_converged', False))
    n1_checked = bool(info.get('n1_checked', True))
    def violated(key, checked, default=999):
        value = info.get(key, default)
        return bool(value > 0) if checked and value is not None else None
    return {
        'acpf_not_converged': not ac,
        'voltage_violation': violated('v_violations', ac),
        'thermal_overload': violated('overloaded_lines', ac),
        'transformer_overload': violated('trafo_overloaded', ac, 0),
        'n1_violation': violated('n1_violations', n1_checked),
    }


def accepted(info):
    flags = constraint_flags(info)
    return all(value is False for value in flags.values())


def primary_reason(info):
    if accepted(info):
        return 'full_sced_pass'
    for name, flag in constraint_flags(info).items():
        if flag:
            return name
    return 'incomplete_verification'


def verify_state(env, mod, case):
    """Check the current controls without executing a zero action or updating state."""
    if case == 118:
        info = dict(env._run_ac_pf())
        info['ac_converged'] = bool(info.pop('converged'))
        count, _ = env._check_n1_security(env.current_power, env.dcnet)
    elif case == 39:
        import pandapower as pp
        from pandapower.networks import case39
        net = case39()
        mod.apply_case39_route_b_limits(net)
        net.gen.p_mw = np.clip(env.current_power, 0, env.PMAX)
        net.gen.vm_pu = env.current_gen_vm_pu[:len(net.gen)]
        net.ext_grid.vm_pu = env.current_ext_vm_pu
        scale = env.target_demand / net.load.p_mw.sum()
        net.load.p_mw *= scale
        net.load.q_mvar *= scale
        net.bus.min_vm_pu, net.bus.max_vm_pu = mod.VMIN, mod.VMAX
        info = dict(ac_converged=False, v_violations=999, overloaded_lines=999,
                    trafo_overloaded=999, v_violation_max_pu=None)
        try:
            pp.runpp(net, calculate_voltage_angles=True, numba=False,
                     max_iteration=100, enforce_q_lims=True)
        except pp.LoadflowNotConverged as exc:
            info['error'] = str(exc)
        else:
            vm = net.res_bus.vm_pu.to_numpy()
            loading = net.res_line.loading_percent.to_numpy()
            trafo = net.res_trafo.loading_percent.to_numpy()
            finite = all(np.isfinite(values).all() for values in (vm, loading, trafo))
            if net.converged and finite:
                gaps = np.maximum(mod.VMIN - vm, 0) + np.maximum(vm - mod.VMAX, 0)
                trafo_count = int((trafo > 100).sum())
                info.update(ac_converged=True, v_violations=int((gaps > 0).sum()),
                            overloaded_lines=int((loading > 100).sum()) + trafo_count,
                            trafo_overloaded=trafo_count, v_min=float(vm.min()), v_max=float(vm.max()),
                            v_violation_max_pu=float(gaps.max()), v_violation_mag_pu=float(gaps.sum()),
                            loading_max=float(max(loading.max(initial=0), trafo.max(initial=0))))
            elif not finite:
                info['error'] = 'Nonfinite power-flow results'
        count, _ = env._check_n1(env.current_power, env.dcnet)
    else:
        raise ValueError(case)
    info.update(n1_violations=int(count), ac_checked=True, n1_checked=True)
    info['constraint_flags'] = constraint_flags(info)
    info['passed'] = accepted(info)
    info['primary_reason'] = primary_reason(info)
    return info

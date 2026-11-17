"""Retain overlapping violations separately from a priority-based failure label."""
import numpy as np


def operation_flags(info):
    ac = bool(info.get('converged', False))
    if 'bus_vm_pu' in info:
        ac = ac and bool(np.isfinite(info['bus_vm_pu']).all())
    mapping = {'voltage_violation': 'v_violations', 'line_overload': 'overloaded_lines',
               'transformer_overload': 'trafo_overloaded', 'reference_source_violation': 'reference_source_violations',
               'generator_bound_violation': 'generator_bound_violations', 'scheduled_balance_violation': 'scheduled_balance_violations',
               'n1_violation': 'n1_violations'}
    algebraic = {'n1_violation', 'generator_bound_violation', 'scheduled_balance_violation'}
    flags = {'ac_nonconvergence': not ac}
    for name, key in mapping.items():
        value = info.get(key)
        if not ac and name in {'generator_bound_violation', 'scheduled_balance_violation'} and value == 999:
            value = None
        flags[name] = bool(value > 0) if value is not None and (ac or name in algebraic) else None
    return flags

"""Single-use acceptance receipts, bound to the complete verification context.

No solver result is trusted without a completed independent physical check.
Fingerprint creation and comparison are charged to the online measurement.
"""
import copy
import hashlib
import pickle
import numpy as np


def fingerprint(env, mod, model, low, high):
    # Include input tables, not mutable solver outputs or cache dictionaries.
    network = {k: v for k, v in env._ac_net_template.items()
               if not k.startswith(('_', 'res_'))}
    constants = {k: v for k, v in vars(mod.core).items()
                 if k.isupper() and isinstance(v, (str, bool, int, float, np.ndarray, tuple, list, type(None)))}
    dc = {k: getattr(env.dcnet, k, None)
          for k in ('S', 'lodf', 'effective_limit', 'flow_intercept')}
    settings = {k: getattr(env, k, None) for k in
                ('PMIN', 'PMAX', 'generator_upper_margin', 'valid_contingency_indices',
                 'target_demand', 'current_sample', 'current_power',
                 'current_gen_vm_pu', 'current_ext_grid_vm_pu')}
    payload = (network, constants, vars(mod.ARGS), dc, settings, model, low, high,
               id(mod.core.adopted_operation_pass), id(env._run_ac_pf_check.__func__)
               if hasattr(env._run_ac_pf_check, '__func__') else id(env._run_ac_pf_check),
               id(type(env)._vectorized_n1_state))
    return hashlib.sha256(pickle.dumps(payload, protocol=5)).hexdigest()


class AcceptanceReceipt:
    def __init__(self, env, mod, model, low, high, passed, info):
        if not passed:
            raise ValueError('Only an accepted independently checked candidate can be receipted')
        self.digest = fingerprint(env, mod, model, low, high)
        self.info = copy.deepcopy(info)
        self.used = False

    def consume(self, env, mod, model, low, high):
        if self.used:
            return None
        self.used = True
        if self.digest != fingerprint(env, mod, model, low, high):
            return None
        return True, copy.deepcopy(self.info)

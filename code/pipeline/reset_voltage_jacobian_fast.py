"""Optional, context-keyed acceleration of the deployed reset-voltage feature.

Keeps the original one-sided finite differences, Q-limit handling and step size.
Install on a fresh environment before reset; the feature remains fixed per reset.
The historical training/evaluation module and its legacy caches are untouched.
"""
import copy
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import time
from types import MethodType

import numpy as np
import pandas as pd

VERSION = 'recycled_finite_v2'


def context_fingerprint(net, epsilon, gen_max, ext_max, recycle=True):
    import pandapower as pp
    digest = hashlib.sha256()
    digest.update(json.dumps({'version':VERSION,'pandapower':pp.__version__,
        'epsilon':float(epsilon),'gen_max':float(gen_max),'ext_max':float(ext_max),'recycle':bool(recycle),
        'sn_mva':float(net.sn_mva),'f_hz':float(net.f_hz),
        'user_pf_options':net.get('user_pf_options',{})},sort_keys=True,default=str).encode())
    for key in sorted(net.keys()):
        table = net[key]
        if key.startswith(('_','res_')) or not isinstance(table,pd.DataFrame):
            continue
        digest.update(key.encode())
        digest.update(repr([(c,str(dtype)) for c,dtype in table.dtypes.items()]).encode())
        try:
            digest.update(pd.util.hash_pandas_object(table,index=True).to_numpy().tobytes())
        except TypeError:
            digest.update(pickle.dumps(table,protocol=5))
    return digest.hexdigest()


def finite_voltage_jacobian(net, epsilon, gen_max, ext_max, numba=True, recycle=False):
    """Keep the finite-difference stencil; recycling starts from nominal results."""
    import pandapower as pp
    nominal = copy.deepcopy(net)
    pp.runpp(nominal,calculate_voltage_angles=True,enforce_q_lims=True,numba=numba)
    if not bool(nominal.converged):
        raise RuntimeError('reset AC-PF did not converge')
    vm0 = nominal.res_bus.vm_pu.to_numpy(dtype=np.float64).copy()
    count = len(nominal.gen)+1
    jacobian = np.zeros((len(nominal.bus),count),dtype=np.float64)
    for index in range(count):
        trial = copy.deepcopy(nominal)
        if index < len(trial.gen):
            trial.gen.vm_pu.values[index] = min(float(trial.gen.vm_pu.values[index]+epsilon),gen_max)
        else:
            trial.ext_grid.vm_pu.values[0] = min(float(trial.ext_grid.vm_pu.values[0]+epsilon),ext_max)
        if recycle:
            if trial._options['enforce_q_lims'] is not True:
                raise RuntimeError('Recycled finite differences require Q-limit enforcement')
            pp.runpp(trial,recycle={'bus_pq':False,'trafo':False,'gen':True})
        else:
            pp.runpp(trial,calculate_voltage_angles=True,enforce_q_lims=True,numba=numba)
        if bool(trial.converged):
            jacobian[:,index] = (trial.res_bus.vm_pu.to_numpy(dtype=np.float64)-vm0)/epsilon
    if not np.isfinite(jacobian).all() or np.max(abs(jacobian)) < 1e-10:
        raise RuntimeError('invalid reset voltage Jacobian')
    return jacobian


def install_fast_voltage_jacobian(env, mod, cache_dir, recycle=True):
    """Attach an opt-in backend with reset-scoped memory and checked disk keys."""
    if getattr(env,'_fast_voltage_installed',False):
        raise ValueError('Fast reset-voltage backend already installed')
    directory = Path(cache_dir)
    directory.mkdir(parents=True,exist_ok=True)
    original_reset = env.reset
    memory = {}
    session = {'matrix':None}
    env.fast_voltage_stats = []

    def reset(self,*args,**kwargs):
        session['matrix'] = None
        return original_reset(*args,**kwargs)

    def jacobian(self):
        if session['matrix'] is not None:
            return session['matrix']
        started = time.perf_counter()
        net = self._configure_voltage_jacobian_net()
        epsilon = float(mod.ARGS.voltage_step_pu)
        key = context_fingerprint(net,epsilon,mod.core.GEN_VM_MAX,mod.core.EXT_VM_MAX,recycle)
        source = 'memory'
        fallback = None
        value = memory.get(key)
        path = directory / (key+'.npz')
        if value is None and path.exists():
            try:
                with np.load(path,allow_pickle=False) as saved:
                    candidate = saved['jacobian']
                    checksum = hashlib.sha256(candidate.tobytes()).hexdigest()
                    if (saved['context_key'].item()==key and saved['sha256'].item()==checksum
                            and candidate.shape==(self._n_bus,self.N+1) and np.isfinite(candidate).all()):
                        value = candidate.copy()
                        source = 'disk'
            except (OSError,ValueError,KeyError):
                pass
        if value is None:
            source = 'computed'
            try:
                value = finite_voltage_jacobian(net,epsilon,mod.core.GEN_VM_MAX,mod.core.EXT_VM_MAX,recycle=recycle)
            except Exception as exc:
                fallback = str(exc)
                # Recompute this exact context with the historical numerical path.
                try:
                    value = finite_voltage_jacobian(net,epsilon,mod.core.GEN_VM_MAX,mod.core.EXT_VM_MAX,numba=False)
                    source = 'legacy_fallback'
                except Exception as legacy_error:
                    value = np.zeros((self._n_bus,self.N+1),dtype=np.float64)
                    fallback += '; legacy: '+str(legacy_error)
                    source = 'unavailable_zero'
            if source != 'unavailable_zero':
                checksum = hashlib.sha256(value.tobytes()).hexdigest()
                with tempfile.NamedTemporaryFile(dir=directory,suffix='.npz',delete=False) as handle:
                    temporary = Path(handle.name)
                    np.savez_compressed(handle,jacobian=value,context_key=np.array(key),sha256=np.array(checksum))
                temporary.replace(path)
        memory[key] = value
        session['matrix'] = value
        self.fast_voltage_stats.append({'context_key':key,'source':source,
            'runtime_sec':time.perf_counter()-started,'fallback':fallback})
        return value

    env.reset = MethodType(reset,env)
    env._reset_voltage_jacobian = MethodType(jacobian,env)
    env._fast_voltage_installed = True
    return env

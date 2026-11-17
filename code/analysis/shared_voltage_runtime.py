"""Serial evaluator adapter sharing an already verified AC operating point.

The captured network lives only within one case. The existing verifier still
extracts every limit and a final check bypasses its memo. No physics is replaced.
"""
import threading
import time
from types import MethodType
import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import splu

_LOCK=threading.RLock()


def operating_key(env):
    return (np.asarray(env.current_power).tobytes(),np.asarray(env.current_gen_vm_pu).tobytes(),
            np.asarray(env.current_ext_grid_vm_pu).tobytes(),float(env.target_demand))


def derivative_from_solved(nominal):
    from pandapower.pypower.dSbus_dV import dSbus_dV
    if not nominal.converged or nominal._options['enforce_q_lims'] is not True:
        raise ValueError('Require a converged Q-limited AC solution')
    internal=nominal['_ppc']['internal'];bus=internal['bus']
    pq=np.asarray(internal['pq'],dtype=int);pv=np.asarray(internal['pv'],dtype=int)
    ref=np.asarray(internal['ref'],dtype=int);pvpq=np.r_[pv,pq]
    dvm,dva=dSbus_dV(internal['Ybus'],internal['V'])
    jac=bmat([[dva[pvpq,:][:,pvpq].real,dvm[pvpq,:][:,pq].real],
              [dva[pq,:][:,pvpq].imag,dvm[pq,:][:,pq].imag]],format='csc')
    lookup=nominal._pd2ppc_lookups['bus']
    buses=lookup[np.r_[nominal.gen.bus.to_numpy(),nominal.ext_grid.bus.to_numpy()]]
    if len(bus)!=len(nominal['_ppc']['bus']):
        raise ValueError('Unsupported out-of-service mapping')
    live=np.flatnonzero(np.isin(buses,np.r_[pv,ref]))
    result=np.zeros((len(bus),len(buses)))
    if live.size:
        cols=buses[live]
        rhs=-np.vstack([dvm[pvpq,:][:,cols].real.toarray(),dvm[pq,:][:,cols].imag.toarray()])
        result[np.ix_(pq,live)]=splu(jac).solve(rhs)[len(pvpq):,:]
        result[buses[live],live]=1.
    return result[lookup[nominal.bus.index.to_numpy()],:]


def install(env,mod):
    """Use only in the matched serial workflow, never concurrent env calls."""
    import pandapower as pp
    from reset_voltage_jacobian_fast import finite_voltage_jacobian
    old_check=env._run_ac_pf_check;old_reset=env.reset
    env._shared_last_solution=None;env._shared_reset_matrix=None

    def checked(self):
        key=operating_key(self);captured={}
        # The original verifier starts a daemon PF thread; the wrapper is bound
        # to that call and does not publish late/failed/timeout results.
        with _LOCK:
            original=pp.runpp
            def solve(net,*args,**kwargs):
                kwargs['numba']=True
                result=original(net,*args,**kwargs)
                if net.converged and net._options['enforce_q_lims'] is True:
                    captured['net']=net
                return result
            pp.runpp=solve
            try: result=old_check()
            finally: pp.runpp=original
        if result.get('converged') and not result.get('timeout') and 'net' in captured:
            self._shared_last_solution=(key,captured['net'])
        return result

    def reset(self,*args,**kwargs):
        self._shared_reset_matrix=None
        return old_reset(*args,**kwargs)

    def matrix(self):
        if self._shared_reset_matrix is not None:return self._shared_reset_matrix
        started=time.perf_counter();key=operating_key(self)
        record=self._shared_last_solution;source='shared_solved_ac';error=None
        try:
            if record is not None and record[0]==key:
                net=record[1]
            else:
                source='new_ac'
                net=self._configure_voltage_jacobian_net()
                pp.runpp(net,calculate_voltage_angles=True,enforce_q_lims=True,numba=True)
            value=derivative_from_solved(net)
            if value.shape!=(self._n_bus,self.N+1) or not np.isfinite(value).all():
                raise ValueError('Invalid voltage derivative')
        except Exception as exc:
            source='finite_fallback';error=str(exc)
            value=finite_voltage_jacobian(self._configure_voltage_jacobian_net(),mod.ARGS.voltage_step_pu,
                mod.core.GEN_VM_MAX,mod.core.EXT_VM_MAX,recycle=True)
        self._shared_reset_matrix=value
        self.implicit_feature_calls.append({'source':source,'runtime_sec':time.perf_counter()-started,'error':error})
        return value
    env._run_ac_pf_check=MethodType(checked,env)
    env.reset=MethodType(reset,env)
    env._reset_voltage_jacobian=MethodType(matrix,env)


def clear_case(env):
    env._case_pf_memo={}
    env._shared_last_solution=None
    env._shared_reset_matrix=None

"""Check case isolation and mandatory fresh final checks in the serial adapter."""
import unittest
import os
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from revision_common import RESULTS
os.environ.setdefault('NUMBA_CACHE_DIR',str(RESULTS/'numba_cache'))
import pandapower as pp
from evaluate_implicit_short_policy import install_case_memo
from shared_voltage_runtime import install, clear_case, operating_key


class VerifierCacheTests(unittest.TestCase):
    def make_env(self):
        env=SimpleNamespace(current_power=np.array([1.]),current_gen_vm_pu=np.array([1.]),
            current_ext_grid_vm_pu=np.array([1.]),target_demand=1.,reset=lambda:None)
        def check():
            net=SimpleNamespace(converged=False,_options={'enforce_q_lims':True})
            pp.runpp(net,numba=False,enforce_q_lims=True)
            return {'converged':True,'timeout':False,'vm':[1.]}
        env._run_ac_pf_check=check
        install_case_memo(env)
        install(env,None)
        return env

    def test_final_check_bypasses_memo_and_preserves_numba(self):
        calls=[]
        def solve(net,**kwargs):
            calls.append(kwargs)
            net.converged=True
        with patch.object(pp,'runpp',side_effect=solve) as original:
            env=self.make_env()
            env._run_ac_pf_check();env._run_ac_pf_check()
            self.assertEqual(len(calls),1)
            self.assertEqual(env._shared_last_solution[0],operating_key(env))
            env._verification_fresh=True
            env._run_ac_pf_check()
            self.assertEqual(len(calls),2)
            self.assertIs(pp.runpp,original)
            self.assertTrue(all(c['numba'] and c['enforce_q_lims'] for c in calls))

    def test_clear_case_forces_new_solve_at_same_controls(self):
        def solve(net,**kwargs):net.converged=True
        with patch.object(pp,'runpp',side_effect=solve) as solver:
            env=self.make_env();env._run_ac_pf_check()
            clear_case(env)
            self.assertIsNone(env._shared_last_solution)
            env._run_ac_pf_check()
            self.assertEqual(solver.call_count,2)

    def test_exception_restores_original_runpp(self):
        with patch.object(pp,'runpp',side_effect=RuntimeError('test failure')) as solver:
            env=self.make_env()
            with self.assertRaises(RuntimeError):env._run_ac_pf_check()
            self.assertIs(pp.runpp,solver)
            self.assertIsNone(env._shared_last_solution)


if __name__=='__main__':unittest.main()

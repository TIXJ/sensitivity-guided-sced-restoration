"""Reuse only identical in-case checks; final acceptance must remain fresh."""
import unittest
import numpy as np
from evaluate_implicit_short_policy import install_case_memo


class Dummy:
    def __init__(self):
        self.current_power=np.array([10.])
        self.current_gen_vm_pu=np.array([1.])
        self.current_ext_grid_vm_pu=np.array([1.02])
        self.target_demand=10.
        self.calls=0
        self.converged=True

    def _run_ac_pf_check(self):
        self.calls+=1
        return {'converged':self.converged,'bus_vm_pu':[1.], 'call':self.calls}


class MemoTests(unittest.TestCase):
    def setUp(self):
        self.env=Dummy();install_case_memo(self.env)

    def test_identical_check_and_defensive_copy(self):
        self.env._run_ac_pf_check()['bus_vm_pu'][0]=9.
        result=self.env._run_ac_pf_check()
        self.assertEqual(self.env.calls,1)
        self.assertEqual(result['bus_vm_pu'],[1.])

    def test_state_change_invalidates(self):
        self.env._run_ac_pf_check()
        self.env.current_power[0]+=1e-9
        self.env._run_ac_pf_check()
        self.assertEqual(self.env.calls,2)

    def test_final_verification_is_fresh(self):
        self.env._run_ac_pf_check();self.env._verification_fresh=True
        self.env._run_ac_pf_check()
        self.assertEqual(self.env.calls,2)

    def test_new_case_clears_cache(self):
        self.env._run_ac_pf_check();self.env._case_pf_memo={}
        self.env._run_ac_pf_check()
        self.assertEqual(self.env.calls,2)

    def test_failed_checks_are_not_cached(self):
        self.env.converged=False
        self.env._run_ac_pf_check();self.env._run_ac_pf_check()
        self.assertEqual(self.env.calls,2)


if __name__=='__main__':unittest.main()

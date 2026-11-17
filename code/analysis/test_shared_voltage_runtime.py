"""Numerical and immutability checks for sharing the nominal AC solution."""
import copy
import unittest
import numpy as np
from revision_common import load_aligned
from test_implicit_voltage import ImplicitTests
from shared_voltage_runtime import derivative_from_solved,operating_key,clear_case
from probe_voltage_kernel_v5 import implicit_voltage


class SharedTests(ImplicitTests):
    def test_shared_free_and_limited_points(self):
        for limit in (100.,-50.):
            net=self.network(limit)
            self.pp.runpp(net,calculate_voltage_angles=True,enforce_q_lims=True,numba=True)
            original=copy.deepcopy(net['_ppc'])
            value=derivative_from_solved(net)
            np.testing.assert_allclose(value,implicit_voltage(net),atol=1e-8,rtol=1e-7)
            np.testing.assert_array_equal(net['_ppc']['bus'],original['bus'])
            np.testing.assert_array_equal(net['_ppc']['gen'],original['gen'])

    def test_state_key_and_case_clear(self):
        from types import SimpleNamespace
        env=SimpleNamespace(current_power=np.array([1.]),current_gen_vm_pu=np.array([1.]),
            current_ext_grid_vm_pu=np.array([1.]),target_demand=1.)
        key=operating_key(env);env.current_gen_vm_pu[0]+=.001
        self.assertNotEqual(key,operating_key(env))
        clear_case(env)
        self.assertEqual(env._case_pf_memo,{})
        self.assertIsNone(env._shared_last_solution)


if __name__=='__main__':unittest.main()

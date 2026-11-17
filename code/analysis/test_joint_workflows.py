"""Focused physical-control and acceptance tests for the matched workflow."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from shared_joint_optimizer import voltage_box
from compare_joint_workflows import check


class JointWorkflowTests(unittest.TestCase):
    def test_voltage_box_is_independent_of_terminal_policy_state(self):
        class Devices(SimpleNamespace):
            def __len__(self):
                return len(self.vm_pu)
        template = SimpleNamespace(gen=Devices(vm_pu=np.array([1.,1.07])),
                                   ext_grid=Devices(vm_pu=np.array([1.])))
        env = SimpleNamespace(_ac_net_template=template, N=2, current_gen_vm_pu=np.array([.95,.95]))
        mod = SimpleNamespace(core=SimpleNamespace(GEN_VM_MIN=.92, GEN_VM_MAX=1.08,
                                                    EXT_VM_MIN=.92, EXT_VM_MAX=1.08),
                              ARGS=SimpleNamespace(voltage_target_range_pu=.02))
        nominal, low, high = voltage_box(env, mod)
        np.testing.assert_allclose(nominal,[1,1.07,1])
        np.testing.assert_allclose(low,[.98,1.05,.98])
        np.testing.assert_allclose(high,[1.02,1.08,1.02])

    def test_finite_verifier_does_not_override_control_bounds(self):
        env = SimpleNamespace(current_gen_vm_pu=np.array([1.03]),current_ext_grid_vm_pu=np.array([1.]))
        with patch('compare_joint_workflows.verify',return_value=(True,{'bus_vm_pu':[1.,1.]})):
            passed, info = check(env,None,np.array([.98,.98]),np.array([1.02,1.02]))
        self.assertFalse(passed)
        self.assertFalse(info['controls_within_original_voltage_box'])

    def test_nonfinite_results_are_rejected(self):
        env = SimpleNamespace(current_gen_vm_pu=np.array([1.]),current_ext_grid_vm_pu=np.array([1.]))
        with patch('compare_joint_workflows.verify',return_value=(True,{'bus_vm_pu':[float('nan')]})):
            passed, _ = check(env,None,np.array([.98,.98]),np.array([1.02,1.02]))
        self.assertFalse(passed)


if __name__=='__main__':
    unittest.main()

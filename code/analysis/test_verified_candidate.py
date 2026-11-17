import unittest
from types import SimpleNamespace
import numpy as np
from verified_candidate import AcceptanceReceipt


class Env:
    def _run_ac_pf_check(self):
        pass

    def _vectorized_n1_state(self):
        pass


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.env = Env()
        self.env._ac_net_template = {'bus': np.array([1., 2.]), 'sn_mva': 100.}
        self.env.dcnet = SimpleNamespace(S=np.eye(2), lodf=np.eye(2),
            effective_limit=np.array([100., 100.]), flow_intercept=np.zeros(2))
        for key, value in dict(PMIN=np.zeros(2), PMAX=np.ones(2), generator_upper_margin=.99,
            valid_contingency_indices=[0], target_demand=1., current_sample={'sample_id':'a'},
            current_power=np.array([.5,.5]), current_gen_vm_pu=np.ones(2),
            current_ext_grid_vm_pu=np.ones(1)).items():
            setattr(self.env, key, value)
        self.mod = SimpleNamespace(core=SimpleNamespace(RL_VMIN_PU=.94,
            adopted_operation_pass=lambda x: True), ARGS=SimpleNamespace(voltage_step_pu=.002))
        self.model = {'S': np.eye(2)}
        self.low, self.high = np.ones(3)*.9, np.ones(3)*1.1

    def receipt(self):
        return AcceptanceReceipt(self.env, self.mod, self.model, self.low, self.high,
                                 True, {'bus_vm_pu': [1., 1.]})

    def consume(self, receipt):
        return receipt.consume(self.env, self.mod, self.model, self.low, self.high)

    def test_unchanged_once_and_deepcopy(self):
        receipt = self.receipt()
        accepted, info = self.consume(receipt)
        self.assertTrue(accepted)
        info['bus_vm_pu'][0] = 0
        self.assertEqual(receipt.info['bus_vm_pu'][0], 1)
        self.assertIsNone(self.consume(receipt))

    def test_context_changes_invalidate(self):
        changes = [lambda: self.env.current_power.__setitem__(0, .6),
                   lambda: self.env.current_gen_vm_pu.__setitem__(0, 1.01),
                   lambda: self.env.current_ext_grid_vm_pu.__setitem__(0, 1.01),
                   lambda: self.env._ac_net_template['bus'].__setitem__(0, 2),
                   lambda: setattr(self.env, 'target_demand', 2),
                   lambda: self.env.dcnet.effective_limit.__setitem__(0, 99),
                   lambda: self.env.dcnet.lodf.__setitem__((0, 0), .5),
                   lambda: self.env.valid_contingency_indices.append(1),
                   lambda: setattr(self.mod.core, 'RL_VMIN_PU', .95),
                   lambda: setattr(self.mod.ARGS, 'voltage_step_pu', .003),
                   lambda: self.model['S'].__setitem__((0, 0), .1),
                   lambda: self.low.__setitem__(0, .91),
                   lambda: setattr(self.env, '_run_ac_pf_check', lambda: None)]
        for change in changes:
            with self.subTest(change=change):
                self.setUp()
                receipt = self.receipt()
                change()
                self.assertIsNone(self.consume(receipt))

    def test_rejected_candidate_cannot_be_receipted(self):
        with self.assertRaises(ValueError):
            AcceptanceReceipt(self.env, self.mod, self.model, self.low, self.high, False, {})


if __name__ == '__main__':
    unittest.main()

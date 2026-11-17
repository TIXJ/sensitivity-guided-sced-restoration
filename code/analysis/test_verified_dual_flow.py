"""Isolated control-flow tests: actual physics is tested by the timed study."""
import unittest
from unittest.mock import patch
import numpy as np
from test_verified_candidate import ReceiptTests
import remeasure_verified_dual as runner


class FlowTests(unittest.TestCase):
    def setUp(self):
        fixture=ReceiptTests()
        fixture.setUp()
        self.env,self.mod,self.model=fixture.env,fixture.mod,fixture.model
        self.env.N=2
        self.mod.core.RL_VMAX_PU=1.09
        self.env._predict_gat=lambda _:np.array([.5,.5])
        self.mod.core.project_box_sum=lambda *args:np.array([.5,.5])
        self.sample={'sample_id':'a','demand_mw':1.,'sample_cp0':[0,0],
                     'sample_cp1':[1,1],'sample_cp2':[0,0]}
        self.env.current_sample=self.sample
        self.voltage=(np.ones(3), np.ones(3)*.9, np.ones(3)*1.1)
        self.candidate={'returned_candidate':True,'power':[.6,.4],
                        'gen_vm':[1.,1.], 'ext_vm':[1.], 'status':'test'}

    def evaluate(self,checks):
        with patch.object(runner.common,'sync'), \
             patch.object(runner.common,'voltage_box',return_value=self.voltage), \
             patch('shared_joint_optimizer.voltage_box',return_value=self.voltage), \
             patch('shared_joint_optimizer.solve',return_value=self.candidate), \
             patch.object(runner.common,'check',side_effect=checks) as check:
            row=runner.evaluate(self.env,self.mod,None,self.sample,0,self.model,None)
            return row,check.call_count

    def test_success_checks_once_not_twice(self):
        row,calls=self.evaluate([(False,{'bus_vm_pu':[1,1]}),(True,{'bus_vm_pu':[1,1]})])
        self.assertEqual(calls,2) # Initial point and accepted optimizer candidate.
        self.assertEqual(row['fresh_candidate_checks'],1)
        self.assertTrue(row['acceptance_receipt_reused'])

    def test_distinct_restarts_checked_separately(self):
        row,calls=self.evaluate([(False,{'bus_vm_pu':[1,1]}),
                                (False,{'bus_vm_pu':[1,1]}),
                                (True,{'bus_vm_pu':[1,1]})])
        self.assertEqual(calls,3)
        self.assertEqual(len(row['optimizer']['initializer_attempts']),2)
        self.assertEqual(row['fresh_candidate_checks'],2)

    def test_mutated_network_requires_new_check(self):
        consume=runner.AcceptanceReceipt.consume
        def mutate(receipt,*args):
            self.env._ac_net_template['sn_mva']=200.
            return consume(receipt,*args)
        with patch.object(runner.AcceptanceReceipt,'consume',mutate):
            row,calls=self.evaluate([(False,{'bus_vm_pu':[1,1]}),
                                    (True,{'bus_vm_pu':[1,1]}),
                                    (True,{'bus_vm_pu':[1,1]})])
        self.assertEqual(calls,3)
        self.assertFalse(row['acceptance_receipt_reused'])


if __name__=='__main__':
    unittest.main()

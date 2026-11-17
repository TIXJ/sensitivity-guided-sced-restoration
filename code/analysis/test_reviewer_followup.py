"""Ensure that both initializations share the declared solver-call budget."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import reviewer_followup as study
import pandapower as pp
import shared_joint_optimizer as solver


class DualBudgetTests(unittest.TestCase):
    def run_case(self,accepted):
        env=SimpleNamespace(N=1,current_power=np.array([1.]),current_gen_vm_pu=np.array([1.]),
                            current_ext_grid_vm_pu=np.array([1.]))
        def fake_solve(*args):
            for _ in range(1 if accepted else 6):
                try:pp.runopp(None,init='flat',PDIPM_MAX_IT=100)
                except pp.OPFNotConverged:break
            return {'returned_candidate':accepted,'power':[1.],'gen_vm':[1.],'ext_vm':[1.]}
        with patch.object(pp,'runopp') as run,patch.object(solver,'solve',side_effect=fake_solve),\
             patch.object(solver,'voltage_box',return_value=(np.ones(2),np.zeros(2),np.ones(2)*2)),\
             patch.object(study.common,'check',return_value=(accepted,{})):
            result=study.two_initializers(env,None,None,np.ones(1),1.)
            self.assertIs(pp.runopp,run)
            self.assertEqual(run.call_count,1 if accepted else 8)
            self.assertTrue(all(call.kwargs['PDIPM_MAX_IT']==100 for call in run.call_args_list))
        return result

    def test_both_failed_starts_capped_at_four_each(self):
        result=self.run_case(False)
        self.assertEqual([r['actual_opf_calls'] for r in result['initializer_attempts']],[4,4])
        self.assertEqual(result['actual_opf_calls'],8)

    def test_verified_success_does_not_restart(self):
        result=self.run_case(True)
        self.assertEqual(len(result['initializer_attempts']),1)
        self.assertTrue(result['initializer_attempts'][0]['accepted'])


if __name__=='__main__':unittest.main()

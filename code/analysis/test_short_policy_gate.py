"""Check conservative invocation boundaries for the deployment-only gate."""
import unittest
from types import SimpleNamespace
from evaluate_short_policy import selected, feature


class GateTests(unittest.TestCase):
    rule={'cap':4,'max_gap_pu':.01,'sum_gap_pu':.05}

    def test_inclusive_boundaries(self):
        self.assertTrue(selected({'demand_mw':16000,'max_gap_pu':.01,'sum_gap_pu':.05},self.rule))
        self.assertTrue(selected({'demand_mw':16400,'max_gap_pu':.01,'sum_gap_pu':.05},self.rule))

    def test_out_of_range_and_large_gaps(self):
        for change in ({'demand_mw':15999},{'demand_mw':16401},{'max_gap_pu':.011},{'sum_gap_pu':.051},{'valid':False}):
            f={'demand_mw':16200,'max_gap_pu':.005,'sum_gap_pu':.02}
            f.update(change)
            self.assertFalse(selected(f,self.rule))

    def test_invalid_voltage_does_not_invoke_policy(self):
        mod=SimpleNamespace(core=SimpleNamespace(RL_VMIN_PU=.933,RL_VMAX_PU=1.09))
        for info in ({},{'bus_vm_pu':[float('nan')]}):
            f=feature(info,{'demand_mw':16000},mod)
            self.assertFalse(selected(f,self.rule))

    def test_validation_selected_demand_upper_bound(self):
        rule=dict(self.rule,max_demand_mw=16000)
        f={'demand_mw':16000,'max_gap_pu':.005,'sum_gap_pu':.02}
        self.assertTrue(selected(f,rule))
        f['demand_mw']=16200
        self.assertFalse(selected(f,rule))


if __name__=='__main__': unittest.main()

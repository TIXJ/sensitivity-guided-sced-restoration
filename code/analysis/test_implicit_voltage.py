"""Check implicit derivatives against small-step Q-limited AC power flow."""
import unittest
import numpy as np
from revision_common import load_aligned
from probe_voltage_kernel_v5 import implicit_voltage
from probe_voltage_kernel_v4 import solver_finite


class ImplicitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_aligned()
        import pandapower as pp
        cls.pp=pp

    def network(self,limit=100.):
        pp=self.pp
        net=pp.create_empty_network(sn_mva=100.)
        for _ in range(3):pp.create_bus(net,vn_kv=110.)
        pp.create_ext_grid(net,0,vm_pu=1.02)
        pp.create_gen(net,1,p_mw=20.,vm_pu=1.01,min_q_mvar=-100.,max_q_mvar=limit)
        pp.create_load(net,2,p_mw=60.,q_mvar=30.)
        for f,t in ((0,1),(1,2),(0,2)):
            pp.create_line_from_parameters(net,f,t,20.,.1,.4,0.,1.)
        return net

    def test_free_reactive_control(self):
        net=self.network()
        analytic=implicit_voltage(net)
        finite=solver_finite(net,1e-6,1.1,1.1)
        np.testing.assert_allclose(analytic,finite,atol=1e-4,rtol=1e-4)

    def test_q_limited_generator_not_treated_as_free_voltage(self):
        net=self.network(limit=-50.)
        analytic=implicit_voltage(net)
        finite=solver_finite(net,1e-6,1.1,1.1)
        np.testing.assert_allclose(analytic,finite,atol=1e-4,rtol=1e-4)
        np.testing.assert_allclose(analytic[:,0],0.,atol=1e-10)


if __name__=='__main__':unittest.main()

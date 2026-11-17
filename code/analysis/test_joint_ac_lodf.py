import unittest
import numpy as np
from pilot_joint_ac_lodf import constraint_rows


class CallbackRowsTests(unittest.TestCase):
    def test_units_order_and_signed_outage(self):
        model={'S':[[1.,-1.],[2.,1.]],'flow_intercept':[3.,-2.],
               'lodf':[[0.,.2],[.5,0.]],'effective_limit':[20.,40.]}
        lookup=np.array([2,1])
        a,lo,hi=constraint_rows([(-1,0,1),(0,1,-1)],model,lookup,3,100.,10.)
        # External source is column 0 and is not part of scheduled generation.
        p_pu=np.array([.03,.04,.06])
        self.assertAlmostEqual(float((a@p_pu)[0]),.1)
        self.assertEqual(lo[0],hi[0])
        self.assertAlmostEqual(float((a@p_pu)[1]-hi[1]),(3+6-4-20)/20)
        flow=-2+2*6+4+.5*(3+6-4)
        self.assertAlmostEqual(float((a@p_pu)[2]-hi[2]),(-flow-40)/40)


if __name__=='__main__':
    unittest.main()

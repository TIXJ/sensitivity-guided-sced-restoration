import unittest
import numpy as np
from pilot_validation_feasibility import minimum_linear_dispatch


class MinimumRedispatchTests(unittest.TestCase):
    def setUp(self):
        self.model={'S':[[1.,-1.]],'flow_intercept':[0.], 'lodf':[[0.]],
                    'effective_limit':[4.], 'valid_contingency_indices':[]}
        self.basis=np.array([[1.],[-1.]])/np.sqrt(2)

    def test_minimum_norm_secure(self):
        point,info=minimum_linear_dispatch(np.array([8.,2.]),np.zeros(2),np.ones(2)*10,self.basis,self.model)
        np.testing.assert_allclose(point,[7,3],atol=1e-5)
        self.assertAlmostEqual(info['minimum_delta_l2_mw'],np.sqrt(2),places=5)

    def test_infeasible_bounds_not_solver_success(self):
        p0=np.array([8.,2.])
        point,info=minimum_linear_dispatch(p0,p0,p0,self.basis,self.model)
        self.assertIsNone(point)
        self.assertEqual(info['status'],'infeasible_in_linear_span_and_box')

    def test_radial_budget_is_enforced(self):
        p0=np.array([8.,2.])
        point,info=minimum_linear_dispatch(p0,np.zeros(2),np.ones(2)*10,self.basis,self.model,
                                           radius_center=p0,radius=1.)
        self.assertIsNone(point)

    def test_radial_budget_allows_feasible_point(self):
        p0=np.array([8.,2.])
        point,info=minimum_linear_dispatch(p0,np.zeros(2),np.ones(2)*10,self.basis,self.model,
                                           radius_center=p0,radius=2.)
        np.testing.assert_allclose(point,[7,3],atol=1e-5)


if __name__=='__main__':
    unittest.main()

import unittest
import numpy as np
from diagnose_validation_action_range import necessary_radius


class RadiusTests(unittest.TestCase):
    def test_single_constraint(self):
        self.assertAlmostEqual(necessary_radius(np.array([10.]), np.array([4.]), np.array([2.]), 0), 3)

    def test_already_feasible(self):
        self.assertEqual(necessary_radius(np.array([2.]), np.array([4.]), np.array([0.])), 0)

    def test_uncorrectable(self):
        self.assertTrue(np.isinf(necessary_radius(np.array([-10.]), np.array([4.]), np.array([0.]))))


if __name__ == '__main__':
    unittest.main()

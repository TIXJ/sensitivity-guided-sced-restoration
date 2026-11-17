import unittest
from final_analysis import projection_stats


class ProjectionTests(unittest.TestCase):
    def test_failed_episode_steps_are_included(self):
        rows=[{'passed':False,'trace':[{'offspan_l2_mw':2e-6,'projection_change_l2_mw':3.,'ac_checked':True}]},
              {'passed':True,'trace':[{'offspan_l2_mw':1e-12,'projection_change_l2_mw':1e-5,'ac_checked':False}]}]
        result=projection_stats(rows)
        self.assertEqual(result['steps'],2)
        self.assertEqual(result['above_1e_minus6_mw'],1)
        self.assertEqual(result['ac_checked_steps'],1)
        self.assertEqual(result['max_recovery_change_mw'],3.)

    def test_missing_measurements_are_not_zero(self):
        with self.assertRaises(ValueError):projection_stats([])
        with self.assertRaises(ValueError):
            projection_stats([{'trace':[{'offspan_l2_mw':float('nan'),'projection_change_l2_mw':0.,'ac_checked':False}]}])


if __name__=='__main__':unittest.main()

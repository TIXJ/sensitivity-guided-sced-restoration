"""Keep failed seeds in acceptance statistics and avoid invented costs."""
import statistics
import unittest
from summarize_revision import across_seeds


class SeedSummaryTests(unittest.TestCase):
    def test_sample_standard_deviation(self):
        rates=[44/75,45/75,41/75]
        summary=across_seeds([{'pass_rate':r} for r in rates])
        self.assertAlmostEqual(summary['pass_rate_mean'],statistics.mean(rates))
        self.assertAlmostEqual(summary['pass_rate_std'],statistics.stdev(rates))

    def test_zero_acceptance_seed_is_retained(self):
        rows=[{'pass_rate':1.,'scheduled_cost':{'mean':10.}},
              {'pass_rate':1.,'scheduled_cost':{'mean':12.}}, {'pass_rate':0.}]
        summary=across_seeds(rows)
        self.assertEqual(summary['n_seeds'],3)
        self.assertAlmostEqual(summary['pass_rate_mean'],2/3)
        self.assertEqual(summary['scheduled_cost']['n_valid_seeds'],2)
        self.assertIsNone(summary['scheduled_cost']['mean'])

    def test_identical_seed_rates(self):
        self.assertEqual(across_seeds([{'pass_rate':1.}]*3)['pass_rate_std'],0.)


if __name__=='__main__':
    unittest.main()

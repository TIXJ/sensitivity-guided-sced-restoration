"""Check seed-level aggregation without accepting smoke or mismatched tests."""
import unittest
from summarize_smallcase_multiseed import summarize_case, archive_scope


def payload(seed, passed, updates=100):
    return {'complete': True, 'contract': 'common_smallcase_v2', 'seed': seed,
            'protocol': {'updates': updates, 'partition_seed': 42},
            'rows': [{'mode': 'policy', 'sample_id': str(i), 'passed': i < passed,
                      'steps': 2 + i, 'runtime_sec': .1} for i in range(3)]}


class SeedSummaryTests(unittest.TestCase):
    def test_archive_scope_requires_completed_public_download_checks(self):
        self.assertIn('local', archive_scope(None))
        with self.assertRaises(AssertionError):
            archive_scope({'version': 'v1.2.0', 'fresh_public_clone': True, 'tests': {}})
        tests = {f'case{case}_seed{seed}': {'completed': True, 'same_ids_pass_flags_steps_reasons': True}
                 for case in (39, 118) for seed in (1, 2)}
        self.assertIn('v1.2.0', archive_scope({'version': 'v1.2.0', 'fresh_public_clone': True, 'tests': tests}))

    def test_mean_and_sample_sd_are_over_three_seed_rates(self):
        summary = summarize_case(payload(42, 3), [payload(1, 2), payload(2, 1)])
        self.assertEqual(summary['counts'], [3, 2, 1])
        self.assertAlmostEqual(summary['mean_pass_rate_pct'], 200/3)
        self.assertAlmostEqual(summary['sample_sd_percentage_points'], 100/3)
        self.assertEqual(summary['all_case_mean_steps'], [3., 3., 3.])

    def test_smoke_and_test_id_changes_are_rejected(self):
        with self.assertRaises(AssertionError):
            summarize_case(payload(42, 3), [payload(1, 3, updates=2), payload(2, 3)])
        changed = payload(2, 3)
        changed['rows'][0]['sample_id'] = 'different'
        with self.assertRaises(AssertionError):
            summarize_case(payload(42, 3), [payload(1, 3), changed])


if __name__ == '__main__':
    unittest.main()

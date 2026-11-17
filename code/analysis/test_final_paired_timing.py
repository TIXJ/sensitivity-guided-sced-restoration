"""Validate the publication plot against saved case-level timing records only."""
from pathlib import Path
import json
import sys
import unittest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from final_evidence_documents import paired_timing_data


class FinalPairedTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.populations=paired_timing_data(HERE)

    def test_case_denominators_and_unpooled_repeats(self):
        self.assertEqual([p['n'] for p in self.populations],[75,100])
        for p in self.populations:
            self.assertEqual(len(p['delta']),p['n'])
            self.assertEqual(len(set(p['ids'])),p['n'])

    def test_saved_intervals_have_different_evidence_strength(self):
        historical,spatial=self.populations
        self.assertAlmostEqual(historical['mean_sec'],.18693853777833283,places=11)
        self.assertGreater(historical['ci95_sec'][0],0)
        self.assertAlmostEqual(spatial['mean_sec'],.039605233665594534,places=11)
        self.assertLess(spatial['ci95_sec'][0],0)
        self.assertGreater(spatial['ci95_sec'][1],0)

    def test_spatial_failures_precede_correction(self):
        raw=json.loads((HERE/'experiment_results/spatial_dual_final_v1/results.json').read_text(encoding='utf-8'))
        for repeat in range(3):
            for arm in ('direct','selected'):
                failed=[r for r in raw['rows'] if r['repeat']==repeat and r['arm']==arm and r['preparation_failed']]
                self.assertEqual(len(failed),7)
                for row in failed:
                    self.assertFalse(row.get('policy_attempted',False))
                    self.assertFalse(row.get('optimizer_called',False))


if __name__=='__main__':
    unittest.main()

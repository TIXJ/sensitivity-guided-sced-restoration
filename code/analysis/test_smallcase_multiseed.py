"""Protocol tests that do not load torch, predictors or physical networks."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import train_smallcase_multiseed as training


class SmallcaseSeedTests(unittest.TestCase):
    def test_partitions_stay_at_seed42_when_training_seed_changes(self):
        mod = SimpleNamespace(SEED=42)
        mod.split_samples = lambda samples: ([mod.SEED], [43], [44])
        split = training.freeze_split(mod, 39, [object()])
        mod.SEED = 2
        first = split([object()])
        self.assertEqual(first, ([42], [43], [44]))
        first[0].append(999)
        self.assertEqual(split([]), ([42], [43], [44]))
        self.assertEqual(mod.SEED, 2)

    def test_118_partition_seed_does_not_change_physical_basis_seed(self):
        mod = SimpleNamespace(SEED=42)
        mod.split_samples_stratified = lambda samples: ([mod.SEED], [43], [44])
        self.assertEqual(training.freeze_split(mod, 118, [])([])[0], [42])
        self.assertEqual(mod.SEED, 42)

    def test_checkpoint_selection_uses_pass_count_then_validation_reward(self):
        best = {'passed': 10, 'avg_reward': 20}
        self.assertTrue(training.validation_improves({'passed': 11, 'avg_reward': -99}, best))
        self.assertFalse(training.validation_improves({'passed': 9, 'avg_reward': 99999}, best))
        self.assertTrue(training.validation_improves({'passed': 10, 'avg_reward': 21}, best))
        self.assertFalse(training.validation_improves(dict(best), best))

    def test_validation_subsets_match_the_original_training_entry_points(self):
        self.assertEqual(training.validation_subset(39, list(range(30))), list(range(15)))
        self.assertEqual(training.validation_subset(118, list(range(16))), list(range(16)))

    def test_original_output_directories_are_rejected(self):
        root = Path(tempfile.gettempdir()).resolve() / 'seed-test-results'
        training.validate_output(root / 'case39_seed1', root)
        with self.assertRaises(ValueError):
            training.validate_output(root.parent / 'rl39_routeb_main', root)
        with self.assertRaises(ValueError):
            training.validate_output(root, root)

    def test_smoke_checks_one_case_but_full_training_evaluates_every_case(self):
        ids = [1, 2, 3]
        self.assertEqual(training.evaluation_ids(ids, 2), [1])
        self.assertEqual(training.evaluation_ids(ids, 100), ids)


if __name__ == '__main__':
    unittest.main()

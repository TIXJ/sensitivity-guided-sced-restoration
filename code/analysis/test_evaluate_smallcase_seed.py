"""Select new-seed checkpoints explicitly and reject unsupported seeds."""
from pathlib import Path
import unittest
from evaluate_smallcase_seed import seed_checkpoint


class SeedCheckpointTests(unittest.TestCase):
    def test_new_seed_checkpoint_names_are_explicit(self):
        expected = Path('results/smallcase_multiseed_v1/case39_seed2/selected_validation_actor.pth')
        self.assertEqual(seed_checkpoint(39, 2, Path('results')), expected)
        with self.assertRaises(ValueError):
            seed_checkpoint(39, 42, Path('results'))
        with self.assertRaises(ValueError):
            seed_checkpoint(300, 1, Path('results'))


if __name__ == '__main__':
    unittest.main()

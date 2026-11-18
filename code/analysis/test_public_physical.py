"""Regression tests for public input isolation, independent of physics packages."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from distribution_integrity import allowed_read, verify, write_only
import physical_reproduce as physical


class PublicPhysicalTests(unittest.TestCase):
    def test_seed_statistics_allow_roundoff_but_not_count_changes(self):
        from public_reproduce import verify_seed_statistics
        expected = {'counts': [30, 30, 24], 'test_total': 30,
                    'mean_pass_rate_pct': 93.33333333333333,
                    'sample_sd_percentage_points': 11.547005383792516}
        actual = dict(expected, sample_sd_percentage_points=11.547005383792515)
        verify_seed_statistics(actual, expected)
        with self.assertRaises(AssertionError):
            verify_seed_statistics(dict(actual, counts=[30, 30, 25]), expected)
        with self.assertRaises(AssertionError):
            verify_seed_statistics(dict(actual, sample_sd_percentage_points=11.6), expected)

    def fixture(self, root):
        for folder in ('code/pipeline/root', 'code/analysis', 'data/cases', 'data/partitions',
                       'data/voltage_context/none', 'data/protocols', 'data/operating_points', 'models'):
            (root / folder).mkdir(parents=True, exist_ok=True)
        sample = root / 'data/cases/input.json'
        sample.write_bytes(b'{"input":1}')
        (root / 'MANIFEST.json').write_text(json.dumps({'files': {
            'data/cases/input.json': hashlib.sha256(sample.read_bytes()).hexdigest()}}))

    def test_hash_mismatch_stops_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            self.assertEqual(verify(root)['verified_files'], 1)
            (root / 'data/cases/input.json').write_bytes(b'{"input":2}')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                verify(root)

    def test_missing_input_stops_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            (root / 'data/cases/input.json').unlink()
            with self.assertRaisesRegex(ValueError, 'missing'):
                verify(root)

    def test_materializations_do_not_share_stale_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            with patch.object(physical, 'REPO', root):
                first, _, _ = physical.prepare()
                (first / 'stale.pth').write_bytes(b'stale')
                second, _, _ = physical.prepare()
            self.assertNotEqual(first, second)
            self.assertEqual(len(first.parent.name), 8)
            self.assertFalse((second / 'stale.pth').exists())
            self.assertTrue((second / 'execution_inputs.json').is_file())

    def test_readwrite_modes_are_not_write_only(self):
        for mode in ('r+b', 'w+b', 'a+b'):
            self.assertFalse(write_only(mode, os.O_RDWR))
        self.assertFalse(write_only(None, os.O_RDWR))
        self.assertTrue(write_only('wb', os.O_WRONLY))

    def test_unmanifested_module_cannot_enter_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            (root / 'code/pipeline/numpy.py').write_bytes(b'raise RuntimeError("shadow")')
            with patch.object(physical, 'REPO', root):
                with self.assertRaisesRegex(ValueError, 'Undeclared'):
                    physical.prepare()

    def test_external_input_suffix_cannot_bypass_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for suffix in ('.json.gz', '.pt', '.pth', '.anything'):
                self.assertFalse(allowed_read(root / ('external' + suffix), (root / 'allowed',)))
                self.assertTrue(allowed_read(root / 'allowed' / ('input' + suffix), (root / 'allowed',)))

    def test_invalid_arguments_fail_before_materialization(self):
        script = Path(physical.__file__)
        for options in (['--name', '..'], ['--repeats', '0'], ['--max-samples', '-1']):
            run = subprocess.run([sys.executable, str(script), 'historical', *options], capture_output=True)
            self.assertEqual(run.returncode, 2)

    def test_added_small_seed_stages_use_real_evaluation_and_training(self):
        for stage, script in (('small-seed', 'evaluate_smallcase_seed.py'),
                              ('train-small', 'train_smallcase_multiseed.py')):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / 'ED_RL'
                revision = root / 'GAT-SAC-SCED-Framework/first revise'
                analysis = revision / 'experiments'
                argv = ['physical_reproduce.py', stage, '--case', '39', '--seed', '1', '--name', 'test_new_seed']
                with patch.object(sys, 'argv', argv), patch.object(physical, 'prepare', return_value=(root, revision, analysis)), patch.object(physical, 'launch') as launch:
                    physical.main()
                self.assertEqual(launch.call_args.args[2], script)
                self.assertIn('--seed', launch.call_args.args[3])

    def test_small_training_rejects_unimplemented_method_and_sample_overrides(self):
        script = Path(physical.__file__)
        for options in (['--method', 'td3'], ['--max-samples', '1']):
            run = subprocess.run([sys.executable, str(script), 'train-small', '--seed', '1', *options], capture_output=True)
            self.assertEqual(run.returncode, 2)


if __name__ == '__main__':
    unittest.main()

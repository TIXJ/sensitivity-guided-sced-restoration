"""Regression tests for failed numerical measurements and partial evaluations."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from revision_common import REV, json_safe, read, resume_rows, write, voltage_cache_directory


class ResultTests(unittest.TestCase):
    def test_voltage_cache_isolation(self):
        settings={'stress':'renewable','initialization':'gat','output':'a'}
        path=voltage_cache_directory(settings,1,{'gat':'hash'},'actor')
        self.assertNotEqual(path,voltage_cache_directory(settings,2,{'gat':'hash'},'actor'))
        self.assertEqual(path,voltage_cache_directory({**settings,'output':'b'},1,{'gat':'hash'},'actor'))
        self.assertNotEqual(path,voltage_cache_directory(settings,1,{'gat':'changed'},'actor'))
        label={'stress':'none','initialization':'label'}
        economic={'stress':'none','initialization':'economic'}
        self.assertNotEqual(voltage_cache_directory(label,1,{},'actor'),voltage_cache_directory(economic,1,{},'actor'))
        original={'stress':'none','initialization':'gat'}
        self.assertEqual(voltage_cache_directory(original,1,{},'actor'),voltage_cache_directory(original,2,{},'actor'))

    def test_nonfinite_is_missing_not_zero(self):
        raw = {'rows': [{'passed': False, 'reason': 'ac_nonconvergence',
                         'reference_p_mw': np.float64(np.nan),
                         'trace': np.array([1., np.inf, -np.inf])}]}
        safe = json_safe(raw)
        self.assertEqual(safe['rows'][0]['trace'], [1., None, None])
        self.assertIsNone(safe['rows'][0]['reference_p_mw'])
        self.assertFalse(safe['rows'][0]['passed'])
        self.assertEqual(len(safe['nonfinite_fields']), 3)
        json.dumps(safe, allow_nan=False)
        self.assertTrue(np.isnan(raw['rows'][0]['reference_p_mw']))
        self.assertEqual(json_safe(safe), safe)

    def test_atomic_roundtrip_preserves_diagnostics(self):
        with tempfile.TemporaryDirectory(dir=REV) as folder:
            path = Path(folder) / 'results.json'
            write(path, {'rows': [{'value': float('nan')}]})
            saved = read(path)
            saved['rows'].append({'value': float('inf')})
            write(path, saved)
            restored = read(path)
            self.assertEqual(restored['nonfinite_fields'],
                             ['$.rows[0].value', '$.rows[1].value'])
            self.assertFalse(path.with_suffix('.json.tmp').exists())

    def test_resume_validation(self):
        settings = {'method': 'sac', 'output': 'old/path'}
        row = {'sample_id': 'a', 'passed': False, 'trace': [1, 2]}
        saved = {'settings': settings, 'checkpoint_sha256': 'hash',
                 'complete': False, 'rows': [row]}
        self.assertEqual(resume_rows(saved, {**settings, 'output': 'new/path'},
                                     'hash', ['a', 'b']), [row])
        for changes, requested, checkpoint, ids in (
            ({}, {**settings, 'method': 'td3'}, 'hash', ['a', 'b']),
            ({}, settings, 'different-hash', ['a', 'b']),
            ({}, settings, 'hash', ['b', 'a']),
            ({'complete': True}, settings, 'hash', ['a', 'b']),
            ({'rows': [row, row]}, settings, 'hash', ['a', 'a']),
        ):
            with self.subTest(changes=changes, ids=ids):
                with self.assertRaises(ValueError):
                    resume_rows({**copy.deepcopy(saved), **changes}, requested, checkpoint, ids)


if __name__ == '__main__':
    unittest.main()

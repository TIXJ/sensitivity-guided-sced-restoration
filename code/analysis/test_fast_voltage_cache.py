"""Check cache hits, invalidation and failure fallback independently of PF speed."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import revision_common
from reset_voltage_jacobian_fast import install_fast_voltage_jacobian


class Network(dict):
    def __getattr__(self,name):
        try: return self[name]
        except KeyError as exc: raise AttributeError(name) from exc


class Environment:
    def __init__(self):
        net=Network(sn_mva=100.,f_hz=50.,
            bus=pd.DataFrame({'vn_kv':[110.,110.]}),
            gen=pd.DataFrame({'p_mw':[20.],'vm_pu':[1.]}),
            ext_grid=pd.DataFrame({'vm_pu':[1.]}),
            load=pd.DataFrame({'p_mw':[20.],'q_mvar':[5.]}))
        self.net=net; self._n_bus=2; self.N=1; self.power=20.

    def _configure_voltage_jacobian_net(self):
        net=copy.deepcopy(self.net); net.gen.p_mw.values[0]=self.power
        return net

    def reset(self):
        return self._reset_voltage_jacobian()


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.mod=SimpleNamespace(ARGS=SimpleNamespace(voltage_step_pu=.002),
            core=SimpleNamespace(GEN_VM_MAX=1.1,EXT_VM_MAX=1.1))
        modules=patch.dict('sys.modules',{'pandapower':SimpleNamespace(__version__='cache-unit-test')})
        modules.start()
        self.addCleanup(modules.stop)

    def install(self):
        return install_fast_voltage_jacobian(Environment(),self.mod,Path(self.directory.name))

    def test_reset_scope_memory_disk_and_changed_dispatch(self):
        with patch('reset_voltage_jacobian_fast.finite_voltage_jacobian',return_value=np.eye(2)) as compute:
            env=self.install(); env.reset(); self.assertEqual(compute.call_count,1)
            env.power=21.
            env._reset_voltage_jacobian()
            self.assertEqual(compute.call_count,1)
            env.reset(); self.assertEqual(compute.call_count,2)
            env.power=20.; env.reset()
            self.assertEqual(env.fast_voltage_stats[-1]['source'],'memory')
            another=self.install(); another.reset()
            self.assertEqual(another.fast_voltage_stats[-1]['source'],'disk')
            self.assertEqual(compute.call_count,2)

    def test_changed_reactive_profile_misses(self):
        with patch('reset_voltage_jacobian_fast.finite_voltage_jacobian',return_value=np.eye(2)) as compute:
            env=self.install(); env.reset()
            env.net.load.q_mvar.values[0]+=1
            env.reset(); self.assertEqual(compute.call_count,2)
            self.assertNotEqual(env.fast_voltage_stats[0]['context_key'],env.fast_voltage_stats[1]['context_key'])

    def test_invalid_checksum_is_recomputed(self):
        with patch('reset_voltage_jacobian_fast.finite_voltage_jacobian',return_value=np.eye(2)) as compute:
            self.install().reset()
            path=next(Path(self.directory.name).glob('*.npz'))
            with np.load(path) as archive: fields={k:archive[k].copy() for k in archive.files}
            fields['jacobian'][0,0]+=10
            np.savez_compressed(path,**fields)
            env=self.install(); env.reset()
            self.assertEqual(compute.call_count,2)
            self.assertEqual(env.fast_voltage_stats[-1]['source'],'computed')

    def test_accelerated_failure_retries_same_context(self):
        with patch('reset_voltage_jacobian_fast.finite_voltage_jacobian',side_effect=[RuntimeError('probe'),np.eye(2)]) as compute:
            env=self.install(); env.reset()
            self.assertEqual(compute.call_count,2)
            self.assertFalse(compute.call_args.kwargs['numba'])
            self.assertEqual(env.fast_voltage_stats[-1]['source'],'legacy_fallback')

    def test_both_paths_unavailable_keep_zero_feature_without_disk_cache(self):
        with patch('reset_voltage_jacobian_fast.finite_voltage_jacobian',side_effect=RuntimeError('no convergence')) as compute:
            env=self.install()
            self.assertTrue(np.array_equal(env.reset(),np.zeros((2,2))))
            self.assertEqual(compute.call_count,2)
            self.assertEqual(env.fast_voltage_stats[-1]['source'],'unavailable_zero')
            self.assertEqual(list(Path(self.directory.name).glob('*.npz')),[])


if __name__=='__main__': unittest.main()

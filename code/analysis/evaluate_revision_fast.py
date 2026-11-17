"""Opt-in SAC evaluation with the numerically checked reset-voltage backend.

Accepts evaluate_revision.py options. Defaults to validation and requires a new
output file. The original evaluator, checkpoints and reported results stay intact.
"""
import sys
from pathlib import Path
import numpy as np
from revision_common import CODE,read,write,sha
import evaluate_revision as legacy
from reset_voltage_jacobian_fast import VERSION,install_fast_voltage_jacobian


def option(arguments,name,default=None):
    return arguments[arguments.index(name)+1] if name in arguments else default


def main():
    arguments=sys.argv[1:]
    if option(arguments,'--method','sac')!='sac':
        raise ValueError('This opt-in entry point has been validated for SAC only')
    if '--scope' not in arguments: arguments+=['--scope','val']
    target=option(arguments,'--output')
    if target is None: raise ValueError('Provide --output with a new result path')
    output=Path(target).resolve()
    if output.exists(): raise ValueError('Use a new output file; previous records will not be overwritten')
    original_loader=legacy.load_aligned
    environments=[]
    def load(extra=()):
        mod=original_loader(extra)
        base=mod.CleanN1Env
        class FastEnv(base):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                install_fast_voltage_jacobian(self,mod,output.parent/(output.stem+'_voltage_cache'))
                environments.append(self)
                # The evaluator's untimed shape probe never supplies an actor
                # action. Defer actual context construction to the timed reset.
                actual_reset=self.reset
                probe_pending=[True]
                def reset(*a,**kw):
                    if not probe_pending[0]: return actual_reset(*a,**kw)
                    probe_pending[0]=False
                    feature=self._reset_voltage_jacobian
                    self._reset_voltage_jacobian=lambda:np.zeros((self._n_bus,self.N+1))
                    try: return actual_reset(*a,**kw)
                    finally: self._reset_voltage_jacobian=feature
                self.reset=reset
        mod.CleanN1Env=FastEnv
        return mod
    legacy.load_aligned=load
    sys.argv=[str(legacy.__file__),*arguments]
    legacy.main()
    payload=read(output)
    payload['voltage_backend']={'name':VERSION,'source_sha256':sha(CODE/'reset_voltage_jacobian_fast.py'),
        'entrypoint_sha256':sha(__file__),'cache_key':'Full configured network and reset dispatch, not demand alone',
        'timing':'Per-case first-use sensitivity is included. Initial shape-probe feature is unused by the actor.',
        'calls':[item for env in environments for item in env.fast_voltage_stats]}
    write(output,payload)


if __name__=='__main__': main()

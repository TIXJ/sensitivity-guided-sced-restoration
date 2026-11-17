"""Short-policy deployment with online implicit voltage derivatives.

Reuse the frozen physical evaluator, but explicitly change its voltage feature.
Selection uses the development validation cases; holdout outcomes stay unseen.
"""
import argparse
import copy
import shutil
import time
from types import MethodType
import numpy as np
import torch
import evaluate_short_policy as base
from revision_common import RESULTS, CHECKPOINTS, load_aligned, read, write, sha
from probe_voltage_kernel_v5 import implicit_voltage
from reset_voltage_jacobian_fast import finite_voltage_jacobian

DEST=RESULTS/'implicit_short_policy_v2'


def setup(samples,data,bank):
    mod=load_aligned()
    envs={key:mod.CleanN1Env(samples,list(range(len(samples))),data) for key in ('direct','short4','short8','selected')}
    env=envs['direct']
    original=env._reset_voltage_jacobian
    env._reset_voltage_jacobian=lambda:np.zeros((env._n_bus,env.N+1))
    obs=env.reset(0)
    env._reset_voltage_jacobian=original
    pdim=env._n_shift_dirs+env._n_key_gens
    mod.core.SAC_LOG_STD_MAX_VECTOR=np.r_[np.full(pdim,mod.ARGS.active_log_std_max),
        np.full(env.action_dim-pdim,mod.ARGS.voltage_log_std_max)].astype(np.float32)
    actor=mod.BranchedGaussianActor(len(obs),pdim,env.action_dim-pdim).to(mod.core.DEVICE)
    actor.load_state_dict(torch.load(CHECKPOINTS['sac'],map_location=mod.core.DEVICE,weights_only=True));actor.eval()
    with torch.no_grad(): actor.select_action(torch.as_tensor(obs,device=mod.core.DEVICE).unsqueeze(0),deterministic=True)
    for env in envs.values():
        install(env,mod)
        install_case_memo(env)
    return mod,envs,actor


def install(env,mod):
    state={'matrix':None}
    original_reset=env.reset
    env.implicit_feature_calls=[]
    def reset(self,*args,**kwargs):
        state['matrix']=None
        return original_reset(*args,**kwargs)
    def matrix(self):
        if state['matrix'] is not None: return state['matrix']
        start=time.perf_counter();net=self._configure_voltage_jacobian_net()
        source='implicit';error=None
        try:
            value=implicit_voltage(net)
            if value.shape!=(self._n_bus,self.N+1) or not np.isfinite(value).all():
                raise ValueError('Invalid implicit matrix')
        except Exception as exc:
            source='finite_fallback';error=str(exc)
            value=finite_voltage_jacobian(net,mod.ARGS.voltage_step_pu,mod.core.GEN_VM_MAX,mod.core.EXT_VM_MAX,recycle=True)
        state['matrix']=value
        self.implicit_feature_calls.append({'source':source,'runtime_sec':time.perf_counter()-start,'error':error})
        return value
    env.reset=MethodType(reset,env)
    env._reset_voltage_jacobian=MethodType(matrix,env)


def install_case_memo(env):
    original=env._run_ac_pf_check
    def checked(self):
        cache=getattr(self,'_case_pf_memo',{})
        key=(np.asarray(self.current_power).tobytes(),np.asarray(self.current_gen_vm_pu).tobytes(),
             np.asarray(self.current_ext_grid_vm_pu).tobytes(),float(self.target_demand))
        if not getattr(self,'_verification_fresh',False) and key in cache:
            return copy.deepcopy(cache[key])
        result=original()
        if result.get('converged') and not result.get('timeout'):
            cache[key]=copy.deepcopy(result)
        self._case_pf_memo=cache
        return result
    env._run_ac_pf_check=MethodType(checked,env)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','val','freeze','test'))
    args=parser.parse_args()
    base.DEST=DEST;base.setup=setup
    original_evaluate=base.evaluate
    def measured(env,*args,**kwargs):
        env._case_pf_memo={}
        before=len(env.implicit_feature_calls)
        row=original_evaluate(env,*args,**kwargs)
        row['voltage_backend_calls']=env.implicit_feature_calls[before:]
        return row
    base.evaluate=measured
    if args.stage=='prepare':
        if DEST.exists(): raise RuntimeError('Preserve prior variant')
        DEST.mkdir()
        for scope in ('val','test'):
            shutil.copy2(RESULTS/'short_policy_deployment_v1'/(scope+'_conditions.json'),DEST/(scope+'_conditions.json'))
        write(DEST/'bank.json',{'complete':True,'rows':[],
            'scope':'No offline feature bank: online implicit derivative, calculated on each policy reset and fully timed.'})
        return
    hashes={str(p):sha(p) for p in (__file__, base.__file__,
        __import__('probe_voltage_kernel_v5').__file__)}
    if args.stage=='freeze':
        base.freeze()
        protocol=read(DEST/'protocol.json');protocol['runner_hashes']=hashes
        protocol['interpretation']='New analytic-feature/short-attempt deployment variant. Not exact finite differences or original-policy trajectory equivalence. All physical constraints, fallback initialization and final checks unchanged.'
        write(DEST/'protocol.json',protocol)
    else:
        if args.stage=='test':
            assert hashes==read(DEST/'protocol.json')['runner_hashes']
        base.run(args.stage)
        result=read(DEST/(args.stage+'.json'))
        result['runner_hashes']=hashes
        result['voltage_feature']='online implicit AC derivative; fresh calculation cost included; original finite differences used on error'
        write(DEST/(args.stage+'.json'),result)


if __name__=='__main__': main()

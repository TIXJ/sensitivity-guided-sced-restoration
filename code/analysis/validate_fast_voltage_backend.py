"""Validation-only paired timing and numerical checks for the optional backend."""
import argparse
import copy
import os
from pathlib import Path
import time
import numpy as np
import torch
from revision_common import RESULTS,CHECKPOINTS,CODE,DATASET,SPLIT,CANDIDATE_SPLIT,read,write,load_aligned,sha
from diagnose_runtime_and_tight_validation import select
from evaluate_independent_conditions import pure_policy
from reset_voltage_jacobian_fast import install_fast_voltage_jacobian,context_fingerprint

DEST=RESULTS/'fast_voltage_validation_20261018'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',default='numba_v1')
    parser.add_argument('--all-validation',action='store_true')
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    out=DEST/args.tag
    if out.exists(): raise RuntimeError('Output exists; preserve prior run')
    out.mkdir(parents=True)
    os.environ.setdefault('NUMBA_CACHE_DIR',str(RESULTS/'numba_cache'))
    data,samples,ids=select()
    if args.all_validation:
        retained=read(DATASET)['samples']
        wanted={retained[i]['sample_id'] for i in read(SPLIT)['val_ids']}
        upstream=read(CANDIDATE_SPLIT)
        ids=sorted((i for i in upstream['val_ids'] if samples[i]['sample_id'] in wanted),
                   key=lambda i:samples[i]['sample_id'])
        assert len(ids)==25 and not set(ids).intersection(upstream['train_ids']+upstream['test_ids'])
    assert args.repeats>0
    mod=load_aligned()
    prototype=mod.CleanN1Env(samples,ids,data)
    prototype._voltage_jacobian_dir=out/'setup'; prototype._voltage_jacobian_dir.mkdir()
    obs=prototype.reset(ids[0]); dim=prototype._n_shift_dirs+prototype._n_key_gens
    mod.core.SAC_LOG_STD_MAX_VECTOR=np.r_[np.full(dim,mod.ARGS.active_log_std_max),
        np.full(prototype.action_dim-dim,mod.ARGS.voltage_log_std_max)].astype(np.float32)
    actor=mod.BranchedGaussianActor(len(obs),dim,prototype.action_dim-dim).to(mod.core.DEVICE)
    actor.load_state_dict(torch.load(CHECKPOINTS['sac'],map_location=mod.core.DEVICE,weights_only=True)); actor.eval()
    with torch.no_grad(): actor.select_action(torch.as_tensor(obs,device=mod.core.DEVICE).unsqueeze(0),deterministic=True)
    protocol={'sample_ids':[samples[i]['sample_id'] for i in ids],'scope':'Preselected retained validation IDs only',
        'repeats':args.repeats,'timing':'Fresh empty per-run caches; complete reset/rollout/final checks. A shared prototype reset and actor warmup precede timing; any remaining backend-specific first-use compilation is included. Rotated serial arm order. No changed model, action or reward.',
        'backend_sha256':sha(CODE/'reset_voltage_jacobian_fast.py'),'script_sha256':sha(__file__),
        'checkpoint_sha256':sha(CHECKPOINTS['sac'])}
    payload={'complete':False,'protocol':protocol,'rows':[],'comparisons':[]}
    write(out/'results.json',payload)
    for repeat in range(args.repeats):
        for ordinal,sid in enumerate(ids):
            observations={}; matrices={}; actions={}; results={}
            for arm in (('legacy','fast') if (repeat+ordinal)%2==0 else ('fast','legacy')):
                env=mod.CleanN1Env(samples,ids,data)
                folder=out/'cache'/str(repeat)/samples[sid]['sample_id']/arm
                if arm=='fast': install_fast_voltage_jacobian(env,mod,folder)
                else:
                    folder.mkdir(parents=True); env._voltage_jacobian_dir=folder
                    env._voltage_jacobian_memory={}
                sequence=[]; asequence=[]
                old_obs=env._get_obs; old_action=actor.select_action
                def capture_obs():
                    value=old_obs(); sequence.append(value.copy()); return value
                def capture_action(*a,**kw):
                    value=old_action(*a,**kw); asequence.append(value.detach().cpu().numpy().copy()); return value
                env._get_obs=capture_obs; actor.select_action=capture_action
                try:
                    row=pure_policy(env,mod,actor,samples[sid],sid)
                finally: actor.select_action=old_action
                matrices[arm]=env._reset_voltage_jacobian().copy()
                observations[arm]=np.asarray(sequence); actions[arm]=np.asarray(asequence)
                row.update(arm=arm,repeat=repeat)
                if arm=='fast':
                    row['backend_stats']=env.fast_voltage_stats
                    assert all(v['source']=='computed' and v['fallback'] is None for v in env.fast_voltage_stats)
                results[arm]=row; payload['rows'].append(row)
                print('RUN',repeat,samples[sid]['sample_id'],arm,row['passed'],row['runtime_sec'],flush=True)
            comparison={'sample_id':samples[sid]['sample_id'],'repeat':repeat,
                'jacobian_max_abs_error':float(np.max(abs(matrices['legacy']-matrices['fast']))),
                'observation_max_abs_error':float(np.max(abs(observations['legacy']-observations['fast']))),
                'action_max_abs_error':float(np.max(abs(actions['legacy']-actions['fast'])))}
            assert results['legacy']['passed']==results['fast']['passed']
            assert results['legacy']['policy_steps']==results['fast']['policy_steps']
            assert np.allclose(matrices['legacy'],matrices['fast'],atol=1e-8,rtol=1e-7),comparison
            assert comparison['observation_max_abs_error']<1e-6,comparison
            assert comparison['action_max_abs_error']<1e-6,comparison
            for field in ('final_dispatch_mw','final_generator_vm_pu','final_reference_vm_pu'):
                assert np.allclose(results['legacy'][field],results['fast'][field],atol=1e-6,rtol=0),field
            payload['comparisons'].append(comparison); write(out/'results.json',payload)
    # Same total demand with changed spatial/reactive distribution must miss.
    net=prototype._configure_voltage_jacobian_net()
    def key(n): return context_fingerprint(n,mod.ARGS.voltage_step_pu,mod.core.GEN_VM_MAX,mod.core.EXT_VM_MAX)
    original=key(net)
    changed=copy.deepcopy(net); changed.load.p_mw.iloc[0]+=1; changed.load.p_mw.iloc[1]-=1
    assert key(changed)!=original
    changed=copy.deepcopy(net); changed.load.q_mvar.iloc[0]+=.01
    assert key(changed)!=original
    changed=copy.deepcopy(net); changed.gen.p_mw.iloc[0]+=.01
    assert key(changed)!=original
    changed=copy.deepcopy(net); changed.line.in_service.iloc[0]=not changed.line.in_service.iloc[0]
    assert key(changed)!=original
    changed=copy.deepcopy(net); changed.res_bus.loc[:,'vm_pu']=.5
    assert key(changed)==original
    payload['cache_invalidation_checks']=['spatial_active_load','reactive_load','reference_dispatch','topology','ignore_result_tables']
    payload['complete']=True; write(out/'results.json',payload)


if __name__=='__main__': main()

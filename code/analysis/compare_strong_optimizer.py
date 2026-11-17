"""Frozen four-arm, three-repeat comparison of flat/PF initialized optimizers."""
import os
import time
from pathlib import Path
import numpy as np
import torch
import compare_joint_workflows as common
from compare_joint_gate_workflows import training_range
from revision_common import RESULTS, CANDIDATES, CANDIDATE_SPLIT, CHECKPOINTS, load_aligned, read, write, sha

DEST = RESULTS/'strong_optimizer_v1'
ARMS = ('direct_flat', 'direct_pf', 'gated_flat', 'gated_pf')


def main():
    os.environ.setdefault('NUMBA_CACHE_DIR', str(RESULTS/'numba_cache'))
    DEST.mkdir(exist_ok=True)
    protocol = {'arms':list(ARMS),'repetitions':3,'range_mw':list(training_range()),
                'hashes':{**common.hashes(),str(Path(__file__)):sha(__file__)},
                'scope':'Same historical 75 cases. Rotated serial order. Three paired repeats. All failed cases and PF initialization counted. No restarts or result-dependent selection.',
                'budget':'8 rounds; 100 interior-point iterations per round; 0.0001 pu solver-only voltage inset; original verifier'}
    pp_path=DEST/'protocol.json'
    if pp_path.exists():
        assert read(pp_path)==protocol
    else:
        write(pp_path,protocol)
    output=DEST/'results.json'
    payload=read(output) if output.exists() else {'complete':False,'protocol':protocol,'rows':[]}
    assert payload['protocol']==protocol
    if payload['complete']:
        print('Completed strong optimizer comparison retained')
        return
    data=read(CANDIDATES)
    samples=[s for s in data['samples'] if s.get('is_complete_sced',True)]
    ids=read(CANDIDATE_SPLIT)['test_ids']
    mod=load_aligned()
    envs={a:mod.CleanN1Env(samples,ids,data) for a in ARMS}
    for env in envs.values():
        env._voltage_jacobian_dir=RESULTS/'jacobian_cache/none'
    env=envs['gated_pf']; obs=env.reset(ids[0])
    pdim=env._n_shift_dirs+env._n_key_gens
    mod.core.SAC_LOG_STD_MAX_VECTOR=np.r_[np.full(pdim,mod.ARGS.active_log_std_max),np.full(env.action_dim-pdim,mod.ARGS.voltage_log_std_max)].astype(np.float32)
    actor=mod.BranchedGaussianActor(len(obs),pdim,env.action_dim-pdim).to(mod.core.DEVICE)
    actor.load_state_dict(torch.load(CHECKPOINTS['sac'],map_location=mod.core.DEVICE,weights_only=True)); actor.eval()
    for arm in ARMS:
        for sid in [next(i for i in ids if samples[i]['demand_mw']==d) for d in sorted({samples[i]['demand_mw'] for i in ids})]:
            envs[arm].reset(sid)
    with torch.no_grad():
        actor.select_action(torch.as_tensor(obs,device=mod.core.DEVICE).unsqueeze(0),deterministic=True)
    import pandapower as pp
    original=pp.runopp
    done={(r['repetition'],r['sample_id'],r['arm']) for r in payload['rows']}
    for rep in range(3):
        for j,sid in enumerate(ids):
            offset=(j+rep)%4
            for arm in ARMS[offset:]+ARMS[:offset]:
                if (rep,samples[sid]['sample_id'],arm) in done:
                    continue
                def configured(net,**kwargs):
                    if arm.endswith('_pf') and kwargs.get('init')=='flat':
                        kwargs['init']='pf'
                    return original(net,**kwargs)
                pp.runopp=configured
                t0=time.perf_counter()
                gated=arm.startswith('gated')
                inside=protocol['range_mw'][0]<=samples[sid]['demand_mw']<=protocol['range_mw'][1]
                effective='gat_sac_optimizer' if gated and inside else 'gat_optimizer'
                gate_sec=time.perf_counter()-t0
                try:
                    row=common.evaluate(envs[arm],mod,actor,samples[sid],sid,data['linear_security_model'],effective)
                finally:
                    pp.runopp=original
                row.update(arm=arm,repetition=rep,range_gate_sec=gate_sec)
                row['runtime_sec']+=gate_sec
                payload['rows'].append(row)
                write(output,payload)
            print(f'repeat={rep+1}/3 case={j+1}/75 {samples[sid]["sample_id"]}',flush=True)
    assert len(payload['rows'])==900
    payload['complete']=True
    write(output,payload)


if __name__=='__main__':
    main()

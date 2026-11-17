"""Three-arm comparison including a training-range gate selected on validation."""
import argparse
import os
import time
from pathlib import Path
import numpy as np
import torch

import compare_joint_workflows as common
from revision_common import CANDIDATES, CANDIDATE_SPLIT, CHECKPOINTS, DATASET, SPLIT, HERE, RESULTS, load_aligned, read, sha, write

DEST = RESULTS/'joint_workflow_comparison_v2'
ARMS = (*common.ARMS, 'gat_range_gated_sac_optimizer')


def training_range():
    samples = read(DATASET)['samples']
    ids = read(SPLIT)['train_ids']
    demands = [float(samples[i]['demand_mw']) for i in ids]
    return min(demands), max(demands)


def hashes():
    return {**common.hashes(), str(Path(__file__)): sha(__file__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=('val','test'), default='val')
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('NUMBA_CACHE_DIR', str(RESULTS/'numba_cache'))
    provenance = hashes()
    bounds = training_range()
    protocol_path = DEST/'protocol.json'
    if args.freeze:
        if protocol_path.exists():
            parser.error('Protocol already frozen')
        validation = read(DEST/'val.json')
        assert validation['complete'] and validation['hashes']==provenance
        write(protocol_path, {'hashes': provenance, 'validation_sha256':sha(DEST/'val.json'),
                              'training_demand_range_mw':bounds, 'arms':ARMS,
                              'objective':'Same squared distance from GAT for every invoked optimizer',
                              'budget':'8 cut rounds; 100 interior-point iterations per round; no restarts',
                              'development':'Ungated two-arm validation showed no aggregate speed gain. '
                                            'The additional gate uses min/max retained training demand, not test outcomes. '
                                            'Original ungated arms are retained and remeasured alongside the gated arm.'})
        print('Three-arm protocol frozen before test.', flush=True)
        return
    if args.scope=='test':
        protocol = read(protocol_path)
        assert protocol['hashes']==provenance and protocol['validation_sha256']==sha(DEST/'val.json')
        assert list(bounds)==protocol['training_demand_range_mw']
    path = DEST/(args.scope+'.json')
    if path.exists():
        saved = read(path)
        if saved.get('complete') and saved['hashes']==provenance:
            print('Completed output preserved:',path)
            return
        parser.error('Output exists; preserve before restarting.')
    data = read(CANDIDATES)
    samples = [s for s in data['samples'] if s.get('is_complete_sced',True)]
    ids = read(CANDIDATE_SPLIT)[args.scope+'_ids']
    mod = load_aligned()
    envs = {a:mod.CleanN1Env(samples,ids,data) for a in ARMS}
    for env in envs.values():
        env._voltage_jacobian_dir = RESULTS/'jacobian_cache/none'
    env = envs[ARMS[1]]
    obs = env.reset(ids[0])
    pdim = env._n_shift_dirs+env._n_key_gens
    mod.core.SAC_LOG_STD_MAX_VECTOR = np.r_[np.full(pdim,mod.ARGS.active_log_std_max),
        np.full(env.action_dim-pdim,mod.ARGS.voltage_log_std_max)].astype(np.float32)
    actor = mod.BranchedGaussianActor(len(obs),pdim,env.action_dim-pdim).to(mod.core.DEVICE)
    actor.load_state_dict(torch.load(CHECKPOINTS['sac'],map_location=mod.core.DEVICE,weights_only=True))
    actor.eval()
    # Identical offline feature preparation for the two SAC-capable arms.
    for arm in ARMS[1:]:
        warmed = set()
        for sid in ids:
            demand = samples[sid]['demand_mw']
            if demand not in warmed:
                envs[arm].reset(sid)
                warmed.add(demand)
    with torch.no_grad():
        actor.select_action(torch.as_tensor(obs,device=mod.core.DEVICE).unsqueeze(0),deterministic=True)
    payload = {'complete':False,'scope':args.scope,'arms':ARMS,'hashes':provenance,
               'protocol_sha256':sha(protocol_path) if args.scope=='test' else None,
               'training_demand_range_mw':bounds,'sample_ids':[samples[i]['sample_id'] for i in ids],
               'timing_scope':'Resident setup and fixed-feature warm-up excluded; all per-case stages '
                               'and final checks included, including failed cases. Three-arm order rotates.',
               'rows':[]}
    write(path,payload)
    for i,sid in enumerate(ids):
        for arm in ARMS[i%3:]+ARMS[:i%3]:
            t0 = time.perf_counter()
            outside = not(bounds[0]<=samples[sid]['demand_mw']<=bounds[1])
            effective = common.ARMS[0] if arm==ARMS[2] and outside else (common.ARMS[1] if arm==ARMS[2] else arm)
            gate_sec = time.perf_counter()-t0
            row = common.evaluate(envs[arm],mod,actor,samples[sid],sid,data['linear_security_model'],effective)
            row.update(arm=arm,range_gate_outside=outside,range_gate_skipped_policy=(arm==ARMS[2] and outside),
                       range_gate_sec=gate_sec)
            row['runtime_sec'] += gate_sec
            payload['rows'].append(row)
            write(path,payload)
            print(f"{i+1}/{len(ids)} {samples[sid]['sample_id']} {arm} pass={row['passed']} opt={row['optimizer_called']} time={row['runtime_sec']:.3f}s",flush=True)
    payload['complete']=True
    write(path,payload)


if __name__=='__main__':
    main()

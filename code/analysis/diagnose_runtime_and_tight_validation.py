"""Read-only timing audit plus preselected validation-only feasibility probes.

No training, test-based tuning, modified checkpoints or changed verifier limits.
Local repair failure is inconclusive, not a certificate of infeasibility.
"""
import argparse
import os
import time
from collections import defaultdict
import numpy as np
import torch
from revision_common import RESULTS, CANDIDATES, CANDIDATE_SPLIT, DATASET, SPLIT, CHECKPOINTS, read, write, sha, load_aligned
from evaluate_independent_conditions import pure_policy
from pilot_validation_feasibility import minimum_linear_dispatch, bounded_voltage_repair
import compare_joint_workflows as common

DEST = RESULTS / 'validation_diagnostics_20261018'


def timing_audit():
    source = RESULTS / 'independent_conditions_v1/results.json'
    data = read(source)
    keys = ('preparation_sec', 'prediction_projection_sec', 'initial_verification_sec',
            'policy_and_gate_sec', 'optimizer_stage_sec', 'final_verification_sec')
    summary = {}
    for arm in ('direct_pf', 'gated_pf'):
        rows = [r for r in data['rows'] if r['arm'] == arm]
        assert len(rows) == 100
        parts = {k: sum(r.get(k, 0.) for r in rows)/100 for k in keys}
        parts['failed_preparation_sec'] = sum(r['runtime_sec'] for r in rows if r['preparation_failed'])/100
        total = sum(r['runtime_sec'] for r in rows)/100
        assert abs(sum(parts.values()) - total) < 1e-8
        summary[arm] = {'passed': sum(r['passed'] for r in rows), 'mean_total_sec': total,
                        'optimizer_calls': sum(r['optimizer_called'] for r in rows),
                        'components_mean_all_100_sec': parts}
    write(DEST/'existing_timing_audit.json', {'source_sha256': sha(source), 'summary': summary,
        'scope': 'Existing fixed independent evaluation; no reruns or tuning. Pure-policy records lack equivalent fine-grained instrumentation.'})
    print(summary, flush=True)


def select():
    data = read(CANDIDATES)
    samples = data['samples']
    retained = read(DATASET)['samples']
    retained_val = {retained[i]['sample_id'] for i in read(SPLIT)['val_ids']}
    split = read(CANDIDATE_SPLIT)
    ids = [min((i for i in split['val_ids'] if samples[i]['sample_id'] in retained_val
                and samples[i]['demand_mw'] == d), key=lambda i: samples[i]['sample_id'])
           for d in (16000., 16200., 16400.)]
    assert not set(ids).intersection(split['train_ids'] + split['test_ids'])
    return data, samples, ids


def profile(env, mod, samples, ids):
    dim = env._n_shift_dirs + env._n_key_gens
    env._voltage_jacobian_dir = DEST/'shape_cache'
    env._voltage_jacobian_dir.mkdir(exist_ok=True)
    obs = env.reset(ids[0])
    mod.core.SAC_LOG_STD_MAX_VECTOR = np.r_[np.full(dim,mod.ARGS.active_log_std_max),
        np.full(env.action_dim-dim,mod.ARGS.voltage_log_std_max)].astype(np.float32)
    actor = mod.BranchedGaussianActor(len(obs),dim,env.action_dim-dim).to(mod.core.DEVICE)
    actor.load_state_dict(torch.load(CHECKPOINTS['sac'],map_location=mod.core.DEVICE,weights_only=True))
    actor.eval()
    with torch.no_grad():
        actor.select_action(torch.as_tensor(obs,dtype=torch.float32,device=mod.core.DEVICE).unsqueeze(0),deterministic=True)
    import pandapower as pp
    rows = []
    for sid in ids:
        env._voltage_jacobian_dir = DEST/'profile_cache'/samples[sid]['sample_id']
        env._voltage_jacobian_dir.mkdir(parents=True)
        env._voltage_jacobian_memory = {}
        for mode in ('cold_context', 'same_context_memory_hit'):
            totals = defaultdict(float)
            counts = defaultdict(int)
            depth = [False]
            originals = []
            def wrap(obj, name, label, jacobian=False):
                old = getattr(obj,name)
                def measured(*args, **kwargs):
                    common.sync()
                    started = time.perf_counter()
                    was = depth[0]
                    if jacobian: depth[0] = True
                    try:
                        return old(*args, **kwargs)
                    finally:
                        common.sync()
                        totals[label] += time.perf_counter()-started
                        counts[label] += 1
                        if label == 'all_runpp' and was: counts['runpp_in_jacobian'] += 1
                        depth[0] = was
                originals.append((obj,name,old))
                setattr(obj,name,measured)
            wrap(env,'_reset_voltage_jacobian','jacobian',True)
            wrap(env,'_run_ac_pf_check','environment_ac_check')
            wrap(actor,'select_action','actor')
            wrap(pp,'runpp','all_runpp')
            try:
                row = pure_policy(env,mod,actor,samples[sid],sid)
            finally:
                for obj,name,old in reversed(originals): setattr(obj,name,old)
            row.update(mode=mode, measured_inclusive_sec=dict(totals), calls=dict(counts))
            rows.append(row)
            write(DEST/'validation_profile.json', {'complete':False,'rows':rows,
                'timing_note':'Instrumented validation diagnostic with synchronization overhead. Nested times must not be summed; warm run reuses exactly the same sample/context. Not a new test-speed result.'})
            print('PROFILE',row['sample_id'],mode,row['runtime_sec'],dict(totals),dict(counts),flush=True)
    write(DEST/'validation_profile.json', {'complete':True,'rows':rows,
        'timing_note':'Nested instrumentation; not additive. Same sample and context in paired warm run; do not reuse by demand across changed profiles.'})


def reachability(env,mod,data,samples,ids):
    basis = env._build_orthogonal_physics_action_basis()
    rows = []
    for sid in ids:
        env.current_sample = samples[sid]
        demand = float(samples[sid]['demand_mw'])
        p0 = mod.core.project_box_sum(env._predict_gat(demand),env.PMIN,env.PMAX*.99,demand)
        dispatch, linear = minimum_linear_dispatch(p0,env.PMIN,env.PMAX*.99,basis,
            data['linear_security_model'],radius_center=p0,radius=39.999,timeout=45.)
        row = {'sample_id':samples[sid]['sample_id'],'linear':linear,'initial_dispatch_mw':p0.tolist()}
        if dispatch is not None:
            delta = dispatch-p0
            row.update(final_dispatch_mw=dispatch.tolist(),norm_mw=float(np.linalg.norm(delta)),
                       offspan_mw=float(np.linalg.norm(delta-basis@(basis.T@delta))))
            row['voltage_probes'] = {}
            for label,low,high in (('original',.933,1.090),('tight',.94,1.08)):
                mod.core.RL_VMIN_PU, mod.core.RL_VMAX_PU = low,high
                result = bounded_voltage_repair(env,mod,dispatch,demand,seconds=45.,calls=600)
                row['voltage_probes'][label] = result
                print('REACHABILITY',row['sample_id'],label,result['passed'],result['best_residual_squared'],flush=True)
        rows.append(row)
        write(DEST/'validation_reachability.json', {'complete':False,'rows':rows})
    write(DEST/'validation_reachability.json', {'complete':True,'rows':rows,
        'scope':'Validation only, 32D and strictly inside 40 MW; original device/cumulative voltage bounds. Fixed minimum-linear-norm P followed by bounded local voltage repair, 45 seconds/600 calls each band. A passing point is a witness; failure does not exclude other P/V points or certify infeasibility. No test cases or rescue labels used.'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('profile','reachability'),required=True)
    args = parser.parse_args()
    DEST.mkdir(exist_ok=True)
    output = DEST/('validation_profile.json' if args.stage=='profile' else 'validation_reachability.json')
    if output.exists(): raise RuntimeError('Preserve existing output; do not silently overwrite or resume incomplete probes')
    timing_audit()
    data,samples,ids = select()
    protocol = {'selection':'First retained validation ID at each of 16.0,16.2,16.4 GW, selected before solving',
        'sample_ids':[samples[i]['sample_id'] for i in ids],
        'hashes':{**common.hashes(),__file__:sha(__file__)},
        'no_training_or_test_tuning':True}
    path = DEST/'protocol.json'
    if path.exists(): assert read(path)==protocol
    else: write(path,protocol)
    os.environ.setdefault('NUMBA_CACHE_DIR',str(RESULTS/'numba_cache'))
    mod = load_aligned()
    env = mod.CleanN1Env(samples,ids,data)
    if args.stage=='profile': profile(env,mod,samples,ids)
    else: reachability(env,mod,data,samples,ids)


if __name__ == '__main__': main()

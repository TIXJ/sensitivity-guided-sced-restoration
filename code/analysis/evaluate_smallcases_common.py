"""Matched small-system timing from reset through independent final verification."""
import argparse
import copy
import importlib.util
import os
import time
from pathlib import Path

import numpy as np
from revision_common import CODE, HERE, ROOT, RESULTS, read, sha, write
import sys
from smallcase_evaluation_contract import accepted, constraint_flags, verify_state


def load_case(case):
    import torch
    if case == 39:
        spec = importlib.util.spec_from_file_location('train_gat39_v1', HERE/'legacy_dependencies/train_gat39_v1.py')
        gat = importlib.util.module_from_spec(spec)
        sys.modules['train_gat39_v1'] = gat
        spec.loader.exec_module(gat)
        import train_rl39_v1 as mod
        import eval_case39_action_diagnosis as diagnosis
        dataset = ROOT/'case39_routeb_rescuable_dataset.json'
        samples = read(dataset)['samples']
        _, _, ids = mod.split_samples(samples)
        env = mod.Env39(samples, ids)
        checkpoint = ROOT/'rl39_routeb_main/best_model.pth'
        trainer = mod.SACTrainer(env._get_obs_dim(), env.action_dim)
        diagnosis._load_actor(trainer, checkpoint)
        action = lambda obs: diagnosis._policy_action(trainer, env, obs, 'full')[1]
    else:
        import train_rl118_v1 as mod
        dataset = ROOT/'case118_routeb_rescuable_dataset.json'
        samples = [s for s in read(dataset)['samples'] if s.get('is_complete_sced')]
        _, _, ids = mod.split_samples_stratified(samples)
        env = mod.SCEDEnv118(samples, ids)
        checkpoint = ROOT/'rl118_routeb_main/best_model.pth'
        actor = mod.GaussianActor(env._get_obs_dim(), env.action_dim).to(mod.DEVICE)
        actor.load_state_dict(torch.load(checkpoint, map_location=mod.DEVICE, weights_only=False)['actor'])
        actor.eval()
        def action(obs):
            raw = actor.select_action(obs, deterministic=True)
            low, high = env.get_action_bounds()
            return low + (raw + 1.) * .5 * (high - low)
    mod.AC_PF_CHECK_EVERY = mod.N1_CHECK_EVERY_STEPS = 1
    return mod, samples, ids, env, action, dataset, checkpoint


def rank(info):
    flags = constraint_flags(info)
    return (not accepted(info), flags['acpf_not_converged'],
            sum(int(info.get(k, 999)) for k in ('v_violations', 'overloaded_lines', 'n1_violations')),
            float(info.get('v_violation_max_pu') or 0) + float(info.get('n1_overload_mw') or 0))


def evaluate(env, mod, case, sid, mode, action):
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    obs = env.reset(sid)
    after_reset = time.perf_counter()
    initial = verify_state(env, mod, case)
    after_initial = time.perf_counter()
    steps, calls = 0, 0
    if not initial['passed'] and mode != 'prediction_only':
        for step in range(1, env.max_steps + 1):
            if mode == 'policy':
                obs, _, done, _, info = env.step(action(obs))
                calls += 1
            elif mode == 'myopic_search':
                _, high = env.get_action_bounds()
                pdim = env._n_shift_dirs + env._n_key_gens
                candidates = [np.zeros(env.action_dim)]
                for sign in (-1, 1):
                    voltage = np.zeros(env.action_dim)
                    voltage[pdim:] = sign * .5
                    candidates.append(voltage * high)
                    active = np.zeros(env.action_dim)
                    active[(step - 1) % pdim] = sign * .5
                    candidates.append(active * high)
                evaluated = []
                for control in candidates:
                    trial = copy.deepcopy(env)
                    _, _, _, _, info = trial.step(control)
                    evaluated.append((rank(info), trial, info))
                _, env, info = min(evaluated, key=lambda row: row[0])
                calls += len(candidates)
                done = accepted(info)
            else:
                raise ValueError(mode)
            steps = step
            if done:
                break
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    after_correction = time.perf_counter()
    final = verify_state(env, mod, case)
    ended = time.perf_counter()
    return {'mode': mode, 'passed': final['passed'], 'reason': final['primary_reason'],
            'steps': steps, 'initial_passed': initial['passed'], 'runtime_sec': ended-started,
            'reset_prediction_sec': after_reset-started,
            'initial_verification_sec': after_initial-after_reset,
            'correction_sec': after_correction-after_initial,
            'final_verification_sec': ended-after_correction,
            'in_loop_checks': calls, 'final_info': final, 'initial_info': initial,
            'constraint_flags': final['constraint_flags'], 'dispatch_mw': env.current_power.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=int, choices=(39, 118), required=True)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    os.environ.setdefault('NUMBA_CACHE_DIR', str(RESULTS/'numba_cache'))
    if os.path.exists(args.output):
        parser.error('Output exists; preserve it and use a new output path.')
    mod, samples, ids, env, action, dataset, checkpoint = load_case(args.case)
    if args.max_samples:
        ids = ids[:args.max_samples]
    # Common setup/warm-up is excluded, while every measured reset is included.
    obs = env.reset(ids[0])
    action(obs)
    payload = {'complete': False, 'contract': 'common_smallcase_v2', 'settings': vars(args),
               'timing_scope': 'Resident model/network setup excluded; reset/prediction, initial checks, '
                               'correction and independent final checks included for every method. '
                               'Initially accepted points skip correction; all failures retained.',
               'input_hashes': {str(p): sha(p) for p in (dataset, checkpoint, CODE/f'train_rl{args.case}_v1.py',
                                                       CODE/'smallcase_evaluation_contract.py', HERE/Path(__file__).name)},
               'rows': []}
    modes = ('prediction_only', 'myopic_search', 'policy')
    for i, sid in enumerate(ids):
        # Rotate method order to avoid always assigning first-call effects to one method.
        order = modes[i % 3:] + modes[:i % 3]
        for mode in order:
            row = evaluate(env, mod, args.case, sid, mode, action)
            row.update(case=args.case, sample_id=str(samples[sid].get('sample_id', samples[sid].get('id', sid))),
                       sid=sid, demand_mw=samples[sid]['demand_mw'])
            payload['rows'].append(row)
            write(args.output, payload)
            print(f"{i+1}/{len(ids)} {args.case} {mode}: {row['passed']} {row['runtime_sec']:.3f}s", flush=True)
    payload['complete'] = True
    write(args.output, payload)


if __name__ == '__main__':
    main()

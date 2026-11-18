"""Repeat the original small-case policies without changing their data partitions."""
import argparse
from datetime import datetime, timezone
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REV = HERE.parent
PROJECT = REV.parents[1]
OUTPUT = REV / 'experiment_results/smallcase_multiseed_v1'


def freeze_split(mod, case, samples):
    name = 'split_samples' if case == 39 else 'split_samples_stratified'
    old_seed = mod.SEED
    try:
        mod.SEED = 42
        partitions = tuple(tuple(ids) for ids in getattr(mod, name)(samples))
    finally:
        mod.SEED = old_seed
    return lambda *_args, **_kwargs: tuple(list(ids) for ids in partitions)


def validation_improves(candidate, best):
    return best is None or (candidate['passed'], candidate['avg_reward']) > (best['passed'], best['avg_reward'])


def validation_subset(case, ids):
    return list(ids[:15 if case == 39 else 20])


def evaluation_ids(ids, updates):
    return list(ids) if updates == 100 else list(ids[:1])


def validate_output(path, root=OUTPUT):
    path, root = Path(path).resolve(), Path(root).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError('Every run must have its own new output directory.')


def load_original(case):
    from revision_common import CODE
    if case == 39:
        spec = importlib.util.spec_from_file_location('train_gat39_v1', HERE / 'legacy_dependencies/train_gat39_v1.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return importlib.import_module(f'train_rl{case}_v1')


def protocol(case, seed, updates):
    from revision_common import CODE, read, sha
    mod = load_original(case)
    dataset = PROJECT / f'case{case}_routeb_rescuable_dataset.json'
    samples = read(dataset)['samples']
    if case == 118:
        samples = [s for s in samples if s.get('is_complete_sced')]
    fixed = freeze_split(mod, case, samples)
    train, val, test = fixed()
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    assert len(train) + len(val) + len(test) == len(samples)
    baseline = read(REV / f'inputs/smallcase_case{case}_partition_ids.json')
    reference_ids = baseline['test_sample_ids']
    sample_id = lambda i: str(samples[i].get('sample_id', samples[i].get('id', i)))
    assert reference_ids == [sample_id(i) for i in test], 'Test IDs differ from the published common evaluation.'
    predictor = PROJECT / ('gat39_results/gat39_best.pth' if case == 39 else 'gatv118_results/gatv118_predictor_best.pth')
    record = {'case': case, 'training_seed': seed, 'partition_seed': 42, 'updates': updates,
              'episodes_per_update': 10, 'gradient_steps_per_update': 30,
              'training_ac_check_every': mod.AC_PF_CHECK_EVERY,
              'training_n1_check_every': mod.N1_CHECK_EVERY_STEPS,
              'voltage_band': [mod.VMIN, mod.VMAX], 'voltage_step_pu': mod.VOLTAGE_STEP_PU,
              'partitions': {'train': train, 'validation': val, 'test': test},
              'validation_subset': validation_subset(case, val), 'test_sample_ids': reference_ids,
              'checkpoint_selection': 'validation accepted count, then mean terminal validation reward; no test selection',
              'smoke_evaluation_limit': None if updates == 100 else 1,
              'baseline_reference': str(REV / f'inputs/smallcase_case{case}_partition_ids.json'),
              'input_sha256': {str(p): sha(p) for p in (dataset, predictor, CODE / f'train_rl{case}_v1.py',
                                                       CODE / 'smallcase_evaluation_contract.py',
                                                       HERE / 'evaluate_smallcases_common.py', Path(__file__))}}
    return mod, samples, fixed, record


def run(args):
    configure_runtime()
    from revision_common import read, write, sha
    import numpy as np
    import torch
    out = Path(args.output).resolve() if args.output else OUTPUT / f'case{args.case}_seed{args.seed}'
    validate_output(out)
    mod, samples, fixed, config = protocol(args.case, args.seed, args.updates)
    if args.preflight:
        print(json.dumps(config, indent=2), flush=True)
        return
    if out.exists():
        raise ValueError(f'Output already exists; preserve it rather than overwriting: {out}')
    out.mkdir(parents=True)
    write(out / 'protocol.json', config)
    import pandapower as pp
    receipt = {'complete': False, 'status': 'training', 'started_utc': datetime.now(timezone.utc).isoformat(),
               'case': args.case, 'seed': args.seed, 'python': sys.version,
               'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(),
               'pandapower': pp.__version__, 'outer_updates': 0,
               'actor_optimizer_steps': 0, 'critic_optimizer_steps': 0,
               'runpp_calls': 0, 'policy_forward_calls': 0}
    original_runpp = pp.runpp
    def counted_runpp(*pos, **kw):
        receipt['runpp_calls'] += 1
        return original_runpp(*pos, **kw)
    pp.runpp = counted_runpp
    original_action = mod.GaussianActor.select_action
    def counted_action(actor, *pos, **kw):
        receipt['policy_forward_calls'] += 1
        return original_action(actor, *pos, **kw)
    mod.GaussianActor.select_action = counted_action
    if torch.cuda.is_available():
        receipt['device'] = torch.cuda.get_device_name()
    write(out / 'execution_receipt.json', receipt)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train, val, test = fixed()
    setattr(mod, 'split_samples' if args.case == 39 else 'split_samples_stratified', fixed)
    if args.case == 39:
        mod.SEED = args.seed
        mod.RESULTS = out
    else:
        # SEED also fixes the physical direction construction on case118.
        mod.RESULTS_DIR = out
        mod.TOTAL_UPDATES = args.updates
    original_evaluate = mod.evaluate
    original_init = mod.SACTrainer.__init__
    original_update = mod.SACTrainer.update
    state = {'best': None, 'history': []}

    def instrument_init(trainer, *pos, **kw):
        original_init(trainer, *pos, **kw)
        for field, metric in (('ao' if args.case == 39 else 'actor_opt', 'actor_optimizer_steps'),
                              ('co' if args.case == 39 else 'critic_opt', 'critic_optimizer_steps')):
            optimizer = getattr(trainer, field)
            old_step = optimizer.step
            def step(*a, _old=old_step, _metric=metric, **k):
                result = _old(*a, **k)
                receipt[_metric] += 1
                return result
            optimizer.step = step

    def instrument_update(trainer, *pos, **kw):
        result = original_update(trainer, *pos, **kw)
        receipt['outer_updates'] += 1
        write(out / 'execution_receipt.json', receipt)
        return result

    def capture_validation(trainer, env, ids):
        actual_ids = evaluation_ids(ids, args.updates)
        result = original_evaluate(trainer, env, actual_ids)
        if list(ids) == validation_subset(args.case, val):
            selected = validation_improves(result, state['best'])
            state['history'].append(dict(result, update=receipt['outer_updates'], selected=selected,
                                         sample_ids=actual_ids))
            if selected:
                state['best'] = dict(result)
                # Serialization captures the weights now, before later optimization.
                torch.save({'actor': trainer.actor.state_dict(), 'update': receipt['outer_updates'],
                            'validation': result, 'seed': args.seed}, out / 'selected_validation_actor.pth')
            write(out / 'validation_history.json', state['history'])
        elif list(ids) != test:
            raise AssertionError('Unexpected evaluation IDs in the original training entry point.')
        return result

    mod.SACTrainer.__init__ = instrument_init
    mod.SACTrainer.update = instrument_update
    mod.evaluate = capture_validation
    old_argv = sys.argv[:]
    sys.argv = [str(Path(mod.__file__)), '--updates', str(args.updates), '--output-dir', str(out)] if args.case == 39 else [str(Path(mod.__file__))]
    started = time.perf_counter()
    try:
        mod.main()
    finally:
        sys.argv = old_argv
    assert receipt['outer_updates'] == args.updates
    selected = torch.load(out / 'selected_validation_actor.pth', map_location=mod.DEVICE, weights_only=False)
    original_best = torch.load(out / 'best_model.pth', map_location=mod.DEVICE, weights_only=False)
    assert all(torch.equal(v, original_best['actor'][k]) for k, v in selected['actor'].items())
    receipt.update(status='independent_test', training_runtime_sec=time.perf_counter() - started,
                   selected_update=selected['update'], selected_checkpoint_sha256=sha(out / 'selected_validation_actor.pth'))
    write(out / 'execution_receipt.json', receipt)
    from evaluate_smallcases_common import evaluate
    env = mod.Env39(samples, test) if args.case == 39 else mod.SCEDEnv118(samples, test)
    actor = mod.GaussianActor(env._get_obs_dim(), env.action_dim).to(mod.DEVICE)
    actor.load_state_dict(selected['actor'])
    actor.eval()
    mod.AC_PF_CHECK_EVERY = mod.N1_CHECK_EVERY_STEPS = 1
    def action(obs):
        raw = actor.select_action(obs, **({'det': True} if args.case == 39 else {'deterministic': True}))
        low, high = env.get_action_bounds()
        return low + (raw + 1.) * .5 * (high - low)
    obs = env.reset(test[0])
    action(obs)
    payload = {'complete': False, 'contract': 'common_smallcase_v2', 'case': args.case,
               'seed': args.seed, 'protocol': config, 'selected_update': selected['update'],
               'checkpoint_sha256': receipt['selected_checkpoint_sha256'], 'rows': []}
    measured_ids = evaluation_ids(test, args.updates)
    for i, sid in enumerate(measured_ids):
        row = evaluate(env, mod, args.case, sid, 'policy', action)
        row.update(sid=sid, sample_id=config['test_sample_ids'][i], case=args.case, demand_mw=samples[sid]['demand_mw'])
        payload['rows'].append(row)
        write(out / 'common_test.json', payload)
        print(f"COMMON TEST {i+1}/{len(measured_ids)} pass={row['passed']} steps={row['steps']}", flush=True)
    payload['complete'] = True
    write(out / 'common_test.json', payload)
    receipt.update(complete=True, status='complete', test_passed=sum(r['passed'] for r in payload['rows']),
                   test_total=len(measured_ids), finished_utc=datetime.now(timezone.utc).isoformat())
    write(out / 'execution_receipt.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)


def suite():
    configure_runtime()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for case in (39, 118):
        for seed in (1, 2):
            out = OUTPUT / f'case{case}_seed{seed}'
            receipt = out / 'execution_receipt.json'
            if receipt.exists() and json.loads(receipt.read_text())['complete']:
                print(f'SKIP completed case={case} seed={seed}', flush=True)
                continue
            log = OUTPUT / f'case{case}_seed{seed}.console.log'
            if log.exists() or out.exists():
                raise ValueError(f'Incomplete prior run exists: {out}; inspect it before restarting.')
            print(f'START case={case} seed={seed}; log={log}', flush=True)
            with log.open('x', encoding='utf-8') as stream:
                subprocess.run([sys.executable, '-u', str(Path(__file__)), '--case', str(case), '--seed', str(seed)],
                               stdout=stream, stderr=subprocess.STDOUT, check=True, cwd=REV)
            print(f'COMPLETE case={case} seed={seed}', flush=True)


def configure_runtime():
    runtime = OUTPUT / '_runtime'
    for name in ('tmp', 'numba'):
        (runtime / name).mkdir(parents=True, exist_ok=True)
    os.environ['TEMP'] = os.environ['TMP'] = str(runtime / 'tmp')
    os.environ['NUMBA_CACHE_DIR'] = str(runtime / 'numba')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=int, choices=(39, 118))
    parser.add_argument('--seed', type=int, choices=(1, 2))
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--output')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--suite', action='store_true')
    args = parser.parse_args()
    if args.suite:
        suite()
    else:
        if args.case is None or args.seed is None:
            parser.error('--case and --seed are required outside --suite')
        if args.updates != 100 and (not args.output or 'smoke' not in Path(args.output).name):
            parser.error('Reduced training is allowed only in a separate smoke output.')
        run(args)


if __name__ == '__main__':
    main()

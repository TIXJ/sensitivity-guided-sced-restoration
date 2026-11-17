"""Run real policy/AC/LODF experiments using only distributed inputs."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from functools import partial

from distribution_integrity import digest, verify

REPO = Path(__file__).resolve().parents[2]


def copy_tree(source, target, pattern='*', *, manifest):
    for path in sorted(source.rglob(pattern)):
        if path.is_file() and '__pycache__' not in path.parts:
            destination = target / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            key = path.relative_to(REPO).as_posix()
            if key not in manifest or not path.resolve().is_relative_to(REPO.resolve()):
                raise ValueError('Undeclared materialization input: ' + key)
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != manifest[key]:
                raise ValueError('Input changed during materialization: ' + key)
            destination.write_bytes(content)


def prepare():
    verified = verify(REPO)
    manifest_bytes = (REPO / 'MANIFEST.json').read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == verified['manifest_sha256']
    copy = partial(copy_tree, manifest=json.loads(manifest_bytes)['files'])
    root = REPO / '_runtime' / uuid.uuid4().hex / 'ED_RL'
    revision = root / 'GAT-SAC-SCED-Framework/first revise'
    analysis = revision / 'experiments'
    # Preserve the historical relative layout rather than modifying physics code.
    copy(REPO / 'code/pipeline', root / 'paper_results/code', '*.py')
    copy(REPO / 'code/pipeline/root', root, '*.py')
    copy(REPO / 'code/analysis', analysis, '*.py')
    copy(REPO / 'data/cases', root, '*.json')
    copy(REPO / 'data/partitions', root / 'paper_results', '*.json')
    copy(REPO / 'data/voltage_context', revision / 'experiment_results/jacobian_cache', '*.npz')
    copy(REPO / 'data/voltage_context/none', root / 'paper_results/.ac_voltage_jacobian_cache', '*.npz')
    copy(REPO / 'data/protocols', revision / 'inputs', '*.json')
    copy(REPO / 'data/operating_points', revision / 'inputs', '*.json')
    for directory in (REPO / 'models').iterdir():
        if directory.is_dir():
            copy(directory, revision / 'experiment_results' if directory.name == 'revision_runs' else root / directory.name)
    (revision / 'experiment_results').mkdir(parents=True, exist_ok=True)
    inventory = {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob('*')) if p.is_file()}
    (root / 'execution_inputs.json').write_text(json.dumps({**verified, 'files': inventory}, indent=2), encoding='utf-8')
    print('Verified inputs:', verified['verified_files'], 'Runtime snapshot:', root, flush=True)
    return root, revision, analysis


def launch(root, analysis, script, args, timing=False):
    env = os.environ.copy()
    env['PYTHONPATH'] = ''
    env['PYTHONIOENCODING'] = 'utf-8'
    env['SCED_PUBLIC_REPO'] = str(REPO)
    env['SCED_RUNTIME_ROOT'] = str(root)
    env['SCED_TIMING_ONLY'] = '1' if timing else '0'
    cache = Path(os.environ.get('SCED_NUMBA_CACHE', str(Path(tempfile.gettempdir()) / 'sced_public_numba_v1_1')))
    cache.mkdir(parents=True, exist_ok=True)
    env['NUMBA_CACHE_DIR'] = str(cache)
    env['MPLCONFIGDIR'] = str(root / '.matplotlib')
    env['TORCH_HOME'] = str(root / '.torch')
    # These legacy entry points set this before importing pandapower. The
    # instrumentation bootstrap imports it first, so preserve that order here.
    if Path(script).name in ('eval_route_b_acopf_baseline.py', 'eval_case300_ac_scopf_baseline.py'):
        env.setdefault('NUMBA_DISABLE_JIT', '1')
    command = [sys.executable, '-I', str(analysis / '_physical_bootstrap.py'), str(analysis / script), *map(str, args)]
    print('Physical execution:', script, *map(str, args), flush=True)
    result = subprocess.run(command, cwd=root, env=env)
    if result.returncode:
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'original', 'small', 'train', 'ac-audit', 'historical', 'spatial', 'baseline'))
    parser.add_argument('--baseline', choices=('repair', 'acopf', 'scopf'), default='repair')
    parser.add_argument('--method', choices=('sac', 'full68', 'ddpg', 'td3'), default='sac')
    parser.add_argument('--configuration', help='Frozen evaluation key from data/protocols/policy_evaluation_specs.json')
    parser.add_argument('--case', choices=(39, 118), type=int, default=39)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--scope', choices=('test', 'val'), default='test')
    parser.add_argument('--name', default='public_rerun')
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--episodes-per-update', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--family', choices=('physics', 'modes', 'random', 'keys', 'full68'), default='physics')
    parser.add_argument('--dimension', type=int, default=32)
    parser.add_argument('--shifts', type=int, default=12)
    parser.add_argument('--smoke-training', action='store_true')
    parser.add_argument('--timing', action='store_true', help='Workflow timing without per-call audit instrumentation; input hashes are still checked')
    args = parser.parse_args()
    if not args.name or args.name in ('.', '..') or Path(args.name).name != args.name:
        parser.error('--name must be a single output-directory name')
    if args.max_samples < 0 or args.repeats < 1 or args.updates < 1 or args.episodes_per_update < 1:
        parser.error('Sample limit must be nonnegative; repeats and training budgets must be positive')
    if args.stage == 'train' and args.method != 'sac':
        parser.error('The train stage supports SAC only; see docs/PHYSICAL_REPRODUCTION.md for deterministic trainers')
    if args.timing and args.stage not in ('historical', 'spatial'):
        parser.error('--timing is supported only for historical/spatial comparisons')
    root, revision, analysis = prepare()
    if args.stage == 'prepare':
        print(root)
        return
    out = revision / 'experiment_results' / args.name
    if not out.resolve().is_relative_to((revision / 'experiment_results').resolve()):
        parser.error('Output must remain inside experiment_results')
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'original':
        output = out / f'{args.configuration or args.method}_{args.scope}.json'
        if output.exists():
            parser.error('Output already exists; use a different --name')
        options = ['--method', args.method]
        if args.configuration:
            specs = json.loads((REPO / 'data/protocols/policy_evaluation_specs.json').read_text(encoding='utf-8'))
            if args.configuration not in specs:
                parser.error('Unknown configuration; inspect data/protocols/policy_evaluation_specs.json')
            spec = specs[args.configuration]
            checkpoint = REPO / spec['checkpoint']
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == spec['checkpoint_sha256']
            relative = checkpoint.relative_to(REPO / 'models')
            runtime_checkpoint = revision / 'experiment_results' / Path(*relative.parts[1:]) if relative.parts[0] == 'revision_runs' else root / relative
            options = ['--checkpoint', runtime_checkpoint]
            for key, value in spec['arguments'].items():
                options.extend(['--' + key.replace('_', '-'), value])
        launch(root, analysis, 'evaluate_revision.py', [*options, '--scope', args.scope,
            '--max-samples', args.max_samples, '--output', output])
    elif args.stage == 'small':
        output = out / f'case{args.case}.json'
        if output.exists():
            parser.error('Output already exists; use a different --name')
        launch(root, analysis, 'evaluate_smallcases_common.py', ['--case', args.case,
            '--max-samples', args.max_samples, '--output', output])
    elif args.stage == 'train':
        options = ['--family', args.family, '--dimension', args.dimension, '--shifts', args.shifts,
            '--updates', 1 if args.smoke_training else args.updates, '--episodes-per-update', 1 if args.smoke_training else args.episodes_per_update,
            '--seed', args.seed, '--dataset', root / 'case300_active_required_fixed_slack_v1.json',
            '--split-file', root / 'paper_results/case300_fixed_slack_v1_split.json',
            '--gat-checkpoint', root / 'gat_case300_fixed_slack_v1/gatv10_predictor_best.pth',
            '--output-dir', out / 'sac_train']
        if args.smoke_training:
            options.extend(['--learning-starts', 4, '--batch-size', 4, '--gradient-steps', 1, '--val-interval', 1])
        launch(root, analysis, 'train_revision.py', options)
    elif args.stage == 'ac-audit':
        launch(root, analysis, 'ac_contingency_audit.py', ['--input', revision / 'inputs/ac_audit_original.json',
            '--max-samples', args.max_samples, '--output', out / 'ac_outages.json'])
    elif args.stage == 'baseline':
        if args.baseline == 'repair':
            script = root / 'paper_results/code/eval_case300_deterministic_repair.py'
            options = ['--candidate-dataset', root / 'case300_sced_candidates_fixed_slack_v1.json',
                '--candidate-split', root / 'paper_results/case300_fixed_slack_v1_candidate_split.json',
                '--screen-report', root / 'paper_results/case300_fixed_slack_v1_screen.json',
                '--gat-checkpoint', root / 'gat_case300_fixed_slack_v1/gatv10_predictor_best.pth']
        else:
            script = root / 'paper_results/code' / ('eval_route_b_acopf_baseline.py' if args.baseline == 'acopf' else 'eval_case300_ac_scopf_baseline.py')
            options = ['--dataset', root / 'case300_active_required_fixed_slack_v1.json',
                '--split-file', root / 'paper_results/case300_fixed_slack_v1_split.json']
        launch(root, analysis, script, [*options, '--scope', args.scope, '--max-samples', args.max_samples,
            '--output', out / (args.baseline + '.json')])
    else:
        launch(root, analysis, 'public_workflow_runner.py', [args.stage, '--max-samples', args.max_samples,
            '--repeats', args.repeats, '--output', out / f'{args.stage}.json'], timing=args.timing)


if __name__ == '__main__':
    main()

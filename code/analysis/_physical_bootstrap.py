"""Deny hidden project inputs and record actual physics/model execution."""
from pathlib import Path
import atexit
import json
import os
import sys
import time
import hashlib

ROOT = Path(os.environ['SCED_RUNTIME_ROOT']).resolve()
REPO = Path(os.environ['SCED_PUBLIC_REPO']).resolve()
SCRIPT = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(SCRIPT.parent), str(Path(__file__).resolve().parent), str(ROOT / 'paper_results/code'), str(ROOT)]
from distribution_integrity import allowed_read, write_only
PREFIX = Path(sys.prefix).resolve()
SYSTEM = Path(os.environ.get('SystemRoot', '/nonexistent')).resolve()
JIT_CACHE = Path(os.environ['NUMBA_CACHE_DIR']).resolve()
inventory = json.loads((ROOT / 'execution_inputs.json').read_text(encoding='utf-8'))
for name, expected in inventory['files'].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
stats = {'script': SCRIPT.name, 'runpp_calls': 0, 'runopp_calls': 0,
         'policy_forward_calls': 0, 'gat_forward_calls': 0, 'denied_inputs': [],
         'neural_layer_forward_calls': 0, 'input_files': [], 'completed': False,
         'distribution_manifest_sha256': inventory['manifest_sha256'],
         'verified_distribution_files': inventory['verified_files'],
         'materialized_input_files': len(inventory['files'])}
stats['arguments'] = sys.argv[2:]
TIMING_ONLY = os.environ.get('SCED_TIMING_ONLY') == '1'
stats['instrumented'] = not TIMING_ONLY
opened = set()
AUDITING = True


def audit(event, args):
    if not AUDITING or event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    mode = args[1]
    if write_only(mode, args[2]):
        return
    if allowed_read(path, (PREFIX, SYSTEM, JIT_CACHE)):
        return
    allowed = path.is_relative_to(ROOT)
    if not allowed:
        stats['denied_inputs'].append(str(path))
        raise PermissionError('Physical reproduction attempted an undeclared external input: ' + str(path))
    if path.suffix.lower() not in ('.py', '.pyc'):
        opened.add(path.relative_to(ROOT).as_posix())


if not TIMING_ONLY:
    sys.addaudithook(audit)


def receipt():
    global AUDITING
    AUDITING = False
    stats['input_files'] = sorted(opened)
    stats['changed_materialized_inputs'] = [name for name, expected in inventory['files'].items()
        if not (ROOT / name).is_file() or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected]
    if stats['changed_materialized_inputs']:
        stats['completed'] = False
    if TIMING_ONLY:
        for name in ('runpp_calls', 'runopp_calls', 'policy_forward_calls', 'gat_forward_calls', 'neural_layer_forward_calls', 'optimizer_step_calls'):
            stats[name] = None
    if '--output' in stats['arguments']:
        output = Path(stats['arguments'][stats['arguments'].index('--output') + 1])
        if output.is_file():
            stats['output_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    folder = ROOT / 'GAT-SAC-SCED-Framework/first revise/experiment_results/execution_receipts'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f'{SCRIPT.stem}_{time.time_ns()}.json'
    target.write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print('Physical execution receipt:', target, flush=True)


atexit.register(receipt)
import pandapower as pp
import torch
stats['optimizer_step_calls'] = 0
adam_step = torch.optim.Adam.step
def optimizer_step(self, *args, **kwargs):
    stats['optimizer_step_calls'] += 1
    return adam_step(self, *args, **kwargs)
if not TIMING_ONLY:
    torch.optim.Adam.step = optimizer_step

for name in ('runpp', 'runopp'):
    original = getattr(pp, name)
    def instrumented(*args, _name=name, _original=original, **kwargs):
        stats[_name + '_calls'] += 1
        return _original(*args, **kwargs)
    if not TIMING_ONLY:
        setattr(pp, name, instrumented)

original_call = torch.nn.Module._call_impl
original_init = torch.nn.Module.__init__
def module_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    name = type(self).__name__
    if name in ('BranchedGaussianActor', 'GaussianActor', 'DeterministicActor', 'Actor'):
        forward = self.forward
        def counted_forward(*values, **options):
            stats['policy_forward_calls'] += 1
            return forward(*values, **options)
        self.forward = counted_forward

if not TIMING_ONLY:
    torch.nn.Module.__init__ = module_init
def module_call(self, *args, **kwargs):
    name = type(self).__name__
    if name in ('GATv10Predictor', 'GAT39', 'GAT118', 'GATPredictor', 'GATv118Predictor'):
        stats['gat_forward_calls'] += 1
    if name in ('Linear', 'GATv2ConvLayer'):
        stats['neural_layer_forward_calls'] += 1
    return original_call(self, *args, **kwargs)
if not TIMING_ONLY:
    torch.nn.Module._call_impl = module_call

sys.argv = [str(SCRIPT), *sys.argv[2:]]
import runpy
runpy.run_path(str(SCRIPT), run_name='__main__')
AUDITING = False
changed = [name for name, expected in inventory['files'].items()
    if not (ROOT / name).is_file() or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected]
if stats['denied_inputs'] or changed:
    raise RuntimeError('Physical outputs invalidated by external reads or changed materialized inputs')
stats['completed'] = True

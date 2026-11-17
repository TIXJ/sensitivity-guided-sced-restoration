"""Shared revision paths and strictly separate output locations."""
from pathlib import Path
import importlib.util
import json
import sys
import hashlib
import math

HERE = Path(__file__).resolve().parent
REV = HERE.parent
ROOT = REV.parents[1]
CODE = ROOT / 'paper_results' / 'code'
RESULTS = REV / 'experiment_results'
CANDIDATES = ROOT / 'case300_sced_candidates_fixed_slack_v1.json'
CANDIDATE_SPLIT = ROOT / 'paper_results/case300_fixed_slack_v1_candidate_split.json'
DATASET = ROOT / 'case300_active_required_fixed_slack_v1.json'
SPLIT = ROOT / 'paper_results/case300_fixed_slack_v1_split.json'
GAT = ROOT / 'gat_case300_fixed_slack_v1/gatv10_predictor_best.pth'
CHECKPOINTS = {
    'sac': ROOT / 'rl_sac_fixed_slack_v1_v31_p40_voltage_sensitivity/best_actor.pth',
    'full68': ROOT / 'rl_full68_fixed_slack_v1_p40_raw_action_v4/best_actor.pth',
    'ddpg': ROOT / 'rl_ddpg_fixed_slack_v1_p40_raw_action_v9/best_actor.pth',
    'td3': ROOT / 'rl_td3_fixed_slack_v1_p40_unified_noise_v5/best_actor.pth',
}
sys.path.insert(0, str(CODE))
sys.path.insert(1, str(ROOT))
sys.path.append(str(HERE / 'legacy_dependencies'))

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def needs_sample_voltage_cache(settings):
    return settings.get('stress') == 'renewable' or settings.get('initialization', 'gat') != 'gat'

def voltage_cache_directory(settings, sample_index, input_hashes, checkpoint_hash):
    if not needs_sample_voltage_cache(settings):
        return RESULTS / 'jacobian_cache' / settings.get('stress', 'none')
    contract = {key: value for key, value in settings.items() if key != 'output'}
    fingerprint = hashlib.sha256(json.dumps(
        [contract, input_hashes, checkpoint_hash], sort_keys=True).encode('utf-8')).hexdigest()[:20]
    return RESULTS / 'jacobian_cache_v2' / fingerprint / str(int(sample_index))

def json_safe(value):
    """Keep failed numerical measurements missing, with auditable JSON paths."""
    missing = []
    def visit(item, path):
        if hasattr(item, 'tolist'):
            item = item.tolist()
        if isinstance(item, float) and not math.isfinite(item):
            missing.append(path)
            return None
        if isinstance(item, dict):
            return {key: visit(val, f'{path}.{key}') for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [visit(val, f'{path}[{i}]') for i, val in enumerate(item)]
        return item
    clean = visit(value, '$')
    if missing:
        if not isinstance(clean, dict):
            raise ValueError('Non-finite diagnostics require a dictionary result')
        clean['nonfinite_fields'] = sorted(set(clean.get('nonfinite_fields', [])) | set(missing))
    return clean

def resume_rows(saved, settings, checkpoint_hash, sample_ids):
    """Resume only an exact, ordered prefix of the same evaluation."""
    old_settings = dict(saved.get('settings', {}))
    new_settings = dict(settings)
    old_settings.pop('output', None)
    new_settings.pop('output', None)
    if old_settings != new_settings:
        raise ValueError('Existing result settings differ; use a new output path')
    if saved.get('checkpoint_sha256') != checkpoint_hash:
        raise ValueError('Existing result checkpoint differs; use a new output path')
    rows = saved.get('rows')
    if not isinstance(rows, list) or len(rows) > len(sample_ids):
        raise ValueError('Invalid saved evaluation rows')
    if [row.get('sample_id') for row in rows] != list(sample_ids[:len(rows)]):
        raise ValueError('Saved samples do not match the selected evaluation prefix')
    if len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate saved samples')
    if saved.get('complete') and len(rows) != len(sample_ids):
        raise ValueError('Saved completion flag disagrees with sample count')
    return rows

def write(path, value):
    path = Path(path)
    if not path.resolve().is_relative_to(REV.resolve()):
        raise ValueError(f'Revision outputs must stay in {REV}: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_aligned(extra=()):
    saved = sys.argv[:]
    sys.argv = [str(CODE / 'train_sac_n1_active_aligned.py'), '--dataset', str(DATASET), '--split-file', str(SPLIT), '--gat-checkpoint', str(GAT), '--output-dir', str(RESULTS / 'scratch'), *map(str, extra)]
    try:
        spec = importlib.util.spec_from_file_location('revision_aligned', CODE / 'train_sac_n1_active_aligned.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.argv = saved

def configure_env(mod, family='physics', dimension=32, shifts=12, basis_seed=20260721, success_scale=1.):
    """Adapt only action representation/reward bonus; reuse the original transition."""
    import numpy as np
    base = mod.CleanN1Env
    use_original_basis = family == 'physics' and dimension == 32 and shifts == 12
    if family == 'physics' and dimension == 32 and shifts == 12 and success_scale == 1:
        return base

    class RevisionEnv(base):
        def __init__(self, samples, sample_ids, dataset_meta):
            self._revision_basis = None
            super().__init__(samples, sample_ids, dataset_meta)
            if use_original_basis:
                return
            self._n_shift_dirs = 0 if family == 'full68' else dimension
            self._n_key_gens = self.N if family == 'full68' else 0
            if family == 'full68':
                self.action_encoding = 'full68'

        def _build_fixed_physics_action_basis(self):
            if use_original_basis:
                return super()._build_fixed_physics_action_basis()
            if family == 'full68':
                self._fixed_shift_directions_cache = np.zeros((0, self.N))
                self._fixed_key_gen_indices_cache = np.arange(self.N)
            else:
                self._fixed_shift_directions_cache = self._build_orthogonal_physics_action_basis().T
                self._fixed_key_gen_indices_cache = np.zeros(0, dtype=int)

        def _build_orthogonal_physics_action_basis(self):
            if use_original_basis:
                return super()._build_orthogonal_physics_action_basis()
            if self._revision_basis is not None:
                return self._revision_basis
            n = self.N
            center = np.eye(n) - np.ones((n, n)) / n
            if family == 'full68':
                basis = center
            elif family == 'random':
                raw = center @ np.random.default_rng(basis_seed).normal(size=(n, dimension))
                basis, _ = np.linalg.qr(raw)
            else:
                s = np.asarray(self.dcnet.S)
                limits = np.maximum(self.dcnet.effective_limit, 1.)
                g = (s @ center) / limits[:, None]
                gram = g.T @ g
                for k in self.valid_contingency_indices:
                    g = s + self.dcnet.lodf[:, k:k+1] * s[k:k+1]
                    g[k] = 0
                    g = (g @ center) / limits[:, None]
                    g = g[np.isfinite(g).all(axis=1)]
                    gram += g.T @ g
                _, vec = np.linalg.eigh((gram + gram.T) / 2)
                modes = vec[:, ::-1]
                keys = np.argsort(-np.diag(gram))
                m = dimension if family == 'modes' else (0 if family == 'keys' else shifts)
                raw = np.column_stack([modes[:, :m], center[:, keys[:dimension-m]] * n/(n-1)])
                u, singular, _ = np.linalg.svd(center @ raw, full_matrices=False)
                if np.count_nonzero(singular > singular[0] * 1e-10) != dimension:
                    raise ValueError('Requested basis is rank deficient')
                basis, _ = np.linalg.qr(center @ u[:, :dimension])
            for j in range(basis.shape[1]):
                if basis[np.argmax(abs(basis[:, j])), j] < 0 and family != 'full68':
                    basis[:, j] *= -1
            self._revision_basis = basis
            return basis

        def _compute_reward(self):
            reward, done, info = super()._compute_reward()
            if done:
                reward *= success_scale
                info['clean_reward'] = reward
            return reward, done, info

    mod.CleanN1Env = RevisionEnv
    return RevisionEnv

def wilson(passed, total):
    if not total:
        return [None, None]
    z = 1.959963984540054
    p = passed / total
    den = 1 + z*z/total
    middle = (p + z*z/(2*total)) / den
    half = z * ((p*(1-p)/total + z*z/(4*total*total)) ** .5) / den
    return [middle-half, middle+half]

"""Validation-only necessary action-range bounds; no training or test selection."""
import argparse
from collections import defaultdict
import os
import subprocess
import numpy as np

from revision_common import (
    CANDIDATES, CANDIDATE_SPLIT, DATASET, GAT, RESULTS, REV,
    load_aligned, read, sha, write,
)


def necessary_radius(flows, limits, gradient_norms, tolerance=0.01):
    """Cauchy--Schwarz lower bound, not a sufficient feasibility radius."""
    gap = np.maximum(np.abs(flows) - limits - tolerance, 0.0)
    norms = np.broadcast_to(gradient_norms, gap.shape)
    blocked = bool(np.any((gap > 0) & (norms <= 1e-12)))
    ratios = np.divide(gap, norms, out=np.zeros_like(gap), where=norms > 1e-12)
    return float('inf') if blocked else float(np.max(ratios))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(RESULTS/'optimization_diagnostics/validation_action_range.json'))
    args = parser.parse_args()
    from pathlib import Path
    if Path(args.output).exists():
        parser.error('Preserve existing diagnostics; choose a new output path')
    os.environ.setdefault('NUMBA_CACHE_DIR', str(RESULTS/'numba_cache'))
    commit = subprocess.check_output(
        ['git', '-C', str(REV/'original_git_snapshot'), 'rev-parse', 'HEAD'],
        text=True).strip()
    data = read(CANDIDATES)
    samples = [s for s in data['samples'] if s.get('is_complete_sced', True)]
    split = read(CANDIDATE_SPLIT)
    ids = split['val_ids']
    assert not set(ids).intersection(split['test_ids'])
    assert not set(ids).intersection(split['train_ids'])
    screened = {s['sample_id'] for s in read(DATASET)['samples']}
    mod = load_aligned()
    env = mod.CleanN1Env(samples, ids, data)
    basis = env._build_orthogonal_physics_action_basis()
    assert np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-10)
    assert np.allclose(basis.sum(axis=0), 0, atol=1e-10)
    s = np.asarray(env.dcnet.S)
    intercept = np.asarray(env.dcnet.flow_intercept)
    lodf = np.asarray(env.dcnet.lodf)
    limits = np.asarray(env.dcnet.effective_limit)
    valid = np.asarray(env.valid_contingency_indices, dtype=int)
    gradients = s[:, None, :] + lodf[:, valid, None] * s[valid][None, :, :]
    gradients[valid, np.arange(len(valid)), :] = 0
    norms = {
        'basis32': (np.linalg.norm(s @ basis, axis=1), np.linalg.norm(gradients @ basis, axis=2)),
        'full_balanced': (np.linalg.norm(s-s.mean(axis=1, keepdims=True), axis=1),
                          np.linalg.norm(gradients-gradients.mean(axis=2, keepdims=True), axis=2)),
    }
    margin = env.generator_upper_margin if env.generator_upper_margin is not None else 0.99
    rows = []
    for sid in ids:
        sample = samples[sid]
        env.current_sample = sample
        demand = float(sample['demand_mw'])
        initial = mod.core.project_box_sum(env._predict_gat(demand), env.PMIN, env.PMAX*margin, demand)
        base = intercept + s @ initial
        post = base[:, None] + lodf[:, valid] * base[valid][None, :]
        post[valid, np.arange(len(valid))] = 0
        row = {'sample_id': sample['sample_id'], 'candidate_index': sid,
               'demand_mw': demand, 'screened': sample['sample_id'] in screened,
               'initial_dispatch_mw': initial.tolist(), 'lower_bounds_mw': {}}
        for name, (base_norm, outage_norm) in norms.items():
            row['lower_bounds_mw'][name] = max(
                necessary_radius(base, limits, base_norm),
                necessary_radius(post, limits[:, None], outage_norm))
        rows.append(row)
    groups = defaultdict(list)
    for row in rows:
        groups[row['demand_mw']].append(row)
    grouped = []
    for demand, items in sorted(groups.items()):
        result = {'demand_mw': demand, 'n': len(items), 'screened': sum(r['screened'] for r in items)}
        for name in norms:
            values = np.asarray([r['lower_bounds_mw'][name] for r in items])
            result[name] = {'min_mw': float(values.min()), 'median_mw': float(np.median(values)),
                            'max_mw': float(values.max()),
                            'ruled_out_at_radius': {str(radius): int(np.sum(values > radius+1e-6))
                                                   for radius in (20, 40, 60)}}
        grouped.append(result)
        print(result, flush=True)
    write(args.output, {
        'scope': 'upstream validation only', 'n': len(ids), 'backup_commit': commit,
        'input_sha256': {str(p.name): sha(p) for p in (CANDIDATES, CANDIDATE_SPLIT, DATASET, GAT)},
        'definition': 'max_j (abs(f_j(P0))-limit_j-0.01)_+ / norm(g_j B)',
        'interpretation': 'Necessary bound for unprojected linear-feasible targets. '
            'A bound below a radius does not show joint feasibility or AC feasibility. '
            'Basis bounds do not certify the recovered dispatch when projection leaves the span. '
            'Full-balanced bounds omit generator limits and are necessary, not sufficient.',
        'groups': grouped, 'rows': rows,
    })


if __name__ == '__main__':
    main()

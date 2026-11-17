"""Portable saved-record checks and table exports; no physical rerun is implied."""
from pathlib import Path
from statistics import mean
import argparse
import hashlib
import gzip
import json
import re

ROOT = Path(__file__).resolve().parents[2]


def read(relative):
    path = ROOT / relative
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return json.loads(gzip.decompress(path.with_suffix(path.suffix + '.gz').read_bytes()))


def verify():
    import numpy as np
    manifest = read('MANIFEST.json')['files']
    for name, expected in manifest.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    policy = read('results/full_sac.json')
    assert len(policy['rows']) == 75 and sum(r['passed'] for r in policy['rows']) == 44
    print('Original unfiltered policy: 44/75')
    for name, total, accepted in [('dual_verified_once_v2/dual.json', 75, 60), ('spatial_dual_final_v1/results.json', 100, 93)]:
        rows = read('results/' + name)['rows']
        records = {(r['repeat'], r['sample_id'], r['arm']): r for r in rows}
        ids = list(dict.fromkeys(r['sample_id'] for r in rows))
        assert len(ids) == total and len(records) == len(rows) == total * 3 * 2
        times = {}
        for arm in ('direct', 'selected'):
            times[arm] = np.array([mean(records[k, sid, arm]['runtime_sec'] for k in range(3)) for sid in ids])
            accepted_ids = [{sid for sid in ids if records[k, sid, arm]['passed']} for k in range(3)]
            assert all(len(x) == accepted for x in accepted_ids)
            if arm == 'direct':
                common = accepted_ids
            else:
                assert accepted_ids == common
        delta = times['direct'] - times['selected']
        rng = np.random.default_rng(2026101805)
        ci = np.quantile(delta[rng.integers(0, total, size=(10000, total))].mean(axis=1), [.025, .975])
        print(f'{name}: same accepted IDs {accepted}/{total}; direct={times["direct"].mean():.3f} s; gated={times["selected"].mean():.3f} s; saving={delta.mean():.3f} s; CI={ci.round(3).tolist()}')
        if total == 75:
            assert f'{times["direct"].mean():.3f}' == '1.912' and f'{times["selected"].mean():.3f}' == '1.725'
            assert np.allclose(ci, [.10416295198041993, .2818816233172612], atol=1e-12)
            assert round(100 * (1 - times['selected'].mean() / times['direct'].mean()), 1) == 9.8
        else:
            assert f'{times["direct"].mean():.3f}' == '1.088' and f'{times["selected"].mean():.3f}' == '1.048'
            assert ci[0] < 0 < ci[1]
    ac = read('results/ac_audit_summary.json')
    assert [ac[k] for k in ('total_pairs', 'passed_pairs', 'nonconverged_pairs', 'voltage_violating_pairs')] == [882, 536, 46, 300]
    print('AC outage diagnostic: 882 pairs; 536 pass / 46 nonconvergent / 300 voltage violations')
    print(f'Distribution integrity: {len(manifest)} hashes verified. Saved-record checks PASS. No training or physical solver rerun was performed.')


def tables(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    stored = read('results/reported_tables.json')
    for i, item in enumerate(stored, 1):
        filename = re.sub(r'[^A-Za-z0-9_.-]', '_', item['label']) + '.tex'
        (output / filename).write_text(item['latex'], encoding='utf-8')
    (output / 'index.json').write_text(json.dumps([{'number': i, 'label': item['label']} for i, item in enumerate(stored, 1)], indent=2), encoding='utf-8')
    print(f'Exported {len(stored)} table snippets from stored tabulation intermediates to {output}.')
    print('This reconstructs the saved tabulations; it does not recompute every table from physical experiments.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('verify', 'tables'))
    parser.add_argument('--output', default='generated_tables')
    args = parser.parse_args()
    verify() if args.stage == 'verify' else tables(args.output)


if __name__ == '__main__':
    main()

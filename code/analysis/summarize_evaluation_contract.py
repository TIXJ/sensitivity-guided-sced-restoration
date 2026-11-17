"""Audit corrected small-system failure labels and common timing measurements."""
from collections import Counter
import csv
import numpy as np
from revision_common import RESULTS, ROOT, read, sha, write


def stats(values):
    return {'mean': float(np.mean(values)), 'std': float(np.std(values, ddof=1)) if len(values)>1 else 0.}


def main():
    root = RESULTS/'evaluation_contract_v2'
    summary = {'contract': 'common_smallcase_v2', 'cases': {}}
    timing = []
    phases = ('reset_prediction_sec', 'initial_verification_sec', 'correction_sec', 'final_verification_sec')
    for case in (39, 118):
        path = root/f'case{case}_common.json'
        payload = read(path)
        assert payload['complete'] and payload['contract'] == 'common_smallcase_v2'
        n = 30 if case == 39 else 20
        assert len(payload['rows']) == 3*n
        groups = {m: [r for r in payload['rows'] if r['mode'] == m]
                  for m in ('prediction_only', 'myopic_search', 'policy')}
        ids = [{r['sample_id'] for r in rows} for rows in groups.values()]
        assert all(len(group)==n and group==ids[0] for group in ids)
        for row in payload['rows']:
            assert abs(sum(row[p] for p in phases)-row['runtime_sec']) < 1e-8
            assert all(row[p]>=0 for p in phases)
            assert row['passed'] == row['final_info']['passed']
        result = {}
        for mode, rows in groups.items():
            result[mode] = {'n': n, 'passed': sum(r['passed'] for r in rows),
                            'steps': stats([r['steps'] for r in rows]),
                            **{p: stats([r[p] for r in rows]) for p in (*phases, 'runtime_sec')}}
            for phase in (*phases, 'runtime_sec'):
                timing.append({'case': case, 'method': mode, 'phase': phase, 'n': n,
                               'passed': result[mode]['passed'], **result[mode][phase]})
        summary['cases'][str(case)] = {'source_sha256': sha(path), 'modes': result}
    original = read(ROOT/'paper_results/case39_action_diagnosis.json')
    corrected_path = root/'case39_actions.json'
    corrected = read(corrected_path)
    lookup = {(r['sample_id'], r['mode']): r for r in original['rows']}
    changed = []
    classes = {}
    for mode in sorted({r['mode'] for r in corrected['rows']}):
        rows = [r for r in corrected['rows'] if r['mode']==mode]
        assert len(rows)==30
        for row in rows:
            old = lookup[(row['sample_id'], mode)]
            if row['passed'] != old['passed']:
                changed.append({'sample_id': row['sample_id'], 'mode': mode,
                                'old': old['passed'], 'new': row['passed']})
        classes[mode] = {'n': len(rows), 'passed': sum(r['passed'] for r in rows),
                         'mean_steps': float(np.mean([r['steps'] for r in rows])),
                         'primary_failures': dict(Counter(r['reason'].split(':')[0] for r in rows if not r['passed'])),
                         'overlapping_failures': dict(Counter(k for r in rows for k, v in r['constraint_flags'].items() if v is True))}
    summary['case39_classification'] = {'source_sha256': sha(corrected_path),
                                       'acceptance_changed': changed, 'modes': classes}
    flags = read(root/'case300_terminal_flags.json')
    assert flags['complete'] and flags['source_sha256']==sha(RESULTS/'full_sac.json')
    summary['case300_multilabel'] = flags['summary']
    summary['case300_multilabel']['voltage_and_n1'] = sum(
        r['constraint_flags'].get('voltage_violation') is True and r['constraint_flags'].get('n1_violation') is True
        for r in flags['rows'])
    write(root/'summary.json', summary)
    with (root/'timing_components.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(timing[0]))
        writer.writeheader()
        writer.writerows(timing)
    for case, row in summary['cases'].items():
        print(case, {k: (v['passed'], v['runtime_sec']['mean']) for k,v in row['modes'].items()})
    print('39-bus failures:', classes)
    print('300-bus overlaps:', flags['summary'])


if __name__ == '__main__':
    main()

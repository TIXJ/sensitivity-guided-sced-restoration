"""Audit the untouched holdout and paired timing of a frozen deployment rule."""
from collections import Counter
from statistics import mean,median
import importlib.metadata
import platform
import sys
import numpy as np
from revision_common import RESULTS,read,write,sha

DEST=RESULTS/'implicit_short_policy_v2'


def main():
    data=read(DEST/'test.json');protocol=read(DEST/'protocol.json')
    assert data['complete'] and len(data['rows'])==600
    assert data['runner_hashes']==protocol['runner_hashes']
    for path,digest in {**protocol['input_hashes'],**protocol['runner_hashes']}.items():
        assert sha(path)==digest,path
    assert sha(DEST/'val.json')==protocol['validation_sha256']
    assert sha(DEST/'test_conditions.json')==protocol['test_conditions_sha256']
    rows={(r['repeat'],r['sample_id'],r['arm']):r for r in data['rows']}
    samples=read(DEST/'test_conditions.json')['samples']
    ids=[s['sample_id'] for s in samples]
    assert len(rows)==600 and len(ids)==len(set(ids))==100
    assert not set(ids).intersection(r['sample_id'] for r in read(DEST/'val_conditions.json')['samples'])
    for row in data['rows']:
        if row['passed']:
            assert all(value is False for value in row['constraint_flags'].values()),row['sample_id']
            assert row['final_verification']['controls_within_original_voltage_box']
            assert row['final_verification']['n1_overload_mw']<.01
    stats={};per_case={};outcome_stable={}
    for arm in ('direct','selected'):
        cases=[[rows[r,sid,arm] for r in range(3)] for sid in ids]
        times=np.array([[c['runtime_sec'] for c in case] for case in cases])
        per_case[arm]=times.mean(axis=1)
        outcome_stable[arm]=all(len({c['passed'] for c in case})==1 for case in cases)
        backend=[b for case in cases for c in case for b in c.get('voltage_backend_calls',[])]
        stats[arm]={'n_cases':100,'repeats':3,'accepted_per_repeat':[sum(case[r]['passed'] for case in cases) for r in range(3)],
            'mean_sec':float(times.mean()),'median_case_mean_sec':float(np.median(times.mean(axis=1))),
            'p90_case_mean_sec':float(np.quantile(times.mean(axis=1),.9)),
            'mean_per_repeat_sec':times.mean(axis=0).tolist(),
            'optimizer_calls_per_repeat':[sum(case[r]['optimizer_called'] for case in cases) for r in range(3)],
            'policy_attempts_per_repeat':[sum(case[r]['policy_attempted'] for case in cases) for r in range(3)],
            'preparation_failures_per_repeat':[sum(case[r]['preparation_failed'] for case in cases) for r in range(3)],
            'backend_sources':dict(Counter(b['source'] for b in backend)),
            'feature_mean_sec_all_cases':sum(b['runtime_sec'] for b in backend)/300,
            'fresh_final_checks':sum(not c['preparation_failed'] and c.get('final_check_sec',0)>0 for case in cases for c in case)}
        assert stats[arm]['fresh_final_checks']==300-sum(stats[arm]['preparation_failures_per_repeat'])
    differences=per_case['direct']-per_case['selected']
    rng=np.random.default_rng(2026101803)
    boot=np.mean(differences[rng.integers(0,100,size=(10000,100))],axis=1)
    lost=[[sid for sid in ids if rows[r,sid,'direct']['passed'] and not rows[r,sid,'selected']['passed']] for r in range(3)]
    gained=[[sid for sid in ids if not rows[r,sid,'direct']['passed'] and rows[r,sid,'selected']['passed']] for r in range(3)]
    eligible=[sid for sid in ids if rows[0,sid,'selected']['policy_attempted']]
    indices=[ids.index(sid) for sid in eligible]
    subgroup={arm:{'n_cases':len(indices),'mean_sec':float(per_case[arm][indices].mean()) if indices else None}
              for arm in ('direct','selected')}
    summary={'complete':True,'source_sha256':sha(DEST/'test.json'),'protocol_sha256':sha(DEST/'protocol.json'),
        'analysis_environment':{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
            'packages':{p:importlib.metadata.version(p) for p in ('numpy','scipy','pandapower','torch')}},
        'rule':protocol['rule'],'stats':stats,'outcomes_stable_across_repeats':outcome_stable,
        'lost_acceptances_per_repeat':lost,'additional_acceptances_per_repeat':gained,
        'mean_paired_saving_sec':float(differences.mean()),
        'paired_case_bootstrap_ci95_sec':np.quantile(boot,[.025,.975]).tolist(),
        'relative_mean_reduction_percent':100*(1-stats['selected']['mean_sec']/stats['direct']['mean_sec']),
        'cases_faster_on_three_repeat_mean':int(np.count_nonzero(differences>0)),
        'gate_selected_subgroup':subgroup,
        'mean_paired_saving_excluding_first_case_sec':float(differences[1:].mean()),
        'last_two_repeat_mean_sec':{a:mean(s['mean_per_repeat_sec'][1:]) for a,s in stats.items()},
        'last_two_repeat_paired_saving_sec':mean(stats['direct']['mean_per_repeat_sec'][1:])-mean(stats['selected']['mean_per_repeat_sec'][1:]),
        'positive_mean_with_no_acceptance_loss':bool(differences.mean()>0 and not any(lost)),
        'interpretation':'One frozen validation-selected analytic-feature/short-attempt variant, evaluated on a separate seed. Three timing repeats are not 300 independent operating conditions. All first-use online feature costs and fresh final checks included.'}
    write(DEST/'summary.json',summary)
    lines=['# Frozen short-policy deployment timing','',summary['interpretation'],'',
        '| Workflow | Accepted by repeat | Mean s/case | Median case mean | P90 case mean |',
        '|---|---|---:|---:|---:|']
    for arm,s in stats.items():
        lines.append(f'| {arm} | {s["accepted_per_repeat"]} | {s["mean_sec"]:.6f} | {s["median_case_mean_sec"]:.6f} | {s["p90_case_mean_sec"]:.6f} |')
    lines+=['',f'Rule: {summary["rule"]}',
        f'Mean reduction: {summary["relative_mean_reduction_percent"]:.3f}%.',
        f'Paired case-bootstrap 95% interval (direct minus selected, seconds): {summary["paired_case_bootstrap_ci95_sec"]}.',
        f'Lost acceptances: {lost}. Additional acceptances: {gained}.',
        'Original results and rejected development variants are retained. This deployment variant changes the voltage feature; it is not exact finite-difference replay.',
        'The Q-limit active-set test and final-fresh-verification test must pass before interpreting these measurements.']
    (DEST/'TIMING_REVIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(summary)


if __name__=='__main__':main()

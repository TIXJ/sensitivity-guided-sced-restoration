"""Audit targeted follow-ups, keeping conditional diagnoses separate from claims."""
from collections import Counter
import importlib.metadata
import sys
import argparse
import numpy as np
from reviewer_followup import DEST,source_hashes
from revision_common import RESULTS,read,write,sha
from constraint_diagnostics import operation_flags


def initializer_counts(rows):
    counts=Counter()
    for row in rows:
        if row['passed']:counts['accepted']+=1
        elif row.get('preparation_failed'):counts['initializer_failure']+=1
        elif not (row.get('optimizer') or {}).get('returned_candidate',False):counts['nonlinear_solver_failure']+=1
        else:counts['final_verifier_rejection']+=1
    return {key:counts[key] for key in ('accepted','initializer_failure','nonlinear_solver_failure','final_verifier_rejection')}


def paired_summary(payload,n,repeats):
    records={(r['repeat'],r['sample_id'],r['arm']):r for r in payload['rows']}
    ids=list(dict.fromkeys(r['sample_id'] for r in payload['rows']))
    assert len(ids)==n and len(records)==2*n*repeats
    per_case={};stats={}
    for arm in ('direct','selected'):
        cases=[[records[r,sid,arm] for r in range(repeats)] for sid in ids]
        times=np.array([[r['runtime_sec'] for r in case] for case in cases])
        per_case[arm]=times.mean(axis=1)
        stats[arm]={'accepted_per_repeat':[sum(c[r]['passed'] for c in cases) for r in range(repeats)],
            'mean_sec':float(times.mean()),'p90_sec':float(np.quantile(times.mean(axis=1),.9)),
            'mean_per_repeat_sec':times.mean(axis=0).tolist(),
            'actual_opf_calls_per_repeat':[sum((c[r].get('optimizer') or {}).get('actual_opf_calls',0) for c in cases) for r in range(repeats)]}
    delta=per_case['direct']-per_case['selected'];rng=np.random.default_rng(2026101805)
    boot=delta[rng.integers(0,n,size=(10000,n))].mean(axis=1)
    return {'stats':stats,'mean_saving_sec':float(delta.mean()),'ci95_sec':np.quantile(boot,[.025,.975]).tolist(),
        'lost_acceptances_per_repeat':[[sid for sid in ids if records[r,sid,'direct']['passed'] and not records[r,sid,'selected']['passed']] for r in range(repeats)]}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--balanced-sequence',action='store_true');args=parser.parse_args()
    protocol=read(DEST/'protocol.json');assert protocol['source_hashes']==source_hashes()
    paths={name:DEST/(name+'.json') for name in ('dual','warm','tight')}
    if args.balanced_sequence:paths['warm']=RESULTS/'reviewer_sequence_balanced_v2/warm.json'
    data={name:read(path) for name,path in paths.items()}
    assert all(d['complete'] and d['protocol_sha256']==sha(paths[name].parent/'protocol.json') for name,d in data.items())
    if args.balanced_sequence:
        old={(r['sample_id'],r['arm']):r['passed'] for r in read(DEST/'warm.json')['rows']}
        assert all(old[r['sample_id'],r['arm']]==r['passed'] for r in data['warm']['rows'])
    for path,digest in protocol['input_hashes'].items():assert sha(path)==digest,path
    for row in data['dual']['rows']:
        opt=row.get('optimizer')
        if opt:
            assert opt['actual_opf_calls']<=8
            assert all(a['actual_opf_calls']<=4 for a in opt['initializer_attempts'])
            assert all(c['max_ip_iterations']==100 for a in opt['initializer_attempts'] for c in a['calls'])
        if row['passed']:
            assert not any(row['constraint_flags'].values())
            assert row['final_check_sec']>0
    dual=paired_summary(data['dual'],75,3)
    warm={};sequence={}
    assert len(data['warm']['rows'])==200
    for arm in ('direct','selected'):
        rows=[r for r in data['warm']['rows'] if r['arm']==arm]
        assert len({(r['pattern'],r['period']) for r in rows})==100
        sequence[arm]=np.array([sum(r['runtime_sec'] for r in rows if r['pattern']==p) for p in range(10)])
        warm[arm]={'accepted':sum(r['passed'] for r in rows),'mean_sec':float(np.mean([r['runtime_sec'] for r in rows])),
            'first_period_mean_sec':float(np.mean([r['runtime_sec'] for r in rows if r['period']==0])),
            'later_period_mean_sec':float(np.mean([r['runtime_sec'] for r in rows if r['period']>0])),
            'mean_ten_period_total_sec':float(sequence[arm].mean()),
            'mean_sequence_total_excluding_first_pattern_sec':float(sequence[arm][1:].mean()),
            'preparation_failures':sum(r['preparation_failed'] for r in rows),
            'policy_attempts':sum(r.get('policy_attempted',False) for r in rows),
            'feature_calls':sum(len(r.get('voltage_backend_calls',[])) for r in rows)}
        assert warm[arm]['policy_attempts']==warm[arm]['feature_calls']
        for row in rows:
            if row['passed']:assert not any(row['constraint_flags'].values()) and row['final_check_sec']>0
    delta=sequence['direct']-sequence['selected'];rng=np.random.default_rng(2026101806)
    warm['paired_mean_sequence_saving_sec']=float(delta.mean())
    warm['sequence_cluster_ci95_sec']=np.quantile(delta[rng.integers(0,10,size=(10000,10))].mean(axis=1),[.025,.975]).tolist()
    tight={}
    assert len(data['tight']['rows'])==45
    for arm in ('basis_40','basis_80','full_joint'):
        rows=[r for r in data['tight']['rows'] if r['arm']==arm]
        assert len(rows)==15
        tight[arm]={'accepted':sum(r['passed'] for r in rows),'n':15,
            'linear_status_counts':dict(Counter(r['linear']['status'] for r in rows if 'linear' in r))}
        for r in rows:
            if r['passed']:
                info=r['final_verification'] if arm=='full_joint' else r['voltage_repair']['final_verification']
                assert not any(operation_flags(info).values()),r['sample_id']
            if arm!='full_joint' and r['passed']:
                assert r['actual_l2_mw']<=r['radius_mw']+1e-5 and r['span_residual_mw']<1e-6
    old=read(RESULTS/'initialization_economic_v1/results.json')
    initialization={arm:initializer_counts([r for r in old['rows'] if r['arm']==arm]) for arm in ('gat','online_linear_sced')}
    summary={'complete':True,'sources_sha256':{name:sha(path) for name,path in paths.items()},
        'source_paths':{name:str(path) for name,path in paths.items()},'balanced_sequence_origins':args.balanced_sequence,
        'analysis_environment':{'python':sys.version,'packages':{p:importlib.metadata.version(p) for p in ('numpy','scipy','pandapower','torch')}},
        'protocol_sha256':sha(DEST/'protocol.json'),'initializer_source_sha256':sha(RESULTS/'initialization_economic_v1/results.json'),
        'dual':dual,'warm':warm,'tight':tight,'initialization_failure_stages':initialization,
        'interpretation':'No new policy training or gate selection. Dual timing repeats are 75 conditions, not 225 independent cases. Warm uncertainty is clustered by ten spatial patterns. Tight-band sequential-search failures are not infeasibility proofs.'}
    output=RESULTS/'reviewer_sequence_balanced_v2/combined_summary.json' if args.balanced_sequence else DEST/'summary.json'
    write(output,summary)
    print(summary)


if __name__=='__main__':main()

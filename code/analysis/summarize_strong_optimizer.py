"""Paired summaries without treating repeated trials as independent scenarios."""
import numpy as np
from revision_common import RESULTS,read,write,sha


def stats(values):
    x=np.asarray(values,dtype=float)
    return {'mean':float(x.mean()),'median':float(np.median(x)),'p10':float(np.quantile(x,.1)),
            'p90':float(np.quantile(x,.9)),'std':float(x.std(ddof=1))}


def main():
    root=RESULTS/'strong_optimizer_v1'; source=root/'results.json'; data=read(source)
    assert data['complete'] and len(data['rows'])==900
    rows=data['rows']; ids=sorted({r['sample_id'] for r in rows}); arms=data['protocol']['arms']
    summary={'source_sha256':sha(source),'n_independent_cases':75,'repeats_per_case':3,'arms':{},'paired':{}}
    case_means={}
    for arm in arms:
        group=[r for r in rows if r['arm']==arm]
        accepted=[sum(r['passed'] for r in group if r['repetition']==rep) for rep in range(3)]
        calls=[sum(r['optimizer_called'] for r in group if r['repetition']==rep) for rep in range(3)]
        case_means[arm]={sid:np.mean([r['runtime_sec'] for r in group if r['sample_id']==sid]) for sid in ids}
        summary['arms'][arm]={'acceptance_by_repeat':accepted,'optimizer_calls_by_repeat':calls,
            'failure_rate_by_repeat':[(75-n)/75 for n in accepted],
            'runtime_all_trials':stats([r['runtime_sec'] for r in group]),
            'runtime_case_means':stats(list(case_means[arm].values()))}
    for init in ('flat','pf'):
        direct,gated='direct_'+init,'gated_'+init
        differences=[case_means[direct][sid]-case_means[gated][sid] for sid in ids]
        summary['paired'][init]={'direct_minus_gated_sec':stats(differences),
            'faster_cases':sum(v>0 for v in differences),
            'mean_reduction_pct':100*np.mean(differences)/np.mean(list(case_means[direct].values())),
            'case_differences':dict(zip(ids,differences))}
    write(root/'summary.json',summary)
    print({a:(r['acceptance_by_repeat'],r['runtime_case_means']) for a,r in summary['arms'].items()},flush=True)
    print('Paired reductions:',{k:v['mean_reduction_pct'] for k,v in summary['paired'].items()})


if __name__=='__main__': main()

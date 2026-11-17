"""Analyze final-configuration remeasurement without relabeling the test set."""
import numpy as np
from revision_common import RESULTS, read, write, sha
from analyze_reviewer_followup import paired_summary
from analyze_deployment_accounting import analyze
from remeasure_spatial_dual import DEST, CONDITIONS


def main():
    raw=read(DEST/'results.json');protocol=read(DEST/'protocol.json')
    assert raw['complete'] and raw['protocol_sha256']==sha(DEST/'protocol.json')
    assert sha(CONDITIONS)==protocol['conditions_sha256']
    result=paired_summary(raw,100,3)
    records={(r['repeat'],r['sample_id'],r['arm']):r for r in raw['rows']}
    ids=list(dict.fromkeys(r['sample_id'] for r in raw['rows']))
    for arm in ('direct','selected'):
        stats=result['stats'][arm]
        stats['accepted_ids_per_repeat']=[[sid for sid in ids if records[r,sid,arm]['passed']] for r in range(3)]
        stats['outer_optimizer_calls_per_repeat']=[sum(records[r,sid,arm]['optimizer_called'] for sid in ids) for r in range(3)]
        stats['preparation_failures_per_repeat']=[sum(records[r,sid,arm]['preparation_failed'] for sid in ids) for r in range(3)]
        for r in raw['rows']:
            if r['arm']==arm:
                assert (r.get('optimizer') or {}).get('actual_opf_calls',0)<=8
                if r['passed']:
                    assert not any(r['constraint_flags'].values())
    result['common_accepted_ids_per_repeat']=[[sid for sid in ids if all(records[r,sid,a]['passed'] for a in ('direct','selected'))] for r in range(3)]
    result['gained_acceptances_per_repeat']=[[sid for sid in ids if records[r,sid,'selected']['passed'] and not records[r,sid,'direct']['passed']] for r in range(3)]
    result['relative_mean_reduction_percent']=100*(1-result['stats']['selected']['mean_sec']/result['stats']['direct']['mean_sec'])
    result.update(complete=True,protocol_sha256=sha(DEST/'protocol.json'),conditions_sha256=sha(CONDITIONS),
        raw_file=str(DEST/'results.json'),raw_sha256=sha(DEST/'results.json'),analysis_sha256=sha(__file__),
        scope=protocol['scope'],accounting=analyze(raw),resume_segments=raw.get('resume_segments',[]))
    write(DEST/'summary.json',result)
    print({k:v for k,v in result.items() if k not in ('stats','accounting','common_accepted_ids_per_repeat')})
    for a,s in result['stats'].items():
        print(a,{k:v for k,v in s.items() if k!='accepted_ids_per_repeat'})


if __name__=='__main__':
    main()

"""Audit a matched software remeasurement without treating it as a new holdout."""
import numpy as np
from collections import Counter
import analyze_deployment_timing as audit
from revision_common import RESULTS,read,write,sha

DEST=RESULTS/'shared_voltage_speed_v3'


def main():
    original=read(RESULTS/'implicit_short_policy_v2/test.json')
    current=read(DEST/'test.json')
    before={(r['repeat'],r['sample_id'],r['arm']):r for r in original['rows']}
    differences=[];maximum={}
    for new in current['rows']:
        key=(new['repeat'],new['sample_id'],new['arm']);old=before[key]
        for field in ('passed','preparation_failed','policy_steps','policy_attempted','optimizer_called'):
            if new.get(field)!=old.get(field):differences.append([key,field,old.get(field),new.get(field)])
        for field in ('final_dispatch_mw','final_generator_vm_pu','final_reference_vm_pu'):
            if field in old:
                delta=float(np.max(np.abs(np.asarray(old[field])-np.asarray(new[field]))))
                maximum[field]=max(maximum.get(field,0.),delta)
                if delta>1e-6:differences.append([key,field,delta])
    calls=dict(Counter(c['source'] for row in current['rows'] if row['arm']=='selected'
                       for c in row.get('voltage_backend_calls',[])))
    assert calls=={'shared_solved_ac':60},calls
    assert not differences,differences
    audit.DEST=DEST
    audit.main()
    result=read(DEST/'summary.json')
    result.update(previous_measurement_sha256=sha(RESULTS/'implicit_short_policy_v2/test.json'),
        prior_trajectory_differences=differences,maximum_prior_control_difference=maximum,
        interpretation='Software remeasurement on the same 100 previously evaluated conditions, not another independent test. No gate or model tuning. Both current arms use the same Numba verifier; policy features reuse the exact already solved within-case AC point. All online preparation, fallback and fresh final checks included.')
    write(DEST/'summary.json',result)
    path=DEST/'TIMING_REVIEW.md';text=path.read_text(encoding='utf-8')
    text=text.replace('# Frozen short-policy deployment timing','# Shared-solution implementation remeasurement')
    start=text.index('\n\n')+2;end=text.index('\n\n',start)
    text=text[:start]+result['interpretation']+text[end:]
    text+='\nMaximum control differences from the prior implementation: '+str(maximum)+'.\n'
    text+='No prior acceptance, step count, gate decision, or fallback decision changes.\n'
    path.write_text(text,encoding='utf-8')
    print(result)


if __name__=='__main__':main()

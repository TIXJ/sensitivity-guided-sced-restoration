"""Summarize the validation-frozen joint fallback separately from the original paper."""
from pathlib import Path
import numpy as np

from revision_common import RESULTS,REV,read,sha,write


def main():
    root=RESULTS/'optimization_diagnostics'
    joint=root/'joint_fallback'
    protocol=read(joint/'protocol.json')
    assert protocol['worker_sha256']==sha(Path(__file__).with_name('pilot_joint_ac_lodf.py'))
    assert protocol['driver_sha256']==sha(Path(__file__).with_name('evaluate_joint_fallback.py'))
    assert protocol['val_result_sha256']==sha(joint/'val_summary.json')
    archived=REV/'original_git_snapshot/revision_completed_20261015_before_optimization/reported_results/full_sac.json'
    assert sha(archived)==sha(RESULTS/'full_sac.json')
    report={'protocol_sha256':sha(joint/'protocol.json'),'original_results_unchanged':True,
        'contract':'Separate full-generator optimization fallback. Voltage cumulative/device bounds '
            'and original acceptance limits retained. The fallback does not retain the policy 32D/40MW restriction.',
        'evaluations':{}}
    for scope in ('val','test'):
        payload=read(joint/f'{scope}_summary.json')
        assert payload['complete']
        if scope=='test': assert payload['protocol_sha256']==report['protocol_sha256']
        newly=[]; times=[]; minimum_margin=[]
        for row in payload['rows']:
            assert not row['baseline_passed'] or row['passed']
            if not row['fallback_attempted']: continue
            assert row['fallback_sha256']==sha(row['fallback_result'])
            result=read(row['fallback_result'])
            assert result['passed']==row['passed']
            if 'runtime_sec' in result: times.append(result['runtime_sec'])
            if not row['passed']: continue
            info=result['final_verification']
            assert info['converged'] and np.isfinite(info['bus_vm_pu']).all()
            assert all(info[key]==0 for key in ('v_violations','n1_violations','overloaded_lines',
                      'trafo_overloaded','reference_source_violations','generator_bound_violations','scheduled_balance_violations'))
            assert info['n1_overload_mw']<.01 and result['controls_within_bounds']
            newly.append({'sample_id':row['sample_id'],'gat_relative_l2_mw':result['gat_relative_l2_mw']})
            minimum_margin.append(min(info['v_min']-.933,1.09-info['v_max']))
        terminal=read(root/f'{"validation" if scope=="val" else "test"}_terminal_repair.json')
        report['evaluations'][scope]={**payload['summary'],
            'terminal_qp_augmented_passed':terminal['summary']['combined_passed'],
            'newly_accepted':newly,'smallest_voltage_margin_pu':min(minimum_margin,default=None),
            'mean_optimizer_and_verification_sec_per_fallback':float(np.mean(times)) if times else None,
            'timing_caveat':payload['timing_scope']}
    preventive_results=[]
    for demand in (16000,16200,16400):
        preventive=root/f'preventive_voltage_{demand}.json'
        if not preventive.exists(): continue
        raw=read(preventive); assert raw['complete']
        for candidate in raw['candidates']:
            if candidate['base_passed']:
                assert len(candidate['outages'])==len(raw['valid_outages'])
                assert [r['outage_position'] for r in candidate['outages']]==raw['valid_outages']
        preventive_results.append({
            'scope':'one preselected accepted validation point; not a population estimate',
            'sample_id':raw['sample_id'],'demand_mw':demand,'outages':len(raw['valid_outages']),
            'source_sha256':sha(preventive),
            'candidates':[{'name':r['name'],'base_passed':r['base_passed'],
                           'post_contingency_passed':r['post_contingency_passed'],
                           'all_valid_line_ac_checks_passed':r['all_valid_line_ac_checks_passed']}
                          for r in raw['candidates']]})
    report['preventive_voltage_pilots']=preventive_results
    write(root/'joint_improvement_summary.json',report)
    print({k:{'original':v['baseline_passed'],'terminal_qp':v['terminal_qp_augmented_passed'],
              'joint_fallback':v['combined_passed']} for k,v in report['evaluations'].items()})
    for case in preventive_results:
        print(case['sample_id'],[(c['name'],c['post_contingency_passed']) for c in case['candidates']])


if __name__=='__main__':
    main()

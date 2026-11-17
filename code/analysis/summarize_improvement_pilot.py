"""Audit the improvement pilot without changing any reported main-experiment result."""
from collections import Counter
from pathlib import Path
import numpy as np

from revision_common import RESULTS, REV, read, sha, write


def main():
    root=RESULTS/'optimization_diagnostics'
    names=['validation_feasibility_pilot','validation_policy_polish','validation_full_reference',
           'validation_terminal_repair','test_terminal_repair','frozen_range_40_positive_controls']
    names += [f'frozen_range_{radius}' for radius in (40,200,400,600)]
    records={name:read(root/(name+'.json')) for name in names}
    assert all(p['complete'] for p in records.values())
    protocol=read(root/'terminal_repair_protocol.json')
    assert records['test_terminal_repair']['protocol_sha256']==sha(root/'terminal_repair_protocol.json')
    assert protocol['source_sha256']==sha(Path(__file__).with_name('evaluate_terminal_repair.py'))
    assert protocol['solver_source_sha256']==sha(Path(__file__).with_name('pilot_validation_feasibility.py'))
    assert records['validation_full_reference']['checkpoint_sha256']==records['test_terminal_repair']['checkpoint_sha256']
    source=RESULTS/'full_sac.json'
    archived=REV/'original_git_snapshot/revision_completed_20261015_before_optimization/reported_results/full_sac.json'
    assert sha(source)==sha(archived), 'Original test result changed'
    out={'purpose':'Small exploratory improvement pilot; not replacement manuscript results',
         'original_test_result_unchanged':True,'protocol_sha256':sha(root/'terminal_repair_protocol.json'),
         'source_hashes':{name:sha(root/(name+'.json')) for name in names},
         'range_controls':[], 'terminal_repair':{}}
    for radius in (40,200,400,600):
        rows=records[f'frozen_range_{radius}']['rows']
        out['range_controls'].append({'radius_mw':radius,'n':len(rows),'passed':sum(r['passed'] for r in rows),
                                      'outcomes':dict(Counter(r['reason'] for r in rows))})
    for scope in ('validation','test'):
        record=records[scope+'_terminal_repair']
        rows=record['rows']
        assert len(rows)==75
        assert all(r['passed'] for r in rows if r['baseline_passed'])
        accepted=[r for r in rows if r['passed'] and not r['baseline_passed']]
        for row in accepted:
            info=row['final_verification']
            assert info['converged'] and info['n1_violations']==0 and info['n1_overload_mw']<0.01
            assert info['v_violations']==0 and info['reference_source_violations']==0
            assert row['gat_relative_l2_mw']<=40+1e-6
        out['terminal_repair'][scope]=record['summary']
        out['terminal_repair'][scope]['accepted_adjustment_norms_mw']=[r['gat_relative_l2_mw'] for r in accepted]
    out['positive_controls']={'n':2,'passed':sum(r['passed'] for r in records['frozen_range_40_positive_controls']['rows'])}
    out['conclusions']=[
        'Increasing the action scale of the unchanged actor did not improve the three preselected validation stress cases.',
        'The QP solver reports the selected 16.8-GW case infeasible in the strict 32D affine span intersected with generator bounds; this does not prove global physical infeasibility or infeasibility after span-breaking projection.',
        'Terminal QP repair, with unchanged voltages and an explicit 40-MW radius, improves validation 41/75 to 44/75 and the previously reported test partition 44/75 to 45/75.',
        'This is a separate SAC-plus-optimization method. Pure SAC results and their timing are unchanged.',
        'Higher-load AC feasibility and full post-contingency AC security remain unresolved; no broad retraining was started.',
    ]
    write(root/'pilot_improvement_summary.json',out)
    print('Verified original results, frozen protocol, 40-MW cap and acceptance of every newly restored point.')
    print(out['terminal_repair'])


if __name__=='__main__':
    main()

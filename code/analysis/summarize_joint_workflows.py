"""Audit paired workflow outcomes and component-inclusive runtime measurements."""
import argparse
import csv
import numpy as np
from revision_common import RESULTS, read, sha, write
from compare_joint_workflows import ARMS, DEST, hashes


def stats(values):
    a = np.asarray(values, dtype=float)
    return {'n': len(a), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'std': float(a.std(ddof=1)), 'p95': float(np.quantile(a, .95))}


def audit(scope, destination=DEST, arms=ARMS, expected_hashes=None):
    source = destination/(scope+'.json')
    data = read(source)
    assert data['complete'] and data['hashes'] == (hashes() if expected_hashes is None else expected_hashes)
    if scope == 'test':
        protocol = destination/'protocol.json'
        assert data['protocol_sha256'] == sha(protocol)
        assert read(protocol)['validation_sha256'] == sha(destination/'val.json')
    grouped = {arm: {r['sample_id']: r for r in data['rows'] if r['arm']==arm} for arm in arms}
    assert len(data['rows'])==75*len(arms) and all(set(g)==set(data['sample_ids']) and len(g)==75 for g in grouped.values())
    pure_path = RESULTS/('full_sac.json' if scope=='test' else 'optimization_diagnostics/validation_full_reference.json')
    pure = {r['sample_id']: r for r in read(pure_path)['rows']}
    phases = ('prediction_projection_sec', 'initial_verification_sec', 'policy_and_gate_sec',
              'optimizer_stage_sec', 'final_verification_sec')
    for row in data['rows']:
        assert abs(sum(row[k] for k in phases)+row.get('range_gate_sec',0)-row['runtime_sec'])<1e-7
        assert all(row[k]>=0 for k in phases)
        if row['passed']:
            assert all(v is False for v in row['constraint_flags'].values()), row['sample_id']
            assert row['final_verification']['controls_within_original_voltage_box']
    no, yes = (grouped[a] for a in arms[:2])
    mismatch = [sid for sid in yes if yes[sid]['policy_gate_passed']!=pure[sid]['passed']]
    assert not mismatch, ('Pure policy gate differs from original evaluation', mismatch)
    shared_invoked = [sid for sid in no if no[sid]['optimizer_called'] and yes[sid]['optimizer_called']]
    for sid in no:
        assert np.allclose(no[sid]['reference_dispatch_mw'], yes[sid]['reference_dispatch_mw'], atol=1e-8, rtol=0)
    for sid in shared_invoked:
        assert no[sid]['passed']==yes[sid]['passed'], sid
        assert np.allclose(no[sid]['final_dispatch_mw'], yes[sid]['final_dispatch_mw'], atol=1e-6, rtol=0), sid
    if len(arms)==3:
        gated = grouped[arms[2]]
        low, high = data['training_demand_range_mw']
        for sid, row in gated.items():
            outside = not(low<=row['demand_mw']<=high)
            assert row['range_gate_skipped_policy']==outside
            expected = no[sid] if outside else yes[sid]
            assert row['passed']==expected['passed'] and row['optimizer_called']==expected['optimizer_called']
            assert np.allclose(row['final_dispatch_mw'],expected['final_dispatch_mw'],atol=1e-6,rtol=0), sid
            if outside:
                assert row['policy_steps']==0
    common = [sid for sid in no if no[sid]['passed'] and yes[sid]['passed']]
    result = {'scope': scope, 'source_sha256': sha(source), 'pure_policy_source_sha256': sha(pure_path),
              'n': 75, 'same_optimizer_outputs_verified_on_shared_calls': len(shared_invoked),
              'pure_policy_gate_mismatches': mismatch, 'arms': {}}
    csv_rows = []
    for arm, rows in grouped.items():
        items = list(rows.values())
        result['arms'][arm] = {'passed': sum(r['passed'] for r in items),
                              'optimizer_calls': sum(r['optimizer_called'] for r in items),
                              'policy_attempts': sum(r['policy_steps']>0 for r in items),
                              'runtime_sec': stats([r['runtime_sec'] for r in items]),
                              'components': {k: stats([r[k] for r in items]) for k in phases},
                              'by_demand': {str(d): {'n':sum(r['demand_mw']==d for r in items),
                                  'passed':sum(r['demand_mw']==d and r['passed'] for r in items),
                                  'mean_runtime_sec':float(np.mean([r['runtime_sec'] for r in items if r['demand_mw']==d]))}
                                  for d in sorted({r['demand_mw'] for r in items})}}
        csv_rows.extend({k: row[k] for k in ('sample_id','arm','demand_mw','passed','optimizer_called','runtime_sec',*phases,
                                            'scheduled_cost','gat_relative_l2_mw')} for row in items)
    mean_no = result['arms'][arms[0]]['runtime_sec']['mean']
    mean_yes = result['arms'][arms[1]]['runtime_sec']['mean']
    result['mean_runtime_ratio_no_sac_over_sac'] = mean_no/mean_yes
    result['paired_runtime_difference_sac_minus_no_sac_sec'] = stats([yes[s]['runtime_sec']-no[s]['runtime_sec'] for s in no])
    result['outcome_pairs'] = {'both_pass':len(common), 'only_no_sac':sum(no[s]['passed'] and not yes[s]['passed'] for s in no),
                               'only_sac_first':sum(yes[s]['passed'] and not no[s]['passed'] for s in no),
                               'both_fail':sum(not yes[s]['passed'] and not no[s]['passed'] for s in no)}
    result['common_accepted'] = {'n':len(common), 'sample_ids':common,
        'mean_cost_no_sac':float(np.mean([no[s]['scheduled_cost'] for s in common])),
        'mean_cost_sac_first':float(np.mean([yes[s]['scheduled_cost'] for s in common])),
        'cost_difference_sac_minus_no_sac': stats([yes[s]['scheduled_cost']-no[s]['scheduled_cost'] for s in common]),
        'mean_gat_relative_l2_no_sac_mw':float(np.mean([no[s]['gat_relative_l2_mw'] for s in common])),
        'mean_gat_relative_l2_sac_first_mw':float(np.mean([yes[s]['gat_relative_l2_mw'] for s in common]))}
    if len(arms)==3:
        gate = grouped[arms[2]]
        both = [s for s in no if no[s]['passed'] and gate[s]['passed']]
        result['gated_contrast'] = {
            'mean_runtime_ratio_no_sac_over_gated':mean_no/result['arms'][arms[2]]['runtime_sec']['mean'],
            'mean_runtime_reduction_pct':100*(1-result['arms'][arms[2]]['runtime_sec']['mean']/mean_no),
            'jointly_accepted':len(both),
            'mean_cost_no_sac':float(np.mean([no[s]['scheduled_cost'] for s in both])),
            'mean_cost_gated':float(np.mean([gate[s]['scheduled_cost'] for s in both])),
            'mean_paired_cost_delta':float(np.mean([gate[s]['scheduled_cost']-no[s]['scheduled_cost'] for s in both])),
            'mean_l2_no_sac_mw':float(np.mean([no[s]['gat_relative_l2_mw'] for s in both])),
            'mean_l2_gated_mw':float(np.mean([gate[s]['gat_relative_l2_mw'] for s in both]))}
    write(destination/(scope+'_summary.json'), result)
    with (destination/(scope+'_paired.csv')).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    if scope=='test' and len(arms)==3:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from revision_common import REV
        fig, ax = plt.subplots(figsize=(3.5,2.6),layout='constrained')
        for arm, label, color in zip(arms,('Joint optimization','SAC then fallback','Training-range gate'),('#596168','#C75A61','#248A75')):
            values=np.sort([r['runtime_sec'] for r in grouped[arm].values()])
            ax.step(np.r_[0,values],np.r_[0,np.arange(1,len(values)+1)/len(values)],
                    where='post',label=label,color=color,linewidth=1.3)
        ax.set(xlabel='End-to-end time (s)',ylabel='Cumulative fraction',ylim=(0,1.03))
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7,frameon=False,loc='lower right')
        ax.grid(axis='y',alpha=.2)
        fig.savefig(REV/'revision_figures/matched_joint_workflow_time.pdf')
        plt.close(fig)
    print(scope, {a:(v['passed'],v['optimizer_calls'],v['runtime_sec']['mean']) for a,v in result['arms'].items()}, flush=True)
    print('Runtime ratio:', result['mean_runtime_ratio_no_sac_over_sac'], 'paired cost:',result['common_accepted'], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=('val','test'), required=True)
    parser.add_argument('--gated', action='store_true')
    args = parser.parse_args()
    if args.gated:
        from compare_joint_gate_workflows import ARMS as gated_arms, DEST as gated_dest, hashes as gated_hashes
        audit(args.scope,gated_dest,gated_arms,gated_hashes())
    else:
        audit(args.scope)


if __name__=='__main__':
    main()

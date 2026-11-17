"""Audit and render completed ablations, projection diagnostics and economics."""
import csv
import numpy as np
import matplotlib.pyplot as plt
from revision_common import RESULTS, REV, CHECKPOINTS, read, sha, write
from completed_result_sections import table

VARIANTS = [
    ('random32','Basis','Random balanced 32D'), ('modes32','Basis','System-wide 32D'),
    ('keys32','Basis','Generator-centered 32D'), ('dim16','Dimension','16 directions'),
    ('dim24','Dimension','24 directions'), ('dim40','Dimension','40 directions'),
    ('split8_24','Split','8 system-wide + 24 local'), ('split20_12','Split','20 system-wide + 12 local'),
    ('radius20','Scale','20 MW'), ('radius60','Scale','60 MW'),
    ('n1_half','Reward',r'Contingency terms $\times0.5$'), ('n1_double','Reward',r'Contingency terms $\times2$'),
    ('voltage_half','Reward',r'Voltage terms $\times0.5$'), ('voltage_double','Reward',r'Voltage terms $\times2$'),
    ('bonus_half','Reward',r'Terminal reward $\times0.5$'), ('bonus_double','Reward',r'Terminal reward $\times2$'),
    ('ac_every1','Feedback','AC-PF every step'), ('ac_every10','Feedback','AC-PF every 10 steps')]


def projection_stats(rows):
    trace=[step for row in rows for step in row.get('trace',[])]
    values=np.asarray([step['offspan_l2_mw'] for step in trace],dtype=float)
    changes=np.asarray([step['projection_change_l2_mw'] for step in trace],dtype=float)
    if not len(values) or not np.isfinite(values).all() or not np.isfinite(changes).all():
        raise ValueError('Projection measurements are missing or non-finite')
    return {'steps':len(trace),'max_offspan_mw':float(values.max()),
            'mean_offspan_mw':float(values.mean()),'above_1e_minus6_mw':int((values>1e-6).sum()),
            'max_recovery_change_mw':float(changes.max()),
            'ac_checked_steps':sum(bool(step['ac_checked']) for step in trace)}


def sci(value):
    mantissa,exponent=f'{value:.2e}'.split('e')
    return rf'{mantissa}\times10^{{{int(exponent)}}}'


def csv_file(name, records):
    if not records:return
    with (RESULTS/name).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)


def generate_final(summaries, seed_stats):
    if not all(name+'_full' in summaries for name,_,_ in VARIANTS):return
    reference=read(RESULTS/'full_sac.json')
    ids=[r['sample_id'] for r in reference['rows']]
    audit={}; records=[]
    for name,group,label in [('reference','Reference','32D, 12+20, 40 MW')]+VARIANTS:
        key='full_sac' if name=='reference' else name+'_full'
        payload=reference if name=='reference' else read(RESULTS/(key+'.json'))
        if not payload.get('complete') or [r['sample_id'] for r in payload['rows']]!=ids:
            raise ValueError(f'Incomplete/mismatched ablation: {name}')
        if name!='reference':
            for job in (name,key):
                if read(RESULTS/'job_records'/(job+'.json'))['returncode']!=0:
                    raise ValueError(f'Failed task: {job}')
            training=read(RESULTS/name/'training_results.json')
            if [h['update'] for h in training['history']]!=list(range(1,101)):
                raise ValueError(f'Incomplete training: {name}')
            if payload['checkpoint_sha256']!=sha(RESULTS/name/'best_actor.pth'):
                raise ValueError(f'Checkpoint mismatch: {name}')
            selected=max(training['validation'],key=lambda r:(r['passed'],r['avg_reward']))
            audit[name]={'selected_update':selected['update'],'validation_passed':selected['passed'],
                         'result_sha256':sha(RESULTS/(key+'.json')),
                         'checkpoint_sha256':payload['checkpoint_sha256'],
                         'training_sha256':sha(RESULTS/name/'training_results.json')}
        s=summaries[key];p=projection_stats(payload['rows'])
        records.append({'variant':name,'group':group,'all_passed':s['passed'],'all_n':s['n'],
                        'screened_passed':s['screened']['passed'],'screened_n':s['screened']['n'],
                        'mean_steps':s['steps']['mean'],'mean_runtime_sec':s['runtime_sec']['mean'],
                        'ac_checked_steps':p['ac_checked_steps'],'rollout_steps':p['steps'],
                        'ac_check_fraction':p['ac_checked_steps']/p['steps']})
    write(RESULTS/'ablation_audit.json',{'runs':audit,'results':records,
          'note':'Single seed 42 per diagnostic; validation-selected checkpoints; no failed run discarded. Evaluation return uses the common evaluator reward, not each training reward.'})
    csv_file('ablation_comparison.csv',records)
    body=[]
    labels={name:label for name,_,label in VARIANTS}
    for r in records:
        label='Reference (32D, 12+20)' if r['variant']=='reference' else labels[r['variant']]
        body.append([label,f"{r['all_passed']}/75",f"{r['screened_passed']}/27",f"{r['mean_steps']:.2f}"])
    prose=(r'\paragraph{\rev{Action and training controls.}} '
           r'\rev{Table~\ref{tab:revision_ablations} reports one-factor controls with training seed 42, the same partitions, and validation-based checkpoint selection. '
           r'The random balanced basis uses construction seed 20260721. Dimension tests use system-wide/local splits of 6/10, 9/15 and 15/25 for 16, 24 and 40 directions; changing dimension therefore also changes this split. '
           r'The contingency controls jointly scale the potential, progress and residual weights (300, 600, 50); the voltage controls jointly scale the progress, sum-gap and maximum-gap weights (4000, 5000, 10000). '
           r'The terminal controls scale the success reward of 20000. Each group is multiplied by 0.5 or 2 while the other weights and replay scale remain fixed. '
           r'No variant replaces the original reference policy after test evaluation.}'+'\n\n')
    prose+=table('Single-seed controls on all 75 candidates and the 27-case screened subset. Steps include failures.',
                 'tab:revision_ablations','Variant & All & Screened & Steps',body,'lrrr')
    prose+=('\n\n'+r'\rev{The reference accepts 44/75 upstream cases, compared with 34/75 for random directions, 45/75 for system-wide directions alone and 42/75 for generator-centered directions alone. '
            r'All four accept 27/27 screened cases. Thus, this run supports the usefulness of sensitivity-based coordinates on the broader population but does not show that combining both direction types is necessary. '
            r'The 24-direction and 8/24 variants accept 43/75 and 45/75, respectively. The 40-direction run accepts none; its outcome concerns training with the tested parameterization, not physical infeasibility of a larger space. '
            r'The results do not establish 32 directions or the 12/20 split as optimal. Increasing the nominal scale from 20 to 40 to 60~MW gives 26/75, 44/75 and 45/75. '
            r'The six separate reward-weight controls accept 44--45/75 upstream cases and all 27 screened cases. This indicates limited sensitivity within these tested coefficient ranges, not across arbitrary reward designs.}')
    ref=projection_stats(reference['rows'])
    write(RESULTS/'projection_audit.json',{'reference_seed42':ref,'source_sha256':sha(RESULTS/'full_sac.json')})
    prose+='\n\n'+r'\rev{Across the '+str(ref['steps'])+r' recorded reference-policy steps, the largest post-recovery component outside the basis span is $'+sci(ref['max_offspan_mw'])+r'$~MW; none exceeds $10^{-6}$~MW. The largest change from the proposed to the recovered correction is $'+sci(ref['max_recovery_change_mw'])+r'$~MW. This confirms numerical span preservation for these recorded trajectories, without changing the general observation that active bounds can cause departure from the span.}'
    (REV/'revision_ablation_results.tex').write_text(prose,encoding='utf-8')

    cadence=[next(r for r in records if r['variant']==name) for name in ('ac_every1','reference','ac_every10')]
    body=[[str(interval),f"{r['all_passed']}/75",f"{r['ac_checked_steps']}/{r['rollout_steps']}",f"{r['mean_runtime_sec']:.3f}"] for interval,r in zip((1,5,10),cadence)]
    prose=(r'\paragraph{\rev{Power-flow feedback frequency.}} '
           r'\rev{Table~\ref{tab:revision_cadence} changes the nominal refresh interval while retaining event-triggered checks and final verification. '
           r'The measured in-loop check counts, rather than the nominal interval alone, determine computation. Each variant is separately trained; the timing difference therefore also reflects its trajectory. '
           r'Runtime includes reset and final acceptance checks; the check-count column counts only rollout steps with AC feedback. These measurements are from the new evaluation runs, not the earlier timing audit.}'+'\n\n')
    prose+=table('Nominal AC-PF refresh interval, observed checks and measured mean evaluation time over all 75 cases.',
                 'tab:revision_cadence','Interval & Passed & Checks/steps & Time (s)',body,'rrrr')
    (REV/'revision_cadence_results.tex').write_text(prose,encoding='utf-8')

    priced=read(RESULTS/'reference_exchange_price_sensitivity.json')['rows']
    target=[r for r in priced if r['method']=='scopf']
    body=[[f"{r['symmetric_exchange_price']:.0f}",f"{r['sac_mean_cost_including_exchange']:.0f}",f"{r['mean_cost_including_exchange']:.0f}",f"{-r['mean_paired_difference_vs_sac']:.0f}"] for r in target]
    prose=(r'\paragraph{\rev{Reference-source exchange cost.}} '
           r'\rev{Table~\ref{tab:revision_exchange} prices the signed reference-source exchange on the 41 upstream cases accepted by both SAC and preventive optimization. '
           r'Positive exchange denotes supply from the external grid; negative exchange receives an export credit. Equal import and export prices of 0, 25 and 50 cost units per MW per dispatch interval are hypothetical scenarios, not market observations. '
           r'The operating points are held fixed; neither method is re-optimized for these prices. The preventive solutions retain lower priced cost on this paired subset, so the contribution is fast feasibility restoration rather than cost minimization. '
           r'The deterministic-repair records contain only absolute reference-power magnitudes and cannot support this signed-exchange calculation.}'+'\n\n')
    prose+=table('Priced costs on 41 jointly accepted cases. Difference is SAC minus preventive optimization; all costs are per dispatch interval.',
                 'tab:revision_exchange','Price & SAC & Preventive & Difference',body,'rrrr')
    (REV/'revision_exchange_results.tex').write_text(prose,encoding='utf-8')

    baseline={r['sample_id']:r for r in reference['rows'] if r['passed']}
    starts=[]
    for name in ('start_label_v2','start_economic_v2','start_previous_v2'):
        rs=read(RESULTS/(name+'.json'))['rows']
        pairs=[r for r in rs if r['passed'] and r['sample_id'] in baseline]
        starts.append({'initialization':name,'common_accepted_cases':len(pairs),
                       'scheduled_cost_difference_vs_gat':float(np.mean([r['scheduled_cost']-baseline[r['sample_id']]['scheduled_cost'] for r in pairs])) if pairs else None})
    write(RESULTS/'initialization_paired_costs.json',{'results':starts,'note':'Each alternative is paired separately with GAT initialization on jointly accepted IDs; subsets differ. Policy is not retrained.'})
    prose=(r'\rev{Scheduled-cost comparisons for changed initialization use only jointly accepted IDs. The stored-label and previous-point starts share '
           +str(starts[0]['common_accepted_cases'])+' and '+str(starts[2]['common_accepted_cases'])+r' accepted cases with GAT initialization, respectively. Their mean scheduled-cost differences relative to the GAT-started policy are '
           +f"{starts[0]['scheduled_cost_difference_vs_gat']:.0f}"+' and '+f"{starts[2]['scheduled_cost_difference_vs_gat']:.0f}"+r' cost units. These are different paired subsets, not a common-population ranking. No accepted-cost comparison is defined for the economic-dispatch start, which accepts no cases.}')
    (REV/'revision_initialization_costs.tex').write_text(prose,encoding='utf-8')

    def mean_sd(stat,digits=2,scale=1.):
        if stat['mean'] is None:return '---'
        return f"${stat['mean']/scale:.{digits}f}\\pm{stat['between_seed_std']/scale:.{digits}f}$"
    body=[]; detail=[]
    for method,label in [('sac','SAC-32D'),('full68','SAC-68D'),('ddpg','DDPG'),('td3','TD3')]:
        s=seed_stats[method]['screened']
        body.append([label,mean_sd(s['steps']),mean_sd(s['episode_return'],2,1000),mean_sd(s['runtime_sec'],3)])
        for seed in (42,1,2):
            key='full_'+method if seed==42 else f'{method}_seed{seed}_full'
            for scope,stats in [('all',summaries[key]),('screened',summaries[key]['screened'])]:
                detail.append({'method':label,'seed':seed,'scope':scope,'n':stats['n'],'passed':stats['passed'],
                               'mean_steps':stats['steps']['mean'],'mean_return':stats.get('episode_return',{}).get('mean'),
                               'return_valid_n':stats.get('episode_return',{}).get('n_valid'),
                               'mean_runtime_sec':stats['runtime_sec']['mean'],
                               'accepted_mean_scheduled_cost':stats.get('scheduled_cost',{}).get('mean')})
    csv_file('seed_detailed_metrics.csv',detail)
    prose=(r'\rev{Table~\ref{tab:revision_seed_metrics} complements acceptance with seed-level mean correction steps, return and runtime on the same 27 screened IDs. '
           r'Every episode, including failed episodes, contributes to steps and time. Return values use the common unscaled evaluation reward. '
           r'Runtime values are newly measured and need not reproduce the earlier microbenchmark. SAC has a mean accepted scheduled cost of '
           +f"{seed_stats['sac']['screened']['scheduled_cost']['mean']:.0f}"+r' cost units with a between-seed standard deviation of '
           +f"{seed_stats['sac']['screened']['scheduled_cost']['between_seed_std']:.0f}"+r'. The other policies each have a seed with no accepted cases, so their three-seed accepted-cost means are undefined rather than zero. Per-seed costs and valid-return counts for both test populations accompany the results.}'+'\n\n')
    prose+=table('Mean $\pm$ sample standard deviation across three seed-level means on 27 screened test cases. Return is shown in thousands.',
                 'tab:revision_seed_metrics','Policy & Steps & Return/1000 & Time (s)',body,'lrrr')
    (REV/'revision_seed_metric_results.tex').write_text(prose,encoding='utf-8')
    write(RESULTS/'revision_completion_analysis.json',{'complete':True,'ablation_variants':18,
          'reference_projection':ref,'cadence':cadence,'exchange_comparison':target,
          'raw_result_policy':'No checkpoints or raw evaluation files changed. All attempted variants retained.'})

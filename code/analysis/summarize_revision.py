"""Rebuild reviewer figures/tables from measured records, never invented results."""
from collections import Counter, defaultdict
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figure_typography import configure_matplotlib, plot_contingency_errors
configure_matplotlib()
from revision_common import *
from completed_result_sections import generate
from final_analysis import generate_final

def rows(path): return read(path)['rows']

def summarize(rs):
    ok=sum(bool(r['passed']) for r in rs)
    out={'n':len(rs),'passed':ok,'pass_rate':ok/len(rs),'wilson95':wilson(ok,len(rs)),'failures':dict(Counter(r.get('reason','unknown') for r in rs if not r['passed']))}
    for key in ('steps','episode_return','runtime_sec','scheduled_cost'):
        eligible=[r for r in rs if key!='scheduled_cost' or r['passed']]
        v=[float(r[key]) for r in eligible if r.get(key) is not None and np.isfinite(float(r[key]))]
        if v: out[key]={'n_valid':len(v),'n_missing':len(eligible)-len(v),'mean':float(np.mean(v)),'std':float(np.std(v,ddof=1)) if len(v)>1 else 0.}
    return out

def across_seeds(results):
    out={'n_seeds':len(results),'pass_rate_mean':float(np.mean([s['pass_rate'] for s in results])),
         'pass_rate_std':float(np.std([s['pass_rate'] for s in results],ddof=1)) if len(results)>1 else 0.}
    for key in ('steps','episode_return','runtime_sec','scheduled_cost'):
        vals=[s[key]['mean'] for s in results if key in s]
        out[key]={'n_valid_seeds':len(vals),'mean':float(np.mean(vals)) if len(vals)==len(results) else None,
                  'between_seed_std':float(np.std(vals,ddof=1)) if len(vals)==len(results) and len(vals)>1 else None}
    return out

def seed_artifacts(summaries, seed_stats, expected_ids):
    audit={}; csv_rows=[]
    fig,axes=plt.subplots(2,2,figsize=(7.2,4.5),layout='constrained',sharex=True,sharey=True)
    colors={42:'#555555',1:'#248A75',2:'#C75A61'}
    names={'sac':'SAC-32D','full68':'SAC-68D','ddpg':'DDPG','td3':'TD3'}
    for ax,(method,checkpoint) in zip(axes.flat,CHECKPOINTS.items()):
        if method not in seed_stats: continue
        for seed in (42,1,2):
            name='full_'+method if seed==42 else f'{method}_seed{seed}_full'
            folder=checkpoint.parent if seed==42 else RESULTS/f'{method}_seed{seed}'
            evaluation=read(RESULTS/(name+'.json'))
            sample_ids=[r['sample_id'] for r in evaluation['rows']]
            if len(sample_ids)!=len(expected_ids) or set(sample_ids)!=expected_ids:
                raise ValueError(f'Seed evaluation sample IDs differ: {name}')
            if evaluation['checkpoint_sha256']!=sha(folder/'best_actor.pth'):
                raise ValueError(f'Seed checkpoint hash differs: {name}')
            training=read(folder/'training_results.json')
            history=training['history']; validation=training['validation']
            if [r['update'] for r in history]!=list(range(1,101)):
                raise ValueError(f'Incomplete training history: {name}')
            if seed!=42:
                for job in (f'{method}_seed{seed}',name):
                    record=read(RESULTS/'job_records'/(job+'.json'))
                    if record['returncode']!=0: raise ValueError(f'Failed seed job: {job}')
                command=read(RESULTS/'job_records'/f'{method}_seed{seed}.json')['command']
                if command[command.index('--seed')+1]!=str(seed):
                    raise ValueError(f'Seed argument mismatch: {name}')
            selected=max(validation,key=lambda r:(r['passed'],r['avg_reward']))
            audit[name]={'seed':seed,'selected_update_from_validation':selected['update'],
                         'best_validation_passed':selected['passed'],'validation_total':selected['total'],
                         'last_validation_passed':validation[-1]['passed'],'training_updates':len(history),
                         'training_sha256':sha(folder/'training_results.json'),
                         'evaluation_sha256':sha(RESULTS/(name+'.json')),
                         'checkpoint_sha256':evaluation['checkpoint_sha256'],
                         'seed_source':'original reported run' if seed==42 else 'verified command argument'}
            for scope,s in [('all_upstream_test',summaries[name]),('screened_test',summaries[name]['screened'])]:
                csv_rows.append({'method':names[method],'seed':seed,'scope':scope,'n':s['n'],'passed':s['passed'],
                                 'pass_rate_pct':100*s['pass_rate'],'mean_steps':s['steps']['mean'],
                                 'selected_update':selected['update'],'best_validation_passed':selected['passed']})
            ax.plot([r['update'] for r in validation],[100*r['passed']/r['total'] for r in validation],
                    marker='o',markersize=3,color=colors[seed],label=f'Seed {seed}',linewidth=1.2)
        ax.set(title=names[method],ylim=(-3,103),xticks=[20,40,60,80,100])
        ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.supxlabel('Outer training iteration');fig.supylabel('Validation acceptance (%)')
    fig.savefig(REV/'revision_figures/seed_validation_curves.pdf');plt.close(fig)
    write(RESULTS/'seed_audit.json',{'runs':audit,'note':'All attempted seeds retained. Selected iteration follows validation pass count then reward, not test performance.'})
    if csv_rows:
        with (RESULTS/'seed_comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)

def main():
    figures=REV/'revision_figures'; figures.mkdir(exist_ok=True)
    old=rows(ROOT/'paper_results/case300_fixed_slack_v1_full_test_audit.json')
    screened={s['sample_id'] for s in read(DATASET)['samples']}
    candidates=[s for s in read(CANDIDATES)['samples'] if s.get('is_complete_sced',True)]
    candidate_by_id={s['sample_id']:s for s in candidates}
    expected_ids={candidates[i]['sample_id'] for i in read(CANDIDATE_SPLIT)['test_ids']}
    grouped=defaultdict(list)
    for r in old: grouped[float(r['demand_gw'])].append(r)
    audit=[]
    for demand,rs in sorted(grouped.items()):
        selected=[r for r in rs if r['sample_id'] in screened]
        audit.append({'demand_gw':demand,**summarize(rs),'selected_n':len(selected),'selected_passed':sum(r['passed'] for r in selected)})
    write(RESULTS/'existing_full_audit_summary.json',{'source_sha256':sha(ROOT/'paper_results/case300_fixed_slack_v1_full_test_audit.json'),'rows':audit})
    fig,ax=plt.subplots(figsize=(6.6,2.7),layout='constrained')
    x=np.arange(len(audit)); passed=np.array([r['passed'] for r in audit]); failed=np.array([r['n']-r['passed'] for r in audit])
    ax.bar(x,passed,label='Passed',color='#248A75'); ax.bar(x,failed,bottom=passed,label='Failed',color='#C75A61')
    ax.set(xticks=x,xticklabels=[f"{r['demand_gw']:.1f}" for r in audit],xlabel='Total active demand (GW)',ylabel='Upstream test cases')
    for j,r in enumerate(audit): ax.text(j,r['n']+.2,f"{r['passed']}/{r['n']}",ha='center',fontsize=9)
    ax.set_ylim(0,19); ax.legend(frameon=False,ncol=2); fig.savefig(figures/'unfiltered_by_demand.pdf'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7,2.8),layout='constrained')
    for ax,case in zip(axes,(39,118)):
        corrected=RESULTS/'evaluation_contract_v2'/f'case{case}_actions.json'
        rr=rows(corrected if corrected.exists() else ROOT/f'paper_results/case{case}_action_diagnosis.json')
        modes=list(dict.fromkeys(r['mode'] for r in rr))
        modes=[m for m in modes if m in ('full','active_power_only','no_active_power','no_reference_voltage')]
        for m in modes:
            values=np.sort([r['steps'] for r in rr if r['mode']==m]); ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',label=m.replace('_',' '))
        ax.set(title=f'IEEE {case}-bus',xlabel='Correction steps (failures included)',ylabel='Empirical cumulative fraction',xlim=(0,51),ylim=(0,1.05))
    axes[0].legend(fontsize=7,frameon=False); fig.savefig(figures/'smallcase_step_distributions.pdf'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.6,2.8),layout='constrained')
    methods=[('SAC',.7882,447756,1),('Deterministic repair',3.604,447329,1),('Preventive optimization',3.075,438727,1),('Base-case optimization',2.076,434524,0)]
    for name,t,c,rate in methods:
        ax.scatter(t,c/1000,s=75,marker='o' if rate else 'x',label=f'{name} ({rate*100:.0f}% accepted)')
    ax.set(xlabel='End-to-end restoration time (s)',ylabel='Scheduled cost (thousand units)'); ax.legend(fontsize=8,frameon=False)
    fig.savefig(figures/'runtime_cost_feasibility.pdf'); plt.close(fig)
    summaries={}; excluded_runs={}
    for path in sorted(RESULTS.glob('*.json')):
        if path.name.startswith(('existing_','smoke_')) or path.name=='summary.json': continue
        if path.name.startswith(('ac_contingencies','ac_audit_','contract_','excluded_')): continue
        data=read(path)
        if data.get('complete') is False or not data.get('rows'): continue
        if needs_sample_voltage_cache(data.get('settings',{})) and data.get('voltage_cache_contract')!='sample_context_v2':
            excluded_runs[path.stem]='Demand-only voltage Jacobian cache may not match this sample/initialization; use the isolated-cache v2 evaluation.'
            continue
        if data.get('settings',{}).get('initialization')=='previous' and data.get('previous_initialization_contract')!='last_accepted_or_gat':
            excluded_runs[path.stem]='Uses previous rejected terminal dispatches, not the previous accepted point.'
            continue
        if path.stem in {'full_'+name for name in (*CHECKPOINTS,'acopf','scopf','repair')}:
            actual=[r['sample_id'] for r in data['rows']]
            if len(actual)!=len(expected_ids) or set(actual)!=expected_ids:
                raise ValueError(f'Incomplete or mismatched full-test sample IDs: {path}')
        for row in data['rows']:
            if 'scheduled_cost' not in row and 'scheduled_controllable_generation_cost' in row:
                row['scheduled_cost']=row['scheduled_controllable_generation_cost']
        if all('passed' in r for r in data['rows']):
            if all('mode' in r for r in data['rows']):
                summaries[path.stem]={'by_mode':{mode:summarize([r for r in data['rows'] if r['mode']==mode]) for mode in sorted({r['mode'] for r in data['rows']})}}
                continue
            summaries[path.stem]=summarize(data['rows'])
            if path.stem.startswith(('stress_','voltage_0.','start_')):
                preparation_failed=sum(r.get('reason')=='stress_anchor_nonconvergence' for r in data['rows'])
                summaries[path.stem]['preparation_failed']=preparation_failed
                summaries[path.stem]['evaluated_after_preparation']=len(data['rows'])-preparation_failed
                summaries[path.stem]['initial_passed']=sum(bool(r.get('initial_pass',False)) for r in data['rows'])
                totals=[r.get('stress_end_to_end_sec',r['runtime_sec']) for r in data['rows']]
                summaries[path.stem]['including_preparation_mean_sec']=float(np.mean(totals))
            subset=[r for r in data['rows'] if r['sample_id'] in screened]
            if subset: summaries[path.stem]['screened']=summarize(subset)
            if path.stem.startswith('full_'):
                by_demand=defaultdict(list)
                for r in data['rows']:
                    by_demand[float(candidate_by_id[r['sample_id']]['demand_mw'])/1000].append(r)
                summaries[path.stem]['by_demand_gw']={f'{d:.1f}':summarize(rs) for d,rs in sorted(by_demand.items())}
                summaries[path.stem]['source_sha256']=sha(path)
    corrected39=RESULTS/'evaluation_contract_v2/case39_actions.json'
    if corrected39.exists():
        corrected_rows=rows(corrected39)
        summaries['case39_actions']={'by_mode':{mode:summarize([r for r in corrected_rows if r['mode']==mode])
                                                for mode in sorted({r['mode'] for r in corrected_rows})},
                                    'timing_scope':'correction_loop_only; use smallcase_common for matched timings',
                                    'source_sha256':sha(corrected39)}
    common_path=RESULTS/'evaluation_contract_v2/summary.json'
    smallcase_common=read(common_path)['cases'] if common_path.exists() else {}
    seed_stats={}
    for method in CHECKPOINTS:
        names=['full_'+method]+[f'{method}_seed{s}_full' for s in (1,2)]
        available=[summaries[n] for n in names if n in summaries]
        if len(available)==3:
            seed_stats[method]=across_seeds(available)
            seed_stats[method]['screened']=across_seeds([s['screened'] for s in available])
            seed_stats[method]['seed_order']=[42,1,2]
            seed_stats[method]['per_seed']=[{'seed':seed,'passed':s['passed'],'n':s['n'],'screened_passed':s['screened']['passed'],'screened_n':s['screened']['n']} for seed,s in zip((42,1,2),available)]
    if len(seed_stats)==len(CHECKPOINTS): seed_artifacts(summaries,seed_stats,expected_ids)
    write(RESULTS/'summary.json',{'runs':summaries,'smallcase_common':smallcase_common,'excluded_runs':excluded_runs,'between_seed':seed_stats,'note':'Case-level Wilson intervals describe the fixed evaluation pool, not training-seed variation. Steps and times include all evaluated cases; scheduled-cost summaries use accepted cases and must not be compared as paired costs when acceptance sets differ. Stress anchoring failures precede policy execution, not policy failures. Previous initialization uses the last accepted dispatch in fixed evaluation order, with a GAT fallback until one is accepted; it is not a chronological time-series study. Old small-system runs retain historical timing windows; use smallcase_common for cross-method timing comparisons.'})
    with (RESULTS/'stress_and_initialization_comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        fields=['experiment','n','passed','initial_passed','preparation_failed','evaluated_after_preparation','including_preparation_mean_sec','failures']
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for name,s in sorted(summaries.items()):
            if not name.startswith(('stress_','voltage_0.','start_')): continue
            writer.writerow({**{k:s[k] for k in fields if k not in ('experiment','failures')},'experiment':name,'failures':json.dumps(s['failures'])})
    with (RESULTS/'smallcase_comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        fields=['system','source','mode','n','passed','mean_steps','measured_mean_runtime_sec','timing_scope']
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for case in (39,118):
            for source in ('actions','baselines'):
                for mode,s in summaries.get(f'case{case}_{source}',{}).get('by_mode',{}).items():
                    mapped='policy' if source=='actions' and mode=='full' else mode
                    common=smallcase_common.get(str(case),{}).get('modes',{}).get(mapped)
                    measured=common or s
                    writer.writerow({'system':case,'source':source,'mode':mode,'n':measured['n'],'passed':measured['passed'],'mean_steps':measured.get('steps',{}).get('mean'),'measured_mean_runtime_sec':measured.get('runtime_sec',{}).get('mean'),
                                     'timing_scope':'common_smallcase_v2' if common else 'historical_action_mask_timing_not_matched'})
    with (RESULTS/'full_test_comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        fields=['method','scope','n','passed','pass_rate','mean_runtime_sec','mean_steps','failures']
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for method in ('sac','full68','ddpg','td3','acopf','scopf','repair'):
            result=summaries.get('full_'+method)
            if result is None: continue
            scopes={'all_upstream_test':result,'screened_test':result.get('screened')}
            scopes.update({f'demand_{d}_GW':s for d,s in result.get('by_demand_gw',{}).items()})
            for scope,s in scopes.items():
                if s is None: continue
                writer.writerow({'method':method,'scope':scope,'n':s['n'],'passed':s['passed'],'pass_rate':s['pass_rate'],'mean_runtime_sec':s.get('runtime_sec',{}).get('mean'),'mean_steps':s.get('steps',{}).get('mean'),'failures':json.dumps(s['failures'])})
    cost_sources={}
    for name in ('sac','full68','ddpg','td3','acopf','scopf','repair'):
        path=RESULTS/f'full_{name}.json'
        if not path.exists(): continue
        payload=read(path)
        if payload.get('complete') is False: continue
        cost_sources[name]={r['sample_id']:r for r in payload['rows'] if r.get('passed')}
    price_rows=[]
    if 'sac' in cost_sources:
        for method,rs in cost_sources.items():
            if method=='sac': continue
            common=set(cost_sources['sac']) & set(rs)
            for price in (0.,25.,50.):
                totals=[]; sac_totals=[]
                for sid in sorted(common):
                    r=rs[sid]
                    c=r.get('scheduled_cost',r.get('scheduled_controllable_generation_cost'))
                    ref=r.get('reference_p_mw',r.get('reference_source_p_mw'))
                    sac=cost_sources['sac'][sid]
                    sc=sac.get('scheduled_cost'); sr=sac.get('reference_p_mw')
                    # Absolute reference-power magnitudes cannot recover signed exchange.
                    if c is not None and ref is not None and sc is not None and sr is not None and all(np.isfinite(float(v)) for v in (c,ref,sc,sr)):
                        totals.append(float(c)+price*float(ref))
                        sac_totals.append(float(sc)+price*float(sr))
                price_rows.append({'method':method,'reference_method':'sac','common_passed_cases':len(common),'priced_cases':len(totals),'symmetric_exchange_price':price,'mean_cost_including_exchange':float(np.mean(totals)) if totals else None,'sac_mean_cost_including_exchange':float(np.mean(sac_totals)) if sac_totals else None,'mean_paired_difference_vs_sac':float(np.mean(np.asarray(totals)-sac_totals)) if totals else None})
    write(RESULTS/'reference_exchange_price_sensitivity.json',{'assumption':'Hypothetical symmetric import/export prices in the same cost units per MW per dispatch interval; not observed market prices. Each baseline is paired separately with SAC on common accepted IDs having signed reference exchange. Pairwise subsets may differ. Missing signed exchanges are not inferred from absolute magnitudes.','rows':price_rows})
    tab=['\\begin{table}[t]','\\centering','\\caption{Unfiltered upstream test performance by demand level; one fixed policy and 15 candidates per level.}','\\label{tab:revision_unfiltered}','\\scriptsize','\\setlength{\\tabcolsep}{3pt}','\\begin{tabular}{rrrr}','\\toprule','Demand (GW) & All cases & Screened cases & Voltage failures \\\\','\\midrule']
    for r in audit:
        selected=f"{r['selected_passed']}/{r['selected_n']}" if r['selected_n'] else '---'
        tab.append(f"{r['demand_gw']:.1f} & {r['passed']}/{r['n']} & {selected} & {r['failures'].get('voltage_violation',0)} \\\\")
    tab.extend(['\\bottomrule','\\end{tabular}','\\end{table}'])
    tab.insert(1,r'\ifrevisionmarked\color{blue}\fi')
    (REV/'revision_full_audit_table.tex').write_text('\n'.join(tab),encoding='utf-8')
    acpath=RESULTS/'ac_contingencies.json'
    if acpath.exists() and read(acpath).get('complete'):
        ac=read(acpath); rr=ac['rows']; converged=[r for r in rr if r['converged']]
        aggregate={'total_pairs':len(rr),'passed_pairs':sum(r['passed'] for r in rr),'nonconverged_pairs':len(rr)-len(converged),'voltage_violating_pairs':sum(bool(r.get('voltage_violations',0)) for r in rr),'hybrid_mae_mw':float(np.mean([r['hybrid_mae_mw'] for r in converged])),'ac_anchored_mae_mw':float(np.mean([r['ac_anchored_lodf_mae_mw'] for r in converged])),'max_hybrid_error_mw':max(r['hybrid_max_error_mw'] for r in converged),'exhaustive_cases':[]}
        for sid in ac['exhaustive_sample_ids']:
            subset=[r for r in rr if r['sample_id']==sid]
            aggregate['exhaustive_cases'].append({'sample_id':sid,'demand_gw':subset[0]['demand_mw']/1000,'tested_outages':len(subset),'passed_outages':sum(r['passed'] for r in subset),'all_pass':all(r['passed'] for r in subset)})
        write(RESULTS/'ac_audit_summary.json',aggregate)
        prose=(r'\paragraph{Explicit post-contingency checks.} '+r'\rev{The original accepted operating points are held fixed during this audit. One case at each accepted demand level is selected by sample-ID order and checked against all 253 valid line outages. For each of the other accepted cases, the three outages with the highest predicted loading are tested. Reactive-power limits are enforced; each failed solve is retried with a second initialization. '
               +f"Across {aggregate['total_pairs']} tested operating-point/outage pairs, {aggregate['passed_pairs']} pass the post-outage operating checks, {aggregate['nonconverged_pairs']} fail to converge, and {aggregate['voltage_violating_pairs']} converged pairs violate voltage limits. "
               +r'These targeted pairs are not a random contingency sample, so their pass fraction is not an estimate of system-wide reliability. None of the three exhaustively checked operating points passes every outage.}')
        prose+='\n\n'+r'\rev{The combined affine-sensitivity and line-outage prediction has a mean absolute error of '+f"{aggregate['hybrid_mae_mw']:.3f}"+r'~MW over converged pairs and monitored in-service lines. Anchoring the same outage factors to the actual pre-outage alternating-current flows reduces this error to '+f"{aggregate['ac_anchored_mae_mw']:.3f}"+r'~MW. The largest absolute hybrid-model error is '+f"{aggregate['max_hybrid_error_mw']:.2f}"+r'~MW. The difference isolates the contribution of the pre-outage flow approximation under these tested conditions. The linear acceptance test therefore cannot substitute for explicit post-contingency voltage and reactive-power assessment.}'
        (REV/'revision_ac_audit_results.tex').write_text(prose,encoding='utf-8')
        plot_contingency_errors(rr,figures/'contingency_flow_error.pdf')
    generate(summaries,seed_stats)
    generate_final(summaries,seed_stats)
    print('Generated measured summaries, comparison CSVs, figures and completed-result sections.')

if __name__=='__main__': main()

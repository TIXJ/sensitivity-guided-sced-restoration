"""Integrate the final frozen experiment suite only after every required result exists."""
import json
from statistics import mean,stdev,median


def load(here):
    root=here/'experiment_results'
    files={'strong':'strong_optimizer_v1/summary.json','independent':'independent_conditions_v1/results.json',
           'anchor':'final_ac_anchor_v1.json','tight':'tight_full_pf_v1/results.json',
           'initialization':'initialization_economic_v1/results.json',
           'tight_policy':'tight_sac_094_108_seed42_v1_full.json'}
    basis={'hybrid':['full_sac','sac_seed1_full','sac_seed2_full'],
           'modes':['modes32_full','modes32_seed1_v1_full','modes32_seed2_v1_full'],
           'random':['random32_full','random32_seed1_v1_full','random32_seed2_v1_full']}
    paths=[root/p for p in files.values()]+[root/(name+'.json') for names in basis.values() for name in names]
    if not all(p.exists() for p in paths): return None
    result={key:json.loads((root/path).read_text(encoding='utf-8')) for key,path in files.items()}
    if not all(result[k].get('complete') for k in files if k!='strong'): return None
    result['basis']={}
    histories={'hybrid':[here.parents[1]/'rl_sac_fixed_slack_v1_v31_p40_voltage_sensitivity/training_results.json',
                          root/'sac_seed1/training_results.json',root/'sac_seed2/training_results.json'],
               'modes':[root/'modes32/training_results.json',root/'modes32_seed1_v1/training_results.json',root/'modes32_seed2_v1/training_results.json'],
               'random':[root/'random32/training_results.json',root/'random32_seed1_v1/training_results.json',root/'random32_seed2_v1/training_results.json']}
    for family,names in basis.items():
        runs=[json.loads((root/(name+'.json')).read_text(encoding='utf-8')) for name in names]
        if not all(r.get('complete') and len(r['rows'])==75 for r in runs): return None
        rates=[sum(x['passed'] for x in r['rows'])/75 for r in runs]
        result['basis'][family]={'counts':[round(r*75) for r in rates], 'mean_pct':100*mean(rates),'std_pct':100*stdev(rates),
            'steps':[mean(x['steps'] for x in r['rows']) for r in runs],
            'seconds':[mean(x.get('stress_end_to_end_sec',x['runtime_sec']) for x in r['rows']) for r in runs]}
        result['basis'][family]['validation']=[json.loads(p.read_text(encoding='utf-8'))['validation'] for p in histories[family]]
    return result


def table(label,caption,header,rows,columns):
    return '\n'.join([r'\begin{table}[!htbp]',r'\centering',r'\ifrevisionmarked\color{blue}\fi',
        r'\caption{'+caption+'}',r'\label{'+label+'}',r'\scriptsize\setlength{\tabcolsep}{3pt}',
        r'\begin{tabular}{'+columns+'}',r'\toprule',header+r' \\',r'\midrule',
        *[' & '.join(row)+r' \\' for row in rows],r'\bottomrule',r'\end{tabular}',r'\end{table}'])


def grouped(payload,arm,n):
    rows=[r for r in payload['rows'] if r['arm']==arm]
    assert len(rows)==n
    times=sorted(r['runtime_sec'] for r in rows)
    import numpy as np
    return {'passed':sum(r['passed'] for r in rows),'mean':mean(times),'p50':median(times),'p90':float(np.quantile(times,.9)),
            'prep_failed':sum(r.get('preparation_failed',False) for r in rows),
            'calls':sum(r.get('optimizer_called',False) for r in rows)}


def historical_abstract(data):
    from pathlib import Path
    from revision_reviewer_followup import load as followup_load
    current=followup_load(Path(__file__).resolve().parent)
    if current and current.get('verified_once'):
        d=current['dual']['stats']['direct'];g=current['dual']['stats']['selected']
        text=('Security-constrained economic dispatch is repeatedly solved in rolling power-system operation as demand, costs, and operating conditions change. '
            'This paper uses a graph attention network to predict a linear security-constrained dispatch, projects the prediction onto power balance and generator bounds, and then uses soft actor-critic to adjust active power and voltage setpoints. '
            'The policy receives line outage distribution factor-based security feedback at every step and periodically refreshed alternating-current power-flow feedback during correction. '
            'For active-power control, a fixed set of balanced redispatch directions is constructed from line-flow sensitivities and post-contingency gradients. '
            'Acceptance combines base-case alternating-current checks with linear contingency screening. '
            'The reference policy restores 44/75 upstream 300-bus cases; three training seeds restore 41--45. Matched-seed comparisons favor the tested sensitivity-guided bases over the fixed random basis, without establishing an advantage for the mixed allocation. '
            f'With the frozen eight-step gate and a common dual-initialization optimizer, direct and gated workflows accept the same {d["accepted_per_repeat"][0]}/75 cases, averaging {d["mean_sec"]:.3f} and {g["mean_sec"]:.3f}~s. '
            'The original gated workflow is slower on an independent 100-case test. '
            'Separate spatial-profile evaluations assess deployment sensitivity. '
            'Tests on IEEE 39-bus and 118-bus systems further assess the prediction-and-correction workflow with system-specific implementations.')
        from revision_final_consistency import spatial_summary
        spatial=spatial_summary(Path(__file__).resolve().parent)
        if spatial:
            a=spatial['stats']['direct'];b=spatial['stats']['selected'];lo,hi=spatial['ci95_sec']
            outcome=('the paired time-saving interval includes zero' if lo<=0<=hi else
                     'the gated workflow is slower' if hi<0 else 'the paired saving is positive on this batch')
            replacement=(f'On 100 previously examined spatial profiles, both final workflows accept {a["accepted_per_repeat"][0]}/100; {outcome}.')
            text=text.replace('Separate spatial-profile evaluations assess deployment sensitivity.',replacement)
        assert len(text.split())<=200
        return text
    a=data['strong']['arms']; d=a['direct_pf']; g=a['gated_pf']
    ni=grouped(data['independent'],'gated_pf',100)
    di=grouped(data['independent'],'direct_pf',100)
    text=('Repeated economic dispatch requires rapid correction of predicted setpoints under physical operating checks. '
        'We study fixed, balanced, sensitivity-guided coordinates for active-power redispatch, with voltage controls represented separately. '
        'A graph attention network predicts a dispatch; soft actor-critic corrects it using linear line-outage feedback and periodically refreshed base-case alternating-current power flow. '
        'Final acceptance applies base-case operating constraints and a specified linear contingency criterion, not full alternating-current contingency security. '
        'On the IEEE 300-bus system, the reference policy accepts 44 of 75 upstream cases; its three training seeds accept 41--45. '
        f'Using identical power-flow-initialized optimizers, direct optimization and training-range-gated correction average {d["runtime_case_means"]["mean"]:.3f} and {g["runtime_case_means"]["mean"]:.3f}~s over three paired repetitions per case. '
        f'On 100 newly generated spatial-load, reactive-demand and cost perturbations, direct and gated workflows accept {di["passed"]} and {ni["passed"]} cases in {di["mean"]:.3f} and {ni["mean"]:.3f}~s, including new-profile preparation. '
        'Matched-seed tests compare sensitivity and random bases. All 44 original accepted policy points retain linear acceptance when actual base-case flows are used. '
        'Studies on the 39-bus and 118-bus systems assess the prediction-and-correction workflow with system-specific implementations.')
    from pathlib import Path
    from revision_deployment_followup import load as deployment_load
    deployment=deployment_load(Path(__file__).resolve().parent,'shared_voltage_speed_v3')
    if deployment:
        a=deployment['stats']['direct'];b=deployment['stats']['selected']
        addition=(f'A validation-selected short-policy variant accepts {b["accepted_per_repeat"][0]}/100 separate profiles at {b["mean_sec"]:.3f}~s, '
                  f'versus {a["mean_sec"]:.3f}~s for direct optimization; the mean saving is {deployment["relative_mean_reduction_percent"]:.1f}\\%, with a paired time-saving 95\\% interval including zero. ')
        text=text.replace('Matched-seed tests compare sensitivity and random bases. ',addition+'Matched-seed tests compare sensitivity and random bases. ')
    return text


def historical_conclusion(data):
    text=('The results show that fixed, sensitivity-based active-power directions can be used with soft actor-critic to correct repeated security-constrained economic-dispatch predictions. '
        'The reference policy restores 44 of the 75 unfiltered test cases. The 27/27 result applies only to the screened subset. '
        'In the 300-bus comparison on that subset, the sensitivity-guided action representation passes 27/27 held-out samples, whereas the generator-wise reference passes 22/27 under the same training budget in the original run. '
        'Across the evaluated seeds, sensitivity-guided bases outperform the fixed random basis. System-wide-only directions remain an effective alternative to the mixed allocation. '
        'The final gated workflow reduces end-to-end restoration time on the historical cases with unchanged acceptance, whereas the spatial-profile saving interval includes zero. '
        'Every accepted dispatch is checked by base-case alternating-current power flow and the line-outage distribution-factor criterion. This acceptance is not full alternating-current contingency security.')
    from pathlib import Path
    from revision_final_consistency import spatial_summary
    spatial=spatial_summary(Path(__file__).resolve().parent)
    if spatial:
        lo,hi=spatial['ci95_sec']
        assert lo<=0<=hi, 'Update the concise conclusion if the frozen spatial result changes.'
    return text
def abstract(data):
    from revision_acceptance_scope import ABSTRACT
    assert 150 <= len(ABSTRACT.split()) <= 250
    return ABSTRACT.replace('%', r'\%')


def conclusion(data):
    from revision_acceptance_scope import CONCLUSION
    return CONCLUSION


def plot_basis(here,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.25),sharey=True,layout='constrained')
    for ax,(family,label) in zip(axes,[('hybrid','Mixed sensitivity'),('modes','System-wide'),('random','Random balanced')]):
        for seed,values in zip((42,1,2),data['basis'][family]['validation']):
            ax.plot([v['update'] for v in values],[v['passed']/v['total'] for v in values],marker='o',ms=3,label=f'Seed {seed}')
        ax.set(title=label,xlabel='Outer iteration',ylim=(-.03,1.03)); ax.grid(alpha=.2)
    axes[0].set_ylabel('Validation pass fraction'); axes[-1].legend(fontsize=7,loc='lower right')
    fig.savefig(here/'revision_figures/final_basis_validation.pdf'); plt.close(fig)


def paired_timing_data(here):
    """Join repeats by case; never treat repeated timings as independent cases."""
    import hashlib
    import numpy as np
    from revision_reviewer_followup import load as followup_load
    from revision_final_consistency import spatial_summary
    historical=followup_load(here)['dual'];spatial=spatial_summary(here)
    populations=[]
    for label,n,name,summary in (
            ('Historical cases',75,'dual_verified_once_v2/dual.json',historical),
            ('Spatial profiles',100,'spatial_dual_final_v1/results.json',spatial)):
        path=here/'experiment_results'/name
        raw=json.loads(path.read_text(encoding='utf-8'))
        assert raw['complete'] and len(raw['rows'])==n*2*3
        records={(r['repeat'],r['sample_id'],r['arm']):r for r in raw['rows']}
        ids=list(dict.fromkeys(r['sample_id'] for r in raw['rows']))
        assert len(ids)==n and len(records)==n*6
        times={arm:np.array([[records[repeat,sid,arm]['runtime_sec'] for repeat in range(3)]
                              for sid in ids]).mean(axis=1) for arm in ('direct','selected')}
        delta=times['direct']-times['selected']
        assert np.isfinite(delta).all()
        for repeat in range(3):
            assert [sid for sid in ids if records[repeat,sid,'direct']['passed']]==[
                sid for sid in ids if records[repeat,sid,'selected']['passed']]
        rng=np.random.default_rng(2026101805)
        ci=np.quantile(delta[rng.integers(0,n,size=(10000,n))].mean(axis=1),[.025,.975])
        assert np.allclose(ci,summary['ci95_sec'],rtol=0,atol=1e-12)
        assert abs(delta.mean()-summary['mean_saving_sec'])<1e-12
        populations.append({'label':label,'n':n,'ids':ids,'times':times,'delta':delta,
            'mean_sec':float(delta.mean()),'ci95_sec':ci.tolist(),
            'raw_file':path.relative_to(here).as_posix(),
            'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    return populations


def plot_final_timing(here):
    import csv
    import hashlib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    populations=paired_timing_data(here)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.9),layout='constrained')
    for letter,ax,p,color in zip('ab',axes,populations,('#386d8c','#3b8065')):
        counts,_,_=ax.hist(p['delta'],bins=17,color=color,alpha=.78,edgecolor='white',linewidth=.5)
        peak=max(counts);lo,hi=p['ci95_sec']
        ax.axvline(0,color='black',linewidth=.8,linestyle='--',label='Zero saving')
        ax.scatter(p['delta'],np.full(p['n'],-.05*peak),marker='o',s=6,alpha=.6,color=color,clip_on=False)
        ax.errorbar(p['mean_sec'],1.18*peak,
            xerr=[[p['mean_sec']-lo],[hi-p['mean_sec']]],fmt='o',ms=4,
            color='#9d383d',capsize=3,linewidth=1.4,label='Mean and 95% interval')
        ax.set(title=f'({letter}) {p["label"]} (n = {p["n"]})',
            xlabel='Direct minus gated time (s)',ylabel='Number of cases',ylim=(-.08*peak,1.55*peak))
        ax.text(.5,.94,f'Mean {p["mean_sec"]:.3f} s; 95% CI [{lo:.3f}, {hi:.3f}] s',
            transform=ax.transAxes,ha='center',va='top',fontsize=7)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.15)
    output=here/'revision_figures/final_paired_timing.pdf'
    fig.savefig(output);plt.close(fig)
    audit={
        'analysis':'Case means over three rotating serial repeats; all failures retained; populations not pooled.',
        'bootstrap_seed':2026101805,'bootstrap_replicates':10000,
        'figure_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'populations':[{k:p[k] for k in ('label','n','mean_sec','ci95_sec','raw_file','raw_sha256')} for p in populations]}
    root=here/'experiment_results'
    (root/'final_paired_timing_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    with (root/'final_paired_timing_cases.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['population','sample_id','direct_mean_sec','gated_mean_sec','saving_sec'])
        for p in populations:
            writer.writerows((p['label'],sid,p['times']['direct'][i],p['times']['selected'][i],p['delta'][i])
                for i,sid in enumerate(p['ids']))
    print('Regenerated final paired-time figure from existing records only.')


def apply(here,add_after,data):
    if data is None: return
    strong=data['strong']; rows=[]
    names={'direct_flat':'Direct, flat','direct_pf':'Direct, PF','gated_flat':'Gated, flat','gated_pf':'Gated, PF'}
    for arm in names:
        x=strong['arms'][arm]; t=x['runtime_case_means']
        counts=x['acceptance_by_repeat']; call_counts=x['optimizer_calls_by_repeat']
        passed=(str(counts[0]) if len(set(counts))==1 else '/'.join(map(str,counts)))+'/75'
        calls=str(call_counts[0]) if len(set(call_counts))==1 else '/'.join(map(str,call_counts))
        failure=100*mean(x['failure_rate_by_repeat'])
        rows.append([names[arm],passed,f'{failure:.1f}',calls,f'{t["mean"]:.3f}',f'{t["median"]:.3f}',f'{t["p90"]:.3f}'])
    p=strong['paired']['pf']; diff=p['direct_minus_gated_sec']
    text=(r'\paragraph{\rev{Initialization-controlled optimizer comparison.}} '
          r'\rev{We retain both flat and power-flow (PF) initialization in a four-arm comparison. Direct and gated workflows share each initializer, objective, constraints, iteration budget and final verifier. '
          r'The gate uses the demand range covered by the retained SAC training set, 16.0--16.4 GW. The joint optimizer uses full generator-wise active power and minimizes squared displacement from the projected GAT dispatch, rather than generation cost. '
          r'All 75 historical cases are run three times with rotating serial order; PF initialization, failed cases and final checks are timed. '
          r'Table~\ref{tab:final_strong} gives acceptance and optimizer-call counts (one value when identical across repeats, otherwise repeat order), each out of 75. P50 and P90 denote the 50th and 90th percentiles of per-case average times across repeats, not 225 independent cases. '
          f'The mean paired direct-minus-gated time with PF initialization is {diff["mean"]:.3f}~s (median {diff["median"]:.3f}~s; 10th--90th percentiles {diff["p10"]:.3f}--{diff["p90"]:.3f}~s); gating is faster on {p["faster_cases"]}/75 case averages. '
          r'This is an initialization-sensitivity comparison for the original policy. The primary deployment comparison uses the short-policy configuration and matched dual initialization in Table~\ref{tab:review_followup_runtime}.}')
    text+='\n'+table('tab:final_strong','Repeated optimizer comparisons. Failures are percentages; times are seconds.','Workflow & Passed & Fail. & Calls & Mean & P50 & P90',rows,'lrrrrrr')
    names={'sac_only':'Frozen SAC','direct_pf':'Direct PF','gated_pf':'Gated PF'}
    nrows=[]
    for arm,label in names.items():
        x=grouped(data['independent'],arm,100)
        nrows.append([label,f'{x["passed"]}/100',str(x['prep_failed']),f'{x["mean"]:.3f}',f'{x["p50"]:.3f}',f'{x["p90"]:.3f}'])
    text+='\n\n'+(r'\paragraph{\rev{New unscreened operating conditions.}} '
        r'\rev{After fixing models, the demand-range gate, solver and verifier, we generate 20 new cases at each of five demand levels (100 total; seed 2026101601). '
        r'Costs are independent Gaussian perturbations of mean coefficients in the upstream training partition, with relative standard deviation sampled uniformly from [0.02,0.03]. '
        r'Three load buses are randomly chosen as region centers; each load is assigned to its nearest center by unweighted shortest-path distance in the bus-branch graph. '
        r'Regions receive independent active/reactive scaling factors in [0.9,1.1], with additional per-load reactive factors in [0.97,1.03]; total active demand is normalized. '
        r'No action-span or outcome-based filter is used. Every generated case, including preparation failure, remains in Table~\ref{tab:final_independent}. '
        r'Timing includes new-profile preparation, affine-intercept re-anchoring, first-use sample-specific voltage sensitivities when needed, correction, optimization and final verification. '
        r'These new-profile times therefore include preparation that the resident fixed-profile comparison excludes; they should not be conflated.}')
    text+='\n'+table('tab:final_independent','New operating conditions without outcome filtering; prep. denotes preparation failures.','Workflow & Passed & Prep. & Mean & P50 & P90',nrows,'lrrrrr')
    ni=grouped(data['independent'],'gated_pf',100); di=grouped(data['independent'],'direct_pf',100)
    if ni['mean']>di['mean']:
        text+='\n\n'+r'\rev{The gated workflow is slower on these first-use spatially perturbed profiles. The fixed-profile timing advantage does not carry over when new context construction and unsuccessful policy attempts are included. The demand-only gate does not detect spatial-profile changes; no gate or policy parameter is retuned on these 100 cases.}'
    records={arm:{r['sample_id']:r for r in data['independent']['rows'] if r['arm']==arm} for arm in names}
    common_ids=sorted(sid for sid in records['direct_pf'] if records['direct_pf'][sid]['passed'] and records['gated_pf'][sid]['passed'])
    if common_ids:
        dc=mean(records['direct_pf'][sid]['scheduled_cost'] for sid in common_ids)
        gc=mean(records['gated_pf'][sid]['scheduled_cost'] for sid in common_ids)
        dl=mean(records['direct_pf'][sid]['gat_relative_l2_mw'] for sid in common_ids)
        gl=mean(records['gated_pf'][sid]['gat_relative_l2_mw'] for sid in common_ids)
        text+='\n\n'+r'\rev{On the '+str(len(common_ids))+f' jointly accepted new cases, mean scheduled cost is {dc:.1f} for direct PF optimization and {gc:.1f} for the gated workflow; mean prediction-relative active-power adjustment norms are {dl:.2f} and {gl:.2f}~MW. '+r'These paired costs exclude reference-source exchange and are not averages over differently accepted subsets.}'
    else:
        text+='\n\n'+r'\rev{The two optimization workflows have no jointly accepted new case; a paired accepted-cost comparison is therefore undefined.}'
    failure_counts={}
    for arm in names:
        group=list(records[arm].values())
        flags=sorted({key for r in group for key in r.get('constraint_flags',{})})
        failure_counts[arm]={'preparation_failed':sum(r.get('preparation_failed',False) for r in group),
            **{key:sum(r.get('constraint_flags',{}).get(key) is True for r in group) for key in flags}}
        failure_counts[arm]['optimizer_no_candidate']=sum(bool(r.get('optimizer_called')) and not (r.get('optimizer') or {}).get('returned_candidate',False) for r in group)
        failure_counts[arm]['candidate_rejected']=sum(bool((r.get('optimizer') or {}).get('returned_candidate')) and not r['passed'] for r in group)
    (here/'experiment_results/final_independent_failure_counts.json').write_text(json.dumps(failure_counts,indent=2),encoding='utf-8')
    text+='\n\n'+r'\rev{Per-constraint failures are retained as overlapping flags; preparation failures are separate. The aggregate input does not encode the new spatial pattern, so these cases test the full physical correction workflow, not an explicitly spatially conditioned GAT predictor.}'
    df,gf=failure_counts['direct_pf'],failure_counts['gated_pf']
    text+='\n\n'+r'\rev{For direct PF optimization, '+f'{df["preparation_failed"]} cases fail preparation, {df["optimizer_no_candidate"]} return no optimizer candidate, and {df["candidate_rejected"]} return a candidate rejected by final verification. '+f'The corresponding gated-workflow counts are {gf["preparation_failed"]}, {gf["optimizer_no_candidate"]} and {gf["candidate_rejected"]}. '+r'An unsuccessful solve is not an infeasibility certificate.}'
    (here/'revision_final_core_workflow_results.tex').write_text(text,encoding='utf-8')
    add_after(r'\subsection{Main Pipeline Comparison}',r'\input{revision_final_core_workflow_results}','Make the strong baseline and independent evaluation core efficiency evidence')
    brows=[]
    for family,label in [('hybrid','Mixed sensitivity'),('modes','System-wide'),('random','Random balanced')]:
        x=data['basis'][family]
        brows.append([label, '/'.join(map(str,x['counts'])),f'{x["mean_pct"]:.1f} $\\pm$ {x["std_pct"]:.1f}',f'{mean(x["steps"]):.2f}',f'{mean(x["seconds"]):.3f}'])
    text=(r'\rev{The matched 32-dimensional comparison uses policy seeds 42, 1 and 2 under the same training protocol. '
          r'The random construction seed is fixed at 20260721, so this measures policy-training variation for that basis, not variation over all random bases. '
          r'All seeds, including failed runs, are retained; acceptance variability is computed over the three seeds, and failed cases enter step/time means. '
          r'Time includes measured sample preparation, reset, rollout and final verification, but excludes model loading. These separately collected seed timings are descriptive, not the paired timing experiment used for the optimizer comparison. '
          r'The comparison tests sensitivity-guided coordinates, not superiority of a particular 12/20 allocation.}')
    if data['basis']['modes']['mean_pct']>=data['basis']['hybrid']['mean_pct']:
        text+='\n\n'+r'\rev{The simpler system-wide basis has at least the mean acceptance of the mixed basis in these runs. The mixed basis is one implementation of the sensitivity-guided representation, and the system-wide basis provides a simpler alternative.}'
    text+='\n'+table('tab:final_basis_seeds','Matched action bases across three policy seeds. Rates are mean and sample standard deviation in percentage points.','Basis & Passed /75 & Rate & Steps & Time (s)',brows,'lrrrr')
    assert (here/'revision_figures/final_basis_validation.pdf').is_file(), 'Run final_evidence_documents.py with a compatible matplotlib environment first'
    text+='\n'+r'''\begin{figure*}[!tbp]
\centering
\includegraphics[width=0.95\textwidth]{revision_figures/final_basis_validation.pdf}
\caption{\rev{Validation trajectories for the matched 32D bases under three policy seeds. Every run is retained, and checkpoints are selected by validation rather than final-iteration performance.}}
\label{fig:final_basis_validation}
\end{figure*}'''
    (here/'revision_final_basis_seed_results.tex').write_text(text,encoding='utf-8')
    add_after(r'\input{revision_ablation_results}',r'\input{revision_final_basis_seed_results}','Complete matched-seed action-basis evidence')
    ar=data['anchor']['rows']; retained=sum(r['ac_anchor'] is not None and r['ac_anchor']['passed'] for r in ar)
    gaps=[r['ac_anchor']['max_post_excess_mw'] for r in ar if r['ac_anchor'] is not None]
    audit=(r'\rev{A complete re-anchoring audit holds all 44 originally accepted SAC control points fixed and recomputes linear line-flow predictions for all 253 valid outages. '
        f'With actual base-case AC line flows replacing the affine anchor, {retained}/44 retain linear acceptance; the largest predicted post-outage excess is {max(gaps):.3f}~MW. '
        r'The original affine check is reproduced for each point before comparison. No controls or acceptance limits are retuned. '
        r'Re-anchoring remains a linear active-line-flow check and does not certify post-outage voltage or reactive-power feasibility.}')
    add_after(r'\input{revision_ac_audit_results}',audit,'Check every accepted point against AC-anchored linear outages')
    tight=data['tight']['rows']; accepted=sum(r['passed'] for r in tight)
    nc=sum(r['status']=='solver_nonconvergence' for r in tight)
    rejected=sum(r['status']=='candidate_rejected' for r in tight)
    add_after(r'\input{revision_stress_results}',r'\rev{The compatible 0.94--1.08 pu joint-optimization test is extended to all 75 original cases with PF initialization: '+f'{accepted}/75 pass, with mean end-to-end time {mean(r["runtime_sec"] for r in tight):.3f}~s including failures. There are no incompatible-bound cases, {nc} nonconvergent solves and {rejected} returned candidates rejected by verification. '+r'These accepted points also satisfy the original broader voltage band; differences in acceptance between optimizer configurations therefore must not be read as physical infeasibility under the broader limits. This is broader-control optimization, not a retrained-policy result.}', 'Extend compatible voltage-band evaluation to all cases')
    trained=data['tight_policy']['rows']; assert len(trained)==75
    add_after(r'\input{revision_stress_results}',r'\rev{A separate matched retraining run uses the compatible 0.94--1.08 pu band, original retained training/validation IDs, seed 42, 100 outer iterations and the original 32D/40-MW action interface, without new screening or reward retuning. '+f'The validation-selected policy accepts {sum(r["passed"] for r in trained)}/75 upstream cases, with mean {mean(r["steps"] for r in trained):.2f} steps including failures. '+r'This single retraining run is reported separately from the original frozen-policy test.}', 'Report matched tighter-band retraining without replacing original policy')
    initrows=[]
    for arm in ('gat','online_linear_sced'):
        x=grouped(data['initialization'],arm,75)
        name={'gat':'GAT prediction','online_linear_sced':'Online linear SCED'}[arm]
        initrows.append([name,f'{x["passed"]}/75',f'{x["mean"]:.3f}',f'{x["p50"]:.3f}',f'{x["p90"]:.3f}'])
    text=(r'\rev{A separate initialization comparison uses a common scheduled-generation-cost objective in the joint optimizer, rather than distance from different initial references. '
          r'GAT prediction and online linear-SCED generation are each timed together with projection, PF-initialized optimization and final checks. '
          r'This changes the optimization objective relative to the displacement-based workflow; its times are compared only within this experiment.}')
    text+='\n'+table('tab:final_initializer','Common economic objective with different initialization generators.','Initializer & Passed & Mean (s) & P50 & P90',initrows,'lrrrr')
    (here/'revision_final_initializer_results.tex').write_text(text,encoding='utf-8')
    add_after(r'\input{revision_initialization_costs}',r'\input{revision_final_initializer_results}','Isolate initializer pipelines with a common economic objective')
    from revision_response_layout import QUOTE_PREFIXES,EVIDENCE,LOCATIONS,TABLES,FIGURES
    QUOTE_PREFIXES.update(abstract='Security-constrained economic dispatch is repeatedly solved',
        conclusion='The results show that fixed, sensitivity-based active-power directions',
        final_strong='We retain both flat and power-flow (PF) initialization',
        final_independent='After fixing models, the demand-range gate',
        final_basis='The matched 32-dimensional comparison uses policy seeds',
        final_anchor='A complete re-anchoring audit holds all 44',
        final_initializer='A separate initialization comparison uses a common')
    for key,quote,label in [('R3.6','final_strong','tab:final_strong'),('R2.8','final_independent','tab:final_independent'),
                           ('R2.10','final_basis','tab:final_basis_seeds'),('R2.12','final_initializer','tab:final_initializer')]:
        EVIDENCE[key] += (quote,)
        LOCATIONS[key] += (label,)
        TABLES[key] = (label,)
    EVIDENCE['R2.3'] += ('final_anchor',)
    EVIDENCE['R3.6'] = ('exchange','matched_workflow_timing','final_strong','final_independent')
    FIGURES['R3.6'] = ()
    EVIDENCE['AE.11'] = ('scope','final_basis')
    LOCATIONS['AE.11'] += ('tab:final_basis_seeds',)


def responses(data,items):
    if data is None: return
    items['AE.7']['response'] = ('The single-paragraph abstract now states the physical checks, the full 75-case policy result, '
        'the repeated optimizer comparison and the independently generated 100-case evaluation. '
        'It distinguishes base-case alternating-current feasibility from full alternating-current contingency security, '
        'and does not present screened acceptance as an overall success rate.')
    p=data['strong']['paired']['pf']
    for key in ('R1.2','R3.6','R4.1'):
        items[key]['response'] += (f' The initialization-sensitivity four-arm comparison applies both flat and PF initialization to direct and gated workflows on all 75 cases, with three serial paired repetitions. '
            f'The PF-initialized mean paired direct-minus-gated difference is {p["direct_minus_gated_sec"]["mean"]:.3f} s. '
            'The earlier 25.9% one-pass result is not used as the final efficiency claim. We also report a frozen 100-case independently generated evaluation, retaining all preparation failures.')
    for key in ('R2.10','R1.3','R4.3'):
        items[key]['response'] += ' The mixed, system-wide and fixed-random 32D bases now each have policy seeds 42, 1 and 2. The random construction seed remains 20260721; failed runs are retained. This does not establish the mixed allocation as necessary or optimal.'
    counts={k:', '.join(map(str,v['counts'])) for k,v in data['basis'].items()}
    items['R2.10']['response'] = (f'The matched three-seed comparison gives mixed-basis counts {counts["hybrid"]}, system-wide counts {counts["modes"]}, '
        f'and fixed-random counts {counts["random"]}, each out of 75 in seed order 42, 1, 2. The random construction seed is 20260721. '
        'All training runs and their validation trajectories are reported. These results do not support treating the particular 12/20 mixture as the main innovation; '
        'the contribution is the balanced sensitivity-guided representation, with system-wide-only directions as a simpler alternative. '
        'The original dimension, allocation and nominal action-scale controls remain reported, without a claim that 32 dimensions or the 12/20 allocation is optimal.')
    items['AE.11']['response'] = ('The contribution is a fixed, balanced, sensitivity-guided representation for active-power correction, not a new graph network or reinforcement-learning algorithm. '
        'The completed matched-seed comparison does not establish superiority of the particular mixed allocation. We therefore treat the mixed basis as one implementation and identify system-wide-only directions as a simpler alternative. '
        'The manuscript distinguishes this representation from action masking and from generator-wise policy outputs.')
    for key in ('R2.3','R2.5','R3.4','R4.2'):
        ar=data['anchor']['rows']; n=sum(r['ac_anchor'] is not None and r['ac_anchor']['passed'] for r in ar)
        items[key]['response'] += f' All 44 original SAC-accepted points are rechecked with their actual base-case AC-flow anchor over all 253 valid outages; {n}/44 retain linear acceptance. No original controls or thresholds are changed, and this is not full AC contingency certification.'
    items['R2.7']['response'] += ' The compatible 0.94--1.08 band is additionally evaluated with joint optimization on all 75 cases, retaining solver failures and rejected candidates separately.'
    items['R2.7']['response'] += f' A separate seed-42 retraining run with unchanged split, action range, reward weights and 100-iteration budget accepts {sum(r["passed"] for r in data["tight_policy"]["rows"])}/75; this is not substituted for the original zero-acceptance frozen-policy result.'
    items['R2.12']['response'] += ' A new GAT-versus-online-linear-SCED initialization experiment uses a common economic joint-optimization objective and counts initializer generation and final verification time. It is reported separately from the displacement-objective workflow.'
    items['R1.2']['response'] = ('The abstract, main acceptance comparison and conclusion now report 44/75 for the original policy, separately from 27/27 on the screened subset. '
        'Fresh terminal checks show that 30 failures violate both voltage and the linear contingency limits, while one further failure violates only the linear criterion. '
        'Optimization fallback is reported as a separate workflow, not credited to the unchanged policy.')
    items['R2.1']['response'] = ('The abstract reports the overall 44/75 acceptance, and the main tables distinguish it from the conditional 27/27 result. '
        'The complete upstream test set is the primary evaluation population; the screened result is not a claim of universal restoration.')
    new=grouped(data['independent'],'gated_pf',100)
    direct=grouped(data['independent'],'direct_pf',100); policy=grouped(data['independent'],'sac_only',100)
    items['R2.8']['response'] += (f' A separate frozen-protocol test generates 100 new cases with independent regional load, reactive-demand and cost perturbations, without action-span screening. '
        f'Policy-only, direct PF optimization and gated PF optimization accept {policy["passed"]}, {direct["passed"]} and {new["passed"]} cases, respectively. '
        f'All {direct["prep_failed"]} preparation failures remain in each denominator. No gate or policy settings are adjusted after seeing these results.')
    a=data['strong']['arms']
    items['R3.6']['response'] = ('We separate actor latency, verified correction and optimization fallback. '
        'The main efficiency comparison now runs all 75 cases three times with rotated serial execution order. '
        'Direct and gated workflows use identical objectives, limits, iteration budgets and either flat or PF initialization. '
        f'With PF initialization they accept {a["direct_pf"]["acceptance_by_repeat"][0]}/75 and {a["gated_pf"]["acceptance_by_repeat"][0]}/75, '
        f'with mean complete times {a["direct_pf"]["runtime_case_means"]["mean"]:.3f} and {a["gated_pf"]["runtime_case_means"]["mean"]:.3f} s. '
        'The flat-initialized arms, paired time differences, percentiles, failures and optimizer-call counts are also reported. '
        f'On the new first-use profiles, direct PF optimization and the gated workflow accept {direct["passed"]}/100 and {new["passed"]}/100, '
        f'with mean complete times {direct["mean"]:.3f} and {new["mean"]:.3f} s including profile/context preparation. '
        'Thus the fixed-profile speed benefit is not an unconditional deployment advantage. '
        'Signed-exchange pricing remains separately reported; preventive optimization has lower cost on the jointly accepted points, and no total-system cost saving is claimed.')
    initializer=[grouped(data['initialization'],arm,75) for arm in ('gat','online_linear_sced')]
    items['R2.12']['response'] += (f' The common-objective GAT and online-SCED workflows accept {initializer[0]["passed"]}/75 and {initializer[1]["passed"]}/75, '
        f'with mean complete times {initializer[0]["mean"]:.3f} and {initializer[1]["mean"]:.3f} s. '
        'The acceptance difference is solver-configuration dependent; the times do not establish a substantial GAT speed advantage or make GAT indispensable.')
    items['R4.1']['response'] = ('We agree that the screened result does not establish overall reliability. The primary policy result remains 44/75, with the 27-case subset identified separately. '
        f'We additionally froze the workflow before generating 100 new spatial/reactive/cost perturbations without action-span screening; the policy-only, direct-optimization and gated workflows accept {policy["passed"]}/100, {direct["passed"]}/100 and {new["passed"]}/100, including preparation failures in the denominator. '
        'This is a separate workflow evaluation, not a revised success rate for the original SAC policy. The manuscript no longer claims universal reliable restoration. '
        'The revised contribution is a sensitivity-guided correction representation under the stated acceptance checks, supported by matched-seed comparisons, together with conditional time savings from avoiding optimization in selected cases. The corresponding basis comparison and final workflow results are identified below.')


if __name__ == '__main__':
    from pathlib import Path
    here = Path(__file__).resolve().parent
    data = load(here)
    assert data is not None, 'Final experimental suite is incomplete'
    plot_basis(here,data)
    plot_final_timing(here)

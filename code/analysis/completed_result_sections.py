"""Generate revision additions only from complete, audited summaries."""
from revision_common import REV, RESULTS, read


def table(caption, label, header, body, columns):
    lines = [r'\begin{table}[!htbp]', r'\centering',
             r'\ifrevisionmarked\color{blue}\fi',
             r'\caption{' + caption + '}', r'\label{' + label + '}',
             r'\scriptsize\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular}{' + columns + '}', r'\toprule',
             header + r' \\', r'\midrule']
    lines.extend(' & '.join(row) + r' \\' for row in body)
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    return '\n'.join(lines)


def fraction(result):
    return f"{result['passed']}/{result['n']}"


def generate(summaries, seed_stats=None):
    full = [('sac', 'SAC-32D'), ('full68', 'SAC-68D'), ('ddpg', 'DDPG'),
            ('td3', 'TD3'), ('acopf', 'Base-case AC-OPF'),
            ('scopf', 'Preventive LODF-SCOPF'), ('repair', 'Deterministic repair')]
    if all('full_' + key in summaries for key, _ in full):
        body = [[name, fraction(summaries['full_' + key]),
                 fraction(summaries['full_' + key]['screened'])] for key, name in full]
        text = (r'\paragraph{\rev{Matched unfiltered comparison.}} '
                r'\rev{Table~\ref{tab:revision_full_methods} evaluates all methods on the same 75 upstream test IDs. '
                r'The trained policies are held fixed; the original 27 screened test cases are reported as a subset, not as a second run. '
                r'TD3 accepts one more candidate than SAC-32D in this run. Neither policy nor either repair/optimization baseline accepts a candidate at 16.6 or 16.8~GW. '
                r'These outcomes describe the tested algorithms and solver settings; they do not establish physical infeasibility of those candidates.}' + '\n\n')
        text += table('Acceptance on matched upstream test IDs and their screened subset.',
                      'tab:revision_full_methods', 'Method & All candidates & Screened subset', body, 'lrr')
        (REV / 'revision_full_method_results.tex').write_text(text, encoding='utf-8')

    if all(f'case{case}_{source}' in summaries for case in (39, 118) for source in ('actions', 'baselines')):
        common_path = RESULTS/'evaluation_contract_v2/summary.json'
        common = read(common_path) if common_path.exists() else None
        body = []
        for case in (39, 118):
            action = summaries[f'case{case}_actions']['by_mode']
            baseline = summaries[f'case{case}_baselines']['by_mode']
            if common:
                measured = common['cases'][str(case)]['modes']
                action = {'full': measured['policy']}
                baseline = measured
            body.append([str(case), fraction(baseline['prediction_only']),
                         fraction(baseline['myopic_search']), fraction(action['full'])])
        text = (r'\paragraph{\rev{Common small-system controls.}} '
                r'\rev{Table~\ref{tab:revision_small_baselines} compares the projected prediction, a deterministic one-step-lookahead search, and the trained policy on identical test IDs within each system. '
                r'The search evaluates five candidates per iteration: no change, positive and negative common voltage increments, and positive and negative changes in one cyclically selected active-power coordinate. '
                r'It selects the candidate by acceptance, violation count, and residual magnitude, with at most 40 iterations for the 39-bus system and 50 for the 118-bus system. '
                r'The search and policy both restore every tested small-system case. On the 118-bus system, masking active-power actions still yields 20/20, whereas retaining only active-power actions yields 0/20. '
                r'This result identifies voltage correction as sufficient for this test set, rather than establishing a cross-system benefit of the 300-bus redispatch basis.}' + '\n\n')
        text += table('Small-system acceptance with common test IDs; each implementation retains its system-specific checks.',
                      'tab:revision_small_baselines', 'System & Prediction & Search & Policy', body, 'lrrr')
        timing=[]
        for label,source,mode in [('Prediction only','baselines','prediction_only'),('One-step search','baselines','myopic_search'),('Policy','actions','full')]:
            row=[label]
            for case in (39,118):
                stat=summaries[f'case{case}_{source}']['by_mode'][mode]['runtime_sec']
                if common:
                    stat=common['cases'][str(case)]['modes']['policy' if mode=='full' else mode]['runtime_sec']
                row.append(f"${stat['mean']:.3f}\\pm{stat['std']:.3f}$")
            timing.append(row)
        caption = ('Matched mean end-to-end time $\\pm$ sample standard deviation across all 30 and 20 test cases, including failures. '
                   'All methods include reset/prediction, initial checks, correction and fresh final verification; resident-model setup is excluded.'
                   if common else 'Measured mean evaluation time $\\pm$ sample standard deviation across all 30 and 20 test cases. Search time includes candidate evaluation and environment copies.')
        text+='\n\n'+table(caption,
                            'tab:revision_small_time','Method & 39-bus (s) & 118-bus (s)',timing,'lrr')
        if common:
            text += ('\n\n' + r'\rev{These timings replace the earlier measurements with different start points. '
                     r'A shared evaluator measures every method from reset and prediction to the same final physical checks. '
                     r'Initially accepted points skip correction; final verification reads the operating state without executing another action. '
                     r'The records separate reset/prediction, initial verification, correction and final verification time. '
                     r'Method order rotates across samples, and setup and model loading occur before timing.}')
        (REV / 'revision_small_baseline_results.tex').write_text(text, encoding='utf-8')

    stress = [('stress_demand-low', r'Demand $-3\%$'), ('stress_demand-high', r'Demand $+3\%$'),
              ('stress_regional', 'Regional redistribution'), ('stress_reactive', r'Reactive demand $+10\%$'),
              ('stress_renewable_v2', 'Net-injection perturbation'),
              ('voltage_0.94_1.08', 'Voltage band 0.94--1.08'), ('voltage_0.95_1.05', 'Voltage band 0.95--1.05'),
              ('start_label_v2', 'Stored linear-dispatch start'), ('start_economic_v2', 'Economic-dispatch start'),
              ('start_previous_v2', 'Previous accepted dispatch')]
    if all(key in summaries for key, _ in stress):
        body = [[name, fraction(summaries[key]), str(summaries[key]['preparation_failed']),f"{summaries[key]['including_preparation_mean_sec']:.3f}"] for key, name in stress if key != 'voltage_0.95_1.05']
        text = (r'\paragraph{\rev{Frozen-policy operating-condition tests.}} '
                r'\rev{Table~\ref{tab:revision_stress} uses the same 75 upstream IDs and frozen policy, without selecting a new checkpoint. '
                r'Regional redistribution multiplies active and reactive loads within three graph hops of the first load bus by 1.10 and the remaining loads by 0.90, then normalizes total active demand. '
                r'The net-injection test applies independent factors in $[0.9,1.1]$ to active loads before normalization; it is a forecast-error surrogate, not a renewable-generator dynamics model. '
                r'For changed demands or load distributions, the affine flow intercept is re-anchored using a base-case power flow, while the sensitivity matrix, outage factors and action basis remain fixed. '
                r'Preparation failures are included in the denominator but are distinguished from policy failures. Voltage-sensitivity caches are isolated when the load distribution or initial dispatch varies between samples.}' + '\n\n')
        text += table('Changed conditions without retraining. Prep. failures precede policy execution. Mean time includes measured preparation; voltage bounds are per unit.',
                      'tab:revision_stress', 'Condition & Passed & Prep. fail. & Time (s)', body, 'lrrr')
        strict = summaries['voltage_0.95_1.05']
        text += ('\n\n'+r'\paragraph{\rev{Control-bound compatibility.}} '
                 r'\rev{The historical 0.95--1.05 pu fixed-policy run accepts '+fraction(strict)+r' cases and averages '+f"{strict['including_preparation_mean_sec']:.3f}"+r' s. '
                 r'It is excluded from the robustness ranking because the retained voltage-control intervals are incompatible with imposing these bounds at every controlled bus in the joint formulation. '
                 r'This result mixes constraint compatibility with policy adaptation and is retained as a separate diagnostic, not as evidence of policy weakness alone.}')
        text += ('\n\n' + r'\rev{The tighter voltage-band tests assess the original policy under new limits, not retraining with those limits. '
                 r'The economic start minimizes scheduled generation cost subject to generator bounds and balance. '
                 r'The previous-point control uses the last accepted active dispatch in the fixed evaluation order and falls back to the projected prediction until one is accepted. It is not a chronological time-series study. '
                 r'Stored-label generation time is unavailable and is not included in timing claims.}')
        (REV / 'revision_stress_results.tex').write_text(text, encoding='utf-8')

    if seed_stats and all(m in seed_stats for m in ('sac','full68','ddpg','td3')):
        body=[]
        for method,name in [('sac','SAC-32D'),('full68','SAC-68D'),('ddpg','DDPG'),('td3','TD3')]:
            stats=seed_stats[method]; selected=stats['screened']
            body.append([name,', '.join(str(s['passed']) for s in stats['per_seed']),
                         f"${100*stats['pass_rate_mean']:.1f}\\pm{100*stats['pass_rate_std']:.1f}$",
                         f"${100*selected['pass_rate_mean']:.1f}\\pm{100*selected['pass_rate_std']:.1f}$"])
        text=(r'\paragraph{\rev{Variation across training seeds.}} '
              r'\rev{We add seeds 1 and 2 for each policy to the original reported seed-42 run, retaining the fixed dataset partitions and frozen predictor. '
              r'Each new run uses 100 outer iterations with 10 episodes per iteration and the same method-specific optimization settings. '
              r'Actual environment-step counts can differ because successful episodes terminate early. Checkpoints are selected by validation acceptance count, then validation return; no test result is used for selection. '
              r'Table~\ref{tab:revision_seeds} reports every run, including zero-acceptance runs. Standard deviations use the three seed-level acceptance rates with divisor $3-1$, not the individual test cases.}'+'\n\n')
        text+=table('Seed-level acceptance. Counts use all 75 upstream test cases in seed order 42, 1, 2. Rate columns report mean $\pm$ sample standard deviation (percentage points).',
                    'tab:revision_seeds','Method & Counts /75 & All rate (\%) & Screened rate (\%)',body,'lrrr')
        text+=('\n\n'+r'\rev{SAC-32D accepts all 27 screened cases in each run, with 44, 45 and 41 accepted upstream cases. '
               r'The seed-2 runs of SAC-68D, DDPG and TD3 have zero validation acceptance at all five checkpoints and zero test acceptance; their selected test failures are voltage violations. '
               r'Figure~\ref{fig:revision_seed_validation} also shows that a successful selected checkpoint need not imply a stable training trajectory. '
               r'For example, TD3 with seed 1 reaches 25/25 validation acceptance at iteration 60 but falls to 0/25 at iteration 100, so the earlier checkpoint is used. '
               r'The observations support lower seed-to-seed variation for SAC-32D under this protocol, without establishing a general ordering from three runs.}'+'\n\n')
        text+='\n'.join([r'\begin{figure*}[!tbp]',r'\centering',r'\includegraphics[width=0.95\textwidth]{revision_figures/seed_validation_curves.pdf}',
                         r'\caption{\rev{Validation acceptance over training for all three seeds of each policy. The same 25 validation IDs are used at each checkpoint; curves show individual runs rather than confidence bands.}}',
                         r'\label{fig:revision_seed_validation}',r'\end{figure*}'])
        (REV/'revision_seed_results.tex').write_text(text,encoding='utf-8')

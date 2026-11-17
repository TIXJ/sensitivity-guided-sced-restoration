"""Link major manuscript tables to versioned records without publishing data."""
import re
from pathlib import Path
from revision_common import REV, RESULTS, CANDIDATES, CANDIDATE_SPLIT, read, write, sha


def main():
    aux=(REV/'Manuscript_Revised_Marked.aux').read_text(encoding='utf-8')
    numbered=dict(re.findall(r'\\newlabel\{(tab:[^}]+)\}\{\{([^}]+)\}',aux))
    sources={
        'tab:revision_full_methods': [RESULTS/(name+'.json') for name in
            ('full_sac','full_full68','full_ddpg','full_td3','full_acopf','full_scopf','full_repair')],
        'tab:final_basis_seeds': [RESULTS/(name+'.json') for name in
            ('full_sac','sac_seed1_full','sac_seed2_full','modes32_full','modes32_seed1_v1_full',
             'modes32_seed2_v1_full','random32_full','random32_seed1_v1_full','random32_seed2_v1_full')],
        'tab:review_followup_runtime':[RESULTS/'dual_verified_once_v2/dual.json',
                                     RESULTS/'dual_verified_once_v2/protocol.json',
                                     RESULTS/'spatial_dual_final_v1/results.json',
                                     RESULTS/'spatial_dual_final_v1/protocol.json',
                                     RESULTS/'spatial_dual_final_v1/summary.json'],
        'tab:sequence_runtime':[RESULTS/'reviewer_sequence_balanced_v2/warm.json',
                                RESULTS/'reviewer_sequence_balanced_v2/protocol.json'],
        'tab:final_strong':[RESULTS/'strong_optimizer_v1/results.json',RESULTS/'strong_optimizer_v1/protocol.json',RESULTS/'strong_optimizer_v1/summary.json'],
        'tab:final_independent':[RESULTS/'independent_conditions_v1/results.json'],
        'tab:short_deployment':[RESULTS/'implicit_short_policy_v2/test.json',RESULTS/'shared_voltage_speed_v3/test.json'],
        'tab:final_initializer':[RESULTS/'initialization_economic_v1/results.json'],
        'tab:revision_stress':[RESULTS/'summary.json'],
        'tab:evaluation_versions':[RESULTS/'dual_verified_once_v2/protocol.json',
                                   RESULTS/'implicit_short_policy_v2/protocol.json',
                                   RESULTS/'spatial_dual_final_v1/protocol.json'],
        'tab:final_route_groups':[RESULTS/'dual_verified_once_v2/dual.json',RESULTS/'deployment_accounting_v1/summary.json'],
        'tab:final_workflow_exchange':[RESULTS/'dual_verified_once_v2/dual.json',RESULTS/'deployment_accounting_v1/summary.json'],
    }
    # A summary alone is not silently represented as per-case evidence.
    versions={
        'tab:revision_full_methods':'Original 50-step policies; original matched base/repair optimization runs.',
        'tab:final_basis_seeds':'Original 50-step policy interface; three policy seeds and one fixed random construction.',
        'tab:review_followup_runtime':'Same final short analytic-feature gate, matched dual initialization and verified-once acceptance on the 75 historical cases and the same previously examined 100 spatial profiles. Three rotating serial repeats per case; populations are not pooled.',
        'tab:sequence_runtime':'Separate order-controlled repeated-clearing diagnostic with power-flow-only fallback. Ten spatial patterns, ten periods each; not the final dual-start configuration.',
        'tab:final_strong':'Historical initialization sensitivity: original finite-difference-feature policy, training-range gate, flat/power-flow arms.',
        'tab:final_independent':'Original 50-step policy and training-demand gate; independent-condition seed 2026101601, power-flow initialization.',
        'tab:short_deployment':'Separate 100-profile population, seed 2026101802, short analytic-feature gate, power-flow-only initialization; two implementation measurements of the same cases.',
        'tab:final_initializer':'No SAC: common economic objective; GAT versus online linear SCED followed by joint optimization.',
        'tab:revision_stress':'Original fixed-policy tests; source summary retains detailed run keys and configuration.',
        'tab:evaluation_versions':'Explicit method/configuration definitions, not another performance result.',
        'tab:final_route_groups':'Actual frozen short-policy routes on the historical 75 cases; descriptive groups, not the original policy 44/31 split.',
        'tab:final_workflow_exchange':'Final workflow pair on the same 60 common accepted cases; signed exchange priced at 0/25/50 with fixed controls.'}
    manifest={'publicly_uploaded':False,'availability':'Author-held raw per-case records, weights, network data and code are available upon reasonable request.',
              'candidate_file_sha256':sha(CANDIDATES),'split_file_sha256':sha(CANDIDATE_SPLIT),
              'upstream_test_ids':read(CANDIDATE_SPLIT)['test_ids'],'tables':[]}
    figure_audit=read(RESULTS/'final_paired_timing_audit.json')
    assert figure_audit['figure_sha256']==sha(REV/'revision_figures/final_paired_timing.pdf')
    for population in figure_audit['populations']:
        assert population['raw_sha256']==sha(REV/population['raw_file'])
    manifest['figures']=[{'label':'fig:final_paired_timing','generator':'final_evidence_documents.py',
        'audit':figure_audit}]
    md=['# Table-to-Evidence Index','',
        'Materials are author-held and available upon reasonable request; no public deposit is claimed.',
        'Times from different policy/feature/initialization configurations are not interchangeable.','']
    for label,paths in sources.items():
        assert label in numbered,label
        records=[]
        for path in paths:
            assert path.is_file(),path
            records.append({'file':path.relative_to(REV).as_posix(),'sha256':sha(path)})
        item={'label':label,'table_number':numbered[label],'configuration':versions[label],'sources':records}
        if label=='tab:review_followup_runtime':
            item['detailed_primary_audit']='experiment_results/dual_verified_once_v2/audit_manifest.json'
        manifest['tables'].append(item)
        md+=['## Table '+numbered[label]+' ('+label+')',versions[label]]
        md += ['- `'+r['file']+'` SHA256 `'+r['sha256']+'`' for r in records]
        md+=['']
    write(REV/'Table_Evidence_Manifest.json',manifest)
    md+=['## Final Paired-Time Figure',
         '`fig:final_paired_timing` is regenerated by `final_evidence_documents.py`.',
         'Each point averages three repeats of one case. Populations are not pooled; failures are retained.']
    for p in figure_audit['populations']:
        md+=['- `'+p['raw_file']+'` SHA256 `'+p['raw_sha256']+'`']
    (REV/'Table_Evidence_Index.md').write_text('\n'.join(md),encoding='utf-8')
    print('Indexed',len(sources),'major tables.')


if __name__=='__main__':
    main()

"""Check identities, paired optimizer controls and complete final-suite denominators."""
import numpy as np
from revision_common import RESULTS,CANDIDATES,CANDIDATE_SPLIT,read,write,sha


def main():
    original=read(CANDIDATES)['samples']
    ids={original[i]['sample_id'] for i in read(CANDIDATE_SPLIT)['test_ids']}
    strong=read(RESULTS/'strong_optimizer_v1/results.json'); assert strong['complete']
    for path,expected in strong['protocol']['hashes'].items():
        assert sha(path)==expected, ('Frozen source/data/model changed',path)
    rows=strong['rows']; keys={(r['repetition'],r['sample_id'],r['arm']) for r in rows}
    assert len(keys)==len(rows)==900
    assert {r['sample_id'] for r in rows}==ids
    lookup={(r['repetition'],r['sample_id'],r['arm']):r for r in rows}
    paired=0
    for rep in range(3):
        for sid in ids:
            for init in ('flat','pf'):
                direct=lookup[rep,sid,'direct_'+init]; gated=lookup[rep,sid,'gated_'+init]
                assert np.allclose(direct['reference_dispatch_mw'],gated['reference_dispatch_mw'],atol=1e-8,rtol=0)
                if direct['optimizer_called'] and gated['optimizer_called']:
                    assert direct['optimizer']['status']==gated['optimizer']['status']
                    for field in ('final_dispatch_mw','final_generator_vm_pu','final_reference_vm_pu'):
                        assert np.allclose(direct[field],gated[field],atol=1e-6,rtol=0), (rep,sid,init,field)
                    paired+=1
    conditions=read(RESULTS/'independent_conditions_v1/conditions.json')['samples']
    new_ids={r['sample_id'] for r in conditions}; assert len(new_ids)==100 and not new_ids.intersection(ids)
    for demand in (16000.,16200.,16400.,16600.,16800.):
        assert sum(r['demand_mw']==demand for r in conditions)==20
    independent=read(RESULTS/'independent_conditions_v1/results.json'); assert independent['complete']
    assert independent['conditions_hash']==sha(RESULTS/'independent_conditions_v1/conditions.json')
    for arm in ('sac_only','direct_pf','gated_pf'):
        group=[r for r in independent['rows'] if r['arm']==arm]
        assert len(group)==100 and {r['sample_id'] for r in group}==new_ids
    assert all(len({r['preparation_failed'] for r in independent['rows'] if r['sample_id']==sid})==1 for sid in new_ids)
    for result in ('tight_full_pf_v1/results.json','initialization_economic_v1/results.json'):
        payload=read(RESULTS/result); assert payload['complete'] and {r['sample_id'] for r in payload['rows']}==ids
    anchor=read(RESULTS/'final_ac_anchor_v1.json'); assert anchor['complete'] and len(anchor['rows'])==44
    assert all(r['original_linear']['passed'] for r in anchor['rows'])
    for name in ('modes32_seed1_v1_full','modes32_seed2_v1_full','random32_seed1_v1_full','random32_seed2_v1_full','tight_sac_094_108_seed42_v1_full'):
        payload=read(RESULTS/(name+'.json'))
        assert payload['complete'] and len(payload['rows'])==75 and {r['sample_id'] for r in payload['rows']}==ids
        history=read(RESULTS/name.removesuffix('_full')/'training_results.json')['history']
        assert len(history)==100, ('Incomplete training budget',name,len(history))
    report={'complete':True,'shared_optimizer_control_comparisons':paired,
            'strong_trials':900,'historical_cases':75,'new_conditions':100,'new_case_generation_failures':0,
            'new_workflow_trials':300,'ac_anchor_cases':44,'new_training_runs':5,
            'training_counts_include_failed_runs':True,'script_sha256':sha(__file__)}
    write(RESULTS/'final_suite_integrity.json',report); print(report)


if __name__=='__main__': main()

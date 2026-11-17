"""Paired acceptance, timing and economics from the remeasured frozen workflows."""
import numpy as np
from revision_common import RESULTS, read, write, sha
from analyze_reviewer_followup import paired_summary
from remeasure_verified_dual import DEST


def main():
    path = DEST/'dual.json'
    raw = read(path)
    assert raw['complete'] and raw['protocol_sha256'] == sha(DEST/'protocol.json')
    protocol = read(DEST/'protocol.json')
    for p, digest in {**protocol['input_hashes'], **protocol['source_hashes'],
                      **protocol['source_hashes_v2']}.items():
        assert sha(p) == digest, p
    summary = read(RESULTS/'reviewer_sequence_balanced_v2/combined_summary.json')
    dual = paired_summary(raw, 75, 3)
    by_key = {(r['repeat'], r['sample_id'], r['arm']): r for r in raw['rows']}
    ids = list(dict.fromkeys(r['sample_id'] for r in raw['rows']))
    old = {(r['repeat'], r['sample_id'], r['arm']): r
           for r in read(RESULTS/'reviewer_followup_v1/dual.json')['rows']}
    paired = []
    for repeat in range(3):
        common = [sid for sid in ids if all(by_key[repeat, sid, a]['passed'] for a in ('direct', 'selected'))]
        a = [by_key[repeat, sid, 'direct'] for sid in common]
        b = [by_key[repeat, sid, 'selected'] for sid in common]
        paired.append({'common_accepted_ids': common, 'common_accepted_count': len(common),
            'direct_cost': float(np.mean([r['scheduled_cost'] for r in a])),
            'selected_cost': float(np.mean([r['scheduled_cost'] for r in b])),
            'paired_cost_difference': float(np.mean([y['scheduled_cost']-x['scheduled_cost'] for x,y in zip(a,b)])),
            'direct_gat_relative_l2_mw': float(np.mean([r['gat_relative_l2_mw'] for r in a])),
            'selected_gat_relative_l2_mw': float(np.mean([r['gat_relative_l2_mw'] for r in b])),
            'gained_ids': [sid for sid in ids if by_key[repeat,sid,'selected']['passed'] and not by_key[repeat,sid,'direct']['passed']],
            'lost_ids': [sid for sid in ids if by_key[repeat,sid,'direct']['passed'] and not by_key[repeat,sid,'selected']['passed']]})
    for arm in ('direct','selected'):
        stats = dual['stats'][arm]
        stats['accepted_ids_per_repeat'] = [[sid for sid in ids if by_key[r,sid,arm]['passed']] for r in range(3)]
        stats['outer_optimizer_calls_per_repeat'] = [sum(by_key[r,sid,arm]['optimizer_called'] for sid in ids) for r in range(3)]
        stats['receipt_reuses_per_repeat'] = [sum(by_key[r,sid,arm]['acceptance_receipt_reused'] for sid in ids) for r in range(3)]
        stats['fresh_candidate_checks_per_repeat'] = [sum(by_key[r,sid,arm]['fresh_candidate_checks'] for sid in ids) for r in range(3)]
    for r in raw['rows']:
        if r['passed']:
            assert not any(r['constraint_flags'].values())
        assert (r.get('optimizer') or {}).get('actual_opf_calls', 0) <= 8
        if r['acceptance_receipt_reused']:
            assert r['passed'] and r['optimizer']['initializer_attempts'][-1]['accepted']
    dual.update(paired_economics_per_repeat=paired,
        reference_cost_scope='Scheduled controllable generation only; external-source exchange cost excluded.',
        acceptance_changes_from_old_implementation=[list(k) for k,r in by_key.items() if old[k]['passed'] != r['passed']],
        max_final_dispatch_change_mw=float(max(np.max(np.abs(np.asarray(r['final_dispatch_mw'])-old[k]['final_dispatch_mw'])) for k,r in by_key.items())),
        relative_mean_reduction_percent=100*(1-dual['stats']['selected']['mean_sec']/dual['stats']['direct']['mean_sec']))
    summary.update(dual=dual, verified_once=True, protocol_sha256=sha(DEST/'protocol.json'),
                   analysis_source_sha256=sha(__file__))
    summary['sources_sha256']['dual'] = sha(path)
    summary['source_paths']['dual'] = str(path)
    summary['interpretation'] += ' Dual-start timing is newly measured with single-use context-bound acceptance receipts; no old time is subtracted. Other populations/configurations retain their own measurements.'
    write(DEST/'combined_summary.json', summary)
    write(DEST/'audit_manifest.json', {
        'primary_table_label':'tab:review_followup_runtime', 'protocol_file':str(DEST/'protocol.json'),
        'protocol_sha256':sha(DEST/'protocol.json'), 'raw_file':str(path), 'raw_sha256':sha(path),
        'analysis_source_sha256':sha(__file__),
        'policy_version':'Frozen SAC weights, analytic local voltage feature, validation-selected <=8-step gate',
        'gate':protocol['gate'], 'initializers':protocol['dual'], 'sample_ids':ids,
        'repetitions':3, 'per_case_record_fields':list(raw['rows'][0]),
        'per_repeat_results':dual, 'publicly_uploaded':False,
        'availability':'Per-case records and author-held materials available upon reasonable request.'})
    print({k:v for k,v in dual.items() if k not in ('paired_economics_per_repeat','stats')})
    for arm in ('direct','selected'):
        print(arm, {k:v for k,v in dual['stats'][arm].items() if k!='accepted_ids_per_repeat'})
    print(paired[0])


if __name__ == '__main__':
    main()

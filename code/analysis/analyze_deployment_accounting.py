"""Route groups and signed-exchange pricing, using existing accepted records."""
from statistics import mean
import math
from revision_common import RESULTS, read, write, sha

DEST=RESULTS/'deployment_accounting_v1'
SOURCE=RESULTS/'dual_verified_once_v2/dual.json'


def route(row):
    if row.get('preparation_failed'):
        return 'preparation_failed'
    if row.get('initial_passed'):
        return 'initially_accepted'
    if not row['policy_attempted']:
        return 'gate_direct'
    return 'policy_fallback' if row['optimizer_called'] else 'policy_accepted'


def analyze(payload):
    records={(r['repeat'],r['sample_id'],r['arm']):r for r in payload['rows']}
    repeats=sorted({r['repeat'] for r in payload['rows']})
    ids=list(dict.fromkeys(r['sample_id'] for r in payload['rows']))
    groups={}
    for group in ('gate_direct','policy_accepted','policy_fallback','initially_accepted','preparation_failed'):
        by_repeat=[];selected=[];direct=[]
        for repeat in repeats:
            chosen=[sid for sid in ids if route(records[repeat,sid,'selected'])==group]
            by_repeat.append(chosen)
            selected.extend(records[repeat,sid,'selected'] for sid in chosen)
            direct.extend(records[repeat,sid,'direct'] for sid in chosen)
        groups[group]={'ids_per_repeat':by_repeat,'count_per_repeat':list(map(len,by_repeat)),
            'accepted_per_repeat':[sum(records[r,sid,'selected']['passed'] for sid in chosen) for r,chosen in zip(repeats,by_repeat)],
            'mean_selected_sec':mean(r['runtime_sec'] for r in selected) if selected else None,
            'mean_direct_sec':mean(r['runtime_sec'] for r in direct) if direct else None,
            'mean_policy_sec':mean(r.get('policy_sec',0.) for r in selected) if selected else None,
            'mean_policy_steps':mean(r.get('policy_steps',0) for r in selected) if selected else None,
            'demand_levels_mw':sorted({r.get('initial_features',{}).get('demand_mw',r.get('demand_mw')) for r in selected})}
    economics=[]
    for repeat in repeats:
        common=[sid for sid in ids if all(records[repeat,sid,a]['passed'] for a in ('direct','selected'))]
        arm={a:[records[repeat,sid,a] for sid in common] for a in ('direct','selected')}
        for rows in arm.values():
            assert all(math.isfinite(r['final_verification']['reference_source_p_mw']) for r in rows)
        prices=[]
        for price in (0.,25.,50.):
            values={a:[r['scheduled_cost']+price*max(r['final_verification']['reference_source_p_mw'],0.)
                -price*max(-r['final_verification']['reference_source_p_mw'],0.) for r in rows] for a,rows in arm.items()}
            prices.append({'symmetric_exchange_price':price,'direct_mean_total_cost':mean(values['direct']),
                'selected_mean_total_cost':mean(values['selected']),
                'paired_selected_minus_direct':mean(b-a for a,b in zip(values['direct'],values['selected']))})
        economics.append({'repeat':repeat,'common_accepted_ids':common,'n':len(common),'prices':prices,
            'mean_signed_reference_p_mw':{a:mean(r['final_verification']['reference_source_p_mw'] for r in rows) for a,rows in arm.items()}})
    return {'groups':groups,'signed_exchange_per_repeat':economics,
        'scope':'Paired total costs on common accepted IDs only. Hypothetical symmetric import/export prices per MW per dispatch interval; no re-optimization or market-cost claim.'}


def main():
    raw=read(SOURCE);assert raw['complete']
    result=analyze(raw)
    result.update(complete=True,source_file=str(SOURCE),source_sha256=sha(SOURCE),analysis_sha256=sha(__file__))
    write(DEST/'summary.json',result)
    print({k:{x:v for x,v in d.items() if x!='ids_per_repeat'} for k,d in result['groups'].items()})
    print(result['signed_exchange_per_repeat'][0]['prices'])


if __name__=='__main__':
    main()

"""Audit saved voltage margins and run fixed solver checks on two failed cases."""
import argparse
import csv
import math
import os
from pathlib import Path
from revision_common import RESULTS, CANDIDATES, read, write, sha, load_aligned


DEST = RESULTS/'solver_margin_followup_v1'


def margin_audit():
    paths = [RESULTS/'tight_voltage_diagnostic_v1/results.json',
             RESULTS/'tight_voltage_boundary_v2/results.json']
    rows = []
    for label, path in zip(('inset_1e-4', 'zero_inset'), paths):
        payload = read(path)
        assert payload['complete']
        for case in payload['rows']:
            if case['band'] != [.94, 1.08]:
                continue
            joint = case['joint']
            solver = joint['optimizer']
            vm = joint['final_verification']['bus_vm_pu']
            finite = bool(vm) and all(v is not None and math.isfinite(v) for v in vm)
            lower, upper = (min(vm), max(vm)) if finite else (None, None)
            # If no candidate was returned, these extrema describe the verified
            # fallback GAT point, not a fictitious OPF solution.
            candidate = solver['returned_candidate']
            gap = max(.94-lower, upper-1.08, 0.) if finite else None
            flags = joint['constraint_flags']
            only_voltage = flags.get('voltage_violation') is True and all(
                not value for name, value in flags.items() if name != 'voltage_violation')
            rounds = solver['rounds']
            rows.append(dict(sample_id=case['sample_id'], setting=label,
                returned_candidate=candidate, passed=case['joint_passed'],
                solver_status=solver['status'], rounds=len(rounds),
                first_failed_round=next((r['iteration'] for r in rounds if not r['opf_converged']), None),
                verified_min_vm=lower, verified_max_vm=upper, max_voltage_gap=gap,
                candidate_signed_margin=min(lower-.94,1.08-upper) if candidate and finite else None,
                voltage_only_rejection=bool(candidate and not case['joint_passed'] and only_voltage),
                controls_within_box=joint['final_verification']['controls_within_original_voltage_box']))
    counts = {}
    for label in ('inset_1e-4', 'zero_inset'):
        group = [r for r in rows if r['setting']==label]
        assert len(group)==15
        gaps = [r['max_voltage_gap'] for r in group if r['voltage_only_rejection']]
        counts[label] = dict(passed=sum(r['passed'] for r in group),
            returned=sum(r['returned_candidate'] for r in group),
            voltage_only_rejections=len(gaps),
            rejected_gap_min=min(gaps) if gaps else None,
            rejected_gap_max=max(gaps) if gaps else None,
            nonconvergence=sum(r['solver_status']=='opf_nonconvergence' for r in group))
    write(DEST/'margin_audit.json', {'source_hashes':{str(p):sha(p) for p in paths}, 'rows':rows,'counts':counts})
    with (DEST/'margin_audit.csv').open('w', newline='', encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(counts, flush=True)


def solver_trials():
    os.environ.setdefault('NUMBA_CACHE_DIR', str(RESULTS/'numba_cache'))
    from compare_joint_workflows import evaluate, hashes
    import pandapower as pp
    source = RESULTS/'joint_workflow_comparison_v2/test.json'
    original = read(source)['rows']
    direct = {r['sample_id']:r for r in original if r['arm']=='gat_optimizer'}
    selected = sorted(r['sample_id'] for r in original if r['arm']=='gat_sac_optimizer'
                      and r['policy_gate_passed'] and not direct[r['sample_id']]['passed'])
    assert selected==['v10_0070','v10_0230']
    settings = [('pf_init_100', 'pf', 100), ('flat_300', 'flat', 300)]
    protocol = {'selection':selected, 'settings':settings, 'hashes':hashes(),
                'source_hash':sha(source), 'script_hash':sha(__file__),
                'scope':'Post-hoc fixed solver diagnostics; no replacement of original baseline or retuning of policy'}
    protocol_path = DEST/'solver_protocol.json'
    # JSON round-trip normalizes tuples for resumability.
    import json
    protocol = json.loads(json.dumps(protocol))
    if protocol_path.exists():
        assert read(protocol_path)==protocol
    else:
        write(protocol_path, protocol)
    output = DEST/'solver_trials.json'
    payload = read(output) if output.exists() else {'complete':False,'protocol':protocol,'rows':[]}
    assert payload['protocol']==protocol
    if payload['complete']:
        return
    data=read(CANDIDATES)
    samples=[s for s in data['samples'] if s.get('is_complete_sced', True)]
    ids=[i for i,s in enumerate(samples) if s['sample_id'] in selected]
    mod=load_aligned()
    env=mod.CleanN1Env(samples,ids,data)
    env._voltage_jacobian_dir=RESULTS/'jacobian_cache/none'
    env.reset(ids[0])
    original_runopp=pp.runopp
    for label, init, iterations in settings:
        for sid in ids:
            if any(r['setting']==label and r['sample_id']==samples[sid]['sample_id'] for r in payload['rows']):
                continue
            def configured_runopp(net, **kwargs):
                if kwargs.get('init')=='flat':
                    kwargs['init']=init
                kwargs['PDIPM_MAX_IT']=iterations
                return original_runopp(net, **kwargs)
            pp.runopp=configured_runopp
            try:
                row=evaluate(env,mod,None,samples[sid],sid,data['linear_security_model'],'gat_optimizer')
            finally:
                pp.runopp=original_runopp
            row['setting']=label
            payload['rows'].append(row)
            write(output,payload)
            print(label,row['sample_id'],row['passed'],row['optimizer']['status'],flush=True)
    payload['complete']=True
    write(output,payload)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--margin-only',action='store_true')
    args=parser.parse_args()
    DEST.mkdir(exist_ok=True)
    margin_audit()
    if not args.margin_only:
        solver_trials()

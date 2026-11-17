"""Matched implementation remeasurement with a frozen short-policy gate.

Validation compares the previous analytic-feature workflow against sharing its
already solved AC network. This does not select another gate or change weights.
The previously evaluated holdout is reused for software timing, not a new test.
"""
import argparse
import shutil
import numpy as np
import evaluate_short_policy as base
import evaluate_implicit_short_policy as previous
import shared_voltage_runtime as runtime
from revision_common import RESULTS, read, write, sha

DEST=RESULTS/'shared_voltage_speed_v3'
PREVIOUS=RESULTS/'implicit_short_policy_v2'


def hashes():
    return {str(p):sha(p) for p in (__file__,base.__file__,previous.__file__,runtime.__file__,
        __import__('probe_voltage_kernel_v5').__file__)}


def setup(samples,data,bank):
    mod,envs,actor=previous.setup(samples,data,bank)
    for key in ('direct','short8','selected'):
        runtime.install(envs[key],mod)
    return mod,envs,actor


def validate():
    payload=read(DEST/'val.json')
    assert payload['complete']
    rows={a:{r['sample_id']:r for r in payload['rows'] if r['arm']==a}
          for a in ('direct','short4','short8')}
    differences=[]
    for sid,old in rows['short4'].items():
        new=rows['short8'][sid]
        for field in ('passed','preparation_failed','policy_attempted','policy_steps','optimizer_called'):
            if old.get(field)!=new.get(field): differences.append([sid,field,old.get(field),new.get(field)])
        for field in ('final_dispatch_mw','final_generator_vm_pu','final_reference_vm_pu'):
            if field in old:
                delta=float(np.max(np.abs(np.asarray(old[field])-np.asarray(new[field]))))
                if delta>1e-6:differences.append([sid,field,delta])
    calls=[c for r in rows['short8'].values() for c in r.get('voltage_backend_calls',[])]
    summary={'complete':True,'trajectory_differences':differences,
        'source_counts':{s:sum(c['source']==s for c in calls) for s in {c['source'] for c in calls}},
        'mean_sec':{a:float(np.mean([r['runtime_sec'] for r in rs.values()])) for a,rs in rows.items()},
        'passed':{a:sum(r['passed'] for r in rs.values()) for a,rs in rows.items()},
        'interpretation':'short4 = previous frozen gated workflow; short8 = same gate with shared solved network. Names are evaluator slots, not changed step caps.'}
    write(DEST/'validation_equivalence.json',summary)
    assert not differences, differences
    assert calls and all(c['source']=='shared_solved_ac' for c in calls), summary
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','val','freeze','test'))
    args=parser.parse_args()
    rule=read(PREVIOUS/'protocol.json')['rule']
    if args.stage=='prepare':
        if DEST.exists():raise RuntimeError('Preserve prior measurements')
        DEST.mkdir()
        for name in ('val_conditions.json','test_conditions.json','bank.json'):
            shutil.copy2(PREVIOUS/name,DEST/name)
        write(DEST/'plan.json',{'runner_hashes':hashes(),'rule':rule,
            'scope':'Implementation-only remeasurement on existing cases; no gate, checkpoint, limits, or optimization-budget tuning. Same Numba verifier for both measured test arms; fresh final verification retained.'})
        return
    base.DEST=DEST;base.setup=setup
    if args.stage=='freeze':
        assert not (DEST/'protocol.json').exists()
        summary=validate()
        protocol=dict(read(PREVIOUS/'protocol.json'))
        protocol.update(script_sha256=sha(base.__file__),validation_sha256=sha(DEST/'val.json'),
            test_conditions_sha256=sha(DEST/'test_conditions.json'),bank_sha256=sha(DEST/'bank.json'),
            runner_hashes=hashes(),validation_equivalence_sha256=sha(DEST/'validation_equivalence.json'),
            interpretation='Same-case software timing remeasurement, not new independent test evidence. Gate and checkpoint frozen before this optimization; matched Numba physical verifier, shared solved network, fresh final check.')
        write(DEST/'protocol.json',protocol)
        print(summary,flush=True)
        return
    if args.stage=='test':
        protocol=read(DEST/'protocol.json')
        assert protocol['runner_hashes']==hashes()
        assert protocol['validation_equivalence_sha256']==sha(DEST/'validation_equivalence.json')
    original=base.evaluate
    def measured(env,mod,actor,sample,sid,model,cap,unused=None):
        runtime.clear_case(env)
        before=len(env.implicit_feature_calls)
        row=original(env,mod,actor,sample,sid,model,0 if cap==0 else rule['cap'],None if cap==0 else rule)
        row['voltage_backend_calls']=env.implicit_feature_calls[before:]
        return row
    base.evaluate=measured
    base.run(args.stage)
    path=DEST/(args.stage+'.json');payload=read(path)
    payload.update(runner_hashes=hashes(),rule=rule,
        interpretation='Implementation remeasurement; existing cases and frozen gate; online preparation/feature/fallback/fresh final checks included, resident setup excluded.')
    write(path,payload)
    if args.stage=='val': print(validate(),flush=True)


if __name__=='__main__':main()

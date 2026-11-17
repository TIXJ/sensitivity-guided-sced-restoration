"""A fixed joint AC/LODF fallback; develop on validation and freeze before test."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from revision_common import CANDIDATES,CANDIDATE_SPLIT,RESULTS,read,sha,write

ROOT=RESULTS/'optimization_diagnostics/joint_fallback'
WORKER=Path(__file__).with_name('pilot_joint_ac_lodf.py')
PROTOCOL=ROOT/'protocol.json'


def freeze():
    if PROTOCOL.exists():
        raise ValueError('Protocol already frozen')
    val=read(ROOT/'val_summary.json')
    assert val['complete'] and val['scope']=='val'
    write(PROTOCOL,{'voltage_range':'policy','voltage_margin_pu':0.0001,
        'worker_timeout_sec':90,'selected_on':'validation only',
        'val_result_sha256':sha(ROOT/'val_summary.json'),'worker_sha256':sha(WORKER),
        'driver_sha256':sha(__file__),'candidate_sha256':sha(CANDIDATES),'split_sha256':sha(CANDIDATE_SPLIT),
        'contract':'Retain original SAC accepts; run full-generator joint AC/linear-LODF optimization '
            'only after rejection. Preserve hardware limits and original voltage residual box. '
            'No 40-MW active-power cap in the optimizer. Original verifier remains unchanged.'})
    print('Frozen:',PROTOCOL,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',choices=('val','test'),default='val')
    parser.add_argument('--freeze-protocol',action='store_true')
    args=parser.parse_args()
    os.environ.setdefault('NUMBA_CACHE_DIR',str(RESULTS/'numba_cache'))
    ROOT.mkdir(parents=True,exist_ok=True)
    if args.freeze_protocol:
        freeze(); return
    if args.scope=='test':
        p=read(PROTOCOL)
        assert p['worker_sha256']==sha(WORKER) and p['driver_sha256']==sha(__file__)
        assert p['val_result_sha256']==sha(ROOT/'val_summary.json')
        assert p['candidate_sha256']==sha(CANDIDATES) and p['split_sha256']==sha(CANDIDATE_SPLIT)
    source_path=RESULTS/('optimization_diagnostics/validation_full_reference.json' if args.scope=='val' else 'full_sac.json')
    source=read(source_path)
    assert source['complete'] and source['settings']['scope']==args.scope
    data=read(CANDIDATES); samples=[s for s in data['samples'] if s.get('is_complete_sced',True)]
    ids=read(CANDIDATE_SPLIT)[args.scope+'_ids']
    assert [r['sample_id'] for r in source['rows']]==[samples[i]['sample_id'] for i in ids]
    summary_path=ROOT/f'{args.scope}_summary.json'
    payload={'complete':False,'scope':args.scope,'source_sha256':sha(source_path),
             'worker_sha256':sha(WORKER),'checkpoint_sha256':source['checkpoint_sha256'],
             'protocol_sha256':sha(PROTOCOL) if args.scope=='test' else None,
             'timing_scope':'Worker wall time includes process/model setup; optimizer timing is separate. '
                'Neither is combined with an earlier policy timing to claim a newly measured pipeline.',
             'rows':[]}
    if summary_path.exists():
        old=read(summary_path)
        assert old['source_sha256']==payload['source_sha256'] and old['worker_sha256']==payload['worker_sha256']
        if old['complete']:
            print('Completed result preserved:',summary_path); return
    for i,row in enumerate(source['rows'],1):
        report={'sample_id':row['sample_id'],'demand_mw':row['demand_mw'],
                'baseline_passed':row['passed'],'passed':row['passed'],'fallback_attempted':not row['passed']}
        if not row['passed']:
            folder=ROOT/args.scope; folder.mkdir(exist_ok=True)
            path=folder/(row['sample_id']+'.json')
            if not path.exists():
                cmd=[sys.executable,'-X','utf8',str(WORKER),'--sample-id',row['sample_id'],
                     '--scope',args.scope,'--voltage-range','policy','--voltage-margin','0.0001','--output',str(path)]
                started=time.perf_counter()
                with path.with_suffix('.log').open('a',encoding='utf-8') as log:
                    try:
                        process=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=90)
                        if process.returncode:
                            raise RuntimeError(f'Worker error, not an experimental failure: {path.with_suffix(".log")}')
                    except subprocess.TimeoutExpired:
                        write(path,{'complete':True,'sample_id':row['sample_id'],'passed':False,'status':'time_limit_90s'})
                write(path.with_suffix('.wall.json'),{'worker_wall_sec':time.perf_counter()-started})
            candidate=read(path)
            assert candidate['complete'] and candidate['sample_id']==row['sample_id']
            report.update(passed=candidate['passed'],fallback_result=str(path),fallback_sha256=sha(path),
                          optimizer_sec=candidate.get('runtime_sec'),
                          worker_wall_sec=read(path.with_suffix('.wall.json'))['worker_wall_sec'])
        payload['rows'].append(report)
        write(summary_path,payload)
        if report['fallback_attempted']:
            print(f"{i}/{len(ids)} {row['sample_id']}: fallback {'PASS' if report['passed'] else 'FAIL'}",flush=True)
    payload['complete']=True
    payload['summary']={'n':len(ids),'baseline_passed':sum(r['baseline_passed'] for r in payload['rows']),
        'combined_passed':sum(r['passed'] for r in payload['rows']),
        'by_demand':{str(d):{'n':sum(r['demand_mw']==d for r in payload['rows']),
                            'passed':sum(r['demand_mw']==d and r['passed'] for r in payload['rows'])}
                     for d in sorted({r['demand_mw'] for r in payload['rows']})}}
    write(summary_path,payload)
    print(payload['summary'],flush=True)


if __name__=='__main__':
    main()

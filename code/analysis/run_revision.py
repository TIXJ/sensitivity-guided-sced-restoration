"""Serial, resumable revision suite. Use --dry-run to inspect exact commands."""
import argparse
import os
import subprocess
import sys
from revision_common import *

def jobs(stage):
    py=sys.executable
    ev=str(HERE/'evaluate_revision.py')
    train=str(HERE/'train_revision.py')
    data_args=['--dataset',str(DATASET),'--split-file',str(SPLIT),'--gat-checkpoint',str(GAT)]
    def eval_job(name, opts=()):
        return name,[py,ev,*map(str,opts),'--output',str(RESULTS/(name+'.json'))]
    if stage=='full-test':
        for method in CHECKPOINTS:
            yield eval_job('full_'+method,['--method',method])
        for method, script in [('acopf','eval_route_b_acopf_baseline.py'),('scopf','eval_case300_ac_scopf_baseline.py')]:
            yield 'full_'+method,[py,str(CODE/script),'--dataset',str(CANDIDATES),'--split-file',str(CANDIDATE_SPLIT),'--scope','test','--output',str(RESULTS/('full_'+method+'.json'))]
        yield 'full_repair',[py,str(HERE/'run_unfiltered_repair.py'),'--output',str(RESULTS/'full_repair.json')]
    elif stage=='seeds':
        for method in CHECKPOINTS:
            for seed in (1,2):
                name=f'{method}_seed{seed}'
                out=RESULTS/name
                if method in ('ddpg','td3'):
                    cmd=[py,str(CODE/'train_deterministic_n1_aligned.py'),method.upper()]
                else:
                    cmd=[py,train,'--family','full68' if method=='full68' else 'physics']
                yield name,cmd+data_args+['--seed',str(seed),'--updates','100','--output-dir',str(out)]
                yield eval_job(name+'_full',['--method',method,'--checkpoint',out/'best_actor.pth'])
    elif stage=='ablations':
        variants={
            'random32':['--family','random'], 'modes32':['--family','modes'], 'keys32':['--family','keys'],
            'dim16':['--dimension','16','--shifts','6'], 'dim24':['--dimension','24','--shifts','9'],
            'dim40':['--dimension','40','--shifts','15'], 'split8_24':['--shifts','8'], 'split20_12':['--shifts','20'],
            'radius20':['--active-target-range-mw','20'], 'radius60':['--active-target-range-mw','60'],
            'n1_half':['--n1-potential-weight','150','--n1-progress-weight','300','--n1-residual-weight','25'],
            'n1_double':['--n1-potential-weight','600','--n1-progress-weight','1200','--n1-residual-weight','100'],
            'voltage_half':['--voltage-sum-weight','2500','--voltage-max-weight','5000','--voltage-progress-weight','2000'],
            'voltage_double':['--voltage-sum-weight','10000','--voltage-max-weight','20000','--voltage-progress-weight','8000'],
            'bonus_half':['--success-scale','.5'], 'bonus_double':['--success-scale','2'],
            'ac_every1':['--ac-pf-check-every','1'], 'ac_every10':['--ac-pf-check-every','10'],
        }
        for name,opts in variants.items():
            out=RESULTS/name
            yield name,[py,train,*opts,*data_args,'--seed','42','--updates','100','--output-dir',str(out)]
            paired=[]
            mapping={'--family':'--family','--dimension':'--dimension','--shifts':'--shifts','--active-target-range-mw':'--radius','--ac-pf-check-every':'--ac-every'}
            for key,dest in mapping.items():
                if key in opts: paired.extend([dest,opts[opts.index(key)+1]])
            yield eval_job(name+'_full',['--checkpoint',out/'best_actor.pth',*paired])
    elif stage in ('stress','stress-recheck'):
        for stress in ('regional','reactive','renewable','demand-low','demand-high'):
            if stage=='stress-recheck' and stress!='renewable': continue
            name='stress_'+stress+('_v2' if stress=='renewable' else '')
            yield eval_job(name,['--stress',stress])
        if stage=='stress':
            for low,high in ((.94,1.08),(.95,1.05)):
                yield eval_job(f'voltage_{low}_{high}',['--vmin',low,'--vmax',high])
        for init in ('label','economic','previous'):
            yield eval_job('start_'+init+'_v2',['--initialization',init])
    elif stage=='ac-audit':
        yield 'ac_contingencies',[py,str(HERE/'ac_contingency_audit.py')]
    elif stage=='smallcases':
        for case in (39,118):
            yield f'case{case}_action',[py,str(HERE/'run_smallcase_actions.py'),'--case',str(case),'--output',str(RESULTS/f'case{case}_actions.json')]
            yield f'case{case}_baselines',[py,str(HERE/'smallcase_baselines.py'),'--case',str(case),'--output',str(RESULTS/f'case{case}_baselines.json')]
    elif stage=='evaluation-fixes':
        dest = RESULTS/'evaluation_contract_v2'
        yield 'case39_classification_v2', [py,str(HERE/'run_smallcase_actions.py'),'--case','39',
            '--modes','full,active_power_only,no_active_power,no_generator_voltage,no_reference_voltage',
            '--output',str(dest/'case39_actions.json')]
        for case in (39,118):
            yield f'case{case}_common_v2', [py,str(HERE/'evaluate_smallcases_common.py'),'--case',str(case),
                                           '--output',str(dest/f'case{case}_common.json')]
        yield 'case300_terminal_flags_v2',[py,str(HERE/'audit_saved_terminal_constraints.py'),'--output',str(dest/'case300_terminal_flags.json')]
        yield 'paired_ac_audit_v2',[py,str(HERE/'audit_paired_ac_contingencies.py'),'--output',str(dest/'paired_ac_audit.json')]
        yield 'summarize_evaluation_contract',[py,str(HERE/'summarize_evaluation_contract.py')]
    elif stage=='summarize':
        yield 'summarize',[py,str(HERE/'summarize_revision.py')]
    elif stage=='export':
        yield 'export',[py,str(HERE/'export_reproducibility.py')]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['full-test','seeds','ablations','stress','stress-recheck','ac-audit','smallcases','evaluation-fixes','summarize','export','all'])
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--only',help='Exact job name, e.g. dim16 or dim16_full')
    a=p.parse_args()
    if not (REV/'backup_receipt.json').exists(): p.error('Run backup_original.py first')
    stages=['full-test','ac-audit','stress','smallcases','evaluation-fixes','seeds','ablations','summarize','export'] if a.stage=='all' else [a.stage]
    env=os.environ.copy()
    env['PYTHONUTF8']='1'
    env['NUMBA_CACHE_DIR']=str(RESULTS/'numba_cache')
    env['PYTHONPATH']=os.pathsep.join([str(HERE/'legacy_dependencies'),str(CODE),str(ROOT),env.get('PYTHONPATH','')])
    Path(env['NUMBA_CACHE_DIR']).mkdir(parents=True,exist_ok=True)
    for stage in stages:
        for name,cmd in jobs(stage):
            if a.only and a.only!=name: continue
            print(name+': '+subprocess.list2cmdline(cmd),flush=True)
            if a.dry_run: continue
            if stage=='evaluation-fixes' and '--output' in cmd:
                result_path=Path(cmd[cmd.index('--output')+1])
                if result_path.exists():
                    saved=read(result_path)
                    if saved.get('complete') or (name=='case39_classification_v2' and saved.get('n_eval')==30):
                        print('Existing completed evaluation preserved; use a new output path for a new protocol.',flush=True)
                        continue
            done=RESULTS/'job_records'/(name+'.json')
            if stage not in ('summarize','export') and done.exists() and read(done).get('returncode')==0:
                print('Already completed; preserved.',flush=True); continue
            log=RESULTS/'logs'/(name+'.log'); log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('a',encoding='utf-8') as f:
                f.write('\n=== Job attempt ===\n')
                child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
                for line in child.stdout:
                    print(line,end='',flush=True); f.write(line); f.flush()
                rc=child.wait()
            write(done,{'command':cmd,'returncode':rc,'log':str(log)})
            if rc: raise SystemExit(rc)

if __name__=='__main__': main()

"""Run final reviewer experiments serially, retaining completed records and failures."""
import os
import subprocess
import sys
import shutil
import time
from revision_common import HERE,RESULTS,write,read


def preserve_partial_training(stage):
    names=([f'{family}32_seed{seed}_v1' for family in ('modes','random') for seed in (1,2)]
           if stage=='run_basis_seed_completion.py' else ['tight_sac_094_108_seed42_v1'] if stage=='train_tight_policy.py' else [])
    for name in names:
        out=(RESULTS/name).resolve(); result=out/'training_results.json'
        if not result.exists() or len(read(result).get('history',[]))==100:
            continue
        assert out.is_relative_to(RESULTS.resolve())
        stamp=str(time.time_ns()); backup=(RESULTS/('interrupted_'+name+'_'+stamp)).resolve()
        assert backup.is_relative_to(RESULTS.resolve()) and not backup.exists()
        shutil.copytree(out,backup)
        target=out/('training_results.partial_'+stamp+'.json')
        assert not target.exists()
        result.rename(target)
        print('Preserved incomplete training before a full-budget restart:',backup,flush=True)


def main():
    stages=['compare_strong_optimizer.py','summarize_strong_optimizer.py','audit_final_ac_anchor.py',
            'evaluate_independent_conditions.py','evaluate_tight_full.py','compare_initialization_workflows.py',
            'run_basis_seed_completion.py','train_tight_policy.py','audit_final_suite.py']
    status=RESULTS/'final_evidence_progress.json'
    env=os.environ.copy(); env['PYTHONIOENCODING']='utf-8'; env['NUMBA_CACHE_DIR']=str(RESULTS/'numba_cache')
    completed=[]
    logs=RESULTS/'final_evidence_logs'; logs.mkdir(exist_ok=True)
    for name in stages:
        preserve_partial_training(name)
        write(status,{'complete':False,'completed':completed,'running':name,'remaining':stages[len(completed)+1:]})
        print('START',name,flush=True)
        with (logs/(name+'.log')).open('a',encoding='utf-8') as log:
            process=subprocess.Popen([sys.executable,str(HERE/name)],env=env,stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            for line in process.stdout:
                log.write(line); log.flush(); print(line,end='',flush=True)
            code=process.wait()
            if code:
                write(status,{'complete':False,'completed':completed,'running':None,'failed':name,'exit_code':code,'remaining':stages[len(completed):]})
                raise subprocess.CalledProcessError(code,process.args)
        completed.append(name)
    write(status,{'complete':True,'completed':completed,'running':None,'remaining':[]})


if __name__=='__main__': main()

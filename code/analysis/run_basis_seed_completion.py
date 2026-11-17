"""Complete missing policy seeds with a fixed random-basis seed; never omit failures."""
import os
import subprocess
import sys
import shutil
from datetime import datetime
from revision_common import HERE, RESULTS, DATASET, SPLIT, GAT, read, write, sha


def main():
    protocol={'families':['modes','random'],'training_seeds':[1,2], 'existing_seed':42,
              'random_basis_seed':20260721,'updates':100,'orchestrator_sha256':sha(__file__),
              'scope':'Fixed-basis policy-initialization variability, not random-basis population variability',
              'sources':{str(HERE/name):sha(HERE/name) for name in ['train_revision.py','evaluate_revision.py','revision_common.py']}}
    path=RESULTS/'basis_seed_completion_protocol.json'
    if path.exists(): assert read(path)==protocol
    else: write(path,protocol)
    environment=os.environ.copy(); environment['NUMBA_CACHE_DIR']=str(RESULTS/'numba_cache'); environment['PYTHONIOENCODING']='utf-8'
    for family in protocol['families']:
        for seed in protocol['training_seeds']:
            name=f'{family}32_seed{seed}_v1'; out=RESULTS/name
            train=[sys.executable,str(HERE/'train_revision.py'),'--family',family,'--basis-seed','20260721','--seed',str(seed),
                   '--updates','100','--dataset',str(DATASET),'--split-file',str(SPLIT),'--gat-checkpoint',str(GAT),'--output-dir',str(out)]
            if not (out/'training_results.json').exists():
                if out.exists() and any(out.iterdir()):
                    backup=RESULTS/('interrupted_'+name+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
                    shutil.copytree(out,backup)
                    print('Partial run preserved:',backup,flush=True)
                out.mkdir(exist_ok=True)
                print('START training',name,flush=True)
                with (out/'console.log').open('a',encoding='utf-8') as log:
                    subprocess.run(train,check=True,env=environment,stdout=log,stderr=subprocess.STDOUT)
            result=RESULTS/(name+'_full.json')
            if not result.exists() or not read(result).get('complete'):
                subprocess.run([sys.executable,str(HERE/'evaluate_revision.py'),'--family',family,'--checkpoint',str(out/'best_actor.pth'),'--output',str(result)],check=True,env=environment)
            print('Completed',name,flush=True)


if __name__=='__main__': main()

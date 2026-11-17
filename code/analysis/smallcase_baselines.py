"""Common prediction-only and one-step-search baselines on both smaller systems."""
import argparse
import copy
import time
import numpy as np
from revision_common import *

def passing(info):
    return bool(info.get('converged',False) and all(info.get(k,999)==0 for k in ('v_violations','overloaded_lines','n1_violations')) and info.get('trafo_overloaded',0)==0)

def score(info):
    return (not info.get('converged',False),int(info.get('v_violations',999))+int(info.get('overloaded_lines',999))+int(info.get('n1_violations',999)),float(info.get('v_violation_max_pu',0))+float(info.get('n1_overload_mw',0)))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=int,choices=[39,118],required=True)
    p.add_argument('--max-samples',type=int,default=0)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if Path(a.output).exists(): p.error('Output exists')
    if a.case==39:
        import importlib.util
        spec=importlib.util.spec_from_file_location('train_gat39_v1',HERE/'legacy_dependencies/train_gat39_v1.py')
        gat=importlib.util.module_from_spec(spec)
        sys.modules['train_gat39_v1']=gat
        spec.loader.exec_module(gat)
        import train_rl39_v1 as mod
        data=read(ROOT/'case39_routeb_rescuable_dataset.json')
        samples=data['samples']; _,_,ids=mod.split_samples(samples)
        Env=mod.Env39
    else:
        import train_rl118_v1 as mod
        data=read(ROOT/'case118_routeb_rescuable_dataset.json')
        samples=[s for s in data['samples'] if s.get('is_complete_sced')]
        _,_,ids=mod.split_samples_stratified(samples)
        Env=mod.SCEDEnv118
    mod.AC_PF_CHECK_EVERY=1; mod.N1_CHECK_EVERY_STEPS=1
    if a.max_samples: ids=ids[:a.max_samples]
    rows=[]
    for mode in ('prediction_only','myopic_search'):
        env=Env(samples,ids)
        for sid in ids:
            start=time.perf_counter(); env.reset(sid)
            checked=copy.deepcopy(env)
            _,_,_,_,info=checked.step(np.zeros(env.action_dim))
            nsteps=0; calls=1
            if mode=='myopic_search' and not passing(info):
                for step in range(1,env.max_steps+1):
                    low,high=env.get_action_bounds()
                    pdim=env._n_shift_dirs+env._n_key_gens
                    candidates=[np.zeros(env.action_dim)]
                    for sign in (-1,1):
                        voltage=np.zeros(env.action_dim); voltage[pdim:]=sign*.5
                        candidates.append(voltage*high)
                        active=np.zeros(env.action_dim); active[(step-1)%pdim]=sign*.5
                        candidates.append(active*high)
                    evaluated=[]
                    for action in candidates:
                        trial=copy.deepcopy(env)
                        _,_,_,_,result=trial.step(action)
                        evaluated.append((score(result),trial,result))
                    _,env,info=min(evaluated,key=lambda row:row[0])
                    nsteps=step; calls+=len(candidates)
                    if passing(info): break
            checked=copy.deepcopy(env)
            _,_,_,_,final=checked.step(np.zeros(env.action_dim))
            calls+=1
            row={'sample_id':str(samples[sid].get('sample_id',samples[sid].get('id',sid))),'case':a.case,'mode':mode,'passed':passing(final),'steps':nsteps,'runtime_sec':time.perf_counter()-start,'power_flow_checks':calls,'demand_mw':samples[sid]['demand_mw'],'reason':'pass' if passing(final) else str(score(final)),'dispatch_mw':env.current_power.tolist()}
            rows.append(row)
            write(a.output,{'complete':False,'settings':vars(a),'rows':rows})
            print(f"{mode} {sid} {row['passed']}",flush=True)
    write(a.output,{'complete':True,'settings':vars(a),'rows':rows})

if __name__=='__main__': main()

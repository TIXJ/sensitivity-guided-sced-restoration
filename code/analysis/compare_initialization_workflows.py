"""GAT versus online linear-SCED initialization with a common economic AC objective."""
import os
import time
import numpy as np
from revision_common import RESULTS,CANDIDATES,CANDIDATE_SPLIT,load_aligned,read,write,sha
from compare_joint_workflows import evaluate,hashes
from pilot_joint_ac_lodf import add_violated_cuts


def linear_sced(sample,env,model):
    import gurobipy as gp
    m=gp.Model(); m.Params.OutputFlag=0; m.Params.Threads=1; m.Params.TimeLimit=30
    p=m.addMVar(env.N,lb=env.PMIN,ub=env.PMAX*.99)
    m.addConstr(p.sum()==sample['demand_mw'])
    c2=np.asarray(sample['sample_cp2']); c1=np.asarray(sample['sample_cp1'])
    m.setObjective((c2*p*p).sum()+c1@p)
    s=np.asarray(model['S']); c=np.asarray(model['flow_intercept']); lodf=np.asarray(model['lodf']); limits=np.asarray(model['effective_limit'])
    cuts=[]
    try:
        for _ in range(20):
            m.optimize()
            if m.Status!=gp.GRB.OPTIMAL:
                raise RuntimeError(f'linear_sced_status_{m.Status}')
            power=p.X.copy(); old=len(cuts); excess,added=add_violated_cuts(power,model,cuts)
            if excess<=1e-5: return power
            if not added: break
            for outage,line,sign in cuts[old:]:
                gradient=s[line].copy(); constant=c[line]
                if outage>=0:
                    gradient+=lodf[line,outage]*s[outage]; constant+=lodf[line,outage]*c[outage]
                m.addConstr(sign*(gradient@p+constant)<=limits[line])
        raise RuntimeError('linear_sced_cut_budget_exhausted')
    finally: m.dispose()


def main():
    os.environ.setdefault('NUMBA_CACHE_DIR',str(RESULTS/'numba_cache'))
    root=RESULTS/'initialization_economic_v1'; root.mkdir(exist_ok=True)
    protocol={'hashes':{**hashes(),'script':sha(__file__)},'arms':['gat','online_linear_sced'],
        'joint_objective':'Identical scheduled generator quadratic cost, not distance from different references',
        'joint_budget':'PF initialization, 8 rounds, 100 interior-point iterations/round',
        'linear_budget':'20 cut rounds, 30 s per Gurobi solve, one thread',
        'timing':'Initializer generation, projection, checks, joint optimizer and final verification included; resident setup excluded; order alternates.'}
    path=root/'protocol.json'
    if path.exists(): assert read(path)==protocol
    else: write(path,protocol)
    output=root/'results.json'; payload=read(output) if output.exists() else {'complete':False,'protocol':protocol,'rows':[]}
    assert payload['protocol']==protocol
    if payload['complete']: return
    import pandapower as pp
    import gurobipy as gp
    # Warm the resident solver backend just as the model/network backend is
    # warmed; every per-case model construction and solve remains timed.
    warm_model=gp.Model(); warm_model.Params.OutputFlag=0; warm_model.dispose()
    data=read(CANDIDATES); samples=[s for s in data['samples'] if s.get('is_complete_sced',True)]; ids=read(CANDIDATE_SPLIT)['test_ids']
    mod=load_aligned(); envs={a:mod.CleanN1Env(samples,ids,data) for a in protocol['arms']}
    for env in envs.values():
        env._voltage_jacobian_dir=RESULTS/'jacobian_cache/none'; env.reset(ids[0])
    runopp=pp.runopp; done={(r['sample_id'],r['arm']) for r in payload['rows']}
    for j,sid in enumerate(ids):
        sample=samples[sid]
        for arm in (protocol['arms'] if j%2==0 else protocol['arms'][::-1]):
            if (sample['sample_id'],arm) in done: continue
            env=envs[arm]; predictor=env._predict_gat
            if arm=='online_linear_sced': env._predict_gat=lambda demand:linear_sced(sample,env,data['linear_security_model'])
            def economic_solver(net,**kwargs):
                if kwargs.get('init')=='flat': kwargs['init']='pf'
                for i,index in enumerate(net.gen.index):
                    mask=(net.poly_cost.et=='gen') & (net.poly_cost.element==index)
                    for column,key in [('cp0_eur','sample_cp0'),('cp1_eur_per_mw','sample_cp1'),('cp2_eur_per_mw2','sample_cp2')]:
                        net.poly_cost.loc[mask,column]=sample[key][i]
                return runopp(net,**kwargs)
            pp.runopp=economic_solver; started=time.perf_counter()
            try:
                row=evaluate(env,mod,None,sample,sid,data['linear_security_model'],'gat_optimizer')
            except RuntimeError as exc:
                row={'sample_id':sample['sample_id'],'passed':False,'runtime_sec':time.perf_counter()-started,'preparation_failed':True,'error':str(exc)}
            finally:
                env._predict_gat=predictor; pp.runopp=runopp
            row['arm']=arm; payload['rows'].append(row); write(output,payload)
        print('initializer comparison',j+1,'/75',flush=True)
    assert len(payload['rows'])==150
    payload['complete']=True; write(output,payload)


if __name__=='__main__': main()

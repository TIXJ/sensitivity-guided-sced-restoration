"""All 75 fixed upstream cases under the compatible 0.94--1.08 bus band."""
import os
import numpy as np
from revision_common import RESULTS,CANDIDATES,CANDIDATE_SPLIT,load_aligned,read,write,sha
from compare_joint_workflows import evaluate,hashes
from shared_joint_optimizer import voltage_box


def main():
    os.environ.setdefault('NUMBA_CACHE_DIR',str(RESULTS/'numba_cache'))
    root=RESULTS/'tight_full_pf_v1'; root.mkdir(exist_ok=True)
    protocol={'band':[.94,1.08],'initializer':'pf','margin':.0001,'budget':'8 rounds, 100 IP iterations per round',
              'hashes':{**hashes(),'script':sha(__file__)}}
    output=root/'results.json'
    payload=read(output) if output.exists() else {'complete':False,'protocol':protocol,'rows':[]}
    assert payload['protocol']==protocol
    if payload['complete']: return
    write(root/'protocol.json',protocol)
    data=read(CANDIDATES); samples=[s for s in data['samples'] if s.get('is_complete_sced',True)]
    ids=read(CANDIDATE_SPLIT)['test_ids']; mod=load_aligned()
    env=mod.CleanN1Env(samples,ids,data); env._voltage_jacobian_dir=RESULTS/'jacobian_cache/none'; env.reset(ids[0])
    mod.core.RL_VMIN_PU,mod.core.RL_VMAX_PU=protocol['band']
    _,lo,hi=voltage_box(env,mod)
    assert np.all(np.maximum(lo,.9401)<=np.minimum(hi,1.0799)), 'Protocol requires compatible control bounds'
    import pandapower as pp
    original=pp.runopp
    def initialized(net,**kwargs):
        if kwargs.get('init')=='flat': kwargs['init']='pf'
        return original(net,**kwargs)
    pp.runopp=initialized
    try:
        for sid in ids[len(payload['rows']):]:
            row=evaluate(env,mod,None,samples[sid],sid,data['linear_security_model'],'gat_optimizer')
            solver=row['optimizer']
            row['status']='accepted' if row['passed'] else ('candidate_rejected' if solver and solver['returned_candidate'] else 'solver_nonconvergence')
            payload['rows'].append(row); write(output,payload)
            print(len(payload['rows']),row['sample_id'],row['status'],flush=True)
    finally: pp.runopp=original
    assert len(payload['rows'])==75
    payload['complete']=True; write(output,payload)


if __name__=='__main__': main()

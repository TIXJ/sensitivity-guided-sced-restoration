"""Check basis geometry and physical action decoding before costly training."""
import numpy as np
from revision_common import *

def main():
    data=read(DATASET); samples=[s for s in data['samples'] if s.get('is_complete_sced',True)]
    sid=read(SPLIT)['train_ids'][0]
    records=[]; reference=None
    for family,dim,shifts,bonus in [('physics',32,12,1),('physics',32,12,.5),('physics',16,6,1),('random',32,12,1),('modes',32,12,1),('keys',32,12,1),('full68',32,12,1)]:
        mod=load_aligned()
        Env=configure_env(mod,family,dim,shifts,success_scale=bonus)
        env=Env(samples,[sid],data); obs=env.reset(sid)
        basis=env._build_orthogonal_physics_action_basis()
        p_dim=env._n_shift_dirs+env._n_key_gens
        assert basis.shape==(68,p_dim)
        assert np.max(abs(basis.sum(axis=0)))<1e-10
        if family!='full68': assert np.allclose(basis.T@basis,np.eye(p_dim),atol=1e-10)
        if family=='physics' and dim==32:
            if reference is None: reference=basis.copy()
            else: assert np.array_equal(reference,basis), 'Reward ablation changed basis'
        raw=np.zeros(env.action_dim); raw[0]=.1
        low,high=env.get_action_bounds()
        expected=env.gat_power+40*basis@raw[:p_dim]*(68/67 if family=='full68' else 1)
        expected=mod.core.project_box_sum(expected,env.PMIN,env.PMAX*.99,env.target_demand)
        new_obs,reward,_,_,_=env.step(raw*high)
        assert np.allclose(expected,env.current_power,atol=1e-5)
        assert len(new_obs)==len(obs) and np.isfinite(new_obs).all() and np.isfinite(reward)
        records.append({'family':family,'p_dim':p_dim,'obs_dim':len(obs),'success_scale':bonus,'passed':True})
    write(RESULTS/'contract_check.json',{'rows':records})
    print('All seven action-contract checks passed.')

if __name__=='__main__': main()

"""Bounded joint P/V search on the already preselected three validation cases.

This is a local diagnostic, never an infeasibility certificate or policy score.
"""
import copy
import os
import time
import numpy as np
from scipy.optimize import least_squares
from revision_common import RESULTS, read, write, load_aligned, sha
from diagnose_runtime_and_tight_validation import DEST, select
from pilot_validation_feasibility import linear_excess, ProbeBudgetReached
from compare_joint_workflows import check


def main():
    output=DEST/'joint_tight_probe.json'
    if output.exists(): raise RuntimeError('Preserve existing diagnostic output')
    os.environ.setdefault('NUMBA_CACHE_DIR',str(RESULTS/'numba_cache'))
    import pandapower as pp
    data,samples,ids=select()
    prior=read(DEST/'validation_reachability.json')
    assert prior['complete'] and [r['sample_id'] for r in prior['rows']]==[samples[i]['sample_id'] for i in ids]
    mod=load_aligned(); core=mod.core
    core.RL_VMIN_PU,core.RL_VMAX_PU=.94,1.08
    env=mod.CleanN1Env(samples,ids,data)
    basis=env._build_orthogonal_physics_action_basis(); dim=basis.shape[1]
    protocol={'source_sha256':sha(DEST/'validation_reachability.json'),'script_sha256':sha(__file__),
        'budget':'1800 residual calls / 90 seconds per case; one local start from prior validation-only probe',
        'scope':'No test cases, no training; original 32D/40MW and voltage boxes. Passing endpoint is a witness; unsuccessful search is inconclusive.'}
    payload={'protocol':protocol,'complete':False,'rows':[]}
    write(output,payload)
    for sid,old in zip(ids,prior['rows']):
        sample=samples[sid]; p0=np.asarray(old['initial_dispatch_mw'])
        voltage=old['voltage_probes']['tight']
        lower=np.asarray(voltage['control_lower']); upper=np.asarray(voltage['control_upper'])
        v0=np.r_[voltage['final_generator_vm_pu'],voltage['final_reference_vm_pu']]
        z0=basis.T@(np.asarray(old['final_dispatch_mw'])-p0)
        template=copy.deepcopy(env._ac_net_template)
        scale=sample['demand_mw']/template.load.p_mw.sum()
        template.load.p_mw*=scale; template.load.q_mvar*=scale
        best=[None]; count=[0]; failed=[0]; start=time.perf_counter()
        size=len(template.bus)+2*len(template.line)+len(template.trafo)+2*env.N+3
        def unpack(x):
            z=x[:dim]; z=z*min(1.,39.999/max(np.linalg.norm(z),1e-12))
            return p0+basis@z,x[dim:]
        def residual(x):
            if count[0]>=1800 or time.perf_counter()-start>=90.: raise ProbeBudgetReached()
            count[0]+=1; power,controls=unpack(x)
            trial=copy.deepcopy(template); trial.gen.p_mw=power
            trial.gen.vm_pu=controls[:env.N]; trial.ext_grid.vm_pu=controls[env.N:]
            try:
                pp.runpp(trial,enforce_q_lims=True,calculate_voltage_angles=True,numba=True,max_iteration=50,init='dc')
                vm=trial.res_bus.vm_pu.to_numpy()
                if not np.isfinite(vm).all(): raise pp.LoadflowNotConverged()
            except pp.LoadflowNotConverged:
                failed[0]+=1
                return np.full(size,100.)
            _,_,base,n1=linear_excess(power,data['linear_security_model'])
            gaps=np.maximum(.9401-vm,0)+np.minimum(1.0799-vm,0)
            line=np.maximum(trial.res_line.loading_percent.to_numpy()/100-.9999,0)
            trafo=np.maximum(trial.res_trafo.loading_percent.to_numpy()/100-.9999,0)
            p,q=trial.res_ext_grid[['p_mw','q_mvar']].sum().to_numpy()
            ref=np.array([max(core.REF_SOURCE_P_MIN_MW-p,p-core.REF_SOURCE_P_MAX_MW,0),
                max(core.REF_SOURCE_Q_MIN_MVAR-q,q-core.REF_SOURCE_Q_MAX_MVAR,0),
                max(np.hypot(p,q)-core.REF_SOURCE_S_MAX_MVA,0)])/100
            # Both base and outage excesses contribute to one per-line penalty.
            security=np.maximum(base,np.max(n1,axis=1))
            value=np.r_[100*gaps,line,trafo,security,
                np.maximum(env.PMIN-power,0),np.maximum(power-env.PMAX*.99,0),ref]
            assert len(value)==size
            score=float(value@value)
            if best[0] is None or score<best[0]['score']: best[0]={'x':x.copy(),'score':score}
            return value
        status='completed'
        try:
            result=least_squares(residual,np.r_[z0,v0],
                bounds=(np.r_[np.full(dim,-40.),lower],np.r_[np.full(dim,40.),upper]),
                x_scale=np.r_[np.full(dim,10.),np.full(len(v0),.01)],
                diff_step=2e-4,max_nfev=20,ftol=1e-8,xtol=1e-8,gtol=1e-8)
            status=result.message
        except ProbeBudgetReached: status='budget_reached'
        if best[0] is None: raise RuntimeError('No converged diagnostic point')
        power,controls=unpack(best[0]['x'])
        env.current_sample=sample; env.target_demand=sample['demand_mw']
        env.current_power=power; env.current_gen_vm_pu=controls[:env.N]; env.current_ext_grid_vm_pu=controls[env.N:]
        env._aligned_n1_cache_key=None
        passed,info=check(env,mod,lower,upper)
        row={'sample_id':sample['sample_id'],'passed':passed,'status':str(status),'pf_calls':count[0],
            'pf_failures':failed[0],'runtime_sec':time.perf_counter()-start,'best_score':best[0]['score'],
            'norm_mw':float(np.linalg.norm(power-p0)),
            'offspan_mw':float(np.linalg.norm(power-p0-basis@(basis.T@(power-p0)))),
            'final_dispatch_mw':power.tolist(),'final_generator_vm_pu':controls[:env.N].tolist(),
            'final_reference_vm_pu':controls[env.N:].tolist(),'final_verification':info}
        payload['rows'].append(row); write(output,payload)
        print(sample['sample_id'],passed,info['v_min'],info['v_max'],info['n1_violations'],flush=True)
    payload['complete']=True; write(output,payload)


if __name__=='__main__': main()

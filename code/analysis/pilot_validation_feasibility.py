"""Validation pilot: minimum secure redispatch followed by bounded AC repair.

This diagnoses reachability, not trained-policy performance or full AC N-1.
It never uses test IDs or stored dispatch/voltage rescue labels.
"""
import argparse
import copy
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import least_squares

from revision_common import CANDIDATES, CANDIDATE_SPLIT, GAT, RESULTS, REV, load_aligned, read, sha, write


def linear_excess(dispatch, model):
    s = np.asarray(model['S'])
    valid = np.asarray(model['valid_contingency_indices'], dtype=int)
    lodf = np.asarray(model['lodf'])
    limits = np.asarray(model['effective_limit'])
    base = np.asarray(model['flow_intercept']) + s @ dispatch
    post = base[:, None] + lodf[:, valid] * base[valid][None, :]
    post[valid, np.arange(len(valid))] = 0
    return base, post, np.maximum(abs(base)-limits, 0), np.maximum(abs(post)-limits[:, None], 0)


def minimum_linear_dispatch(p0, lower, upper, basis, model, timeout=45., cut_batch=512, max_rounds=30,
                            radius_center=None, radius=None):
    """Convex cutting-plane QP; scan all constraints before declaring success."""
    import gurobipy as gp
    from gurobipy import GRB
    started = time.perf_counter()
    s = np.asarray(model['S'])
    intercept = np.asarray(model['flow_intercept'])
    lodf = np.asarray(model['lodf'])
    valid = np.asarray(model['valid_contingency_indices'], dtype=int)
    limits = np.asarray(model['effective_limit'])
    assert np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-9)
    assert np.allclose(basis.sum(axis=0), 0, atol=1e-9)
    current = np.asarray(p0).copy()
    seen = set()
    status = 'round_limit'
    with gp.Model('validation_minimum_secure_redispatch') as qp:
        qp.Params.OutputFlag = 0
        qp.Params.Threads = 2
        qp.Params.FeasibilityTol = 1e-8
        qp.Params.OptimalityTol = 1e-8
        qp.Params.DualReductions = 0
        z = qp.addMVar(basis.shape[1], lb=-GRB.INFINITY, name='coordinates')
        power = p0 + basis @ z
        qp.addConstr(power >= lower)
        qp.addConstr(power <= upper)
        if radius is not None:
            if radius_center is None or radius <= 0:
                raise ValueError('A positive radius and its dispatch center are required')
            delta = power-np.asarray(radius_center)
            qp.addConstr(delta @ delta <= radius**2)
        qp.setObjective(z @ z, GRB.MINIMIZE)
        qp_status = None
        for iteration in range(max_rounds):
            base, post, b_excess, n_excess = linear_excess(current, model)
            candidates = []
            for line in np.flatnonzero(b_excess > 1e-5):
                candidates.append((b_excess[line], -1, int(line), 1 if base[line] > 0 else -1))
            ii, jj = np.where(n_excess > 1e-5)
            for line, col in zip(ii, jj):
                candidates.append((n_excess[line, col], int(valid[col]), int(line), 1 if post[line, col] > 0 else -1))
            within_radius = radius is None or np.linalg.norm(current-radius_center) <= radius+1e-7
            if not candidates and within_radius:
                status = 'optimal_secure' if qp_status == GRB.OPTIMAL else 'initial_secure'
                break
            fresh = [c for c in sorted(candidates, reverse=True) if c[1:] not in seen][:cut_batch]
            if not fresh and within_radius:
                status = 'numerical_residual'
                break
            for _, outage, line, sign in fresh:
                gradient = s[line].copy()
                constant = intercept[line]
                if outage >= 0:
                    gradient += lodf[line, outage] * s[outage]
                    constant += lodf[line, outage] * intercept[outage]
                qp.addConstr(sign * (constant + gradient @ power) <= limits[line])
                seen.add((outage, line, sign))
            remaining = timeout-(time.perf_counter()-started)
            if remaining <= 0:
                status = 'time_limit'
                break
            qp.Params.TimeLimit = remaining
            qp.optimize()
            qp_status = int(qp.Status)
            if qp_status != GRB.OPTIMAL:
                status = 'infeasible_in_linear_span_and_box' if qp_status == GRB.INFEASIBLE else f'qp_status_{qp_status}'
                break
            current = p0 + basis @ np.asarray(z.X)
        else:
            # The last solved QP may already satisfy all omitted constraints.
            _, _, b_excess, n_excess = linear_excess(current, model)
            if (max(b_excess.max(initial=0), n_excess.max(initial=0)) <= 1e-5
                    and (radius is None or np.linalg.norm(current-radius_center) <= radius+1e-7)):
                status = 'optimal_secure'
    secure = status in ('initial_secure', 'optimal_secure')
    return (current if secure else None), {
        'status': status, 'qp_status': qp_status, 'rounds': iteration+1, 'cuts': len(seen),
        'runtime_sec': time.perf_counter()-started,
        'minimum_delta_l2_mw': float(np.linalg.norm(current-p0)) if secure else None,
    }


class ProbeBudgetReached(Exception):
    pass


def bounded_voltage_repair(env, mod, dispatch, demand, seconds=45., calls=600, initial_controls=None):
    """Use SciPy within the deployed voltage box; recheck with original verifier."""
    import pandapower as pp
    core = mod.core
    template = copy.deepcopy(env._ac_net_template)
    template.gen.p_mw = dispatch.copy()
    scale = demand / float(template.load.p_mw.sum())
    template.load.p_mw *= scale
    template.load.q_mvar *= scale
    base = np.r_[np.clip(template.gen.vm_pu.to_numpy(), core.GEN_VM_MIN, core.GEN_VM_MAX),
                 np.clip(template.ext_grid.vm_pu.to_numpy(), core.EXT_VM_MIN, core.EXT_VM_MAX)]
    lower = np.maximum(base-mod.ARGS.voltage_target_range_pu, np.r_[np.full(env.N, core.GEN_VM_MIN), core.EXT_VM_MIN])
    upper = np.minimum(base+mod.ARGS.voltage_target_range_pu, np.r_[np.full(env.N, core.GEN_VM_MAX), core.EXT_VM_MAX])
    started = time.perf_counter()
    count = 0
    best = None
    failures = 0
    residual_size = len(template.bus)+len(template.line)+len(template.trafo)+3

    def residual(control):
        nonlocal count, best, failures
        if count >= calls or time.perf_counter()-started > seconds:
            raise ProbeBudgetReached()
        count += 1
        trial = copy.deepcopy(template)
        trial.gen.vm_pu = control[:env.N]
        trial.ext_grid.vm_pu = control[env.N:]
        try:
            pp.runpp(trial, calculate_voltage_angles=True, enforce_q_lims=True,
                     numba=True, max_iteration=50, init='dc')
            vm = trial.res_bus.vm_pu.to_numpy()
            if not np.isfinite(vm).all():
                raise pp.LoadflowNotConverged('Nonfinite bus voltages')
        except pp.LoadflowNotConverged:
            failures += 1
            return np.full(residual_size, 100.)
        # Small interior targets prevent numerical boundary equality failures.
        gap = np.maximum(core.RL_VMIN_PU+1e-4-vm, 0) + np.minimum(core.RL_VMAX_PU-1e-4-vm, 0)
        line = np.maximum(trial.res_line.loading_percent.to_numpy()/100-0.9999, 0)
        trafo = np.maximum(trial.res_trafo.loading_percent.to_numpy()/100-0.9999, 0)
        p, q = trial.res_ext_grid[['p_mw', 'q_mvar']].sum().to_numpy()
        ref = [max(core.REF_SOURCE_P_MIN_MW-p, p-core.REF_SOURCE_P_MAX_MW, 0)/100,
               max(core.REF_SOURCE_Q_MIN_MVAR-q, q-core.REF_SOURCE_Q_MAX_MVAR, 0)/100,
               max(np.hypot(p,q)-core.REF_SOURCE_S_MAX_MVA, 0)/100]
        value = np.r_[100*gap, line, trafo, ref]
        score = float(np.dot(value, value))
        if best is None or score < best['score']:
            best = {'control': control.copy(), 'score': score,
                    'min_bus_vm': float(vm.min()), 'max_bus_vm': float(vm.max())}
        return value

    optimizer_status = 'initial_schedule'
    try:
        if initial_controls is not None:
            assert np.shape(initial_controls)==np.shape(base)
            assert np.all(initial_controls >= lower-1e-8) and np.all(initial_controls <= upper+1e-8)
            residual(np.clip(initial_controls,lower,upper))
        for fraction in (0., 0.5, 1., -0.5, -1.):
            if best is not None and best['score'] <= 1e-16:
                break
            control = base+fraction*(upper-base) if fraction >= 0 else base+(-fraction)*(lower-base)
            residual(control)
        if best is not None and best['score'] > 1e-16:
            result = least_squares(residual, best['control'], bounds=(lower, upper),
                                   diff_step=2e-4, max_nfev=12, ftol=1e-7, xtol=1e-7, gtol=1e-7)
            optimizer_status = str(result.message)
    except ProbeBudgetReached:
        optimizer_status = 'budget_reached_best_retained'
    control = base if best is None else best['control']
    assert np.all(control >= lower-1e-10) and np.all(control <= upper+1e-10)
    env.current_power = dispatch.copy()
    env.target_demand = demand
    env.current_gen_vm_pu = control[:env.N].copy()
    env.current_ext_grid_vm_pu = control[env.N:].copy()
    info = dict(env._run_ac_pf_check())
    state = env._vectorized_n1_state()
    info.update(n1_overload_mw=float(state['total_overload_mw']), n1_violations=int(state['violation_count']))
    return {
        'passed': bool(core.adopted_operation_pass(info)), 'runtime_sec': time.perf_counter()-started,
        'pf_calls': count, 'pf_failures': failures, 'optimizer_status': optimizer_status,
        'best_residual_squared': None if best is None else best['score'],
        'control_lower': lower.tolist(), 'control_upper': upper.tolist(),
        'final_generator_vm_pu': control[:env.N].tolist(), 'final_reference_vm_pu': control[env.N:].tolist(),
        'final_verification': info,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-demand', type=int, default=1)
    parser.add_argument('--ac-seconds', type=float, default=45)
    parser.add_argument('--warm-start-result', help='Frozen-policy validation records; never a test result.')
    parser.add_argument('--output', default=str(RESULTS/'optimization_diagnostics/validation_feasibility_pilot.json'))
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error('Output already exists; use another path to preserve the experiment')
    os.environ.setdefault('NUMBA_CACHE_DIR', str(RESULTS/'numba_cache'))
    data = read(CANDIDATES)
    samples = [s for s in data['samples'] if s.get('is_complete_sced', True)]
    split = read(CANDIDATE_SPLIT)
    ids = []
    for demand in (16400., 16600., 16800.):
        choices = sorted((i for i in split['val_ids'] if samples[i]['demand_mw']==demand),
                         key=lambda i:samples[i]['sample_id'])
        ids.extend(choices[:args.per_demand])
    assert ids and not set(ids).intersection(split['test_ids']+split['train_ids'])
    warm = {}
    if args.warm_start_result:
        saved = read(args.warm_start_result)
        assert saved['complete'] and saved['settings']['scope']=='val'
        warm = {r['sample_id']:r for r in saved['rows']}
        assert set(warm)=={samples[i]['sample_id'] for i in ids}
    mod = load_aligned()
    env = mod.CleanN1Env(samples, ids, data)
    bases = {'physics32': env._build_orthogonal_physics_action_basis(),
             'full_balanced67': null_space(np.ones((1,env.N)))}
    payload = {
        'complete': False, 'scope': 'upstream validation only', 'settings': vars(args),
        'selection_rule': 'first sample IDs per demand, fixed before solving',
        'sample_ids': [samples[i]['sample_id'] for i in ids],
        'input_sha256': {p.name:sha(p) for p in (CANDIDATES,CANDIDATE_SPLIT,GAT)},
        'backup_commit': subprocess.check_output(
            ['git','-c','safe.directory='+(REV/'original_git_snapshot').as_posix(),
             '-C',str(REV/'original_git_snapshot'),'rev-parse','HEAD'],text=True).strip(),
        'interpretation': 'Optimization reachability diagnostic, not RL performance. '
            'Failure of bounded local AC repair is not a proof of AC infeasibility. '
            'Acceptance retains base-case AC and linear-contingency checks; it is not full AC N-1.',
        'rows': [],
    }
    if args.warm_start_result:
        payload['warm_start_sha256']=sha(args.warm_start_result)
    for sid in ids:
        sample = samples[sid]
        env.current_sample = sample
        demand = float(sample['demand_mw'])
        p0 = mod.core.project_box_sum(env._predict_gat(demand),env.PMIN,env.PMAX*0.99,demand)
        warm_row = warm.get(sample['sample_id'])
        solver_start = p0 if warm_row is None else np.asarray(warm_row['final_dispatch_mw'])
        warm_controls = None if warm_row is None else np.r_[warm_row['final_generator_vm_pu'],warm_row['final_reference_vm_pu']]
        for family, basis in bases.items():
            dispatch, qp = minimum_linear_dispatch(solver_start,env.PMIN,env.PMAX*0.99,basis,data['linear_security_model'])
            qp['objective_origin']='projected_gat' if warm_row is None else 'policy_terminal'
            row = {'sample_id':sample['sample_id'], 'candidate_index':sid, 'demand_mw':demand,
                   'family':family, 'initial_dispatch_mw':p0.tolist(), 'linear':qp}
            if dispatch is not None:
                radius = float(np.linalg.norm(dispatch-p0))
                row['gat_relative_delta_l2_mw'] = radius
                row['within_nominal_radius'] = {str(r): radius <= r for r in (40,200,400,600)}
                row['final_dispatch_mw'] = dispatch.tolist()
                if radius <= 600:
                    row['ac'] = bounded_voltage_repair(env,mod,dispatch,demand,seconds=args.ac_seconds,
                                                      initial_controls=warm_controls)
                else:
                    row['ac'] = {'passed':False, 'not_run':'minimum_linear_norm_above_600_mw'}
            row['passed'] = bool(row.get('ac',{}).get('passed',False))
            payload['rows'].append(row)
            write(args.output,payload)
            print(f"{sample['sample_id']} demand={demand:g} {family}: {qp['status']}, "
                  f"L2={qp['minimum_delta_l2_mw']} AC-pass={row['passed']}",flush=True)
    payload['complete'] = True
    write(args.output,payload)


if __name__ == '__main__':
    main()

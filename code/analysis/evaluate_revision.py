"""Unfiltered paired evaluation, projection audit, initialization and load stress."""
import argparse
import copy
import shutil
import time
from collections import Counter
import numpy as np
import torch
from revision_common import *
from constraint_diagnostics import operation_flags

def verify(env, mod):
    info = dict(env._run_ac_pf_check())
    state = env._vectorized_n1_state()
    info.update(n1_overload_mw=float(state['total_overload_mw']), n1_violations=int(state['violation_count']))
    return bool(mod.core.adopted_operation_pass(info)), info

def cost(sample, power):
    return float(np.sum(np.asarray(sample['sample_cp2'])*power**2 + np.asarray(sample['sample_cp1'])*power + np.asarray(sample['sample_cp0'])))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=list(CHECKPOINTS), default='sac')
    p.add_argument('--checkpoint')
    p.add_argument('--family', choices=['physics','random','modes','keys','full68'], default='physics')
    p.add_argument('--dimension', type=int, default=32)
    p.add_argument('--shifts', type=int, default=12)
    p.add_argument('--scope', choices=['test','val'], default='test')
    p.add_argument('--max-samples', type=int, default=0)
    p.add_argument('--initialization', choices=['gat','label','economic','previous'], default='gat')
    p.add_argument('--stress', choices=['none','regional','reactive','renewable','demand-low','demand-high'], default='none')
    p.add_argument('--vmin', type=float, default=.933)
    p.add_argument('--vmax', type=float, default=1.09)
    p.add_argument('--radius', type=float, default=40.)
    p.add_argument('--ac-every', type=int, default=5)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    checkpoint = Path(args.checkpoint) if args.checkpoint else CHECKPOINTS[args.method]
    checkpoint_hash = sha(checkpoint)
    data = read(CANDIDATES)
    samples = [s for s in data['samples'] if s.get('is_complete_sced', True)]
    ids = read(CANDIDATE_SPLIT)[args.scope+'_ids']
    ids = ids[:args.max_samples] if args.max_samples else ids
    input_hashes = {key: sha(path) for key, path in (
        ('candidates', CANDIDATES), ('split', CANDIDATE_SPLIT),
        ('screened_dataset', DATASET), ('gat', GAT))}
    rows, diagnostics = [], []
    if Path(args.output).exists():
        saved = read(args.output)
        if needs_sample_voltage_cache(vars(args)) and saved.get('voltage_cache_contract') != 'sample_context_v2':
            p.error('Existing output used demand-only caches; preserve it and use a new output path')
        if args.initialization == 'previous' and saved.get('previous_initialization_contract') != 'last_accepted_or_gat':
            p.error('Existing output used rejected terminal dispatches; use a new output path')
        rows = resume_rows(saved, vars(args), checkpoint_hash, [samples[sid]['sample_id'] for sid in ids])
        if 'input_sha256' in saved and saved['input_sha256'] != input_hashes:
            p.error('Existing result inputs differ; use a new output path')
        diagnostics = saved.get('nonfinite_fields', [])
        if len(rows) == len(ids):
            print(f'Already evaluated {len(rows)}/{len(ids)} samples; preserved.', flush=True)
            return
        backup = Path(args.output).parent / 'resume_backups' / f'{Path(args.output).stem}_{time.time_ns()}.json'
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.output, backup)
        print(f'Resuming {len(rows)}/{len(ids)} saved samples; backup: {backup}', flush=True)
        if 'input_sha256' not in saved:
            print('Legacy partial result: input hashes were not recorded; settings, checkpoint and sample order verified.', flush=True)
    def save():
        write(args.output, {'complete':len(rows)==len(ids), 'settings':vars(args),
              'checkpoint_sha256':checkpoint_hash, 'input_sha256':input_hashes,
              'voltage_cache_contract':'sample_context_v2',
              'previous_initialization_contract':'last_accepted_or_gat',
              'nonfinite_fields':diagnostics, 'rows':rows})
    mod = load_aligned(['--active-target-range-mw', args.radius, '--ac-pf-check-every', args.ac_every])
    family = 'full68' if args.method == 'full68' else args.family
    Env = configure_env(mod, family, args.dimension, args.shifts)
    mod.core.RL_VMIN_PU, mod.core.RL_VMAX_PU = args.vmin, args.vmax
    retained = {s['sample_id'] for s in read(DATASET)['samples']}
    env = Env(samples, ids, data)
    template = copy.deepcopy(env._ac_net_template)
    original_predict = env._predict_gat
    original_intercept = env.dcnet.flow_intercept.copy()
    obs = env.reset(ids[0])
    pdim = env._n_shift_dirs + env._n_key_gens
    mod.core.SAC_LOG_STD_MAX_VECTOR = np.concatenate([np.full(pdim, mod.ARGS.active_log_std_max), np.full(env.action_dim-pdim, mod.ARGS.voltage_log_std_max)]).astype(np.float32)
    if args.method in ('ddpg','td3'):
        from train_deterministic_n1_aligned import DeterministicActor
        actor = DeterministicActor(len(obs), pdim, env.action_dim-pdim).to(mod.core.DEVICE)
    else:
        actor = mod.BranchedGaussianActor(len(obs), pdim, env.action_dim-pdim).to(mod.core.DEVICE)
    actor.load_state_dict(torch.load(checkpoint, map_location=mod.core.DEVICE, weights_only=True))
    actor.eval()
    def sync():
        if torch.cuda.is_available(): torch.cuda.synchronize()
    previous = None
    previous_id = None
    if args.initialization == 'previous':
        for row in reversed(rows):
            if row.get('passed') and 'final_dispatch_mw' in row:
                previous = np.asarray(row['final_dispatch_mw'], dtype=float)
                previous_id = row['sample_id']
                if not np.isfinite(previous).all():
                    raise ValueError('Cannot resume previous initialization from a non-finite dispatch')
                break
    for sid in ids[len(rows):]:
        sample = samples[sid]
        env.current_sample = sample
        scenario_started = time.perf_counter()
        env._ac_net_template = copy.deepcopy(template)
        env.dcnet.flow_intercept = original_intercept.copy()
        original_demand = sample['demand_mw']
        net = env._ac_net_template
        if args.stress == 'regional':
            # Deterministic electrical region: graph partition, not bus-number distance.
            import networkx as nx
            import pandapower.topology as top
            graph = top.create_nxgraph(net, include_trafos=True)
            region = nx.single_source_shortest_path_length(graph, int(net.load.bus.iloc[0]), cutoff=3)
            weights = np.where(net.load.bus.isin(region), 1.10, .90)
            net.load.p_mw *= weights
            net.load.q_mvar *= weights
        elif args.stress == 'reactive':
            net.load.q_mvar *= 1.10
        elif args.stress == 'renewable':
            # Net-injection forecast error, without claiming a detailed RES model.
            rng = np.random.default_rng(20261014 + int(sid))
            weights = rng.uniform(.9, 1.1, len(net.load))
            net.load.p_mw *= weights
        elif args.stress in ('demand-low','demand-high'):
            sample['demand_mw'] = float(original_demand) * (.97 if args.stress == 'demand-low' else 1.03)
        env._voltage_jacobian_memory = {}
        env._voltage_jacobian_dir = voltage_cache_directory(vars(args), sid, input_hashes, checkpoint_hash)
        env._voltage_jacobian_dir.mkdir(parents=True, exist_ok=True)
        if args.stress != 'none':
            # Re-anchor the affine flow intercept to the actual perturbed demand.
            # Keep S, LODF and the learned basis fixed; audit their remaining errors.
            import pandapower as pp
            anchor = copy.deepcopy(net)
            scale = float(sample['demand_mw']) / float(anchor.load.p_mw.sum())
            anchor.load.p_mw *= scale
            anchor.load.q_mvar *= scale
            anchor_power = mod.core.project_box_sum(original_predict(float(sample['demand_mw'])), env.PMIN, env.PMAX*.99, float(sample['demand_mw']))
            anchor.gen.p_mw = anchor_power
            try:
                pp.runpp(anchor, enforce_q_lims=True, calculate_voltage_angles=True, numba=False,max_iteration=100)
            except pp.LoadflowNotConverged as exc:
                rows.append({'sample_id':sample['sample_id'],'candidate_index':sid,'screened':sample['sample_id'] in retained,'demand_mw':float(sample['demand_mw']),'passed':False,'reason':'stress_anchor_nonconvergence','error':str(exc),'steps':0,'runtime_sec':time.perf_counter()-scenario_started})
                sample['demand_mw']=original_demand
                save()
                continue
            env.dcnet.flow_intercept = anchor.res_line.p_from_mw.to_numpy() - env.dcnet.S @ anchor_power
        scenario_preparation_sec = time.perf_counter()-scenario_started
        preparation_started = time.perf_counter()
        if args.initialization == 'label':
            env._predict_gat = lambda demand: np.asarray(sample['dispatch_68']).copy()
        elif args.initialization == 'economic':
            from scipy.optimize import minimize
            lo, hi = env.PMIN, env.PMAX*.99
            start = mod.core.project_box_sum(original_predict(float(sample['demand_mw'])), lo, hi, float(sample['demand_mw']))
            solved = minimize(lambda x: cost(sample,x), start, bounds=list(zip(lo,hi)), constraints={'type':'eq','fun':lambda x: x.sum()-sample['demand_mw']}, method='SLSQP', options={'maxiter':500,'ftol':1e-8})
            if not solved.success: raise RuntimeError(solved.message)
            env._predict_gat = lambda demand: solved.x.copy()
        elif args.initialization == 'previous' and previous is not None:
            env._predict_gat = lambda demand: previous.copy()
        else:
            env._predict_gat = original_predict
        preparation_sec = time.perf_counter()-preparation_started
        sync()
        started = time.perf_counter()
        obs = env.reset(sid)
        initial = env.current_power.copy()
        initial_pass, initial_info = verify(env, mod)
        trace, total_reward = [], 0.
        basis = env._build_orthogonal_physics_action_basis()
        projector = basis @ np.linalg.pinv(basis)
        if not initial_pass:
            for step in range(1, env.max_steps+1):
                state = torch.as_tensor(obs, dtype=torch.float32, device=mod.core.DEVICE).unsqueeze(0)
                with torch.no_grad():
                    raw = actor(state) if args.method in ('ddpg','td3') else actor.select_action(state, deterministic=True)
                raw = raw.cpu().numpy()[0]
                lo, hi = env.get_action_bounds()
                proposed = basis @ raw[:pdim] * args.radius
                if family == 'full68': proposed *= env.N/(env.N-1)
                obs, reward, done, truncated, info = env.step(lo+(raw+1)*.5*(hi-lo))
                executed = env.current_power - env.gat_power
                offspan = executed - projector @ executed
                total_reward += float(reward)
                trace.append({'step':step, 'n1_overload_mw':float(info.get('n1_overload_mw',0)), 'voltage_gap_pu':float(info.get('v_violation_mag_pu',0)), 'ac_checked':bool(info.get('ac_checked',False)), 'proposed_l2_mw':float(np.linalg.norm(proposed)), 'executed_l2_mw':float(np.linalg.norm(executed)), 'projection_change_l2_mw':float(np.linalg.norm(executed-proposed)), 'offspan_l2_mw':float(np.linalg.norm(offspan)), 'offspan_relative':float(np.linalg.norm(offspan)/max(np.linalg.norm(executed),1e-12))})
                if done or truncated: break
        passed, final = verify(env, mod)
        sync()
        elapsed = time.perf_counter()-started
        row = {'sample_id':sample['sample_id'],'candidate_index':sid,'screened':sample['sample_id'] in retained,'demand_mw':float(sample['demand_mw']), 'passed':passed, 'initial_pass':initial_pass, 'steps':len(trace), 'episode_return':total_reward, 'runtime_sec':elapsed, 'scheduled_cost':cost(sample,env.current_power), 'reference_p_mw':float(final.get('reference_source_p_mw',0)), 'reference_q_mvar':float(final.get('reference_source_q_mvar',0)), 'reason':'pass' if passed else ('ac_nonconvergence' if not final.get('converged') else ('voltage_violation' if final.get('v_violations',0) else ('n1_violation' if final['n1_violations'] else 'other_operating_limit'))), 'final_dispatch_mw':env.current_power.tolist(), 'final_generator_vm_pu':env.current_gen_vm_pu.tolist(), 'final_reference_vm_pu':float(env.current_ext_grid_vm_pu[0]), 'trace':trace}
        row['final_verification'] = final
        row['constraint_flags'] = operation_flags(final)
        row['previous_accepted_sample_id'] = previous_id if args.initialization == 'previous' else None
        row['initialization_used'] = 'gat' if args.initialization == 'previous' and previous is None else args.initialization
        if passed:
            previous = env.current_power.copy()
            previous_id = sample['sample_id']
        rows.append(row)
        row['initialization_preparation_sec'] = preparation_sec
        row['runtime_with_measured_preparation_sec'] = elapsed + preparation_sec
        row['scenario_preparation_sec'] = scenario_preparation_sec
        row['stress_end_to_end_sec'] = elapsed + preparation_sec + scenario_preparation_sec
        row['stored_label_solver_time_included'] = False
        sample['demand_mw'] = original_demand
        save()
        print(f"{len(rows)}/{len(ids)} {row['sample_id']} {row['reason']} steps={len(trace)}", flush=True)

if __name__ == '__main__': main()

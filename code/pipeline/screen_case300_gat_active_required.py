#!/usr/bin/env python3
"""Build an IEEE-300 RL dataset that requires active-power correction.

The screen starts from the deployed GAT warm start, not from the stored SCED
label or rescue dispatch. A sample is retained only when:

1. fixed-P operation cannot pass the dataset's linear base/N-1 constraints and
   common strict AC verifier; and
2. a nearby P redispatch in the deployed physics-guided action span, followed
   by the same voltage optimization, does pass both checks.

Stored rescue fields are never read when constructing candidates. They are
written only as dataset-curation evidence and are not RL observations/labels.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
PROJ_DIR = CODE_DIR.parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(1, str(PROJ_DIR))

from strict_sced_utils import project_box_sum, zero_sum_unit


_VOLTAGE_JACOBIAN_CACHE = {}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Candidate linear-SCED dataset.")
    ap.add_argument("--gat-checkpoint", required=True)
    ap.add_argument("--split-file", required=True,
                    help="Candidate-pool split used to train the supplied GAT.")
    ap.add_argument("--output-dataset", required=True)
    ap.add_argument("--output-split", required=True)
    ap.add_argument("--output-report", required=True)
    ap.add_argument("--attempts", type=int, default=80)
    ap.add_argument("--step-mw", type=float, default=40.0)
    ap.add_argument("--max-delta-mw", type=float, default=600.0)
    ap.add_argument("--min-delta-l1-mw", type=float, default=40.0)
    ap.add_argument("--projection-margin", type=float, default=0.995,
                    help="Target fraction of stored N-1 limits in the span projection.")
    ap.add_argument("--projection-iters", type=int, default=12)
    ap.add_argument("--voltage-iters", type=int, default=35)
    ap.add_argument("--vmin", type=float, default=0.933)
    ap.add_argument("--vmax", type=float, default=1.09)
    ap.add_argument("--seed", type=int, default=20260721)
    ap.add_argument(
        "--action-encoding",
        default="physics32fixed",
        choices=["physics32", "physics32fixed", "physics32orth", "full68"],
        help=(
            "Active-power basis used for both reachability screening and the "
            "resulting RL task. The fixed basis gives stationary action semantics."
        ),
    )
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--allow-empty-partitions", action="store_true",
                    help="Permit partial probe runs that do not cover every split.")
    ap.add_argument("--progress-every", type=int, default=5)
    return ap.parse_args()


def add_unique(directions, vec, cosine_limit=0.92):
    unit = zero_sum_unit(np.asarray(vec, dtype=np.float64))
    if unit is None:
        return
    if any(abs(float(np.dot(unit, old))) > cosine_limit for old in directions):
        return
    directions.append(unit)


def deployed_active_directions(env):
    """Return directions representable by the deployed active-action basis."""
    env._compute_shift_directions()
    env._select_key_generators()
    directions = []
    for direction in np.asarray(env._shift_directions, dtype=np.float64):
        add_unique(directions, direction)
        add_unique(directions, -direction)
    for gen_idx in np.asarray(env._key_gen_indices, dtype=int):
        direction = np.full(env.N, -1.0 / max(env.N - 1, 1), dtype=np.float64)
        direction[int(gen_idx)] = 1.0
        add_unique(directions, direction)
        add_unique(directions, -direction)
    return directions


def make_candidate(p0, lower, upper, demand, directions, rng, attempt,
                   step_mw, max_delta_mw):
    n_dir = len(directions)
    if n_dir == 0:
        raise RuntimeError("No deployed active-power directions were constructed")
    if attempt < 3 * n_dir:
        direction = directions[attempt % n_dir]
        magnitude = step_mw * (1.0 + attempt // n_dir)
    else:
        weights = rng.normal(size=n_dir)
        keep = np.argsort(np.abs(weights))[-min(4, n_dir):]
        mixed = sum(weights[j] * directions[j] for j in keep)
        direction = zero_sum_unit(mixed)
        if direction is None:
            direction = directions[attempt % n_dir]
        magnitude = min(max_delta_mw, step_mw * rng.gamma(2.0, 2.0))
    delta = np.clip(magnitude * direction, -max_delta_mw, max_delta_mw)
    return project_box_sum(p0 + delta, lower, upper, demand)


def solve_n1_in_action_span(p0, lower, upper, directions, dcnet,
                            max_delta_mw, limit_margin, max_iters):
    """Find a linearly secure point using only deployed active directions.

    This cutting-plane QP reads network constraints and the GAT start only. It
    never reads the SCED label or any stored rescue dispatch.
    """
    import gurobipy as gp
    from gurobipy import GRB

    basis = np.asarray(directions, dtype=np.float64).T  # (n_gen, n_action)
    cuts = []
    seen = set()
    current = np.asarray(p0, dtype=np.float64).copy()
    last_info = check_n1_security(current, dcnet)
    for iteration in range(max_iters):
        for cut in last_info.get("qp_constraints", []):
            key = tuple(cut[:4])
            if key not in seen:
                seen.add(key)
                cuts.append(cut)
        if not cuts and int(last_info.get("n_violations", 0)) == 0:
            return current, {
                "status": "already_secure", "iterations": iteration,
                "cuts": 0, "linear_n1": last_info,
            }

        model = gp.Model("gat_action_span_n1_projection")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = 15.0
        z = model.addMVar(
            shape=basis.shape[1], lb=-float(max_delta_mw),
            ub=float(max_delta_mw), name="action_coeff",
        )
        p_expr = np.asarray(p0, dtype=np.float64) + basis @ z
        delta_expr = basis @ z
        model.addConstr(p_expr >= lower)
        model.addConstr(p_expr <= upper)
        # Match the environment's per-generator cumulative residual clip.
        # Bounding the basis coefficients alone is insufficient because
        # several physics directions can add on the same generator.
        model.addConstr(delta_expr >= -float(max_delta_mw))
        model.addConstr(delta_expr <= float(max_delta_mw))

        for kind, k, line, sign, _stored_limit in cuts:
            line = int(line)
            sign = float(sign)
            if kind == "base":
                flow = float(dcnet.flow_intercept[line]) + dcnet.S[line, :] @ p_expr
            else:
                k = int(k)
                flow_l = float(dcnet.flow_intercept[line]) + dcnet.S[line, :] @ p_expr
                flow_k = float(dcnet.flow_intercept[k]) + dcnet.S[k, :] @ p_expr
                flow = flow_l + float(dcnet.lodf[line, k]) * flow_k
            model.addConstr(
                sign * flow <= float(dcnet.effective_limit[line]) * float(limit_margin)
            )

        model.setObjective(delta_expr @ delta_expr + 1e-6 * (z @ z), GRB.MINIMIZE)
        model.optimize()
        if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            return None, {
                "status": f"qp_status_{model.Status}",
                "iterations": iteration + 1, "cuts": len(cuts),
                "linear_n1": last_info,
            }
        current = np.asarray(p0, dtype=np.float64) + basis @ np.asarray(z.X)
        last_info = check_n1_security(current, dcnet)
        if int(last_info.get("n_violations", 999999)) == 0:
            return current, {
                "status": "secure", "iterations": iteration + 1,
                "cuts": len(cuts), "linear_n1": last_info,
            }
    return None, {
        "status": "max_projection_iterations", "iterations": max_iters,
        "cuts": len(cuts), "linear_n1": last_info,
    }


def strict_voltage_result(verifier, net, dispatch, demand, lower, upper,
                          vmin, vmax, voltage_iters):
    gen_vm = np.asarray(net.gen.vm_pu.values, dtype=np.float64).copy()
    ext_vm = np.asarray(net.ext_grid.vm_pu.values, dtype=np.float64).copy()
    optimizer_result = {"status": "replaced_by_ac_voltage_sensitivity"}
    report = verifier.verify_dispatch(
        dispatch, demand, check_base_ac=True,
        check_all_ac_contingencies=False,
        gen_vm_pu=gen_vm, ext_grid_vm_pu=ext_vm,
    )
    if report.passed:
        return True, gen_vm, ext_vm, {
            "strategy": "iterative_local_voltage",
            "optimizer": optimizer_result,
            "common_verifier": report.to_dict(),
        }

    # Use the full deployed voltage-control space before broad scalar sweeps.
    # This normally resolves remote load-bus violations in one Jacobian round
    # and avoids dozens of redundant complete-verifier calls.
    direct_passed, direct_gen, direct_ext, direct_audit = (
        optimize_voltage_sensitivity(
            verifier, dispatch, demand, gen_vm, ext_vm, vmin, vmax,
        )
    )
    if direct_passed:
        return True, direct_gen, direct_ext, {
            "strategy": "iterative_local_then_ac_voltage_sensitivity",
            "optimizer": optimizer_result,
            "common_verifier": direct_audit["common_verifier"],
            "voltage_sensitivity": direct_audit,
        }
    # The local optimizer only reacts when a generator bus itself violates a
    # voltage bound. Remote load-bus undervoltage can therefore remain even
    # though coordinated generator/reference setpoint movement is feasible.
    # Sweep coordinated controls within exactly the deployed [0.92, 1.08]
    # range; no stored rescue voltage is consulted.
    base_gen = np.asarray(gen_vm, dtype=np.float64)
    base_ext = np.asarray(ext_vm, dtype=np.float64)
    offsets = [0.005 * i for i in range(1, 17)] + [-0.005 * i for i in range(1, 17)]
    schedules = []
    for offset in offsets:
        schedules.append(("global", base_gen + offset, base_ext + offset))
    for offset in offsets:
        schedules.append(("generator_only", base_gen + offset, base_ext))
    for offset in offsets:
        schedules.append(("reference_only", base_gen, base_ext + offset))

    best_report = report
    best_gen = base_gen
    best_ext = base_ext
    best_strategy = "iterative_local_voltage"
    best_score = float("inf")
    for strategy, gen_candidate, ext_candidate in schedules:
        gen_candidate = np.clip(gen_candidate, 0.92, 1.08)
        ext_candidate = np.clip(ext_candidate, 0.92, 1.08)
        candidate_report = verifier.verify_dispatch(
            dispatch, demand, check_base_ac=True,
            check_all_ac_contingencies=False,
            gen_vm_pu=gen_candidate, ext_grid_vm_pu=ext_candidate,
        )
        metrics = candidate_report.metrics
        v_lo = max(0.0, vmin - float(metrics.get("base_v_min", -1.0)))
        v_hi = max(0.0, float(metrics.get("base_v_max", 2.0)) - vmax)
        score = (
            1000.0 * float(metrics.get("base_v_violations", 999))
            + 1000.0 * (v_lo + v_hi)
            + 100.0 * len(candidate_report.failures)
        )
        if score < best_score:
            best_score = score
            best_report = candidate_report
            best_gen = gen_candidate.copy()
            best_ext = ext_candidate.copy()
            best_strategy = strategy
        if candidate_report.passed:
            return True, gen_candidate, ext_candidate, {
                "strategy": strategy,
                "optimizer": optimizer_result,
                "common_verifier": candidate_report.to_dict(),
            }

    # Refine the best coordinated schedule locally. Starting the original
    # generator-bus rule from this raised-voltage point lets it lower only the
    # high-voltage generator areas while preserving remote low-bus support.
    refine_net = copy.deepcopy(net)
    refine_net.gen.vm_pu.values[:] = best_gen
    if len(refine_net.ext_grid):
        refine_net.ext_grid.vm_pu.values[:] = best_ext
    _refined_pass, refined_gen, refined_ext, refined_optimizer = optimize_voltage_strict(
        refine_net, dispatch, demand, lower, upper,
        vmin_strict=vmin, vmax_strict=vmax, max_iter=voltage_iters,
    )
    refined_report = verifier.verify_dispatch(
        dispatch, demand, check_base_ac=True,
        check_all_ac_contingencies=False,
        gen_vm_pu=refined_gen, ext_grid_vm_pu=refined_ext,
    )
    if refined_report.passed:
        return True, refined_gen, refined_ext, {
            "strategy": f"{best_strategy}_then_local_refinement",
            "optimizer": refined_optimizer,
            "common_verifier": refined_report.to_dict(),
        }

    sensitivity_passed, sensitivity_gen, sensitivity_ext, sensitivity_audit = (
        optimize_voltage_sensitivity(
            verifier, dispatch, demand, best_gen, best_ext, vmin, vmax,
        )
    )
    if sensitivity_passed:
        return True, sensitivity_gen, sensitivity_ext, {
            "strategy": f"{best_strategy}_then_ac_voltage_sensitivity",
            "optimizer": optimizer_result,
            "common_verifier": sensitivity_audit["common_verifier"],
            "voltage_sensitivity": sensitivity_audit,
        }

    return False, best_gen, best_ext, {
        "strategy": best_strategy,
        "optimizer": optimizer_result,
        "common_verifier": best_report.to_dict(),
        "local_refinement": {
            "optimizer": refined_optimizer,
            "common_verifier": refined_report.to_dict(),
        },
        "initial_voltage_sensitivity": direct_audit,
        "voltage_sensitivity": sensitivity_audit,
    }


def optimize_voltage_sensitivity(verifier, dispatch, demand, initial_gen,
                                 initial_ext, vmin, vmax, max_rounds=4):
    """Restore bus voltages with the deployed generator/reference controls.

    The Jacobian is estimated from AC-PF perturbations at the current operating
    point.  This is a reachability screen only: it uses neither SCED labels nor
    stored rescue controls.
    """
    try:
        from scipy.optimize import lsq_linear
    except ImportError:
        report = verifier.verify_dispatch(
            dispatch, demand, check_base_ac=True,
            check_all_ac_contingencies=False,
            gen_vm_pu=initial_gen, ext_grid_vm_pu=initial_ext,
        )
        return False, initial_gen, initial_ext, {
            "status": "scipy_unavailable", "rounds": [],
            "common_verifier": report.to_dict(),
        }

    gen_vm = np.clip(np.asarray(initial_gen, dtype=np.float64), 0.92, 1.08)
    ext_vm = np.clip(np.asarray(initial_ext, dtype=np.float64), 0.92, 1.08)
    controls = np.concatenate([gen_vm, ext_vm])
    n_gen = len(gen_vm)
    perturb = 0.002
    target_margin = min(0.001, max((vmax - vmin) * 0.05, 1e-4))
    rounds = []

    def evaluate(control_vector):
        g = control_vector[:n_gen]
        e = control_vector[n_gen:]
        report = verifier.verify_dispatch(
            dispatch, demand, check_base_ac=True,
            check_all_ac_contingencies=False,
            gen_vm_pu=g, ext_grid_vm_pu=e,
        )
        return report

    def score(report):
        metrics = report.metrics
        lo = max(0.0, vmin - float(metrics.get("base_v_min", -1.0)))
        hi = max(0.0, float(metrics.get("base_v_max", 2.0)) - vmax)
        return (1000.0 * float(metrics.get("base_v_violations", 999))
                + 1000.0 * (lo + hi) + 100.0 * len(report.failures))

    best_controls = controls.copy()
    best_report = evaluate(best_controls)
    best_score = score(best_report)
    if best_report.passed:
        return True, gen_vm, ext_vm, {
            "status": "initial_pass", "rounds": rounds,
            "common_verifier": best_report.to_dict(),
        }

    for round_idx in range(max_rounds):
        base_net = verifier._fresh_net(
            dispatch, demand,
            gen_vm_pu=controls[:n_gen], ext_grid_vm_pu=controls[n_gen:],
        )
        if verifier._runpp(base_net) is not None:
            rounds.append({"round": round_idx + 1, "status": "base_acpf_failed"})
            break
        base_bus_vm = np.asarray(base_net.res_bus.vm_pu.values, dtype=np.float64)
        bus_ids = list(base_net.res_bus.index)
        violating_positions = np.flatnonzero(
            (base_bus_vm < vmin) | (base_bus_vm > vmax)
        )
        if len(violating_positions) == 0:
            violating_positions = np.argsort(
                np.minimum(base_bus_vm - vmin, vmax - base_bus_vm)
            )[:2]
        critical_buses = [bus_ids[int(pos)] for pos in violating_positions]
        selected_gen_controls = []
        try:
            import networkx as nx
            import pandapower.topology as top
            graph = top.create_nxgraph(
                base_net, respect_switches=True, include_trafos=True,
            )
            gen_buses = list(base_net.gen.bus.values)
            distances = []
            for gen_idx, gen_bus in enumerate(gen_buses):
                best_distance = min(
                    nx.shortest_path_length(graph, source=bus, target=gen_bus)
                    for bus in critical_buses
                    if bus in graph and gen_bus in graph
                )
                distances.append((best_distance, gen_idx))
            selected_gen_controls = [idx for _, idx in sorted(distances)[:12]]
        except Exception:
            selected_gen_controls = list(range(min(n_gen, 12)))
        selected_controls = selected_gen_controls + list(range(n_gen, len(controls)))
        cache_key = (round(float(demand), 3), tuple(selected_controls))
        cached = _VOLTAGE_JACOBIAN_CACHE.get(cache_key)
        if cached is None or cached[0].shape != (len(base_bus_vm), len(controls)):
            jacobian = np.zeros((len(base_bus_vm), len(controls)), dtype=np.float64)
            valid_columns = 0
            for control_idx in selected_controls:
                direction = 1.0 if controls[control_idx] <= 1.08 - perturb else -1.0
                delta = direction * perturb
                trial = controls.copy()
                trial[control_idx] = np.clip(trial[control_idx] + delta, 0.92, 1.08)
                actual_delta = trial[control_idx] - controls[control_idx]
                if abs(actual_delta) < 1e-10:
                    continue
                # Reuse the initialized network while estimating Jacobian
                # columns. Deep-copying case300 for every 0.002-pu perturbation
                # dominates runtime and does not change the operating model.
                base_net.gen.vm_pu.values[:] = trial[:n_gen]
                if len(base_net.ext_grid):
                    base_net.ext_grid.vm_pu.values[:] = trial[n_gen:]
                if verifier._runpp(base_net) is not None:
                    continue
                jacobian[:, control_idx] = (
                    np.asarray(base_net.res_bus.vm_pu.values, dtype=np.float64)
                    - base_bus_vm
                ) / actual_delta
                valid_columns += 1
            _VOLTAGE_JACOBIAN_CACHE[cache_key] = (jacobian.copy(), valid_columns)
            jacobian_source = "computed"
        else:
            jacobian, valid_columns = cached
            jacobian_source = "cached"

        if valid_columns == 0:
            rounds.append({"round": round_idx + 1, "status": "no_valid_sensitivity"})
            break

        target_bus_vm = np.clip(
            base_bus_vm, vmin + target_margin, vmax - target_margin,
        )
        residual = target_bus_vm - base_bus_vm
        distance_to_bound = np.minimum(base_bus_vm - vmin, vmax - base_bus_vm)
        weights = np.full(len(base_bus_vm), 0.15, dtype=np.float64)
        weights[distance_to_bound < 0.005] = 2.0
        weights[(base_bus_vm < vmin) | (base_bus_vm > vmax)] = 10.0
        weighted_jacobian = weights[:, None] * jacobian
        weighted_residual = weights * residual
        regularization = 0.03
        matrix = np.vstack([
            weighted_jacobian,
            regularization * np.eye(len(controls), dtype=np.float64),
        ])
        rhs = np.concatenate([weighted_residual, np.zeros(len(controls))])
        lower_delta = 0.92 - controls
        upper_delta = 1.08 - controls
        solution = lsq_linear(
            matrix, rhs, bounds=(lower_delta, upper_delta),
            method="trf", lsmr_tol="auto", max_iter=100,
        )

        accepted = False
        round_best_score = best_score
        round_best_controls = controls.copy()
        round_best_report = best_report
        for damping in (1.0, 0.5, 0.25):
            trial = np.clip(controls + damping * solution.x, 0.92, 1.08)
            trial_report = evaluate(trial)
            trial_score = score(trial_report)
            if trial_score < round_best_score:
                round_best_score = trial_score
                round_best_controls = trial
                round_best_report = trial_report
                accepted = True
            if trial_report.passed:
                rounds.append({
                    "round": round_idx + 1, "status": "passed",
                    "valid_columns": valid_columns,
                    "jacobian_source": jacobian_source, "damping": damping,
                    "max_control_delta": float(np.max(np.abs(trial - controls))),
                })
                return True, trial[:n_gen], trial[n_gen:], {
                    "status": "passed", "rounds": rounds,
                    "common_verifier": trial_report.to_dict(),
                }

        rounds.append({
            "round": round_idx + 1,
            "status": "improved" if accepted else "no_improvement",
            "valid_columns": valid_columns,
            "jacobian_source": jacobian_source,
            "max_proposed_delta": float(np.max(np.abs(solution.x))),
        })
        if not accepted:
            break
        controls = round_best_controls
        best_controls = round_best_controls.copy()
        best_report = round_best_report
        best_score = round_best_score

    return False, best_controls[:n_gen], best_controls[n_gen:], {
        "status": "not_restored", "rounds": rounds,
        "common_verifier": best_report.to_dict(),
    }


def demand_key(sample):
    return f"{float(sample['demand_mw']) / 1000.0:g}"


def stratified_probe_indices(samples, max_samples):
    if max_samples is None or max_samples >= len(samples):
        return list(range(len(samples)))
    groups = defaultdict(list)
    for idx, sample in enumerate(samples):
        groups[demand_key(sample)].append(idx)
    ordered_groups = [groups[k] for k in sorted(groups, key=float)]
    selected = []
    depth = 0
    while len(selected) < max_samples:
        added = False
        for ids in ordered_groups:
            if depth < len(ids):
                selected.append(ids[depth])
                added = True
                if len(selected) == max_samples:
                    break
        if not added:
            break
        depth += 1
    return selected


def main():
    global check_n1_security, optimize_voltage_strict
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Delay pandapower/Gurobi imports until after CLI parsing so --help and
    # malformed commands return immediately.
    from full_sced_verifier import FullSCEDVerifier
    from generate_case300_gt_v10 import check_n1_security, optimize_voltage_strict

    dataset_path = Path(args.dataset).resolve()
    checkpoint_path = Path(args.gat_checkpoint).resolve()
    with dataset_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    samples = [s for s in data["samples"] if s.get("is_complete_sced", False)]
    with Path(args.split_file).open("r", encoding="utf-8") as fh:
        old_split = json.load(fh)
    partition = {}
    for name in ("train", "val", "test"):
        for idx in old_split[f"{name}_ids"]:
            partition[int(idx)] = name
    if set(partition) != set(range(len(samples))):
        raise RuntimeError("Input split must partition every complete candidate sample exactly once")

    os.environ["ROUTE_B_GAT_CHECKPOINT"] = str(checkpoint_path)
    from train_rlv35_complete import (
        SCEDEnvV21,
        REF_SOURCE_P_MIN_MW, REF_SOURCE_P_MAX_MW,
        REF_SOURCE_Q_MIN_MVAR, REF_SOURCE_Q_MAX_MVAR,
    )

    env = SCEDEnvV21(
        samples, list(range(len(samples))), dataset_meta=data,
        action_encoding=args.action_encoding, action_basis_seed=args.seed,
    )
    lower = np.asarray(env.PMIN, dtype=np.float64)
    upper = np.asarray(env.PMAX, dtype=np.float64) * 0.99
    verifier_net = copy.deepcopy(env._ac_net_template)
    if len(verifier_net.ext_grid):
        verifier_net.ext_grid["min_p_mw"] = REF_SOURCE_P_MIN_MW
        verifier_net.ext_grid["max_p_mw"] = REF_SOURCE_P_MAX_MW
        verifier_net.ext_grid["min_q_mvar"] = REF_SOURCE_Q_MIN_MVAR
        verifier_net.ext_grid["max_q_mvar"] = REF_SOURCE_Q_MAX_MVAR
    verifier = FullSCEDVerifier(
        net_template=verifier_net,
        voltage_min_pu=args.vmin,
        voltage_max_pu=args.vmax,
        check_ext_grid_bounds=True,
        check_dispatch_sum_balance=True,
    )

    selected = []
    selected_old_indices = []
    records = []
    counts = Counter()
    by_demand = defaultdict(Counter)
    screen_indices = stratified_probe_indices(samples, args.max_samples)
    n_screen = len(screen_indices)
    started = time.time()

    for screen_pos, old_idx in enumerate(screen_indices, 1):
        sample = samples[old_idx]
        sid = sample.get("sample_id", old_idx)
        demand = float(sample["demand_mw"])
        group = demand_key(sample)
        ts = time.time()
        env.reset(old_idx)
        p0 = np.asarray(env.gat_power, dtype=np.float64).copy()

        lin0 = check_n1_security(p0, env.dcnet)
        initial_linear_secure = int(lin0.get("n_violations", 999999)) == 0
        if initial_linear_secure:
            voltage_pass, _g0, _e0, voltage_only = strict_voltage_result(
                verifier, env._ac_net_template, p0, demand, lower, upper,
                args.vmin, args.vmax, args.voltage_iters,
            )
        else:
            voltage_pass = False
            voltage_only = {
                "skipped_ac_voltage_search": True,
                "reason": "fixed active power already violates linear base/N-1 security",
                "linear_n1": lin0,
            }

        found = None
        projection_audit = None
        projection_strict_audit = None
        last_candidate_strict_audit = None
        linear_rejects = 0
        if voltage_pass:
            status = "voltage_only_recoverable"
        else:
            directions = deployed_active_directions(env)
            projected, projection_audit = solve_n1_in_action_span(
                p0, lower, upper, directions, env.dcnet,
                args.max_delta_mw, args.projection_margin,
                args.projection_iters,
            )
            if projected is not None:
                passed, gen_vm, ext_vm, strict = strict_voltage_result(
                    verifier, env._ac_net_template, projected, demand,
                    lower, upper, args.vmin, args.vmax, args.voltage_iters,
                )
                projection_strict_audit = strict
                if passed:
                    found = {
                        "attempt": 0,
                        "dispatch": projected,
                        "gen_vm": gen_vm,
                        "ext_vm": ext_vm,
                        "delta_l1_mw": float(np.abs(projected - p0).sum()),
                        "delta_linf_mw": float(np.abs(projected - p0).max()),
                        "linear_n1": check_n1_security(projected, env.dcnet),
                        "strict": strict,
                    }
            search_center = projected if projected is not None else p0
            for attempt in range(args.attempts if found is None else 0):
                pcand = make_candidate(
                    search_center, lower, upper, demand, directions, rng, attempt,
                    args.step_mw, args.max_delta_mw,
                )
                delta_l1 = float(np.abs(pcand - p0).sum())
                if delta_l1 < args.min_delta_l1_mw:
                    continue
                lin = check_n1_security(pcand, env.dcnet)
                if int(lin.get("n_violations", 999999)):
                    linear_rejects += 1
                    continue
                passed, gen_vm, ext_vm, strict = strict_voltage_result(
                    verifier, env._ac_net_template, pcand, demand,
                    lower, upper, args.vmin, args.vmax, args.voltage_iters,
                )
                last_candidate_strict_audit = strict
                if passed:
                    found = {
                        "attempt": attempt + 1,
                        "dispatch": pcand,
                        "gen_vm": gen_vm,
                        "ext_vm": ext_vm,
                        "delta_l1_mw": delta_l1,
                        "delta_linf_mw": float(np.abs(pcand - p0).max()),
                        "linear_n1": lin,
                        "strict": strict,
                    }
                    break
            status = "active_required_and_recoverable" if found else "hybrid_search_failed"

        counts[status] += 1
        by_demand[group][status] += 1
        if found:
            out = copy.deepcopy(sample)
            out["active_power_required_from_gat"] = True
            out["gat_start_linear_n1_audit"] = lin0
            out["gat_start_dispatch_68_audit"] = p0.tolist()
            out["gat_voltage_only_failure_audit"] = voltage_only
            out["active_required_rescue_evidence"] = {
                "attempt": found["attempt"],
                "delta_l1_mw": found["delta_l1_mw"],
                "delta_linf_mw": found["delta_linf_mw"],
                "strict": found["strict"],
                "note": "Dataset-curation evidence only; excluded from GAT/RL inputs and rewards.",
            }
            out["active_required_rescue_dispatch_68"] = found["dispatch"].tolist()
            out["active_required_rescue_gen_vm_pu"] = np.asarray(found["gen_vm"]).tolist()
            out["active_required_rescue_ext_grid_vm_pu"] = np.asarray(found["ext_vm"]).tolist()
            selected.append(out)
            selected_old_indices.append(old_idx)

        elapsed = time.time() - ts
        records.append({
            "old_index": old_idx, "sample_id": sid, "partition": partition[old_idx],
            "demand_mw": demand, "status": status,
            "elapsed_sec": round(elapsed, 3),
            "rescue_delta_l1_mw": None if not found else found["delta_l1_mw"],
            "gat_start_n1_violations": int(lin0.get("n_violations", 999999)),
            "projection": projection_audit,
            "projection_strict": projection_strict_audit,
            "last_candidate_strict": last_candidate_strict_audit,
            "linear_rejected_candidates": linear_rejects,
        })
        if screen_pos == 1 or screen_pos % args.progress_every == 0 or screen_pos == n_screen:
            print(
                f"[{screen_pos}/{n_screen}] kept={len(selected)} "
                f"last={sid} {status} {elapsed:.1f}s"
            )

    new_split = {"train_ids": [], "val_ids": [], "test_ids": []}
    for new_idx, old_idx in enumerate(selected_old_indices):
        new_split[f"{partition[old_idx]}_ids"].append(new_idx)
    for name in ("train", "val", "test"):
        if not new_split[f"{name}_ids"] and not args.allow_empty_partitions:
            raise RuntimeError(
                f"No retained {name} samples. Generate a larger candidate pool before RL training."
            )

    summary = {
        "candidate_dataset": str(dataset_path),
        "gat_checkpoint": str(checkpoint_path),
        "input_split": str(Path(args.split_file).resolve()),
        "screened": n_screen,
        "retained": len(selected),
        "counts": dict(counts),
        "retained_split_sizes": {k: len(v) for k, v in new_split.items()},
        "by_demand_gw": {k: dict(v) for k, v in sorted(by_demand.items(), key=lambda x: float(x[0]))},
        "criteria": {
            "voltage_only": "same fixed-P voltage optimizer plus common strict verifier must fail",
            "hybrid": (
                f"deployed {args.action_encoding} active span plus same "
                "voltage optimizer/verifier must pass"
            ),
            "action_encoding": args.action_encoding,
            "action_basis_seed": int(args.seed),
            "linear_n1_limit_scale": float(
                data.get("model_config", {}).get("n1_limit_scale", 1.0)
            ),
            "vmin": args.vmin, "vmax": args.vmax,
            "attempts": args.attempts, "step_mw": args.step_mw,
            "max_delta_mw": args.max_delta_mw,
            "projection_margin": args.projection_margin,
            "projection_iters": args.projection_iters,
        },
        "elapsed_sec": round(time.time() - started, 3),
    }
    out_data = copy.deepcopy(data)
    out_data["samples"] = selected
    out_data["gat_active_required_screen"] = summary
    for output_path in (args.output_dataset, args.output_split, args.output_report):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_dataset).write_text(
        json.dumps(out_data, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    split_payload = {
        "version": "gat_active_required_preserved_split_v1",
        "dataset": str(Path(args.output_dataset).resolve()),
        "source_split": str(Path(args.split_file).resolve()),
        **new_split,
    }
    Path(args.output_split).write_text(json.dumps(split_payload, indent=2), encoding="utf-8")
    Path(args.output_report).write_text(
        json.dumps({"summary": summary, "records": records}, indent=2, default=float),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print("Wrote:", Path(args.output_dataset))
    print("Wrote:", Path(args.output_split))
    print("Wrote:", Path(args.output_report))


if __name__ == "__main__":
    main()

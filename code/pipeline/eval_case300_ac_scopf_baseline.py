"""Preventive AC-SCOPF baseline for the active-required case300 benchmark.

The adopted contingency model is hybrid: base-case operation is checked by a
reactive-power-enforced AC power flow, while N-1 line security is represented
by the dataset's affine PTDF/LODF model.  This baseline therefore uses:

1. a convex security master with all generator bounds, scheduled balance, and
   base/valid-contingency LODF constraints; and
2. an AC-OPF voltage/reactive-power subproblem with the secure active schedule
   fixed, followed by the same independent AC-PF and LODF verifier used by the
   reported base-case AC-OPF baseline.

No GAT prediction, RL policy, rescue dispatch, validation result, or test
outcome enters the optimization.
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import time
from collections import Counter

import numpy as np

from eval_route_b_acopf_baseline import (
    PROJ_DIR,
    REF_SOURCE_P_MAX_MW,
    REF_SOURCE_P_MIN_MW,
    REF_SOURCE_Q_MAX_MVAR,
    REF_SOURCE_Q_MIN_MVAR,
    RL_VMAX_PU,
    RL_VMIN_PU,
    build_opf_net,
    extract_opf_voltage_setpoints,
    is_pass,
    linear_n1_check,
    load_linear_model,
    load_samples,
    load_split,
    prepare_base_net,
    run_opf,
    scheduled_controllable_generation_cost,
    verify_ac_pf,
)


def _security_residuals(p: np.ndarray, model: dict, tol: float) -> tuple[np.ndarray, list[tuple]]:
    """Return affine flows and every violated base/contingency inequality."""
    p = np.asarray(p, dtype=np.float64)
    flows = model["F_ref"] + model["S"] @ (p - model["P_ref"])
    limit = model["limit"]
    violations: list[tuple] = []

    base_excess = np.abs(flows) - limit
    for line in np.flatnonzero(base_excess > tol):
        sign = 1.0 if flows[line] >= 0.0 else -1.0
        violations.append((float(base_excess[line]), "base", int(line), -1, sign))

    lodf = model["lodf"]
    for outage in model["valid_contingency_indices"]:
        post = flows + lodf[:, outage] * flows[outage]
        post[outage] = 0.0
        excess = np.abs(post) - limit
        excess[outage] = -np.inf
        for line in np.flatnonzero(excess > tol):
            sign = 1.0 if post[line] >= 0.0 else -1.0
            violations.append((float(excess[line]), "n1", int(line), int(outage), sign))
    violations.sort(key=lambda item: item[0], reverse=True)
    return flows, violations


def solve_security_master(
    sample: dict,
    base_net,
    model: dict,
    timeout_sec: float,
    threads: int,
    cut_batch: int,
    max_cut_rounds: int,
    feasibility_tol_mw: float,
) -> tuple[bool, dict]:
    """Solve the preventive LODF-constrained economic dispatch by cut generation."""
    import gurobipy as gp
    from gurobipy import GRB

    n_gen = len(base_net.gen)
    demand = float(sample["demand_mw"])
    pmin = np.asarray(base_net.gen.min_p_mw.values, dtype=np.float64)
    pmax = 0.99 * np.asarray(base_net.gen.max_p_mw.values, dtype=np.float64)
    cp2 = np.asarray(sample.get("sample_cp2", np.zeros(n_gen)), dtype=np.float64)
    cp1 = np.asarray(sample.get("sample_cp1", np.ones(n_gen) * 30.0), dtype=np.float64)
    cp0 = np.asarray(sample.get("sample_cp0", np.zeros(n_gen)), dtype=np.float64)
    if not (len(cp2) == len(cp1) == len(cp0) == n_gen):
        raise ValueError("Generator cost vectors do not match the case300 generator count")

    master = gp.Model("case300_preventive_lodf_master")
    master.Params.OutputFlag = 0
    master.Params.Threads = max(1, int(threads))
    master.Params.TimeLimit = max(1.0, float(timeout_sec))
    master.Params.FeasibilityTol = 1e-7
    master.Params.OptimalityTol = 1e-7
    pvar = master.addMVar(n_gen, lb=pmin, ub=pmax, name="Pg")
    master.addConstr(pvar.sum() == demand, name="scheduled_balance")
    master.setObjective(
        gp.quicksum(float(cp2[i]) * pvar[i] * pvar[i] + float(cp1[i]) * pvar[i] + float(cp0[i]) for i in range(n_gen)),
        GRB.MINIMIZE,
    )

    warm = np.asarray(sample.get("dispatch_68", np.full(n_gen, demand / n_gen)), dtype=np.float64)
    warm = np.clip(warm, pmin, pmax)
    pvar.Start = warm

    s_mat = model["S"]
    constant = model["F_ref"] - s_mat @ model["P_ref"]
    limit = model["limit"]
    lodf = model["lodf"]
    added: set[tuple] = set()
    t0 = time.perf_counter()
    cut_rounds = 0
    total_cuts = 0
    p_sol = None
    final_violations: list[tuple] = []

    for cut_rounds in range(max(1, int(max_cut_rounds))):
        remaining = float(timeout_sec) - (time.perf_counter() - t0)
        if remaining <= 0.0:
            return False, {"reason": "security_master_timeout", "security_master_sec": time.perf_counter() - t0}
        master.Params.TimeLimit = max(0.1, remaining)
        master.optimize()
        if master.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) or master.SolCount == 0:
            return False, {
                "reason": f"security_master_status_{master.Status}",
                "security_master_sec": time.perf_counter() - t0,
                "security_cut_rounds": cut_rounds + 1,
                "security_cuts": total_cuts,
            }
        p_sol = np.asarray(pvar.X, dtype=np.float64)
        _, final_violations = _security_residuals(p_sol, model, feasibility_tol_mw)
        if not final_violations:
            break

        new_cuts = 0
        for _excess, kind, line, outage, sign in final_violations:
            key = (kind, line, outage, sign)
            if key in added:
                continue
            if kind == "base":
                coeff = sign * s_mat[line]
                rhs = float(limit[line] - sign * constant[line])
            else:
                multiplier = float(lodf[line, outage])
                coeff = sign * (s_mat[line] + multiplier * s_mat[outage])
                rhs = float(limit[line] - sign * (constant[line] + multiplier * constant[outage]))
            master.addConstr(coeff @ pvar <= rhs, name=f"{kind}_{line}_{outage}_{'p' if sign > 0 else 'n'}")
            added.add(key)
            total_cuts += 1
            new_cuts += 1
            if new_cuts >= max(1, int(cut_batch)):
                break
        if new_cuts == 0:
            break

    elapsed = time.perf_counter() - t0
    if p_sol is None:
        return False, {"reason": "security_master_no_solution", "security_master_sec": elapsed}
    audit = linear_n1_check(p_sol, model, tol=feasibility_tol_mw)
    secure = audit["n1_violations"] == 0
    return secure, {
        "reason": "security_master_secure" if secure else "security_master_residual",
        "security_master_sec": elapsed,
        "security_cut_rounds": cut_rounds + 1,
        "security_cuts": total_cuts,
        "security_dispatch_mw": p_sol.tolist(),
        "security_n1_violations": int(audit["n1_violations"]),
        "security_n1_overload_mw": float(audit["n1_overload_mw"]),
        "security_objective": float(master.ObjVal) if master.SolCount else None,
    }


def solve_ac_subproblem(
    base_net, sample: dict, p_secure: np.ndarray, timeout_sec: float,
    verifier, voltage_rounds: int,
) -> tuple[bool, dict]:
    """Restore AC feasibility while holding the secure Pg schedule exactly.

    An OPF with 68 equality-fixed active powers is attempted first. Some
    PYPOWER interior-point solves become numerically degenerate under those
    equality bounds even when a feasible voltage schedule exists. In that
    case, a finite-difference AC-PF Jacobian subproblem updates only generator
    and reference-source voltage controls. The secure active schedule is never
    relaxed, so the master problem's LODF guarantee remains intact.
    """
    started = time.perf_counter()
    net = build_opf_net(base_net, sample, RL_VMIN_PU, RL_VMAX_PU)
    p_secure = np.asarray(p_secure, dtype=np.float64)
    net.gen.p_mw.values[:] = p_secure
    net.gen.min_p_mw.values[:] = p_secure
    net.gen.max_p_mw.values[:] = p_secure
    net.gen.controllable.values[:] = True
    ok, info = run_opf(net, timeout_sec)
    if ok:
        gen_vm, ext_vm = extract_opf_voltage_setpoints(net, RL_VMIN_PU, RL_VMAX_PU)
        return True, {
            **info,
            "reason": "ac_subproblem_converged",
            "ac_subproblem_method": "equality_fixed_p_ac_opf",
            "ac_subproblem_sec": time.perf_counter() - started,
            "gen_vm_pu": gen_vm.tolist(),
            "ext_grid_vm_pu": float(ext_vm),
        }

    # Numerical fallback: solve only for voltage controls with repeated AC-PF
    # evaluations. This is a deterministic optimization baseline, not a
    # learned policy and not a stored rescue action.
    from screen_case300_gat_active_required import optimize_voltage_sensitivity

    initial_gen = np.asarray(base_net.gen.vm_pu.values, dtype=np.float64)
    initial_ext = np.asarray(base_net.ext_grid.vm_pu.values, dtype=np.float64)
    restored, gen_vm, ext_vm, audit = optimize_voltage_sensitivity(
        verifier,
        p_secure,
        float(sample["demand_mw"]),
        initial_gen,
        initial_ext,
        RL_VMIN_PU,
        RL_VMAX_PU,
        max_rounds=max(1, int(voltage_rounds)),
    )
    return bool(restored), {
        **info,
        "reason": "ac_subproblem_converged" if restored else "ac_subproblem_failed",
        "ac_subproblem_method": "fixed_p_ac_pf_jacobian_voltage_restoration",
        "ac_subproblem_sec": time.perf_counter() - started,
        "voltage_restoration_status": audit.get("status"),
        "voltage_restoration_rounds": len(audit.get("rounds", [])),
        "gen_vm_pu": np.asarray(gen_vm, dtype=float).tolist(),
        "ext_grid_vm_pu": float(np.asarray(ext_vm, dtype=float).reshape(-1)[0]),
    }


def failure_reason(row: dict) -> str:
    if row.get("security_n1_violations", 999) > 0:
        return "security_master_residual"
    if not row.get("converged", False):
        return "acpf_not_converged"
    if row.get("v_violations", 999) > 0:
        return "voltage_violation"
    if row.get("overloaded_lines", 999) > 0 or row.get("trafo_overloaded", 999) > 0:
        return "thermal_overload"
    if row.get("reference_source_violations", 999) > 0:
        return "reference_source_limit"
    if row.get("n1_violations", 999) > 0:
        return "n1_violation"
    if row.get("scheduled_balance_violations", 999) > 0:
        return "scheduled_balance"
    if row.get("generator_bound_violations", 999) > 0:
        return "generator_bounds"
    return "full_sced_pass"


def summarize(rows: list[dict]) -> dict:
    passed = int(sum(bool(row.get("passed")) for row in rows))
    def mean(key: str):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return float(np.mean(values)) if values else None
    return {
        "n": len(rows),
        "passed": passed,
        "pass_rate_pct": 100.0 * passed / max(len(rows), 1),
        "avg_runtime_sec": mean("runtime_sec"),
        "avg_security_master_sec": mean("security_master_sec"),
        "avg_ac_opf_sec": mean("opf_runtime_sec"),
        "avg_ac_subproblem_sec": mean("ac_subproblem_sec"),
        "avg_final_verification_sec": mean("verification_sec"),
        "avg_security_cut_rounds": mean("security_cut_rounds"),
        "avg_security_cuts": mean("security_cuts"),
        "avg_scheduled_controllable_generation_cost": mean("scheduled_controllable_generation_cost"),
        "fail_reasons": dict(Counter(row["reason"] for row in rows if not row.get("passed"))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(PROJ_DIR / "case300_gt_n1s0575_active_required_fixed32.json"))
    parser.add_argument("--split-file", default=str(PROJ_DIR / "paper_results" / "case300_n1s0575_active_fixed32_split.json"))
    parser.add_argument("--scope", choices=["val", "test"], default="test")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--master-timeout-sec", type=float, default=120.0)
    parser.add_argument("--ac-timeout-sec", type=float, default=120.0)
    parser.add_argument("--voltage-rounds", type=int, default=6)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cut-batch", type=int, default=1000)
    parser.add_argument("--max-cut-rounds", type=int, default=50)
    parser.add_argument("--security-tol-mw", type=float, default=1e-5)
    parser.add_argument("--output", default=str(PROJ_DIR / "paper_results" / "case300_n1s0575_active_ac_scopf_test.json"))
    args = parser.parse_args()

    data, samples = load_samples(args.dataset)
    _train_ids, val_ids, test_ids = load_split(args.split_file, samples)
    ids = val_ids if args.scope == "val" else test_ids
    if args.max_samples > 0:
        ids = ids[: args.max_samples]

    print(f"Preventive AC-SCOPF: {len(ids)} {args.scope} samples", flush=True)
    base_net = prepare_base_net()
    model = load_linear_model(data, base_net)
    from full_sced_verifier import FullSCEDVerifier
    verifier_net = copy.deepcopy(base_net)
    verifier_net.ext_grid["min_p_mw"] = REF_SOURCE_P_MIN_MW
    verifier_net.ext_grid["max_p_mw"] = REF_SOURCE_P_MAX_MW
    verifier_net.ext_grid["min_q_mvar"] = REF_SOURCE_Q_MIN_MVAR
    verifier_net.ext_grid["max_q_mvar"] = REF_SOURCE_Q_MAX_MVAR
    verifier = FullSCEDVerifier(
        net_template=verifier_net,
        voltage_min_pu=RL_VMIN_PU,
        voltage_max_pu=RL_VMAX_PU,
        check_ext_grid_bounds=True,
        check_dispatch_sum_balance=True,
    )
    rows = []
    for position, sid in enumerate(ids, 1):
        sample = samples[sid]
        sample_id = sample.get("sample_id", str(sid))
        t0 = time.perf_counter()
        print(f"[{position:02d}/{len(ids)}] {sample_id} security master", flush=True)
        secure, master_info = solve_security_master(
            sample, base_net, model, args.master_timeout_sec, args.threads,
            args.cut_batch, args.max_cut_rounds, args.security_tol_mw,
        )
        row = {"sid": int(sid), "sample_id": sample_id, **master_info}
        if secure:
            p_secure = np.asarray(master_info["security_dispatch_mw"], dtype=np.float64)
            print(f"[{position:02d}/{len(ids)}] {sample_id} AC subproblem", flush=True)
            ac_ok, ac_info = solve_ac_subproblem(
                base_net, sample, p_secure, args.ac_timeout_sec,
                verifier, args.voltage_rounds,
            )
            row.update(ac_info)
            if ac_ok:
                gen_vm = np.asarray(ac_info["gen_vm_pu"], dtype=np.float64)
                ext_vm = float(ac_info["ext_grid_vm_pu"])
                verify_t0 = time.perf_counter()
                ac = verify_ac_pf(base_net, p_secure, gen_vm, ext_vm, sample["demand_mw"])
                n1 = linear_n1_check(p_secure, model, tol=args.security_tol_mw)
                row.update(ac)
                row.update(n1)
                row["verification_sec"] = time.perf_counter() - verify_t0
                row["scheduled_controllable_generation_cost"] = scheduled_controllable_generation_cost(sample, p_secure)
            else:
                row.update({"converged": False, "n1_violations": 999, "verification_sec": None, "scheduled_controllable_generation_cost": None})
        else:
            row.update({"converged": False, "n1_violations": 999, "opf_runtime_sec": None, "verification_sec": None, "scheduled_controllable_generation_cost": None})

        row["reason"] = failure_reason(row)
        row["passed"] = bool(is_pass(row))
        row["runtime_sec"] = time.perf_counter() - t0
        rows.append(row)
        print(
            f"[{position:02d}/{len(ids)}] {sample_id} "
            f"{'PASS' if row['passed'] else 'FAIL'} {row['reason']} {row['runtime_sec']:.2f}s",
            flush=True,
        )

    result = {
        "method": "preventive_ac_scopf_affine_lodf",
        "formulation": "LODF-constrained convex active-power master plus fixed-P AC voltage/reactive feasibility subproblem",
        "dataset": str(args.dataset),
        "split_file": str(args.split_file),
        "scope": args.scope,
        "valid_line_outages": int(len(model["valid_contingency_indices"])),
        "settings": vars(args),
        "summary": summarize(rows),
        "rows": rows,
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"Wrote: {output}", flush=True)


if __name__ == "__main__":
    main()

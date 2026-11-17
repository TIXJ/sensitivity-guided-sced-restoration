"""AC-OPF restoration baseline for the 300-bus correction benchmark.

This script is intentionally independent of the RL environment.  It avoids
loading GAT/SAC or rebuilding the numerical N-1 network.  Each sample is solved
with pandapower AC-OPF, then re-verified by:

1. a fresh AC-PF with reactive-power limits; and
2. the affine LODF model stored in the dataset metadata.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import threading
import time
import traceback
from collections import Counter, defaultdict

import numpy as np

PROJ_DIR = pathlib.Path(__file__).resolve().parents[2]
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

RL_VMIN_PU = 0.933
RL_VMAX_PU = 1.09
GEN_VM_MIN = RL_VMIN_PU
GEN_VM_MAX = RL_VMAX_PU
EXT_VM_MIN = RL_VMIN_PU
EXT_VM_MAX = RL_VMAX_PU
REF_SOURCE_P_MIN_MW = -2500.0
REF_SOURCE_P_MAX_MW = 2500.0
REF_SOURCE_Q_MIN_MVAR = -250.0
REF_SOURCE_Q_MAX_MVAR = 250.0
REF_SOURCE_S_MAX_MVA = 2500.0


def apply_voltage_limits(net, vmin=RL_VMIN_PU, vmax=RL_VMAX_PU):
    """Keep OPF voltage constraints aligned with the final AC-PF verifier."""
    net.bus.min_vm_pu = float(vmin)
    net.bus.max_vm_pu = float(vmax)
    for element in (net.gen, net.ext_grid):
        if "min_vm_pu" not in element:
            element["min_vm_pu"] = float(vmin)
        else:
            element.min_vm_pu = float(vmin)
        if "max_vm_pu" not in element:
            element["max_vm_pu"] = float(vmax)
        else:
            element.max_vm_pu = float(vmax)


def project_box_sum(x, lo, hi, target_sum, max_iter=100):
    x = np.asarray(x, dtype=np.float64).copy()
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    y = np.clip(x, lo, hi)
    target_sum = float(np.clip(target_sum, float(lo.sum()), float(hi.sum())))
    for _ in range(max_iter):
        err = target_sum - float(y.sum())
        if abs(err) < 1e-7:
            break
        if err > 0:
            free = y < hi - 1e-9
            if not np.any(free):
                break
            y[free] += err / float(np.count_nonzero(free))
            y = np.minimum(y, hi)
        else:
            free = y > lo + 1e-9
            if not np.any(free):
                break
            y[free] += err / float(np.count_nonzero(free))
            y = np.maximum(y, lo)
    return np.clip(y, lo, hi)


def load_samples(path):
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    samples = data["samples"] if isinstance(data, dict) and "samples" in data else data
    samples = [s for s in samples if s.get("is_complete_sced", True)]
    return data, samples


def load_split(path, samples):
    path = pathlib.Path(path)
    if path.exists():
        split = json.loads(path.read_text(encoding="utf-8"))
        return split["train_ids"], split["val_ids"], split["test_ids"]
    by = defaultdict(list)
    for i, s in enumerate(samples):
        by[round(float(s.get("demand_gw", s["demand_mw"] / 1000.0)), 3)].append(i)
    rng = np.random.default_rng(42)
    train_ids, val_ids, test_ids = [], [], []
    for key in sorted(by):
        ids = np.asarray(by[key], dtype=int)
        rng.shuffle(ids)
        n = len(ids)
        nt = max(1, int(0.70 * n))
        nv = max(1, int(0.15 * n))
        train_ids.extend(ids[:nt].tolist())
        val_ids.extend(ids[nt:nt + nv].tolist())
        test_ids.extend(ids[nt + nv:].tolist())
    return train_ids, val_ids, test_ids


def load_linear_model(data, net):
    meta = data.get("linear_security_model", {})
    lodf = np.asarray(meta["lodf"], dtype=np.float64)
    valid = np.asarray(meta.get("valid_contingency_indices", []), dtype=int)
    if lodf.shape != (len(net.line), len(net.line)) or np.count_nonzero(np.abs(lodf) > 1e-8) == 0:
        from generate_case300_gt_v10 import compute_dc_line_lodf
        lodf, valid = compute_dc_line_lodf(net)
        print(
            "[WARN] Stored LODF is degenerate; AC-OPF verification uses "
            f"fresh DC LODF ({len(valid)}/{len(net.line)} valid line outages).",
            flush=True,
        )
    elif len(valid) == 0:
        valid = np.flatnonzero(np.isfinite(lodf).all(axis=0)).astype(int)
    return {
        "P_ref": np.asarray(meta["P_ref"], dtype=np.float64),
        "F_ref": np.asarray(meta["F_ref"], dtype=np.float64),
        "S": np.asarray(meta["S"], dtype=np.float64),
        "lodf": lodf,
        "valid_contingency_indices": valid,
        "limit": np.asarray(meta["effective_limit"], dtype=np.float64),
    }


def linear_n1_check(p, model, tol=1e-6):
    p = np.asarray(p, dtype=np.float64)
    flows = model["F_ref"] + model["S"].dot(p - model["P_ref"])
    limit = model["limit"]
    base_over = np.maximum(np.abs(flows) - limit, 0.0)
    n_viol = int(np.sum(base_over > tol))
    total_over = float(np.sum(base_over))
    lodf = model["lodf"]
    for k in model["valid_contingency_indices"]:
        post = flows + lodf[:, k] * flows[k]
        post[k] = 0.0
        over = np.maximum(np.abs(post) - limit, 0.0)
        n_viol += int(np.sum(over > tol))
        total_over += float(np.sum(over))
    return {"n1_violations": n_viol, "n1_overload_mw": total_over}


def prepare_base_net():
    import pandapower.networks as pn

    net = pn.case300()
    apply_voltage_limits(net)
    if "min_q_mvar" not in net.gen:
        net.gen["min_q_mvar"] = -1e4
    if "max_q_mvar" not in net.gen:
        net.gen["max_q_mvar"] = 1e4
    if "min_p_mw" not in net.ext_grid:
        net.ext_grid["min_p_mw"] = -1e4
    if "max_p_mw" not in net.ext_grid:
        net.ext_grid["max_p_mw"] = 1e4
    if "min_q_mvar" not in net.ext_grid:
        net.ext_grid["min_q_mvar"] = -1e4
    if "max_q_mvar" not in net.ext_grid:
        net.ext_grid["max_q_mvar"] = 1e4
    return net


def build_opf_net(base_net, sample, opf_vmin=RL_VMIN_PU, opf_vmax=RL_VMAX_PU):
    import pandapower as pp

    net = copy.deepcopy(base_net)
    apply_voltage_limits(net, opf_vmin, opf_vmax)
    demand = float(sample["demand_mw"])
    base_load = float(net.load.p_mw.sum())
    if base_load > 0:
        scale = demand / base_load
        net.load.p_mw = net.load.p_mw * scale
        net.load.q_mvar = net.load.q_mvar * scale

    p0 = np.asarray(sample["dispatch_68"], dtype=np.float64)
    pmin = np.asarray(net.gen.min_p_mw.values, dtype=np.float64)
    pmax = np.asarray(net.gen.max_p_mw.values, dtype=np.float64) * 0.99
    # Use the same generator upper bound as the strict post-OPF verifier.
    net.gen.max_p_mw.values[:] = pmax
    net.gen["controllable"] = True
    net.gen.p_mw.values[:] = np.clip(p0, pmin, pmax)
    gen_vm = np.asarray(sample.get("gen_vm_pu") or net.gen.vm_pu.values, dtype=np.float64)
    if len(gen_vm) == len(net.gen):
        net.gen.vm_pu.values[:] = np.clip(gen_vm, opf_vmin, opf_vmax)
    if len(net.ext_grid):
        net.ext_grid["controllable"] = True
        net.ext_grid["min_p_mw"] = REF_SOURCE_P_MIN_MW
        net.ext_grid["max_p_mw"] = REF_SOURCE_P_MAX_MW
        net.ext_grid["min_q_mvar"] = REF_SOURCE_Q_MIN_MVAR
        net.ext_grid["max_q_mvar"] = REF_SOURCE_Q_MAX_MVAR
        ext_vm = sample.get("ext_grid_vm_pu") or [float(net.ext_grid.vm_pu.iloc[0])]
        net.ext_grid.vm_pu.values[0] = float(np.clip(ext_vm[0], opf_vmin, opf_vmax))

    if hasattr(net, "poly_cost"):
        net.poly_cost.drop(net.poly_cost.index, inplace=True)
    cp2 = np.asarray(sample.get("sample_cp2", np.zeros(len(net.gen))), dtype=np.float64)
    cp1 = np.asarray(sample.get("sample_cp1", np.ones(len(net.gen)) * 30.0), dtype=np.float64)
    cp0 = np.asarray(sample.get("sample_cp0", np.zeros(len(net.gen))), dtype=np.float64)
    for i, gen_idx in enumerate(net.gen.index):
        pp.create_poly_cost(
            net,
            int(gen_idx),
            "gen",
            cp1_eur_per_mw=float(cp1[i]),
            cp2_eur_per_mw2=float(cp2[i]),
            cp0_eur=float(cp0[i]),
        )
    if len(net.ext_grid):
        pp.create_poly_cost(
            net,
            int(net.ext_grid.index[0]),
            "ext_grid",
            cp1_eur_per_mw=0.0,
            cp2_eur_per_mw2=1e4,
            cp0_eur=0.0,
        )
    return net


def scheduled_controllable_generation_cost(sample, p_mw):
    """Eq. (1a) cost, excluding the artificial reference-source penalty."""
    p = np.asarray(p_mw, dtype=np.float64)
    cp2 = np.asarray(sample.get("sample_cp2", np.zeros(len(p))), dtype=np.float64)
    cp1 = np.asarray(sample.get("sample_cp1", np.ones(len(p)) * 30.0), dtype=np.float64)
    cp0 = np.asarray(sample.get("sample_cp0", np.zeros(len(p))), dtype=np.float64)
    if not (len(cp2) == len(cp1) == len(cp0) == len(p)):
        raise ValueError("Generator cost vectors do not match the optimized dispatch dimension")
    return float(np.sum(cp2 * p**2 + cp1 * p + cp0))


def run_opf(net, timeout_sec):
    import pandapower as pp

    holder = {"error": None, "traceback": None}

    def inner():
        try:
            pp.runopp(net, calculate_voltage_angles=True, init="pf", numba=False, verbose=False)
        except Exception as exc:
            holder["error"] = repr(exc)
            holder["traceback"] = traceback.format_exc()

    t0 = time.perf_counter()
    th = threading.Thread(target=inner, daemon=True)
    th.start()
    th.join(timeout=timeout_sec)
    dt = time.perf_counter() - t0
    if th.is_alive():
        return False, {"opf_error": f"timeout_after_{timeout_sec}s", "opf_runtime_sec": dt}
    if holder["error"] is not None:
        return False, {"opf_error": holder["error"], "opf_traceback": holder["traceback"], "opf_runtime_sec": dt}
    if not bool(getattr(net, "OPF_converged", False)):
        return False, {"opf_error": "OPF_not_converged", "opf_runtime_sec": dt}
    return True, {"opf_error": None, "opf_runtime_sec": dt}


def run_balanced_opf(net, demand_mw, timeout_sec):
    """Solve AC-OPF while enforcing the SCED scheduled-generation balance.

    A free reference source lets runopp choose a generator total different
    from the scheduled demand. Projecting that result afterwards changes the
    AC operating point and can invalidate the OPF voltage solution. Instead,
    iteratively fix the reference-source P value and resolve OPF until the
    generator total matches the SCED demand.
    """
    t0 = time.perf_counter()
    last_info = None
    max_iter = 6
    balance_tol = 0.001 * max(float(demand_mw), 1.0)
    for iteration in range(max_iter):
        remaining = max(5.0, float(timeout_sec) - (time.perf_counter() - t0))
        ok, info = run_opf(net, remaining)
        last_info = dict(info)
        if not ok:
            last_info["balance_opf_iterations"] = iteration + 1
            last_info["opf_runtime_sec"] = time.perf_counter() - t0
            return False, last_info

        gen_sum = float(net.res_gen.p_mw.sum())
        error = gen_sum - float(demand_mw)
        last_info["scheduled_generation_error_mw"] = error
        if abs(error) <= balance_tol:
            last_info["balance_opf_iterations"] = iteration + 1
            last_info["opf_runtime_sec"] = time.perf_counter() - t0
            return True, last_info

        if not len(net.ext_grid):
            break
        ext_p = float(net.res_ext_grid.p_mw.iloc[0])
        # Under pandapower's sign convention, moving the reference-source P
        # in the same direction as the generator-total error brings the
        # scheduled generator sum toward demand.
        new_ext_p = float(np.clip(ext_p + error, REF_SOURCE_P_MIN_MW, REF_SOURCE_P_MAX_MW))
        net.ext_grid.min_p_mw.values[0] = new_ext_p
        net.ext_grid.max_p_mw.values[0] = new_ext_p
        if abs(new_ext_p - ext_p) < 1e-5:
            break

    if last_info is None:
        last_info = {"opf_error": "balanced_opf_no_attempt"}
    last_info["balance_opf_iterations"] = max_iter
    last_info["opf_runtime_sec"] = time.perf_counter() - t0
    last_info["opf_error"] = "scheduled_generation_balance_not_reached"
    return False, last_info


def extract_opf_voltage_setpoints(net, opf_vmin=RL_VMIN_PU, opf_vmax=RL_VMAX_PU):
    """Use optimized OPF bus voltages as the PF voltage-control setpoints."""
    vm = np.asarray(net.res_bus.vm_pu.values, dtype=np.float64)
    gen_buses = np.asarray(net.gen.bus.values, dtype=int)
    gen_vm = np.clip(vm[gen_buses], opf_vmin, opf_vmax)
    if len(net.ext_grid):
        ext_bus = int(net.ext_grid.bus.iloc[0])
        ext_vm = float(np.clip(vm[ext_bus], opf_vmin, opf_vmax))
    else:
        ext_vm = 1.0
    return gen_vm, ext_vm


def verify_ac_pf(base_net, p, gen_vm, ext_vm, demand_mw):
    import pandapower as pp

    net = copy.deepcopy(base_net)
    apply_voltage_limits(net)
    base_load = float(net.load.p_mw.sum())
    if base_load > 0:
        scale = float(demand_mw) / base_load
        net.load.p_mw = net.load.p_mw * scale
        net.load.q_mvar = net.load.q_mvar * scale
    net.gen.p_mw.values[:] = np.asarray(p, dtype=np.float64)
    net.gen.vm_pu.values[:] = np.clip(np.asarray(gen_vm, dtype=np.float64), GEN_VM_MIN, GEN_VM_MAX)
    if len(net.ext_grid):
        net.ext_grid.vm_pu.values[0] = float(np.clip(float(ext_vm), EXT_VM_MIN, EXT_VM_MAX))
    t0 = time.perf_counter()
    err = None
    try:
        pp.runpp(net, calculate_voltage_angles=True, enforce_q_lims=True, numba=False, max_iteration=50)
    except Exception as exc:
        err = repr(exc)
    dt = time.perf_counter() - t0
    converged = bool(getattr(net, "converged", False)) and err is None
    out = {"converged": converged, "ac_error": err, "ac_runtime_sec": dt}
    if not converged:
        out.update({"v_violations": 999, "overloaded_lines": 999, "trafo_overloaded": 999, "reference_source_violations": 999, "v_gap": 999.0})
        return out
    vm = net.res_bus.vm_pu.values
    low = np.maximum(net.bus.min_vm_pu.values - vm, 0.0)
    high = np.maximum(vm - net.bus.max_vm_pu.values, 0.0)
    ref_p = float(net.res_ext_grid.p_mw.iloc[0]) if len(net.ext_grid) else 0.0
    ref_q = float(net.res_ext_grid.q_mvar.iloc[0]) if len(net.ext_grid) else 0.0
    ref_s = float(np.hypot(ref_p, ref_q))
    ref_p_v = bool(ref_p < REF_SOURCE_P_MIN_MW or ref_p > REF_SOURCE_P_MAX_MW)
    ref_q_v = bool(ref_q < REF_SOURCE_Q_MIN_MVAR or ref_q > REF_SOURCE_Q_MAX_MVAR)
    ref_s_v = bool(ref_s > REF_SOURCE_S_MAX_MVA)
    p_arr = np.asarray(p, dtype=np.float64)
    pmin = np.asarray(net.gen.min_p_mw.values, dtype=np.float64)
    pmax = np.asarray(net.gen.max_p_mw.values, dtype=np.float64) * 0.99
    balance_error = float(abs(p_arr.sum() - float(demand_mw)))
    balance_violations = int(balance_error > 0.001 * max(float(demand_mw), 1.0))
    bound_violations = int(np.count_nonzero((p_arr < pmin - 1e-6) | (p_arr > pmax + 1e-6)))
    out.update({
        "v_min": float(vm.min()),
        "v_max": float(vm.max()),
        "v_violations": int((low > 0).sum() + (high > 0).sum()),
        "v_gap": float(max(low.max(initial=0.0), high.max(initial=0.0))),
        "overloaded_lines": int((net.res_line.loading_percent.values > 100.0).sum()),
        "trafo_overloaded": int((net.res_trafo.loading_percent.values > 100.0).sum()) if len(net.trafo) else 0,
        "reference_source_p_mw": ref_p,
        "reference_source_q_mvar": ref_q,
        "reference_source_s_mva": ref_s,
        "reference_source_p_violation": ref_p_v,
        "reference_source_q_violation": ref_q_v,
        "reference_source_s_violation": ref_s_v,
        "reference_source_violations": int(ref_p_v) + int(ref_q_v) + int(ref_s_v),
        "reference_source_s_utilization_pct": 100.0 * ref_s / REF_SOURCE_S_MAX_MVA,
        "scheduled_balance_error_mw": balance_error,
        "scheduled_balance_violations": balance_violations,
        "generator_bound_violations": bound_violations,
    })
    return out


def is_pass(row):
    return (
        row.get("converged", False)
        and row.get("v_violations", 999) == 0
        and row.get("overloaded_lines", 999) == 0
        and row.get("trafo_overloaded", 999) == 0
        and row.get("reference_source_violations", 999) == 0
        and row.get("n1_violations", 999) == 0
        and row.get("scheduled_balance_violations", 999) == 0
        and row.get("generator_bound_violations", 999) == 0
    )


def summarize(rows):
    n = len(rows)
    passed = sum(r["passed"] for r in rows)
    successful_costs = [
        float(r["scheduled_controllable_generation_cost"])
        for r in rows
        if r.get("passed") and r.get("scheduled_controllable_generation_cost") is not None
    ]
    return {
        "n": n,
        "passed": int(passed),
        "pass_rate_pct": round(100.0 * passed / max(n, 1), 3),
        "avg_runtime_sec": round(float(np.mean([r["runtime_sec"] for r in rows])), 4) if rows else 0.0,
        "avg_opf_runtime_sec": round(float(np.mean([r["opf_runtime_sec"] for r in rows])), 4) if rows else 0.0,
        "avg_scheduled_controllable_generation_cost": (
            round(float(np.mean(successful_costs)), 6) if successful_costs else None
        ),
        "fail_reasons": dict(Counter(r["reason"] for r in rows if not r["passed"]).most_common(10)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(PROJ_DIR / "case300_gt_v10_dataset.json"))
    ap.add_argument("--split-file", default=str(PROJ_DIR / "rl_scratch_sac" / "split_ids.json"))
    ap.add_argument("--scope", choices=["val", "test"], default="test")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--timeout-sec", type=float, default=60.0)
    ap.add_argument("--opf-vmin", type=float, default=RL_VMIN_PU)
    ap.add_argument("--opf-vmax", type=float, default=RL_VMAX_PU)
    ap.add_argument("--output", default=str(PROJ_DIR / "paper_results" / "route_b_acopf_baseline.json"))
    args = ap.parse_args()

    data, samples = load_samples(args.dataset)
    _tr, val_ids, test_ids = load_split(args.split_file, samples)
    ids = val_ids if args.scope == "val" else test_ids
    if args.max_samples:
        ids = ids[: args.max_samples]
    print(
        f"AC-OPF baseline: samples={len(ids)} scope={args.scope} timeout={args.timeout_sec}s "
        f"opf_v=[{args.opf_vmin},{args.opf_vmax}] verify_v=[{RL_VMIN_PU},{RL_VMAX_PU}]",
        flush=True,
    )
    print("Preparing pandapower case300 base network...", flush=True)
    base_net = prepare_base_net()
    print("Base network ready.", flush=True)
    model = load_linear_model(data, base_net)

    rows = []
    for i, sid in enumerate(ids, 1):
        sample = samples[sid]
        t0 = time.perf_counter()
        print(f"[{i}/{len(ids)}] {sample.get('sample_id')} building OPF...", flush=True)
        net = build_opf_net(base_net, sample, args.opf_vmin, args.opf_vmax)
        print(f"[{i}/{len(ids)}] {sample.get('sample_id')} running OPF...", flush=True)
        ok, opf_info = run_balanced_opf(net, sample["demand_mw"], args.timeout_sec)
        if ok:
            # Do not project the solved OPF dispatch after the fact: that
            # changes the voltage-feasible operating point. Balance is
            # enforced inside run_balanced_opf instead.
            p = np.asarray(net.res_gen.p_mw.values, dtype=np.float64)
            scheduled_cost = scheduled_controllable_generation_cost(sample, p)
            gen_vm, ext_vm = extract_opf_voltage_setpoints(net, args.opf_vmin, args.opf_vmax)
            ac = verify_ac_pf(base_net, p, gen_vm, ext_vm, sample["demand_mw"])
            n1 = linear_n1_check(p, model)
            row = {
                **opf_info,
                **ac,
                **n1,
                "optimized_dispatch_mw": p.tolist(),
                "scheduled_controllable_generation_cost": scheduled_cost,
            }
            if not ac.get("converged", False):
                reason = "acpf_not_converged"
            elif ac.get("v_violations", 999) > 0:
                reason = "voltage_violation"
            elif ac.get("overloaded_lines", 999) > 0 or ac.get("trafo_overloaded", 999) > 0:
                reason = "thermal_overload"
            elif ac.get("reference_source_violations", 999) > 0:
                reason = "reference_source_limit"
            elif n1.get("n1_violations", 999) > 0:
                reason = "n1_violation"
            else:
                reason = "full_sced_pass"
        else:
            row = {**opf_info, "converged": False, "v_violations": 999, "overloaded_lines": 999, "trafo_overloaded": 999, "reference_source_violations": 999, "n1_violations": 999, "optimized_dispatch_mw": None, "scheduled_controllable_generation_cost": None}
            reason = "ac_opf_failed"
        row.update({
            "sid": sid,
            "sample_id": sample.get("sample_id"),
            "demand_gw": float(sample.get("demand_gw", sample["demand_mw"] / 1000.0)),
            "runtime_sec": float(time.perf_counter() - t0),
            "reason": reason,
        })
        row["passed"] = bool(is_pass(row))
        rows.append(row)
        print(f"[{i}/{len(ids)}] {row['sample_id']} {'PASS' if row['passed'] else 'FAIL'} {row['reason']} {row['runtime_sec']:.1f}s")

    result = {
        "dataset": args.dataset,
        "split_file": args.split_file,
        "scope": args.scope,
        "method": "ac_opf_restoration",
        "opf_voltage_limits": [args.opf_vmin, args.opf_vmax],
        "verify_voltage_limits": [RL_VMIN_PU, RL_VMAX_PU],
        "reference_source_limits": {
            "p_mw": [REF_SOURCE_P_MIN_MW, REF_SOURCE_P_MAX_MW],
            "q_mvar": [REF_SOURCE_Q_MIN_MVAR, REF_SOURCE_Q_MAX_MVAR],
            "s_mva_max": REF_SOURCE_S_MAX_MVA,
        },
        "reference_source_objective": {
            "form": "c_ref * P_ref^2",
            "c_ref": 10000.0,
            "excluded_from_reported_scheduled_cost": True,
        },
        "valid_line_outages": int(len(model["valid_contingency_indices"])),
        "summary": {"ac_opf_restoration": summarize(rows)},
        "rows": rows,
    }
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print("Wrote:", args.output)


if __name__ == "__main__":
    main()

"""Traditional restoration baselines for the 300-bus correction benchmark.

Baselines:
1. ac_opf_restoration: solves an AC-OPF feasibility/redispatch problem with
   pandapower, then re-verifies the resulting operating point with the same
   AC-PF plus linear N-1 checker used for GAT+SAC.
2. online_local_search: starts from the GAT warm start and runs a deterministic
   non-learning local repair over voltage setpoints and small active-power
   moves.  It does not read stored rescue solutions.

The stored offline rescue solution is retained only as a reachability
certificate/oracle check and should not be reported as an online deployable
baseline.

Both baselines are evaluated with the same AC-PF + linear N-1 verifier used by
the GAT+SAC pipeline.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict

import numpy as np

CODE_DIR = pathlib.Path(__file__).resolve().parent
PROJ_DIR = CODE_DIR.parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(1, str(PROJ_DIR))

from eval_route_b_pipeline import is_pass, verify_current
from train_rlv35_complete import (
    EXT_VM_MAX,
    EXT_VM_MIN,
    GEN_VM_MAX,
    GEN_VM_MIN,
    RL_VMAX_PU,
    RL_VMIN_PU,
    SCEDEnvV21,
    split_samples_stratified,
)


def _load_samples(path: pathlib.Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data["samples"] if isinstance(data, dict) and "samples" in data else data
    samples = [s for s in samples if s.get("is_complete_sced", True)]
    return data, samples


def _load_split(path: pathlib.Path, samples):
    if path.exists():
        split = json.loads(path.read_text(encoding="utf-8"))
        return split["train_ids"], split["val_ids"], split["test_ids"]
    return split_samples_stratified(samples)


def _dispatch_label(sample):
    if "dispatch_68" in sample:
        return np.asarray(sample["dispatch_68"], dtype=np.float64)
    return np.asarray(sample["dispatch"], dtype=np.float64)


def _project_box_sum(x, lo, hi, target_sum, max_iter=100):
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


def _rescue_dispatch(sample):
    if sample.get("rescue_dispatch_68") is not None:
        return np.asarray(sample["rescue_dispatch_68"], dtype=np.float64)
    if sample.get("rescue_dispatch") is not None:
        return np.asarray(sample["rescue_dispatch"], dtype=np.float64)
    return None


def _base_vm(env):
    gen = np.asarray(env.dcnet.net_base.gen.vm_pu.values, dtype=np.float64).copy()
    ext = np.asarray(
        env.dcnet.net_base.ext_grid.vm_pu.values if len(env.dcnet.net_base.ext_grid) else [1.0],
        dtype=np.float64,
    ).copy()
    return gen, ext


def _fit_mean_deltas(samples, train_ids, env):
    base_gen, base_ext = _base_vm(env)
    global_items = []
    by_demand = defaultdict(list)
    for sid in train_ids:
        s = samples[sid]
        p0 = _dispatch_label(s)
        pr = _rescue_dispatch(s)
        if pr is None or s.get("rescue_gen_vm_pu") is None or s.get("rescue_ext_grid_vm_pu") is None:
            continue
        item = {
            "dp": pr - p0,
            "dgen": np.asarray(s["rescue_gen_vm_pu"], dtype=np.float64) - base_gen,
            "dext": np.asarray(s["rescue_ext_grid_vm_pu"], dtype=np.float64) - base_ext,
        }
        global_items.append(item)
        by_demand[round(float(s.get("demand_gw", s["demand_mw"] / 1000.0)), 3)].append(item)

    def mean_item(items):
        if not items:
            return {
                "dp": np.zeros_like(_dispatch_label(samples[train_ids[0]]), dtype=np.float64),
                "dgen": np.zeros_like(base_gen),
                "dext": np.zeros_like(base_ext),
            }
        return {
            "dp": np.mean([x["dp"] for x in items], axis=0),
            "dgen": np.mean([x["dgen"] for x in items], axis=0),
            "dext": np.mean([x["dext"] for x in items], axis=0),
        }

    return mean_item(global_items), {k: mean_item(v) for k, v in by_demand.items()}


def _apply_candidate(env, sample, sid, mode, global_mean, demand_mean):
    env.reset(sample_idx=sid)
    base_gen, base_ext = _base_vm(env)
    if mode == "offline_rescue_search":
        pr = _rescue_dispatch(sample)
        if pr is None:
            raise RuntimeError(f"sample {sid} has no rescue dispatch")
        env.current_power = pr.copy()
        env.current_gen_vm_pu = np.asarray(sample["rescue_gen_vm_pu"], dtype=np.float64).copy()
        env.current_ext_grid_vm_pu = np.asarray(sample["rescue_ext_grid_vm_pu"], dtype=np.float64).copy()
    elif mode in {"global_mean_rescue_delta", "demand_mean_rescue_delta"}:
        p0 = _dispatch_label(sample)
        key = round(float(sample.get("demand_gw", sample["demand_mw"] / 1000.0)), 3)
        fit = demand_mean.get(key, global_mean) if mode == "demand_mean_rescue_delta" else global_mean
        env.current_power = p0 + fit["dp"]
        env.current_power = _project_box_sum(env.current_power, env.PMIN, env.PMAX * 0.99, sample["demand_mw"])
        env.current_gen_vm_pu = base_gen + fit["dgen"]
        env.current_ext_grid_vm_pu = base_ext + fit["dext"]
    else:
        raise ValueError(mode)


def _build_case300_acopf_net(env, sample):
    import pandapower as pp
    import pandapower.networks as pn

    net = pn.case300()
    net.bus.min_vm_pu = RL_VMIN_PU
    net.bus.max_vm_pu = RL_VMAX_PU

    base_load = float(net.load.p_mw.sum())
    if base_load > 0:
        scale = float(sample["demand_mw"]) / base_load
        net.load.p_mw = net.load.p_mw * scale
        net.load.q_mvar = net.load.q_mvar * scale

    n_gen = min(len(net.gen), len(env.PMIN))
    net.gen["controllable"] = True
    net.gen["min_p_mw"] = net.gen["min_p_mw"].astype(float)
    net.gen["max_p_mw"] = net.gen["max_p_mw"].astype(float)
    net.gen["p_mw"] = net.gen["p_mw"].astype(float)
    net.gen.loc[net.gen.index[:n_gen], "min_p_mw"] = np.asarray(env.PMIN[:n_gen], dtype=np.float64)
    net.gen.loc[net.gen.index[:n_gen], "max_p_mw"] = np.asarray(env.PMAX[:n_gen], dtype=np.float64) * 0.99
    gen_vm = np.asarray(sample.get("gen_vm_pu") or net.gen.vm_pu.values, dtype=np.float64)
    if len(gen_vm) != len(net.gen):
        gen_vm = np.asarray(net.gen.vm_pu.values, dtype=np.float64)
    net.gen.loc[net.gen.index[:n_gen], "vm_pu"] = np.clip(gen_vm[:n_gen], GEN_VM_MIN, GEN_VM_MAX)
    p_init = _dispatch_label(sample)
    net.gen.loc[net.gen.index[:n_gen], "p_mw"] = np.clip(p_init[:n_gen], env.PMIN[:n_gen], env.PMAX[:n_gen] * 0.99)

    if len(net.ext_grid):
        net.ext_grid["controllable"] = True
        if "min_p_mw" not in net.ext_grid:
            net.ext_grid["min_p_mw"] = -1e4
        if "max_p_mw" not in net.ext_grid:
            net.ext_grid["max_p_mw"] = 1e4
        if "min_q_mvar" not in net.ext_grid:
            net.ext_grid["min_q_mvar"] = -1e4
        if "max_q_mvar" not in net.ext_grid:
            net.ext_grid["max_q_mvar"] = 1e4
        ext_vm = sample.get("ext_grid_vm_pu") or [float(net.ext_grid.vm_pu.iloc[0])]
        net.ext_grid.loc[net.ext_grid.index[0], "vm_pu"] = float(np.clip(ext_vm[0], EXT_VM_MIN, EXT_VM_MAX))

    if hasattr(net, "poly_cost"):
        net.poly_cost.drop(net.poly_cost.index, inplace=True)
    cp2 = np.asarray(sample.get("sample_cp2", np.zeros(n_gen)), dtype=np.float64)
    cp1 = np.asarray(sample.get("sample_cp1", np.ones(n_gen) * 30.0), dtype=np.float64)
    cp0 = np.asarray(sample.get("sample_cp0", np.zeros(n_gen)), dtype=np.float64)
    for i in range(n_gen):
        pp.create_poly_cost(net, i, "gen", cp1_eur_per_mw=float(cp1[i]), cp2_eur_per_mw2=float(cp2[i]), cp0_eur=float(cp0[i]))
    if len(net.ext_grid):
        pp.create_poly_cost(net, int(net.ext_grid.index[0]), "ext_grid", cp1_eur_per_mw=1e3, cp2_eur_per_mw2=0.0, cp0_eur=0.0)
    return net


def _run_acopf_restoration(env, sample, sid, timeout_sec):
    import pandapower as pp

    env.reset(sample_idx=sid)
    net = _build_case300_acopf_net(env, sample)
    result = {"opf_converged": False, "opf_error": None, "opf_runtime_sec": None}
    holder = {"error": None}

    def _inner():
        try:
            pp.runopp(net, calculate_voltage_angles=True, init="pf", numba=False, verbose=False)
        except Exception as exc:
            holder["error"] = repr(exc)
            holder["traceback"] = traceback.format_exc()

    t0 = time.perf_counter()
    th = threading.Thread(target=_inner, daemon=True)
    th.start()
    th.join(timeout=timeout_sec)
    result["opf_runtime_sec"] = float(time.perf_counter() - t0)
    if th.is_alive():
        result["opf_error"] = f"timeout_after_{timeout_sec}s"
        return result
    if holder["error"] is not None:
        result["opf_error"] = holder["error"]
        result["opf_traceback"] = holder.get("traceback")
        return result
    if not bool(getattr(net, "OPF_converged", False)):
        result["opf_error"] = "OPF_not_converged"
        return result

    result["opf_converged"] = True
    env.current_power = np.asarray(net.res_gen.p_mw.values[: env.N], dtype=np.float64).copy()
    env.current_power = _project_box_sum(env.current_power, env.PMIN, env.PMAX * 0.99, sample["demand_mw"])
    env.current_gen_vm_pu = np.asarray(net.gen.vm_pu.values[: env.N], dtype=np.float64).copy()
    if len(net.ext_grid):
        env.current_ext_grid_vm_pu = np.asarray([float(net.ext_grid.vm_pu.iloc[0])], dtype=np.float64)
    return result


def _candidate_pass(env):
    info, verify_dt = verify_current(env)
    return is_pass(info), info, verify_dt


def _run_online_local_search(env, sample, sid, max_trials):
    env.reset(sample_idx=sid)
    p0 = np.asarray(env.gat_power, dtype=np.float64).copy()
    base_gen, base_ext = _base_vm(env)
    env.current_power = p0.copy()
    env.current_gen_vm_pu = base_gen.copy()
    env.current_ext_grid_vm_pu = base_ext.copy()

    # Deterministic repair grid: voltage support first, then low-dimensional
    # active-power balancing moves over sensitivity-selected key generators.
    gen_steps = [0.0, 0.004, 0.008, 0.012, -0.004]
    ext_steps = [0.0, 0.004, 0.008, -0.004]
    p_steps = [0.0, 10.0, -10.0, 20.0, -20.0]
    key = getattr(env, "_key_gens", np.arange(min(20, env.N)))
    key = np.asarray(key[: min(8, len(key))], dtype=int)
    trials = 0
    best = None
    best_score = 1e18
    t0 = time.perf_counter()

    for vg in gen_steps:
        for ve in ext_steps:
            for pg in p_steps:
                for kg in ([-1] + key.tolist()):
                    if trials >= max_trials:
                        return {
                            "search_trials": trials,
                            "search_runtime_sec": float(time.perf_counter() - t0),
                            "found": False,
                            "best_info": best,
                        }
                    env.current_power = p0.copy()
                    if kg >= 0 and pg != 0.0:
                        env.current_power[kg] += pg
                        env.current_power = _project_box_sum(env.current_power, env.PMIN, env.PMAX * 0.99, sample["demand_mw"])
                    env.current_gen_vm_pu = np.clip(base_gen + vg, GEN_VM_MIN, GEN_VM_MAX)
                    env.current_ext_grid_vm_pu = np.clip(base_ext + ve, EXT_VM_MIN, EXT_VM_MAX)
                    ok, info, verify_dt = _candidate_pass(env)
                    trials += 1
                    score = (
                        (0 if info.get("converged") else 1e5)
                        + 1000.0 * float(info.get("v_violations", 999))
                        + 10.0 * float(info.get("n1_violations", 999))
                        + 100.0 * float(info.get("overloaded_lines", 999))
                        + 1e5 * float(info.get("trafo_overloaded", 999))
                    )
                    if score < best_score:
                        best_score = score
                        best = dict(info)
                    if ok:
                        return {
                            "search_trials": trials,
                            "search_runtime_sec": float(time.perf_counter() - t0),
                            "found": True,
                            "info": info,
                            "verify_runtime_sec": verify_dt,
                        }
    return {
        "search_trials": trials,
        "search_runtime_sec": float(time.perf_counter() - t0),
        "found": False,
        "best_info": best,
    }


def _run(env, sample, sid, mode, global_mean, demand_mean):
    t0 = time.perf_counter()
    if mode == "ac_opf_restoration":
        opf = _run_acopf_restoration(env, sample, sid, timeout_sec=_run.acopf_timeout_sec)
        if not opf.get("opf_converged"):
            return {
                "passed": False,
                "reason": "ac_opf_failed",
                "converged": False,
                "v_violations": 999,
                "v_gap": 999.0,
                "overloaded_lines": 999,
                "trafo_overloaded": 999,
                "n1_violations": 999,
                "verify_runtime_sec": 0.0,
                "runtime_sec": float(time.perf_counter() - t0),
                "opf_converged": False,
                "opf_error": opf.get("opf_error"),
                "opf_traceback": opf.get("opf_traceback"),
                "opf_runtime_sec": opf.get("opf_runtime_sec"),
                "stored_rescue_attempts": None,
                "stored_rescue_time_sec": None,
            }
    elif mode == "online_local_search":
        search = _run_online_local_search(env, sample, sid, max_trials=_run.local_search_max_trials)
        if not search.get("found"):
            best = search.get("best_info") or {}
            return {
                "passed": False,
                "reason": "local_search_failed",
                "converged": bool(best.get("converged", False)),
                "v_violations": int(best.get("v_violations", 999)),
                "v_gap": float(best.get("v_violation_max_pu", 0.0) or 0.0),
                "overloaded_lines": int(best.get("overloaded_lines", 999)),
                "trafo_overloaded": int(best.get("trafo_overloaded", 999)),
                "n1_violations": int(best.get("n1_violations", 999)),
                "verify_runtime_sec": 0.0,
                "runtime_sec": float(time.perf_counter() - t0),
                "search_trials": int(search.get("search_trials", 0)),
                "search_runtime_sec": float(search.get("search_runtime_sec", 0.0)),
                "opf_converged": None,
                "opf_error": None,
                "opf_runtime_sec": None,
                "stored_rescue_attempts": None,
                "stored_rescue_time_sec": None,
            }
        info = search["info"]
        verify_dt = search.get("verify_runtime_sec", 0.0)
        opf = {}
    else:
        _apply_candidate(env, sample, sid, mode, global_mean, demand_mean)
        opf = {}
        info, verify_dt = verify_current(env)
    total_dt = time.perf_counter() - t0
    evidence = sample.get("rl_rescue_pv_evidence") or sample.get("route_b_rescue_evidence") or {}
    return {
        "passed": bool(is_pass(info)),
        "reason": info.get("reason"),
        "converged": bool(info.get("converged", False)),
        "v_violations": int(info.get("v_violations", 999)),
        "v_gap": float(info.get("v_violation_max_pu", 0.0) or 0.0),
        "overloaded_lines": int(info.get("overloaded_lines", 999)),
        "trafo_overloaded": int(info.get("trafo_overloaded", 999)),
        "n1_violations": int(info.get("n1_violations", 999)),
        "verify_runtime_sec": float(verify_dt),
        "runtime_sec": float(total_dt),
        "opf_converged": opf.get("opf_converged"),
        "opf_error": opf.get("opf_error"),
        "opf_traceback": opf.get("opf_traceback"),
        "opf_runtime_sec": opf.get("opf_runtime_sec"),
        "search_trials": int(search.get("search_trials", 0)) if mode == "online_local_search" else None,
        "search_runtime_sec": float(search.get("search_runtime_sec", 0.0)) if mode == "online_local_search" else None,
        "stored_rescue_attempts": evidence.get("attempts", evidence.get("attempt", evidence.get("acpf_trials"))),
        "stored_rescue_time_sec": evidence.get("elapsed_sec", evidence.get("runtime_sec")),
    }


def _summarize(rows):
    out = {}
    for mode in sorted({r["mode"] for r in rows}):
        rs = [r for r in rows if r["mode"] == mode]
        out[mode] = {
            "n": len(rs),
            "passed": int(sum(r["passed"] for r in rs)),
            "pass_rate_pct": round(100.0 * sum(r["passed"] for r in rs) / max(len(rs), 1), 3),
            "avg_verify_runtime_sec": round(float(np.mean([r["verify_runtime_sec"] for r in rs])), 4),
            "avg_runtime_sec": round(float(np.mean([r["runtime_sec"] for r in rs])), 4),
            "avg_stored_rescue_time_sec": round(float(np.mean([r["stored_rescue_time_sec"] for r in rs if r["stored_rescue_time_sec"] is not None])), 4)
            if any(r["stored_rescue_time_sec"] is not None for r in rs) else None,
            "avg_stored_rescue_attempts": round(float(np.mean([r["stored_rescue_attempts"] for r in rs if r["stored_rescue_attempts"] is not None])), 3)
            if any(r["stored_rescue_attempts"] is not None for r in rs) else None,
            "fail_reasons": dict(Counter(r["reason"] for r in rs if not r["passed"]).most_common(8)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(PROJ_DIR / "case300_gt_v10_dataset.json"))
    ap.add_argument("--split-file", default=str(PROJ_DIR / "rl_scratch_sac" / "split_ids.json"))
    ap.add_argument("--scope", choices=["val", "test"], default="test")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--acopf-timeout-sec", type=float, default=60.0)
    ap.add_argument("--local-search-max-trials", type=int, default=80)
    ap.add_argument("--modes", default="online_local_search")
    ap.add_argument("--output", default=str(PROJ_DIR / "paper_results" / "route_b_restoration_baselines.json"))
    args = ap.parse_args()

    data, samples = _load_samples(pathlib.Path(args.dataset))
    train_ids, val_ids, test_ids = _load_split(pathlib.Path(args.split_file), samples)
    eval_ids = val_ids if args.scope == "val" else test_ids
    if args.max_samples:
        eval_ids = eval_ids[: args.max_samples]

    env = SCEDEnvV21(samples, eval_ids, dataset_meta=data if isinstance(data, dict) else None)
    global_mean, demand_mean = _fit_mean_deltas(samples, train_ids, env)
    _run.acopf_timeout_sec = float(args.acopf_timeout_sec)
    _run.local_search_max_trials = int(args.local_search_max_trials)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    rows = []
    for i, sid in enumerate(eval_ids, 1):
        sample = samples[sid]
        for mode in modes:
            result = _run(env, sample, sid, mode, global_mean, demand_mean)
            rows.append({
                "sid": int(sid),
                "sample_id": sample.get("sample_id", sample.get("id")),
                "demand_gw": float(sample.get("demand_gw", sample["demand_mw"] / 1000.0)),
                "mode": mode,
                **result,
            })
        print(f"[{i}/{len(eval_ids)}] {sample.get('sample_id')} {float(sample.get('demand_gw', sample['demand_mw']/1000.0)):g}GW")

    result = {
        "dataset": args.dataset,
        "split_file": args.split_file,
        "scope": args.scope,
        "n_eval": len(eval_ids),
        "notes": {
            "reachability_certificate": "Stored offline rescue solutions are used only to certify that screened samples have reachable feasible points; they are not treated as online baselines.",
            "ac_opf_restoration": "Pandapower AC-OPF restoration followed by independent AC-PF plus linear N-1 verification.",
            "online_local_search": "Deterministic non-learning local search from the GAT warm start; no stored rescue dispatch or voltage setpoints are read.",
            "offline_rescue_search": "Optional oracle/certificate mode only, not an online deployable baseline.",
            "demand_mean_rescue_delta": "Optional data-driven rule baseline; not used as the traditional strong baseline.",
        },
        "summary": _summarize(rows),
        "rows": rows,
    }
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print("Wrote:", args.output)


if __name__ == "__main__":
    main()

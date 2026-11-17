"""Screen case39/case118 samples for Route-B reachability.

Retained samples must satisfy the linear SCED/N-1 label check and have evidence
that a nearby P+V operating point passes strict AC-PF thermal/voltage checks
while preserving the same linear N-1 check.

The stored rescue_* fields are audit evidence only. GAT labels remain the
original linear-SCED active-power dispatch in sample["dispatch"].
"""
import argparse
import copy
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(r".")
sys.path.insert(0, str(ROOT))


def project_box_sum(x, lo, hi, target, n_iter=80):
    x = np.asarray(x, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    target = float(target)
    if target <= lo.sum():
        return lo.copy()
    if target >= hi.sum():
        return hi.copy()
    low = float(np.min(x - hi)) - abs(target)
    high = float(np.max(x - lo)) + abs(target)
    for _ in range(n_iter):
        mid = 0.5 * (low + high)
        y = np.clip(x - mid, lo, hi)
        if y.sum() > target:
            low = mid
        else:
            high = mid
    return np.clip(x - 0.5 * (low + high), lo, hi)


def zero_sum_unit(vec):
    v = np.asarray(vec, dtype=np.float64).copy()
    v -= v.mean()
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1e-10:
        return None
    return v / n


def load_case(case_name):
    if case_name == "39":
        import generate_case39_gt_v1 as gen
        from pandapower.networks import case39
        from case39_route_b_utils import CASE39_VMIN, CASE39_VMAX

        dataset = ROOT / "case39_routeb_dataset.json"
        if not dataset.exists():
            dataset = ROOT / "case39_gt_v1_dataset.json"
        return {
            "case": "39",
            "dataset": dataset,
            "network_builder": case39,
            "dcnet_cls": gen.Case39Network,
            "check_n1": lambda p, dc: gen.check_n1(p, dc)[0],
            "output_dataset": ROOT / "case39_routeb_rescuable_dataset.json",
            "output_report": ROOT / "paper_results" / "case39_routeb_rescuable_screen.json",
            "default_vmin": CASE39_VMIN,
            "default_vmax": CASE39_VMAX,
        }
    if case_name == "118":
        import generate_case118_gt_v1 as gen
        from pandapower.networks import case118

        return {
            "case": "118",
            "dataset": ROOT / "case118_gt_v1_dataset.json",
            "network_builder": case118,
            "dcnet_cls": gen.Case118Network,
            "check_n1": lambda p, dc: gen.check_n1_security(p, dc)[0],
            "output_dataset": ROOT / "case118_routeb_rescuable_dataset.json",
            "output_report": ROOT / "paper_results" / "case118_routeb_rescuable_screen.json",
        }
    raise ValueError(case_name)


def raw_linear_sced_ok(sample, p, dcnet, check_n1):
    demand = float(sample["demand_mw"])
    p = np.asarray(p, dtype=np.float64)
    if abs(p.sum() - demand) > 1e-3 * max(1.0, demand):
        return False, "balance"
    if np.any(p < np.asarray(dcnet.gen_pmin) - 1e-6) or np.any(p > np.asarray(dcnet.gen_pmax) + 1e-6):
        return False, "gen_bounds"
    n_viol = int(check_n1(p, dcnet))
    if n_viol != 0:
        return False, f"linear_n1={n_viol}"
    return True, "ok"


def strict_acpf(cfg, p, demand, gen_vm=None, ext_vm=None, vmin=0.93, vmax=1.10):
    import pandapower as pp
    if cfg["case"] == "39":
        from case39_route_b_utils import apply_case39_route_b_limits

    net = cfg["network_builder"]()
    if cfg["case"] == "39":
        apply_case39_route_b_limits(net)
    p = np.asarray(p, dtype=np.float64)
    net.gen.p_mw = np.clip(p, net.gen.min_p_mw.values, net.gen.max_p_mw.values)
    if gen_vm is not None:
        net.gen.vm_pu = np.asarray(gen_vm, dtype=np.float64)[: len(net.gen)]
    if ext_vm is not None and len(net.ext_grid):
        net.ext_grid.vm_pu = float(np.asarray(ext_vm).reshape(-1)[0])
    scale = float(demand) / float(net.load.p_mw.sum())
    net.load.p_mw *= scale
    net.load.q_mvar *= scale
    net.bus.min_vm_pu = vmin
    net.bus.max_vm_pu = vmax
    try:
        pp.runpp(net, numba=False, enforce_q_lims=True, max_iteration=80, tolerance_mva=1e-6)
    except Exception as exc:
        return {"passed": False, "reason": f"acpf_error:{type(exc).__name__}", "converged": False}
    if not bool(getattr(net, "converged", False)):
        return {"passed": False, "reason": "acpf_not_converged", "converged": False}
    vm = np.asarray(net.res_bus.vm_pu.values, dtype=np.float64)
    low = vm < vmin - 1e-9
    high = vm > vmax + 1e-9
    line_over = int((net.res_line.loading_percent.values > 100.0 + 1e-9).sum())
    trafo_over = int((net.res_trafo.loading_percent.values > 100.0 + 1e-9).sum()) if len(net.trafo) else 0
    vv = int(low.sum() + high.sum())
    passed = vv == 0 and line_over == 0 and trafo_over == 0
    if passed:
        reason = "strict_pass"
    elif vv:
        gap = max(float(np.maximum(vmin - vm, 0).max()), float(np.maximum(vm - vmax, 0).max()))
        reason = f"voltage:vv={vv}:gap={gap:.4f}"
    elif line_over or trafo_over:
        reason = f"thermal:line={line_over}:trafo={trafo_over}"
    else:
        reason = "unknown"
    return {
        "passed": passed,
        "reason": reason,
        "converged": True,
        "v_min": float(vm.min()),
        "v_max": float(vm.max()),
        "v_violations": vv,
        "line_overloaded": line_over,
        "trafo_overloaded": trafo_over,
        "max_line_loading_pct": float(np.max(net.res_line.loading_percent.values)) if len(net.line) else 0.0,
    }


def directions(p0, dcnet):
    out = []

    def add(v):
        u = zero_sum_unit(v)
        if u is None:
            return
        for old in out:
            if abs(float(np.dot(u, old))) > 0.92:
                return
        out.append(u)

    pmax = np.asarray(dcnet.gen_pmax, dtype=np.float64)
    pmin = np.asarray(dcnet.gen_pmin, dtype=np.float64)
    cp1 = np.asarray(getattr(dcnet, "cp1", np.linspace(30, 20, len(p0))), dtype=np.float64)
    add(cp1 - cp1.mean())
    add(-(cp1 - cp1.mean()))
    add((pmax - p0) - (pmax - p0).mean())
    add(-((p0 - pmin) - (p0 - pmin).mean()))
    try:
        sens = np.sum(np.abs(np.asarray(dcnet.S, dtype=np.float64)), axis=0)
        add(sens - sens.mean())
        add(-(sens - sens.mean()))
        _u, _s, vh = np.linalg.svd(np.asarray(dcnet.S, dtype=np.float64), full_matrices=False)
        for j in range(min(4, vh.shape[0])):
            add(vh[j])
            add(-vh[j])
    except Exception:
        pass
    return out


def make_candidate(p0, lo, hi, demand, dirs, rng, attempt, step_mw, max_delta_mw):
    if attempt == 0:
        return project_box_sum(p0, lo, hi, demand)
    if dirs and attempt <= 3 * len(dirs):
        band = (attempt - 1) // len(dirs)
        direction = dirs[(attempt - 1) % len(dirs)]
        sign = 1.0 if band % 2 == 0 else -1.0
        mag = step_mw * (1.0 + 0.5 * band)
        delta = sign * mag * direction
    else:
        u = zero_sum_unit(rng.normal(size=len(p0)))
        if u is None:
            u = np.zeros_like(p0)
        delta = min(max_delta_mw, step_mw * rng.gamma(2.0, 1.3)) * u
    delta = np.clip(delta, -max_delta_mw, max_delta_mw)
    return project_box_sum(p0 + delta, lo, hi, demand)


def voltage_grid(n_gen, vmin, vmax):
    """Small set of representative voltage profiles.

    Avoid a Cartesian voltage sweep here.  Reachability screening only needs an
    existence witness, and exhaustive P/V search can require thousands of AC-PF
    solves per sample on small cases.
    """
    pairs = [
        (1.00, 1.00),
        (1.02, 1.02),
        (1.04, 1.04),
        (min(1.06, vmax), min(1.06, vmax)),
        (1.04, 1.02),
        (1.02, 1.04),
        (0.98, 1.00),
        (1.00, 0.98),
    ]
    seen = set()
    for gv, ev in pairs:
        gv = float(np.clip(gv, vmin, vmax))
        ev = float(np.clip(ev, vmin, vmax))
        key = (round(gv, 5), round(ev, 5))
        if key in seen:
            continue
        seen.add(key)
        yield np.full(n_gen, gv, dtype=np.float64), np.array([ev], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=["39", "118"])
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--attempts", type=int, default=50)
    ap.add_argument("--max-acpf-per-sample", type=int, default=120)
    ap.add_argument("--step-mw", type=float, default=40.0)
    ap.add_argument("--max-delta-mw", type=float, default=300.0)
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--include-raw-pass", action="store_true")
    ap.add_argument("--seed", type=int, default=20260717)
    ap.add_argument("--print-every", type=int, default=5)
    args = ap.parse_args()

    cfg = load_case(args.case)
    if args.vmin is None:
        args.vmin = cfg.get("default_vmin", 0.93)
    if args.vmax is None:
        args.vmax = cfg.get("default_vmax", 1.10)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    with open(cfg["dataset"], "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = list(data["samples"])
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    print(f"Building linear model for case{args.case}...")
    dcnet = cfg["dcnet_cls"]()
    lo = np.asarray(dcnet.gen_pmin, dtype=np.float64)
    hi = np.asarray(dcnet.gen_pmax, dtype=np.float64) * 0.99
    retained = []
    records = []
    by_gw = defaultdict(lambda: {"n": 0, "kept": 0})
    failures = Counter()
    t0 = time.time()
    print(f"Screen case{args.case}: {len(samples)} samples from {cfg['dataset']}")

    for idx, sample in enumerate(samples, 1):
        ts = time.time()
        demand = float(sample["demand_mw"])
        gw = f"{float(sample.get('demand_gw', demand / 1000.0)):g}"
        by_gw[gw]["n"] += 1
        p0 = project_box_sum(np.asarray(sample["dispatch"], dtype=np.float64), lo, hi, demand)
        lin_ok, lin_reason = raw_linear_sced_ok(sample, p0, dcnet, cfg["check_n1"])
        found = None
        raw = strict_acpf(cfg, p0, demand, vmin=args.vmin, vmax=args.vmax) if lin_ok else {"passed": False, "reason": lin_reason}
        acpf_tries = 1 if lin_ok else 0
        if not lin_ok:
            failures[lin_reason] += 1
        elif raw["passed"] and args.include_raw_pass:
            found = {"dispatch": p0, "gen_vm": None, "ext_vm": None, "attempt": 0, "ac": raw, "raw_pass": True}
        elif not raw["passed"]:
            dirs = directions(p0, dcnet)
            last = raw
            for attempt in range(args.attempts):
                pc = make_candidate(p0, lo, hi, demand, dirs, rng, attempt, args.step_mw, args.max_delta_mw)
                lin2, lin2_reason = raw_linear_sced_ok(sample, pc, dcnet, cfg["check_n1"])
                if not lin2:
                    last = {"passed": False, "reason": lin2_reason}
                    continue
                for gen_vm, ext_vm in voltage_grid(len(p0), args.vmin, args.vmax):
                    if acpf_tries >= args.max_acpf_per_sample:
                        break
                    ac = strict_acpf(cfg, pc, demand, gen_vm, ext_vm, args.vmin, args.vmax)
                    acpf_tries += 1
                    last = ac
                    if ac["passed"]:
                        found = {
                            "dispatch": pc,
                            "gen_vm": gen_vm,
                            "ext_vm": ext_vm,
                            "attempt": attempt,
                            "ac": ac,
                            "raw_pass": False,
                        }
                        break
                if found is not None:
                    break
                if acpf_tries >= args.max_acpf_per_sample:
                    break
            if found is None:
                failures[last.get("reason", "unknown")] += 1
        else:
            failures["raw_pass_excluded"] += 1

        if found is not None:
            out = copy.deepcopy(sample)
            out["linear_sced_feasible"] = True
            out["linear_n1_pass"] = True
            out["linear_base_pass"] = True
            out["route_b_rescuable"] = True
            out["rl_rescuable_pv"] = True
            out["is_complete_sced"] = True
            out["label_dispatch_field"] = "dispatch"
            out["rescue_dispatch_field"] = "rescue_dispatch"
            out["raw_strict_acpf_pass_before_rescue"] = bool(raw.get("passed", False))
            out["raw_strict_acpf_result_before_rescue"] = raw
            out["raw_strict_acpf_failures_before_rescue"] = [] if raw.get("passed", False) else [raw.get("reason", "unknown")]
            out["raw_strict_acpf_metrics_before_rescue"] = raw
            out["rescue_dispatch"] = np.asarray(found["dispatch"], dtype=float).tolist()
            out[f"rescue_dispatch_{len(p0)}"] = np.asarray(found["dispatch"], dtype=float).tolist()
            out["rescue_gen_vm_pu"] = None if found["gen_vm"] is None else np.asarray(found["gen_vm"], dtype=float).tolist()
            out["rescue_ext_grid_vm_pu"] = None if found["ext_vm"] is None else np.asarray(found["ext_vm"], dtype=float).tolist()
            out["dispatch_label_not_rescue"] = True
            out["route_b_rescue_evidence"] = {
                "attempt": int(found["attempt"]),
                "raw_pass": bool(found["raw_pass"]),
                "delta_l1_mw": float(np.abs(found["dispatch"] - p0).sum()),
                "delta_linf_mw": float(np.abs(found["dispatch"] - p0).max()),
                "strict_acpf": found["ac"],
            }
            out["rl_rescue_pv_evidence"] = out["route_b_rescue_evidence"]
            retained.append(out)
            by_gw[gw]["kept"] += 1

        records.append({
            "id": sample.get("id", sample.get("sample_id", idx - 1)),
            "demand_gw": gw,
            "linear_sced_ok": lin_ok,
            "raw_pass": bool(raw.get("passed", False)),
            "kept": found is not None,
            "reason": "kept" if found is not None else (raw.get("reason", lin_reason) if lin_ok else lin_reason),
            "acpf_tries": acpf_tries,
            "elapsed_sec": round(time.time() - ts, 3),
        })
        if idx == 1 or idx % args.print_every == 0 or idx == len(samples):
            print(
                f"[{idx}/{len(samples)}] kept={len(retained)}/{idx} "
                f"last={records[-1]['id']} {'YES' if found is not None else 'NO'} "
                f"{records[-1]['reason']} acpf={acpf_tries} {time.time()-ts:.1f}s",
                flush=True,
            )

    summary = {
        "case": f"case{args.case}",
        "source_dataset": str(cfg["dataset"]),
        "screened": len(samples),
        "kept": len(retained),
        "kept_rate_pct": round(100.0 * len(retained) / max(len(samples), 1), 3),
        "vmin": args.vmin,
        "vmax": args.vmax,
        "attempts": args.attempts,
        "max_acpf_per_sample": args.max_acpf_per_sample,
        "step_mw": args.step_mw,
        "max_delta_mw": args.max_delta_mw,
        "elapsed_sec": round(time.time() - t0, 3),
        "by_demand_gw": {
            k: {"n": v["n"], "kept": v["kept"], "rate_pct": round(100.0 * v["kept"] / max(v["n"], 1), 2)}
            for k, v in sorted(by_gw.items(), key=lambda kv: float(kv[0]))
        },
        "top_failures": failures.most_common(20),
    }
    out_data = dict(data)
    out_data["samples"] = retained
    out_data["version"] = f"route_b_case{args.case}_linear_sced_rescuable_pv"
    out_data["label_generation_method"] = "linear-SCED-QP-LODF"
    out_data["operation_verification_method"] = "strict AC-PF with enforce_q_lims=True"
    out_data["contingency_verification_method"] = "linear LODF N-1 for labels and rescue evidence"
    out_data["route_b_alignment"] = {
        "gat_label_field": "dispatch",
        "rescue_fields_are_audit_only": True,
        "requires_linear_sced_feasible": True,
        "requires_raw_strict_acpf_fail": not args.include_raw_pass,
        "requires_strict_acpf_rescue": True,
    }
    out_data["route_b_rescuable_screen_summary"] = summary
    with open(cfg["output_dataset"], "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    cfg["output_report"].parent.mkdir(parents=True, exist_ok=True)
    with open(cfg["output_report"], "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Wrote dataset: {cfg['output_dataset']}")
    print(f"Wrote report: {cfg['output_report']}")


if __name__ == "__main__":
    main()

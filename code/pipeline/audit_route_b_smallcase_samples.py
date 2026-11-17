"""Audit IEEE 39/118 samples under the current Route-B strict AC-PF protocol.

This script does not train any model. It checks whether existing linear-SCED
labels are meaningful for the Route-B story:
  linear N-1 SCED label -> strict AC-PF may fail -> RL can learn correction.
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(r".")
OUT_DIR = ROOT / "paper_results"


def load_case(case_name):
    if case_name == "39":
        from pandapower.networks import case39
        from case39_route_b_utils import CASE39_VMIN, CASE39_VMAX

        return {
            "case": "39",
            "network_builder": case39,
            "dataset": ROOT / "case39_routeb_rescuable_dataset.json"
            if (ROOT / "case39_routeb_rescuable_dataset.json").exists()
            else (
                ROOT / "case39_routeb_dataset.json"
                if (ROOT / "case39_routeb_dataset.json").exists()
                else ROOT / "case39_gt_v1_dataset.json"
            ),
            "dispatch_key": "dispatch",
            "default_vmin": CASE39_VMIN,
            "default_vmax": CASE39_VMAX,
        }
    if case_name == "118":
        from pandapower.networks import case118

        return {
            "case": "118",
            "network_builder": case118,
            "dataset": ROOT / "case118_routeb_rescuable_dataset.json"
            if (ROOT / "case118_routeb_rescuable_dataset.json").exists()
            else ROOT / "case118_gt_v1_dataset.json",
            "dispatch_key": "dispatch",
        }
    raise ValueError(f"Unsupported case: {case_name}")


def demand_key(sample):
    return f"{float(sample.get('demand_gw', sample['demand_mw'] / 1000.0)):g}"


def audit_sample(sample, cfg, vmin, vmax, enforce_q_lims):
    import pandapower as pp
    if cfg["case"] == "39":
        from case39_route_b_utils import apply_case39_route_b_limits

    net = cfg["network_builder"]()
    if cfg["case"] == "39":
        apply_case39_route_b_limits(net)
    p = np.asarray(sample[cfg["dispatch_key"]], dtype=np.float64)
    if len(p) != len(net.gen):
        return {
            "passed": False,
            "reason": f"dispatch_dim_mismatch:{len(p)}!={len(net.gen)}",
            "converged": False,
        }

    net.gen.p_mw = np.clip(p, net.gen.min_p_mw.values, net.gen.max_p_mw.values)
    load_scale = float(sample["demand_mw"]) / float(net.load.p_mw.sum())
    net.load.p_mw *= load_scale
    net.load.q_mvar *= load_scale
    net.bus.min_vm_pu = vmin
    net.bus.max_vm_pu = vmax

    t0 = time.perf_counter()
    try:
        pp.runpp(
            net,
            numba=False,
            enforce_q_lims=enforce_q_lims,
            max_iteration=80,
            tolerance_mva=1e-6,
            init="auto",
        )
    except Exception as exc:
        return {
            "passed": False,
            "reason": f"acpf_error:{type(exc).__name__}",
            "converged": False,
            "runtime_sec": time.perf_counter() - t0,
        }

    runtime = time.perf_counter() - t0
    if not bool(getattr(net, "converged", False)):
        return {
            "passed": False,
            "reason": "acpf_not_converged",
            "converged": False,
            "runtime_sec": runtime,
        }

    vm = np.asarray(net.res_bus.vm_pu.values, dtype=np.float64)
    low = vm < (vmin - 1e-9)
    high = vm > (vmax + 1e-9)
    v_viol = int(low.sum() + high.sum())
    line_over = int((net.res_line.loading_percent.values > 100.0 + 1e-9).sum())
    trafo_over = 0
    if len(net.trafo):
        trafo_over = int((net.res_trafo.loading_percent.values > 100.0 + 1e-9).sum())

    passed = v_viol == 0 and line_over == 0 and trafo_over == 0
    if passed:
        reason = "raw_pass"
    elif v_viol:
        vgap_low = float(np.maximum(vmin - vm, 0.0).max())
        vgap_high = float(np.maximum(vm - vmax, 0.0).max())
        reason = f"voltage_violation:vv={v_viol}:vgap={max(vgap_low, vgap_high):.4f}"
    elif line_over or trafo_over:
        reason = f"thermal_overload:line={line_over}:trafo={trafo_over}"
    else:
        reason = "unknown_fail"

    return {
        "passed": passed,
        "reason": reason,
        "converged": True,
        "runtime_sec": runtime,
        "v_min": float(vm.min()),
        "v_max": float(vm.max()),
        "v_violations": v_viol,
        "v_low_violations": int(low.sum()),
        "v_high_violations": int(high.sum()),
        "line_overloaded": line_over,
        "trafo_overloaded": trafo_over,
        "max_line_loading_pct": float(np.max(net.res_line.loading_percent.values)) if len(net.line) else 0.0,
        "max_trafo_loading_pct": float(np.max(net.res_trafo.loading_percent.values)) if len(net.trafo) else 0.0,
    }


def audit_case(case_name, args):
    cfg = load_case(case_name)
    vmin = cfg.get("default_vmin", 0.93) if args.vmin is None else args.vmin
    vmax = cfg.get("default_vmax", 1.10) if args.vmax is None else args.vmax
    with open(cfg["dataset"], "r", encoding="utf-8") as f:
        data = json.load(f)
    all_samples = list(data["samples"])
    samples = [s for s in all_samples if s.get("is_complete_sced", True)]
    used_incomplete_fallback = False
    if not samples and all_samples:
        samples = all_samples
        used_incomplete_fallback = True
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    rows = []
    reasons = Counter()
    by_demand = defaultdict(lambda: {"n": 0, "passed": 0, "reasons": Counter()})
    t0 = time.perf_counter()

    print(
        f"Auditing case{case_name}: n={len(samples)} "
        f"v=[{vmin},{vmax}] enforce_q_lims={args.enforce_q_lims}",
        flush=True,
    )
    for i, sample in enumerate(samples, 1):
        result = audit_sample(sample, cfg, vmin, vmax, args.enforce_q_lims)
        gw = demand_key(sample)
        row = {
            "index": i - 1,
            "id": sample.get("id", sample.get("sample_id")),
            "demand_gw": float(gw),
            **result,
        }
        rows.append(row)
        reasons[result["reason"]] += 1
        by_demand[gw]["n"] += 1
        by_demand[gw]["passed"] += int(result["passed"])
        by_demand[gw]["reasons"][result["reason"]] += 1

        if i == 1 or i % args.print_every == 0 or i == len(samples):
            passed = sum(1 for r in rows if r["passed"])
            print(
                f"[case{case_name} {i}/{len(samples)}] raw_pass={passed}/{i} "
                f"last={row['id']} {result['reason']}",
                flush=True,
            )

    elapsed = time.perf_counter() - t0
    passed = sum(1 for r in rows if r["passed"])
    summary = {
        "case": f"case{case_name}",
        "dataset": str(cfg["dataset"]),
        "n": len(rows),
        "passed": passed,
        "pass_rate_pct": round(100.0 * passed / max(len(rows), 1), 3),
        "vmin": vmin,
        "vmax": vmax,
        "enforce_q_lims": args.enforce_q_lims,
        "used_incomplete_fallback": used_incomplete_fallback,
        "elapsed_sec": round(elapsed, 3),
        "reasons": dict(reasons),
        "by_demand": {
            k: {
                "n": v["n"],
                "passed": v["passed"],
                "pass_rate_pct": round(100.0 * v["passed"] / max(v["n"], 1), 3),
                "top_reasons": dict(v["reasons"].most_common(5)),
            }
            for k, v in sorted(by_demand.items(), key=lambda kv: float(kv[0]))
        },
        "rows": rows,
    }
    out = OUT_DIR / f"route_b_case{case_name}_sample_audit.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ["case", "n", "passed", "pass_rate_pct", "reasons"]}, indent=2))
    print(f"Saved: {out}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="39,118", help="Comma-separated cases: 39,118")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--no-enforce-q-lims", action="store_true")
    args = parser.parse_args()
    args.enforce_q_lims = not args.no_enforce_q_lims

    all_summaries = []
    for case_name in [c.strip() for c in args.cases.split(",") if c.strip()]:
        all_summaries.append(audit_case(case_name, args))
    out = OUT_DIR / "route_b_smallcase_sample_audit_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cases": [
                    {k: s[k] for k in ["case", "dataset", "n", "passed", "pass_rate_pct", "reasons", "by_demand"]}
                    for s in all_summaries
                ]
            },
            f,
            indent=2,
        )
    print(f"Saved summary: {out}")


if __name__ == "__main__":
    main()

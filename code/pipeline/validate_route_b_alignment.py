"""Validate Route-B dataset alignment across 300/118/39 cases.

This is a contract checker for the paper pipeline.  It verifies that retained
samples follow the same semantics:
  1. label dispatch is a linear SCED/N-1 dispatch,
  2. raw strict AC-PF does not already pass,
  3. a rescue P+V evidence point exists,
  4. rescue evidence is audit-only and not the GAT label field.

Use --exact to additionally rebuild the small-case network and re-check
linear N-1 plus strict AC-PF rescue evidence for a subset/all samples.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(r".")
sys.path.insert(0, str(ROOT))


def case_cfg(case):
    if case == "300":
        return {
            "dataset": ROOT / "paper_results" / "case300_rl_rescuable_pv_dataset.json",
            "dispatch_key": "dispatch_68",
            "rescue_key": "rescue_dispatch_68",
            "n_gen": 68,
            "exact_supported": False,
        }
    if case == "118":
        return {
            "dataset": ROOT / "case118_routeb_rescuable_dataset.json",
            "dispatch_key": "dispatch",
            "rescue_key": "rescue_dispatch",
            "n_gen": 53,
            "exact_supported": True,
        }
    if case == "39":
        return {
            "dataset": ROOT / "case39_routeb_rescuable_dataset.json",
            "dispatch_key": "dispatch",
            "rescue_key": "rescue_dispatch",
            "n_gen": 9,
            "exact_supported": True,
        }
    raise ValueError(case)


def strict_acpf_small(case, dispatch, demand, gen_vm, ext_vm, vmin=0.93, vmax=1.10):
    import pandapower as pp
    from pandapower.networks import case39, case118

    net = case39() if case == "39" else case118()
    if case == "39":
        from case39_route_b_utils import apply_case39_route_b_limits, CASE39_VMIN, CASE39_VMAX
        apply_case39_route_b_limits(net)
        if vmin == 0.93 and vmax == 1.10:
            vmin, vmax = CASE39_VMIN, CASE39_VMAX
    p = np.asarray(dispatch, dtype=np.float64)
    net.gen.p_mw = np.clip(p, net.gen.min_p_mw.values, net.gen.max_p_mw.values)
    if gen_vm is not None:
        net.gen.vm_pu = np.asarray(gen_vm, dtype=np.float64)[: len(net.gen)]
    if ext_vm is not None and len(net.ext_grid):
        net.ext_grid.vm_pu = float(np.asarray(ext_vm, dtype=np.float64).reshape(-1)[0])
    scale = float(demand) / float(net.load.p_mw.sum())
    net.load.p_mw *= scale
    net.load.q_mvar *= scale
    net.bus.min_vm_pu = vmin
    net.bus.max_vm_pu = vmax
    try:
        pp.runpp(net, numba=False, enforce_q_lims=True, max_iteration=80)
    except Exception as exc:
        return False, f"acpf_error:{type(exc).__name__}"
    if not net.converged:
        return False, "acpf_not_converged"
    vm = np.asarray(net.res_bus.vm_pu.values, dtype=np.float64)
    vv = int((vm < vmin - 1e-9).sum() + (vm > vmax + 1e-9).sum())
    line = int((net.res_line.loading_percent.values > 100.0 + 1e-9).sum())
    trafo = int((net.res_trafo.loading_percent.values > 100.0 + 1e-9).sum()) if len(net.trafo) else 0
    if vv:
        return False, f"voltage:{vv}"
    if line or trafo:
        return False, f"thermal:{line}:{trafo}"
    return True, "ok"


def linear_n1_small(case, dispatch):
    if case == "39":
        import generate_case39_gt_v1 as gen
        dc = gen.Case39Network()
        nviol, _bad = gen.check_n1(np.asarray(dispatch, dtype=np.float64), dc)
        return int(nviol) == 0, f"n1={int(nviol)}"
    if case == "118":
        import generate_case118_gt_v1 as gen
        dc = gen.Case118Network()
        nviol, _bad = gen.check_n1_security(np.asarray(dispatch, dtype=np.float64), dc)
        return int(nviol) == 0, f"n1={int(nviol)}"
    return True, "skipped"


def validate_case(case, args):
    cfg = case_cfg(case)
    path = cfg["dataset"]
    if not path.exists():
        return {"case": case, "dataset": str(path), "status": "missing"}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = list(data.get("samples", []))
    if args.max_samples and args.max_samples > 0:
        check_samples = samples[: args.max_samples]
    else:
        check_samples = samples

    failures = Counter()
    by_demand = defaultdict(lambda: {"n": 0, "ok": 0})
    exact_failures = Counter()
    examples = []

    for i, s in enumerate(check_samples):
        gw = f"{float(s.get('demand_gw', s.get('demand_mw', 0) / 1000.0)):g}"
        by_demand[gw]["n"] += 1
        ok = True
        dispatch = s.get(cfg["dispatch_key"])
        rescue = s.get(cfg["rescue_key"])
        raw_pass = bool(s.get("raw_strict_acpf_pass_before_rescue", True))
        if not s.get("is_complete_sced", False):
            failures["is_complete_false"] += 1; ok = False
        if not (s.get("linear_sced_feasible", False) or s.get("linear_n1_pass", False)):
            failures["linear_sced_flag_missing_or_false"] += 1; ok = False
        if not (s.get("rl_rescuable_pv", False) or s.get("route_b_rescuable", False)):
            failures["rescuable_flag_missing_or_false"] += 1; ok = False
        if raw_pass:
            failures["raw_strict_acpf_already_passed"] += 1; ok = False
        if dispatch is None:
            failures["missing_label_dispatch"] += 1; ok = False
        elif len(dispatch) != cfg["n_gen"]:
            failures["bad_label_dispatch_dim"] += 1; ok = False
        if rescue is None:
            failures["missing_rescue_dispatch"] += 1; ok = False
        elif len(rescue) != cfg["n_gen"]:
            failures["bad_rescue_dispatch_dim"] += 1; ok = False
        if args.exact and cfg["exact_supported"] and dispatch is not None and rescue is not None:
            lin_ok, lin_reason = linear_n1_small(case, dispatch)
            if not lin_ok:
                exact_failures[f"label_{lin_reason}"] += 1; ok = False
            rescue_lin_ok, rescue_lin_reason = linear_n1_small(case, rescue)
            if not rescue_lin_ok:
                exact_failures[f"rescue_{rescue_lin_reason}"] += 1; ok = False
            ac_ok, ac_reason = strict_acpf_small(
                case,
                rescue,
                s["demand_mw"],
                s.get("rescue_gen_vm_pu"),
                s.get("rescue_ext_grid_vm_pu"),
                args.vmin,
                args.vmax,
            )
            if not ac_ok:
                exact_failures[f"rescue_{ac_reason}"] += 1; ok = False

        by_demand[gw]["ok"] += int(ok)
        if not ok and len(examples) < 10:
            examples.append({
                "idx": i,
                "id": s.get("id", s.get("sample_id")),
                "demand_gw": gw,
                "raw_pass": raw_pass,
                "keys": sorted(s.keys()),
            })

    checked = len(check_samples)
    bad = sum(failures.values()) + sum(exact_failures.values())
    return {
        "case": case,
        "dataset": str(path),
        "status": "ok" if bad == 0 else "fail",
        "samples_total": len(samples),
        "samples_checked": checked,
        "basic_failures": dict(failures),
        "exact_failures": dict(exact_failures),
        "by_demand": {
            k: {"n": v["n"], "ok": v["ok"], "ok_rate_pct": round(100.0 * v["ok"] / max(v["n"], 1), 3)}
            for k, v in sorted(by_demand.items(), key=lambda kv: float(kv[0]))
        },
        "examples": examples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="300,118,39")
    ap.add_argument("--exact", action="store_true", help="Re-run exact checks for small cases; slower.")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--vmin", type=float, default=0.93)
    ap.add_argument("--vmax", type=float, default=1.10)
    args = ap.parse_args()
    results = [validate_case(c.strip(), args) for c in args.cases.split(",") if c.strip()]
    out = {"results": results}
    out_path = ROOT / "paper_results" / "route_b_alignment_validation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

"""Audit ext-grid P/Q and voltage actions for Route-B policies.

This script evaluates raw labels, GAT-only, and GAT+SAC on the shared split
and records ext_grid P/Q, voltage setpoints, generator Q margins, and pass/fail
information. It is intended to answer the reviewer concern that an unconstrained
external grid may be providing unrealistic reactive support.
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np
import torch

CODE_DIR = pathlib.Path(__file__).resolve().parent
PROJ_DIR = CODE_DIR.parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(1, str(PROJ_DIR))

from train_rlv35_complete import (
    DEVICE, EXT_VM_MAX, EXT_VM_MIN, GEN_VM_MAX, GEN_VM_MIN, RL_VMAX_PU,
    RL_VMIN_PU, REF_SOURCE_P_MAX_MW, REF_SOURCE_P_MIN_MW,
    REF_SOURCE_Q_MAX_MVAR, REF_SOURCE_Q_MIN_MVAR, REF_SOURCE_S_MAX_MVA,
    SCEDEnvV21, SACTrainer, adopted_operation_pass, split_samples_stratified,
)
from route_b_split_utils import load_master_split


def is_pass(info):
    return bool(adopted_operation_pass(info))


def apply_state_to_net(env):
    import pandapower.networks as pn

    net = pn.case300()
    net.gen.p_mw.values[:] = np.clip(env.current_power, env.PMIN, env.PMAX * 0.99)
    if env.current_gen_vm_pu is not None and len(env.current_gen_vm_pu) == len(net.gen):
        net.gen.vm_pu.values[:] = np.clip(env.current_gen_vm_pu, GEN_VM_MIN, GEN_VM_MAX)
    if env.current_ext_grid_vm_pu is not None and len(net.ext_grid):
        net.ext_grid.vm_pu.values[0] = float(np.clip(env.current_ext_grid_vm_pu[0], EXT_VM_MIN, EXT_VM_MAX))
    base_load = float(net.load.p_mw.sum())
    if base_load > 0:
        scale = float(env.target_demand) / base_load
        net.load.p_mw = net.load.p_mw * scale
        net.load.q_mvar = net.load.q_mvar * scale
    net.bus.min_vm_pu = RL_VMIN_PU
    net.bus.max_vm_pu = RL_VMAX_PU
    return net


def run_ac_ext_audit(env):
    import pandapower as pp

    net = apply_state_to_net(env)
    t0 = time.perf_counter()
    err = None
    try:
        pp.runpp(net, calculate_voltage_angles=True, enforce_q_lims=True, numba=False, max_iteration=50)
    except Exception as exc:
        err = repr(exc)
    dt = time.perf_counter() - t0
    converged = bool(getattr(net, "converged", False)) and err is None
    out = {
        "ac_runtime_sec": dt,
        "converged": converged,
        "ac_error": err,
        "ext_grid_vm_set": float(net.ext_grid.vm_pu.iloc[0]) if len(net.ext_grid) else None,
        "ext_grid_min_p_mw": float(net.ext_grid.min_p_mw.iloc[0]) if len(net.ext_grid) and "min_p_mw" in net.ext_grid else None,
        "ext_grid_max_p_mw": float(net.ext_grid.max_p_mw.iloc[0]) if len(net.ext_grid) and "max_p_mw" in net.ext_grid else None,
        "ext_grid_min_q_mvar": float(net.ext_grid.min_q_mvar.iloc[0]) if len(net.ext_grid) and "min_q_mvar" in net.ext_grid else None,
        "ext_grid_max_q_mvar": float(net.ext_grid.max_q_mvar.iloc[0]) if len(net.ext_grid) and "max_q_mvar" in net.ext_grid else None,
        "scheduled_balance_error_mw": float(abs(env.current_power.sum() - env.target_demand)),
        "scheduled_balance_violations": int(
            abs(env.current_power.sum() - env.target_demand) > 0.001 * max(env.target_demand, 1.0)
        ),
        "generator_bound_violations": int(np.count_nonzero(
            (env.current_power < env.PMIN - 1e-6)
            | (env.current_power > env.PMAX * 0.99 + 1e-6)
        )),
    }
    if converged:
        vm = net.res_bus.vm_pu.values
        low = np.maximum(net.bus.min_vm_pu.values - vm, 0.0)
        high = np.maximum(vm - net.bus.max_vm_pu.values, 0.0)
        line_over = int((net.res_line.loading_percent.values > 100.0).sum())
        trafo_over = int((net.res_trafo.loading_percent.values > 100.0).sum()) if len(net.trafo) else 0
        out.update({
            "v_min": float(vm.min()),
            "v_max": float(vm.max()),
            "v_violations": int((low > 0).sum() + (high > 0).sum()),
            "v_gap_max": float(max(low.max(initial=0.0), high.max(initial=0.0))),
            "overloaded_lines": line_over,
            "trafo_overloaded": trafo_over,
            "ext_grid_p_mw": float(net.res_ext_grid.p_mw.iloc[0]) if len(net.ext_grid) else None,
            "ext_grid_q_mvar": float(net.res_ext_grid.q_mvar.iloc[0]) if len(net.ext_grid) else None,
            "ext_grid_s_mva": float(np.hypot(net.res_ext_grid.p_mw.iloc[0], net.res_ext_grid.q_mvar.iloc[0])) if len(net.ext_grid) else None,
        })
        ref_p = out["ext_grid_p_mw"] or 0.0
        ref_q = out["ext_grid_q_mvar"] or 0.0
        ref_s = out["ext_grid_s_mva"] or 0.0
        out.update({
            "reference_source_p_violation": bool(ref_p < REF_SOURCE_P_MIN_MW or ref_p > REF_SOURCE_P_MAX_MW),
            "reference_source_q_violation": bool(ref_q < REF_SOURCE_Q_MIN_MVAR or ref_q > REF_SOURCE_Q_MAX_MVAR),
            "reference_source_s_violation": bool(ref_s > REF_SOURCE_S_MAX_MVA),
            "reference_source_s_utilization_pct": 100.0 * ref_s / REF_SOURCE_S_MAX_MVA,
        })
        out["reference_source_violations"] = int(out["reference_source_p_violation"]) + int(out["reference_source_q_violation"]) + int(out["reference_source_s_violation"])
        if hasattr(net, "res_gen") and len(net.res_gen):
            q = net.res_gen.q_mvar.values.astype(float)
            qmin = net.gen.min_q_mvar.values.astype(float) if "min_q_mvar" in net.gen else np.full(len(q), np.nan)
            qmax = net.gen.max_q_mvar.values.astype(float) if "max_q_mvar" in net.gen else np.full(len(q), np.nan)
            out.update({
                "gen_q_min_margin_min": float(np.nanmin(q - qmin)),
                "gen_q_max_margin_min": float(np.nanmin(qmax - q)),
                "gen_q_at_min_count": int(np.nansum((q - qmin) <= 1e-3)),
                "gen_q_at_max_count": int(np.nansum((qmax - q) <= 1e-3)),
            })
    return out


def eval_sac(env, trainer, sid, deployment):
    obs = env.reset(sample_idx=sid)
    last_info = {}
    # Match the deployment evaluator: exhaust the voltage-only SAC horizon
    # before escalating to the complete hybrid action.
    first_steps = env.max_steps
    for _ in range(first_steps):
        raw = trainer.select_action(obs, deterministic=True)
        lo, hi = env.get_action_bounds()
        action = lo + (raw + 1.0) * 0.5 * (hi - lo)
        if deployment == "hierarchical":
            action[:env._n_shift_dirs + env._n_key_gens] = 0.0
        obs, _r, done, _trunc, last_info = env.step(action)
        if done:
            break
    if deployment == "hierarchical" and not is_pass(last_info):
        obs = env.reset(sample_idx=sid)
        for _ in range(env.max_steps):
            raw = trainer.select_action(obs, deterministic=True)
            lo, hi = env.get_action_bounds()
            action = lo + (raw + 1.0) * 0.5 * (hi - lo)
            obs, _r, done, _trunc, last_info = env.step(action)
            if done:
                break
    return last_info


def summarize(rows):
    out = {}
    for mode in sorted({r["mode"] for r in rows}):
        rs = [r for r in rows if r["mode"] == mode]
        def vals(key):
            return [r[key] for r in rs if r.get(key) is not None]
        out[mode] = {
            "n": len(rs),
            "passed": int(sum(r["passed"] for r in rs)),
            "ext_grid_p_mean": float(np.mean(vals("ext_grid_p_mw"))) if vals("ext_grid_p_mw") else None,
            "ext_grid_p_min": float(np.min(vals("ext_grid_p_mw"))) if vals("ext_grid_p_mw") else None,
            "ext_grid_p_max": float(np.max(vals("ext_grid_p_mw"))) if vals("ext_grid_p_mw") else None,
            "ext_grid_q_mean": float(np.mean(vals("ext_grid_q_mvar"))) if vals("ext_grid_q_mvar") else None,
            "ext_grid_q_min": float(np.min(vals("ext_grid_q_mvar"))) if vals("ext_grid_q_mvar") else None,
            "ext_grid_q_max": float(np.max(vals("ext_grid_q_mvar"))) if vals("ext_grid_q_mvar") else None,
            "ext_grid_vm_set_mean": float(np.mean(vals("ext_grid_vm_set"))) if vals("ext_grid_vm_set") else None,
            "ext_grid_vm_touch_minmax_count": int(sum(
                r.get("ext_grid_vm_set") is not None
                and (abs(r["ext_grid_vm_set"] - EXT_VM_MIN) < 1e-6 or abs(r["ext_grid_vm_set"] - EXT_VM_MAX) < 1e-6)
                for r in rs
            )),
            "reference_source_p_touch_count": int(sum(r.get("reference_source_p_violation", False) for r in rs)),
            "reference_source_q_touch_count": int(sum(r.get("reference_source_q_violation", False) for r in rs)),
            "reference_source_s_touch_count": int(sum(r.get("reference_source_s_violation", False) for r in rs)),
            "reference_source_s_utilization_max_pct": float(np.max(vals("reference_source_s_utilization_pct"))) if vals("reference_source_s_utilization_pct") else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(PROJ_DIR / "case300_gt_v10_dataset.json"))
    ap.add_argument("--split-file", default=str(PROJ_DIR / "paper_results" / "route_b_master_split.json"))
    ap.add_argument("--checkpoint", default=str(PROJ_DIR / "rl_scratch_sac" / "best_actor.pth"))
    ap.add_argument("--output", default=str(PROJ_DIR / "paper_results" / "route_b_ext_grid_audit.json"))
    ap.add_argument("--scope", choices=["test", "val", "train"], default="test")
    ap.add_argument("--deployment", choices=["hierarchical", "full_hybrid"], default="hierarchical")
    args = ap.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = [s for s in data["samples"] if s.get("is_complete_sced")]
    if pathlib.Path(args.split_file).exists():
        split = load_master_split(args.split_file)
        ids = split[f"{args.scope}_ids"]
    else:
        tr, va, te = split_samples_stratified(samples)
        ids = {"train": tr, "val": va, "test": te}[args.scope]

    env = SCEDEnvV21(samples, ids, dataset_meta=data)
    trainer = SACTrainer(env._get_obs_dim(), env.action_dim, device=DEVICE)
    trainer.actor.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE, weights_only=False))
    trainer.actor.eval()

    rows = []
    for n, sid in enumerate(ids, 1):
        s = samples[sid]
        for mode in ("raw_label", "gat_only", "gat_sac"):
            env.reset(sample_idx=sid)
            if mode == "raw_label":
                env.current_power = np.asarray(s["dispatch_68"], dtype=np.float64).copy()
            elif mode == "gat_only":
                env.current_power = np.asarray(env.gat_power, dtype=np.float64).copy()
            else:
                eval_sac(env, trainer, sid, args.deployment)
            ac = run_ac_ext_audit(env)
            n1_over, n1_viol = env._compute_n1_overload_margin()
            row = {
                "sid": sid,
                "sample_id": s.get("sample_id"),
                "demand_gw": float(s.get("demand_gw", s["demand_mw"] / 1000.0)),
                "mode": mode,
                **ac,
                "n1_violations": int(n1_viol),
                "n1_overload_mw": float(n1_over),
            }
            row["passed"] = bool(is_pass(row))
            rows.append(row)
        print(f"[{n}/{len(ids)}] {s.get('sample_id')}")

    result = {
        "split_file": args.split_file,
        "dataset": args.dataset,
        "scope": args.scope,
        "deployment": args.deployment,
        "reference_source_limits": {
            "p_mw": [REF_SOURCE_P_MIN_MW, REF_SOURCE_P_MAX_MW],
            "q_mvar": [REF_SOURCE_Q_MIN_MVAR, REF_SOURCE_Q_MAX_MVAR],
            "s_mva_max": REF_SOURCE_S_MAX_MVA,
        },
        "summary": summarize(rows),
        "rows": rows,
    }
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print("Wrote:", args.output)


if __name__ == "__main__":
    main()

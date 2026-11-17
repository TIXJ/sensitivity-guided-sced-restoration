"""Supplementary N-1 stress test for the Route-B 300-bus benchmark.

This diagnostic does not retrain any policy and does not modify the dataset.
It tightens the stored affine PTDF/LODF line-flow limits at evaluation time and
checks whether different deployment variants retain the linear N-1 criterion.
The purpose is to isolate the role of active-power/PTDF-LODF actions when the
main screened benchmark has weak N-1 pressure.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch

CODE_DIR = pathlib.Path(__file__).resolve().parent
PROJ_DIR = CODE_DIR.parents[1]
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(1, str(PROJ_DIR))

from route_b_split_utils import load_master_split
from eval_route_b_acopf_baseline import (
    build_opf_net,
    extract_opf_voltage_setpoints,
    prepare_base_net,
    run_opf,
)
from train_rlv35_complete import (
    DEVICE,
    EXT_VM_MAX,
    EXT_VM_MIN,
    GEN_VM_MAX,
    GEN_VM_MIN,
    SCEDEnvV21,
    SACTrainer,
    project_box_sum,
    split_samples_stratified,
)


def is_strict_pass(info: dict) -> bool:
    return (
        bool(info.get("converged", False))
        and int(info.get("v_violations", 999)) == 0
        and int(info.get("overloaded_lines", 999)) == 0
        and int(info.get("trafo_overloaded", 999)) == 0
        and int(info.get("n1_violations", 999)) == 0
    )


def verify_current(env: SCEDEnvV21, check_ac: bool) -> dict:
    t0 = time.perf_counter()
    if check_ac:
        info = dict(env._run_ac_pf_check())
    else:
        info = {
            "converged": True,
            "v_violations": 0,
            "overloaded_lines": 0,
            "trafo_overloaded": 0,
            "ac_skipped": True,
        }
    n1_over, n1_viol = env._compute_n1_overload_margin()
    n1_margin = compute_n1_utilization(env)
    info["n1_overload_mw"] = float(n1_over)
    info["n1_violations"] = int(n1_viol)
    info.update(n1_margin)
    info["runtime_sec"] = time.perf_counter() - t0
    info["strict_pass"] = is_strict_pass(info) if check_ac else None
    return info


def compute_n1_utilization(env: SCEDEnvV21) -> dict:
    flows = env._estimate_base_flows(env.current_power)
    limits = np.asarray(env.dcnet.effective_limit, dtype=np.float64)
    lodf = np.asarray(env.dcnet.lodf, dtype=np.float64)
    safe_limits = np.maximum(limits, 1e-9)
    base_abs = np.abs(flows)
    max_ratio = float(np.max(base_abs / safe_limits)) if len(flows) else 0.0
    min_margin = float(np.min(safe_limits - base_abs)) if len(flows) else 0.0

    for k in env.valid_contingency_indices:
        fk = flows[k]
        if abs(fk) < 0.1:
            continue
        if hasattr(env.dcnet, "estimate_post_contingency_flows"):
            post = env.dcnet.estimate_post_contingency_flows(env.current_power, k)
        else:
            post = flows + lodf[:, k] * fk
            post[k] = 0.0
        abs_post = np.abs(post)
        ratio = float(np.max(abs_post / safe_limits))
        margin = float(np.min(safe_limits - abs_post))
        max_ratio = max(max_ratio, ratio)
        min_margin = min(min_margin, margin)
    return {
        "n1_max_utilization_pct": 100.0 * max_ratio,
        "n1_min_margin_mw": min_margin,
    }


def apply_action_mask(env: SCEDEnvV21, action: np.ndarray, mode: str) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64).copy()
    p_dim = env._n_shift_dirs + env._n_key_gens
    if mode == "voltage_only_sac":
        action[:p_dim] = 0.0
    elif mode == "active_power_only_sac":
        action[p_dim:] = 0.0
    return action


def apply_fast_no_ac_step(env: SCEDEnvV21, action: np.ndarray) -> None:
    """Apply the same hybrid action recovery as env.step, but skip reward/AC-PF."""
    env.corr_step += 1
    env._compute_shift_directions()
    env._select_key_generators()
    lo, hi = env.get_action_bounds()
    action = np.clip(action, lo, hi).astype(np.float64)

    p_dim = env._n_shift_dirs + env._n_key_gens
    shift_scales = action[:env._n_shift_dirs]
    key_gen_deltas = action[env._n_shift_dirs:p_dim]
    voltage_deltas = action[p_dim:]

    delta_p = env._shift_directions.T @ shift_scales
    delta_p[env._key_gen_indices] += key_gen_deltas
    env.cumulative_delta += delta_p

    if env.target_demand >= 21000:
        cum_clip = env.current_residual_scale * 10.0
    elif env.target_demand >= 19000:
        cum_clip = env.current_residual_scale * 5.0
    else:
        cum_clip = env.current_residual_scale * 3.5
    env.cumulative_delta = np.clip(env.cumulative_delta, -cum_clip, cum_clip)
    raw_new_power = env.gat_power + env.cumulative_delta
    env.current_power = project_box_sum(raw_new_power, env.PMIN, env.PMAX * 0.98, env.target_demand)

    if len(voltage_deltas) == env._n_voltage_actions:
        gen_group_deltas = voltage_deltas[:env._n_voltage_groups]
        for gidx, gen_idx in enumerate(env._voltage_group_indices):
            if len(gen_idx):
                env.current_gen_vm_pu[gen_idx] += gen_group_deltas[gidx]
        env.current_ext_grid_vm_pu[0] += voltage_deltas[-1]
        env.current_gen_vm_pu = np.clip(env.current_gen_vm_pu, GEN_VM_MIN, GEN_VM_MAX)
        env.current_ext_grid_vm_pu = np.clip(env.current_ext_grid_vm_pu, EXT_VM_MIN, EXT_VM_MAX)


def run_policy(env: SCEDEnvV21, trainer: SACTrainer, sid: int, mode: str, max_steps: int, check_ac: bool) -> dict:
    obs = env.reset(sample_idx=sid)
    last = {}
    steps = min(max_steps, env.max_steps)
    for _ in range(steps):
        raw = trainer.select_action(obs, deterministic=True)
        lo, hi = env.get_action_bounds()
        action = lo + (raw + 1.0) * 0.5 * (hi - lo)
        action = apply_action_mask(env, action, mode)
        if check_ac:
            obs, _r, done, _trunc, last = env.step(action)
            if done:
                break
        else:
            apply_fast_no_ac_step(env, action)
            obs = env._get_obs()
    if "n1_violations" not in last or not check_ac:
        last = verify_current(env, check_ac=check_ac)
    last["steps"] = int(env.corr_step)
    return last


def run_acopf_baseline(env: SCEDEnvV21, base_net, sample: dict, opf_vmin: float, opf_vmax: float, timeout_sec: float, check_ac: bool) -> dict:
    t0 = time.perf_counter()
    net = build_opf_net(base_net, sample, opf_vmin=opf_vmin, opf_vmax=opf_vmax)
    ok, opf_info = run_opf(net, timeout_sec)
    if not ok:
        return {
            **opf_info,
            "converged": False,
            "v_violations": 999,
            "overloaded_lines": 999,
            "trafo_overloaded": 999,
            "n1_violations": 999,
            "n1_overload_mw": 9999.0,
            "runtime_sec": time.perf_counter() - t0,
            "reason": "ac_opf_failed",
        }

    p = np.asarray(net.res_gen.p_mw.values, dtype=np.float64)
    pmin = np.asarray(net.gen.min_p_mw.values, dtype=np.float64)
    pmax = np.asarray(net.gen.max_p_mw.values, dtype=np.float64) * 0.99
    p = project_box_sum(p, pmin, pmax, float(sample["demand_mw"]))
    gen_vm, ext_vm = extract_opf_voltage_setpoints(net, opf_vmin, opf_vmax)

    env.reset(sample_idx=env.sample_ids[0])
    env.current_sample = sample
    env.target_demand = float(sample["demand_mw"])
    env.current_power = p.copy()
    env.current_gen_vm_pu = np.asarray(gen_vm, dtype=np.float64).copy()
    env.current_ext_grid_vm_pu = np.asarray([ext_vm], dtype=np.float64)
    info = verify_current(env, check_ac=check_ac)
    info.update(opf_info)
    info["runtime_sec"] = time.perf_counter() - t0
    if int(info.get("n1_violations", 999)) > 0:
        info["reason"] = "n1_violation"
    elif check_ac and not is_strict_pass(info):
        info["reason"] = "ac_voltage_or_thermal"
    else:
        info["reason"] = "full_sced_pass"
    return info


def summarize(rows):
    out = {}
    for mode in sorted({r["mode"] for r in rows}):
        rs = [r for r in rows if r["mode"] == mode]
        by = defaultdict(lambda: {"n": 0, "n1_pass": 0, "strict_pass": 0})
        reasons = Counter()
        for r in rs:
            key = f"{float(r['demand_gw']):g}"
            by[key]["n"] += 1
            by[key]["n1_pass"] += int(r["n1_pass"])
            by[key]["strict_pass"] += int(bool(r["strict_pass"]))
            if not r["n1_pass"]:
                reasons["n1_violation"] += 1
            elif r["strict_pass"] is False:
                reasons[str(r.get("reason", "ac_voltage_or_thermal"))] += 1
        n1_over = [float(r["n1_overload_mw"]) for r in rs]
        n1_viol = [int(r["n1_violations"]) for r in rs]
        n1_util = [float(r["n1_max_utilization_pct"]) for r in rs]
        n1_margin = [float(r["n1_min_margin_mw"]) for r in rs]
        strict_vals = [r["strict_pass"] for r in rs if r["strict_pass"] is not None]
        out[mode] = {
            "n": len(rs),
            "n1_pass": int(sum(r["n1_pass"] for r in rs)),
            "strict_pass": int(sum(bool(v) for v in strict_vals)) if strict_vals else None,
            "n1_pass_rate_pct": round(100.0 * sum(r["n1_pass"] for r in rs) / max(len(rs), 1), 3),
            "strict_pass_rate_pct": round(100.0 * sum(bool(v) for v in strict_vals) / max(len(strict_vals), 1), 3) if strict_vals else None,
            "n1_overload_mean_mw": float(np.mean(n1_over)) if n1_over else None,
            "n1_overload_max_mw": float(np.max(n1_over)) if n1_over else None,
            "n1_violations_mean": float(np.mean(n1_viol)) if n1_viol else None,
            "n1_max_utilization_mean_pct": float(np.mean(n1_util)) if n1_util else None,
            "n1_max_utilization_max_pct": float(np.max(n1_util)) if n1_util else None,
            "n1_min_margin_mean_mw": float(np.mean(n1_margin)) if n1_margin else None,
            "n1_min_margin_min_mw": float(np.min(n1_margin)) if n1_margin else None,
            "fail_reasons": dict(reasons.most_common(10)),
            "by_demand": dict(sorted(by.items(), key=lambda kv: float(kv[0]))),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(PROJ_DIR / "case300_gt_v10_dataset.json"))
    ap.add_argument("--split-file", default=str(PROJ_DIR / "paper_results" / "route_b_master_split.json"))
    ap.add_argument("--checkpoint", default=str(PROJ_DIR / "rl_scratch_sac" / "best_actor.pth"))
    ap.add_argument("--scope", choices=["train", "val", "test", "all"], default="test")
    ap.add_argument("--limit-scale", type=float, default=0.90)
    ap.add_argument("--modes", default="raw_label,gat_only,gat_sac,voltage_only_sac,active_power_only_sac,ac_opf")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=1)
    ap.add_argument("--check-ac", action="store_true")
    ap.add_argument("--opf-vmin", type=float, default=0.94)
    ap.add_argument("--opf-vmax", type=float, default=1.08)
    ap.add_argument("--opf-timeout-sec", type=float, default=30.0)
    ap.add_argument("--output", default=str(PROJ_DIR / "paper_results" / "route_b_n1_stress_test.json"))
    args = ap.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = [s for s in data["samples"] if s.get("is_complete_sced")]
    if pathlib.Path(args.split_file).exists():
        split = load_master_split(args.split_file)
        ids = split["train_ids"] + split["val_ids"] + split["test_ids"] if args.scope == "all" else split[f"{args.scope}_ids"]
    else:
        tr, va, te = split_samples_stratified(samples)
        ids = {"train": tr, "val": va, "test": te, "all": tr + va + te}[args.scope]
    if args.max_samples > 0:
        ids = ids[: args.max_samples]

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    print(
        f"N-1 stress test: scope={args.scope} samples={len(ids)} "
        f"limit_scale={args.limit_scale} modes={modes} check_ac={args.check_ac}",
        flush=True,
    )
    env = SCEDEnvV21(samples, ids, dataset_meta=data)
    original_limits = np.asarray(env.dcnet.effective_limit, dtype=np.float64).copy()
    env.dcnet.effective_limit = original_limits * float(args.limit_scale)

    trainer = SACTrainer(env._get_obs_dim(), env.action_dim, device=DEVICE)
    trainer.actor.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE, weights_only=False))
    trainer.actor.eval()
    acopf_base_net = prepare_base_net() if "ac_opf" in modes else None

    rows = []
    for i, sid in enumerate(ids, 1):
        sample = samples[sid]
        for mode in modes:
            env.reset(sample_idx=sid)
            if mode == "raw_label":
                env.current_power = np.asarray(sample["dispatch_68"], dtype=np.float64).copy()
                info = verify_current(env, check_ac=args.check_ac)
            elif mode == "gat_only":
                env.current_power = np.asarray(env.gat_power, dtype=np.float64).copy()
                info = verify_current(env, check_ac=args.check_ac)
            elif mode == "ac_opf":
                info = run_acopf_baseline(
                    env,
                    acopf_base_net,
                    sample,
                    opf_vmin=args.opf_vmin,
                    opf_vmax=args.opf_vmax,
                    timeout_sec=args.opf_timeout_sec,
                    check_ac=args.check_ac,
                )
            else:
                info = run_policy(env, trainer, sid, mode, args.max_steps, check_ac=args.check_ac)
            n1_pass = int(info.get("n1_violations", 999)) == 0
            row = {
                "sid": sid,
                "sample_id": sample.get("sample_id"),
                "demand_gw": float(sample.get("demand_gw", sample["demand_mw"] / 1000.0)),
                "mode": mode,
                "n1_pass": bool(n1_pass),
                "strict_pass": bool(is_strict_pass(info)) if args.check_ac else None,
                "n1_violations": int(info.get("n1_violations", 999)),
                "n1_overload_mw": float(info.get("n1_overload_mw", 9999.0)),
                "n1_max_utilization_pct": float(info.get("n1_max_utilization_pct", 9999.0)),
                "n1_min_margin_mw": float(info.get("n1_min_margin_mw", -9999.0)),
                "v_violations": int(info.get("v_violations", 999)),
                "overloaded_lines": int(info.get("overloaded_lines", 999)),
                "trafo_overloaded": int(info.get("trafo_overloaded", 999)),
                "reason": info.get("reason"),
                "steps": int(info.get("steps", 0)),
            }
            rows.append(row)
        print(f"[{i}/{len(ids)}] {sample.get('sample_id')}", flush=True)

    result = {
        "dataset": args.dataset,
        "split_file": args.split_file,
        "scope": args.scope,
        "limit_scale": args.limit_scale,
        "check_ac": bool(args.check_ac),
        "notes": "Evaluation-only stress test. Stored LODF effective limits are multiplied by limit_scale; no training or dataset labels are changed.",
        "summary": summarize(rows),
        "rows": rows,
    }
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print("Wrote:", args.output)


if __name__ == "__main__":
    main()

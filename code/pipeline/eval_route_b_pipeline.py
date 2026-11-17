"""Route-B evaluation: raw linear-SCED labels, GAT-only, and GAT+SAC.

Uses the current held-out split in rl_scratch_sac/split_ids.json and the same
SCEDEnvV21 AC-PF/N-1 verifier as training.
"""

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
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(1, str(PROJ_DIR))

import train_rlv35_complete as _v35

# Match the final training/evaluation rollout budget.
_v35.MAX_STEPS = 50
_v35.MAX_STEPS_HIGH = 70

from train_rlv35_complete import (
    DEVICE, SCEDEnvV21, SACTrainer, adopted_operation_pass,
    split_samples_stratified,
)


def is_pass(info):
    return bool(adopted_operation_pass(info))


def dispatch_cost(sample, p):
    p = np.asarray(p, dtype=np.float64)
    cp2 = np.asarray(sample.get("sample_cp2", np.zeros_like(p)), dtype=np.float64)
    cp1 = np.asarray(sample.get("sample_cp1", np.zeros_like(p)), dtype=np.float64)
    cp0 = np.asarray(sample.get("sample_cp0", np.zeros_like(p)), dtype=np.float64)
    return float(np.sum(cp2 * p * p + cp1 * p + cp0))


def verify_current(env):
    t0 = time.perf_counter()
    ac = env._run_ac_pf_check()
    n1_overload, n1_viol = env._compute_n1_overload_margin()
    dt = time.perf_counter() - t0
    info = dict(ac)
    info["n1_overload_mw"] = float(n1_overload)
    info["n1_violations"] = int(n1_viol)
    if is_pass(info):
        info["reason"] = "full_sced_pass"
    elif not info.get("converged", False):
        info["reason"] = "acpf_not_converged"
    elif info.get("v_violations", 999) > 0:
        info["reason"] = "voltage_violation"
    elif info.get("overloaded_lines", 999) > 0 or info.get("trafo_overloaded", 999) > 0:
        info["reason"] = "line_overload"
    elif n1_viol > 0:
        info["reason"] = "n1_violation"
    else:
        info["reason"] = "not_full_sced"
    return info, dt


def set_default_voltage(env):
    env.current_gen_vm_pu = np.asarray(env.dcnet.net_base.gen.vm_pu.values, dtype=np.float64).copy()
    env.current_ext_grid_vm_pu = np.asarray(
        env.dcnet.net_base.ext_grid.vm_pu.values if len(env.dcnet.net_base.ext_grid) else [1.0],
        dtype=np.float64,
    ).copy()


def eval_static_dispatch(env, sample, sid, mode):
    env.reset(sample_idx=sid)
    set_default_voltage(env)
    if mode == "raw_label":
        env.current_power = np.asarray(sample["dispatch_68"], dtype=np.float64).copy()
    elif mode == "gat_only":
        env.current_power = np.asarray(env.gat_power, dtype=np.float64).copy()
    else:
        raise ValueError(mode)
    info, runtime = verify_current(env)
    p = env.current_power.copy()
    return {
        "passed": is_pass(info),
        "info": info,
        "runtime_sec": runtime,
        "dispatch": p,
        "cost": dispatch_cost(sample, p),
    }


def eval_sac(env, trainer, sample, sid, deployment="hierarchical"):
    t0 = time.perf_counter()
    obs = env.reset(sample_idx=sid)
    p0 = env.current_power.copy()
    last_info = {}
    total_reward = 0.0
    steps = 0
    total_actor_steps = 0
    deployment_stage = "full_hybrid"
    # Give the voltage-only policy the same correction horizon as the full
    # policy. Escalating after a single action needlessly introduced active-P
    # redispatch for cases that voltage control resolves in later SAC steps.
    first_stage_steps = env.max_steps
    for steps in range(first_stage_steps):
        raw = trainer.select_action(obs, deterministic=True)
        lo, hi = env.get_action_bounds()
        action = lo + (raw + 1.0) * 0.5 * (hi - lo)
        if deployment == "hierarchical":
            p_dim = env._n_shift_dirs + env._n_key_gens
            action[:p_dim] = 0.0
            deployment_stage = "voltage_only"
        obs, r, done, _trunc, last_info = env.step(action)
        total_actor_steps += 1
        total_reward += float(np.nan_to_num(r, nan=-100.0, posinf=1000.0, neginf=-1000.0))
        if done:
            break

    # Escalate to the complete hybrid action only when voltage-only recovery fails.
    if deployment == "hierarchical" and not is_pass(last_info):
        obs = env.reset(sample_idx=sid)
        deployment_stage = "full_hybrid_fallback"
        total_reward = 0.0
        for steps in range(env.max_steps):
            raw = trainer.select_action(obs, deterministic=True)
            lo, hi = env.get_action_bounds()
            action = lo + (raw + 1.0) * 0.5 * (hi - lo)
            obs, r, done, _trunc, last_info = env.step(action)
            total_actor_steps += 1
            total_reward += float(np.nan_to_num(r, nan=-100.0, posinf=1000.0, neginf=-1000.0))
            if done:
                break
    runtime = time.perf_counter() - t0
    p = env.current_power.copy()
    return {
        "passed": is_pass(last_info),
        "info": last_info,
        "runtime_sec": runtime,
        "dispatch": p,
        "cost": dispatch_cost(sample, p),
        "p0_cost": dispatch_cost(sample, p0),
        "p0_l1_mw": float(np.abs(p - p0).sum()),
        "p0_max_delta_mw": float(np.abs(p - p0).max()),
        "reward": float(total_reward),
        "steps": int(total_actor_steps),
        "deployment_stage": deployment_stage,
    }


def summarize(name, rows):
    n = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    reasons = Counter(r["reason"] for r in rows if not r["passed"])
    by_demand = defaultdict(lambda: {"passed": 0, "n": 0})
    for r in rows:
        key = f"{r['demand_gw']:g}"
        by_demand[key]["n"] += 1
        by_demand[key]["passed"] += int(r["passed"])
    return {
        "name": name,
        "n": n,
        "passed": passed,
        "pass_rate_pct": round(100.0 * passed / max(n, 1), 3),
        "avg_runtime_sec": round(float(np.mean([r["runtime_sec"] for r in rows])), 4),
        "max_runtime_sec": round(float(np.max([r["runtime_sec"] for r in rows])), 4),
        "avg_cost": round(float(np.mean([r["cost"] for r in rows])), 3),
        "avg_v_violations": round(float(np.mean([r.get("v_violations", 999) for r in rows])), 3),
        "avg_v_gap": round(float(np.mean([r.get("v_gap", 0.0) or 0.0 for r in rows])), 6),
        "reasons": dict(reasons),
        "by_demand": dict(sorted(by_demand.items(), key=lambda kv: float(kv[0]))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        default=str(PROJ_DIR / "case300_gt_v10_dataset_lodf_corrected_277.json"),
    )
    ap.add_argument(
        "--checkpoint-dir",
        default=str(PROJ_DIR / "rl_scratch_sac_lodf_corrected"),
    )
    ap.add_argument(
        "--gat-checkpoint",
        default=str(PROJ_DIR / "gatv10_results_lodf_corrected" / "gatv10_predictor_best.pth"),
    )
    ap.add_argument("--deployment", choices=["hierarchical", "full_hybrid"], default="hierarchical")
    ap.add_argument("--scope", choices=["val", "test"], default="test")
    ap.add_argument("--checkpoint", default=None,
                    help="Actor checkpoint override; defaults to <checkpoint-dir>/best_actor.pth.")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="Optional stratified cap for quick diagnostics (0 means all).")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    if args.gat_checkpoint:
        os.environ["ROUTE_B_GAT_CHECKPOINT"] = str(pathlib.Path(args.gat_checkpoint).resolve())

    dataset_path = pathlib.Path(args.dataset)
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = [s for s in data["samples"] if s.get("is_complete_sced")]

    split_path = checkpoint_dir / "split_ids.json"
    if split_path.exists():
        split = json.load(open(split_path, "r", encoding="utf-8"))
        recorded_dataset = split.get("dataset")
        if recorded_dataset and pathlib.Path(recorded_dataset).resolve() != dataset_path.resolve():
            raise ValueError(
                f"Split dataset mismatch: split={recorded_dataset}, requested={dataset_path}"
            )
        test_ids = split[f"{args.scope}_ids"]
    else:
        _train_ids, _val_ids, _test_ids = split_samples_stratified(samples)
        test_ids = _val_ids if args.scope == "val" else _test_ids
    if not test_ids or min(test_ids) < 0 or max(test_ids) >= len(samples):
        raise ValueError(f"Test split is invalid for dataset with {len(samples)} samples")
    if args.max_samples > 0 and len(test_ids) > args.max_samples:
        by_demand = {}
        for sid in test_ids:
            gw = round(float(samples[sid].get("demand_gw", samples[sid]["demand_mw"] / 1000.0)), 3)
            by_demand.setdefault(gw, []).append(sid)
        selected = [by_demand[gw][0] for gw in sorted(by_demand)]
        if len(selected) < args.max_samples:
            used = set(selected)
            selected.extend(sid for sid in test_ids if sid not in used)
        test_ids = selected[:args.max_samples]

    env = SCEDEnvV21(samples, test_ids, dataset_meta=data)
    obs_dim, action_dim = env._get_obs_dim(), env.action_dim
    trainer = SACTrainer(obs_dim, action_dim, device=DEVICE)
    ckpt = pathlib.Path(args.checkpoint) if args.checkpoint else checkpoint_dir / "best_actor.pth"
    trainer.actor.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
    trainer.actor.eval()

    all_rows = {"raw_label": [], "gat_only": [], "gat_sac": []}

    for idx, sid in enumerate(test_ids, 1):
        sample = samples[sid]
        demand_gw = float(sample.get("demand_gw", sample["demand_mw"] / 1000.0))
        raw = eval_static_dispatch(env, sample, sid, "raw_label")
        gat = eval_static_dispatch(env, sample, sid, "gat_only")
        sac = eval_sac(env, trainer, sample, sid, deployment=args.deployment)
        for name, result in [("raw_label", raw), ("gat_only", gat), ("gat_sac", sac)]:
            info = result["info"]
            row = {
                "sid": sid,
                "sample_id": sample.get("sample_id"),
                "demand_gw": demand_gw,
                "passed": bool(result["passed"]),
                "reason": info.get("reason"),
                "converged": bool(info.get("converged", False)),
                "v_violations": int(info.get("v_violations", 999)),
                "v_low_violations": int(info.get("v_low_violations", 999)),
                "v_high_violations": int(info.get("v_high_violations", 999)),
                "v_min": info.get("v_min"),
                "v_max": info.get("v_max"),
                "v_gap": info.get("v_violation_max_pu"),
                "overloaded_lines": int(info.get("overloaded_lines", 999)),
                "trafo_overloaded": int(info.get("trafo_overloaded", 999)),
                "reference_source_violations": int(info.get("reference_source_violations", 999)),
                "reference_source_p_mw": info.get("reference_source_p_mw"),
                "reference_source_q_mvar": info.get("reference_source_q_mvar"),
                "reference_source_s_mva": info.get("reference_source_s_mva"),
                "reference_source_s_utilization_pct": info.get("reference_source_s_utilization_pct"),
                "n1_violations": int(info.get("n1_violations", 999)),
                "n1_overload_mw": float(info.get("n1_overload_mw", 9999.0)),
                "runtime_sec": float(result["runtime_sec"]),
                "cost": float(result["cost"]),
            }
            if name == "gat_sac":
                row.update({
                    "p0_cost": float(result["p0_cost"]),
                    "p0_l1_mw": float(result["p0_l1_mw"]),
                    "p0_max_delta_mw": float(result["p0_max_delta_mw"]),
                    "steps": int(result["steps"]),
                    "reward": float(result["reward"]),
                    "deployment_stage": result["deployment_stage"],
                    "cost_gap_vs_raw_pct": (result["cost"] - raw["cost"]) / (raw["cost"] + 1e-9) * 100.0,
                    "cost_gap_vs_gat_pct": (result["cost"] - gat["cost"]) / (gat["cost"] + 1e-9) * 100.0,
                })
            all_rows[name].append(row)
        print(
            f"{idx:02d}/{len(test_ids)} {sample.get('sample_id')} {demand_gw:g}GW "
            f"raw={'OK' if raw['passed'] else 'FAIL'} "
            f"gat={'OK' if gat['passed'] else 'FAIL'} "
            f"sac={'OK' if sac['passed'] else 'FAIL'}"
        )

    summary = {name: summarize(name, rows) for name, rows in all_rows.items()}
    sac_rows = all_rows["gat_sac"]
    summary["gat_sac"]["avg_cost_gap_vs_raw_pct"] = round(float(np.mean([r["cost_gap_vs_raw_pct"] for r in sac_rows])), 4)
    summary["gat_sac"]["max_cost_gap_vs_raw_pct"] = round(float(np.max([r["cost_gap_vs_raw_pct"] for r in sac_rows])), 4)
    summary["gat_sac"]["avg_cost_gap_vs_gat_pct"] = round(float(np.mean([r["cost_gap_vs_gat_pct"] for r in sac_rows])), 4)
    summary["gat_sac"]["avg_l1_delta_mw"] = round(float(np.mean([r["p0_l1_mw"] for r in sac_rows])), 3)
    summary["gat_sac"]["avg_max_delta_mw"] = round(float(np.mean([r["p0_max_delta_mw"] for r in sac_rows])), 3)
    summary["gat_sac"]["avg_steps"] = round(float(np.mean([r["steps"] for r in sac_rows])), 3)
    summary["gat_sac"]["voltage_only_count"] = int(sum(r["deployment_stage"] == "voltage_only" for r in sac_rows))
    summary["gat_sac"]["full_hybrid_fallback_count"] = int(sum(r["deployment_stage"] == "full_hybrid_fallback" for r in sac_rows))

    result = {
        "dataset": str(dataset_path),
        "checkpoint": str(ckpt),
        "split": str(split_path),
        "n_test": len(test_ids),
        "deployment": args.deployment,
        "summary": summary,
        "rows": all_rows,
    }
    out = pathlib.Path(args.output) if args.output else PROJ_DIR / "paper_results" / (
        "route_b_pipeline_eval_hierarchical.json" if args.deployment == "hierarchical"
        else "route_b_pipeline_eval.json"
    )
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(summary, indent=2))
    print("Wrote:", out)


if __name__ == "__main__":
    main()

"""Audit whether Route-B data and policies have meaningful linear N-1 pressure.

The paper now claims a linear N-1 SCED warm start, so we need explicit evidence
that the screened samples and learned policies are still checked against the
same affine PTDF/LODF model used by the RL environment.
"""
from __future__ import annotations

import argparse
import json
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

from route_b_split_utils import load_master_split
from train_rlv35_complete import DEVICE, SCEDEnvV21, SACTrainer, split_samples_stratified


def pass_info(info: dict) -> bool:
    return (
        bool(info.get("converged", False))
        and int(info.get("v_violations", 999)) == 0
        and int(info.get("overloaded_lines", 999)) == 0
        and int(info.get("trafo_overloaded", 999)) == 0
        and int(info.get("n1_violations", 999)) == 0
    )


def verify_env_state(env: SCEDEnvV21) -> dict:
    t0 = time.perf_counter()
    ac = env._run_ac_pf_check()
    n1_over, n1_viol = env._compute_n1_overload_margin()
    dt = time.perf_counter() - t0
    info = dict(ac)
    info["n1_overload_mw"] = float(n1_over)
    info["n1_violations"] = int(n1_viol)
    info["runtime_sec"] = dt
    info["passed"] = pass_info(info)
    return info


def eval_sac(env: SCEDEnvV21, trainer: SACTrainer, sid: int) -> dict:
    obs = env.reset(sample_idx=sid)
    last = {}
    for _ in range(env.max_steps + 1):
        raw = trainer.select_action(obs, deterministic=True)
        lo, hi = env.get_action_bounds()
        action = lo + (raw + 1.0) * 0.5 * (hi - lo)
        obs, _r, done, _trunc, last = env.step(action)
        if done:
            break
    if "n1_violations" not in last:
        last = verify_env_state(env)
    return last


def summarize(rows):
    out = {}
    for mode in sorted({r["mode"] for r in rows}):
        rs = [r for r in rows if r["mode"] == mode]
        n1_over = [float(r["n1_overload_mw"]) for r in rs]
        n1_viol = [int(r["n1_violations"]) for r in rs]
        by_demand = defaultdict(lambda: {"n": 0, "n1_pass": 0, "strict_pass": 0})
        for r in rs:
            key = f"{float(r['demand_gw']):g}"
            by_demand[key]["n"] += 1
            by_demand[key]["n1_pass"] += int(r["n1_violations"] == 0)
            by_demand[key]["strict_pass"] += int(r["passed"])
        out[mode] = {
            "n": len(rs),
            "strict_pass": int(sum(r["passed"] for r in rs)),
            "n1_pass": int(sum(int(v == 0) for v in n1_viol)),
            "n1_violation_rows": int(sum(int(v > 0) for v in n1_viol)),
            "n1_overload_mean_mw": float(np.mean(n1_over)) if n1_over else None,
            "n1_overload_max_mw": float(np.max(n1_over)) if n1_over else None,
            "n1_violations_mean": float(np.mean(n1_viol)) if n1_viol else None,
            "by_demand": dict(sorted(by_demand.items(), key=lambda kv: float(kv[0]))),
        }
    return out


def dataset_pressure(samples):
    fields = [
        "n1_violations_before_corr",
        "n1_violations_after_corr",
        "n1_correction_iters",
        "dc_base_overloads",
        "dc_base_flow_max",
    ]
    summary = {"n": len(samples), "available_fields": {}}
    for field in fields:
        vals = [s.get(field) for s in samples if s.get(field) is not None]
        if vals:
            arr = np.asarray(vals, dtype=float)
            summary["available_fields"][field] = {
                "n": int(arr.size),
                "mean": float(arr.mean()),
                "max": float(arr.max()),
                "positive_count": int((arr > 0).sum()),
            }
    status = Counter()
    for s in samples:
        status[str(s.get("n1_correction_method", "unknown"))] += 1
    summary["n1_correction_method_counts"] = dict(status)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(PROJ_DIR / "case300_gt_v10_dataset.json"))
    ap.add_argument("--split-file", default=str(PROJ_DIR / "paper_results" / "route_b_master_split.json"))
    ap.add_argument("--checkpoint", default=str(PROJ_DIR / "rl_scratch_sac" / "best_actor.pth"))
    ap.add_argument("--scope", choices=["train", "val", "test", "all"], default="test")
    ap.add_argument("--output", default=str(PROJ_DIR / "paper_results" / "route_b_n1_pressure_audit.json"))
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = [s for s in data["samples"] if s.get("is_complete_sced")]

    if pathlib.Path(args.split_file).exists():
        split = load_master_split(args.split_file)
        if args.scope == "all":
            ids = split["train_ids"] + split["val_ids"] + split["test_ids"]
        else:
            ids = split[f"{args.scope}_ids"]
    else:
        tr, va, te = split_samples_stratified(samples)
        ids = {"train": tr, "val": va, "test": te, "all": tr + va + te}[args.scope]
    if args.max_samples > 0:
        ids = ids[:args.max_samples]

    env = SCEDEnvV21(samples, ids, dataset_meta=data)
    trainer = SACTrainer(env._get_obs_dim(), env.action_dim, device=DEVICE)
    trainer.actor.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE, weights_only=False))
    trainer.actor.eval()

    rows = []
    for i, sid in enumerate(ids, 1):
        s = samples[sid]
        for mode in ("raw_label", "gat_only", "gat_sac"):
            env.reset(sample_idx=sid)
            if mode == "raw_label":
                env.current_power = np.asarray(s["dispatch_68"], dtype=np.float64).copy()
                info = verify_env_state(env)
            elif mode == "gat_only":
                env.current_power = np.asarray(env.gat_power, dtype=np.float64).copy()
                info = verify_env_state(env)
            else:
                info = eval_sac(env, trainer, sid)
                if "runtime_sec" not in info:
                    info = {**verify_env_state(env), **info}
            rows.append({
                "sid": sid,
                "sample_id": s.get("sample_id"),
                "demand_gw": float(s.get("demand_gw", s["demand_mw"] / 1000.0)),
                "mode": mode,
                "passed": bool(pass_info(info)),
                "n1_violations": int(info.get("n1_violations", 999)),
                "n1_overload_mw": float(info.get("n1_overload_mw", 9999.0)),
                "v_violations": int(info.get("v_violations", 999)),
                "overloaded_lines": int(info.get("overloaded_lines", 999)),
                "trafo_overloaded": int(info.get("trafo_overloaded", 999)),
            })
        print(f"[{i}/{len(ids)}] {s.get('sample_id')}")

    result = {
        "dataset": args.dataset,
        "split_file": args.split_file,
        "scope": args.scope,
        "dataset_generation_pressure": dataset_pressure(samples),
        "summary": summarize(rows),
        "rows": rows,
    }
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print("Wrote:", args.output)


if __name__ == "__main__":
    main()

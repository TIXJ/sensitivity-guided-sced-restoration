"""Evaluate Route-B policies on the complete 360-sample generated pool.

The conditional RL benchmark uses the 279 samples pre-screened as reachable.
This script reports the broader pool categories so the paper can distinguish
method pass rate from offline reachability.
"""
from __future__ import annotations

import argparse
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

from train_rlv35_complete import DEVICE, SCEDEnvV21, SACTrainer, adopted_operation_pass


def is_pass(info):
    return bool(adopted_operation_pass(info))


def verify_current(env):
    t0 = time.perf_counter()
    ac = env._run_ac_pf_check()
    n1_over, n1_viol = env._compute_n1_overload_margin()
    info = dict(ac)
    info["n1_overload_mw"] = float(n1_over)
    info["n1_violations"] = int(n1_viol)
    info["runtime_sec"] = time.perf_counter() - t0
    info["passed"] = is_pass(info)
    return info


def eval_sac(env, trainer, sid):
    t0 = time.perf_counter()
    obs = env.reset(sample_idx=sid)
    last = {}
    steps = 0
    for steps in range(env.max_steps + 1):
        raw = trainer.select_action(obs, deterministic=True)
        lo, hi = env.get_action_bounds()
        action = lo + (raw + 1.0) * 0.5 * (hi - lo)
        obs, _r, done, _trunc, last = env.step(action)
        if done:
            break
    if "n1_violations" not in last:
        last = verify_current(env)
    last = dict(last)
    last["runtime_sec"] = time.perf_counter() - t0
    last["steps"] = steps + 1
    last["passed"] = is_pass(last)
    return last


def sample_category(sample, reachable_ids):
    if sample.get("sample_id") in reachable_ids:
        return "initially_infeasible_screened_reachable"
    raw_conv = bool(sample.get("ac_pf_converged", False))
    raw_vv0 = bool(sample.get("ac_vv0", False) or sample.get("strict_ac_pass", False))
    if raw_conv and raw_vv0:
        return "initially_strict_ac_feasible"
    return "screen_not_reachable_or_not_screened"


def summarize(rows):
    out = {}
    for mode in sorted({r["mode"] for r in rows}):
        rs_mode = [r for r in rows if r["mode"] == mode]
        cats = {}
        for cat in sorted({r["category"] for r in rs_mode}):
            rs = [r for r in rs_mode if r["category"] == cat]
            cats[cat] = {
                "n": len(rs),
                "passed": int(sum(r["passed"] for r in rs)),
                "pass_rate_pct": round(100.0 * sum(r["passed"] for r in rs) / max(len(rs), 1), 3),
            }
        out[mode] = {
            "n": len(rs_mode),
            "passed": int(sum(r["passed"] for r in rs_mode)),
            "pass_rate_pct": round(100.0 * sum(r["passed"] for r in rs_mode) / max(len(rs_mode), 1), 3),
            "by_category": cats,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dataset", default=str(PROJ_DIR / "case300_sced_linear_labels_160_170_60each.json"))
    ap.add_argument("--reachable-dataset", default=str(PROJ_DIR / "case300_gt_v10_dataset.json"))
    ap.add_argument("--checkpoint", default=str(PROJ_DIR / "rl_scratch_sac" / "best_actor.pth"))
    ap.add_argument("--output", default=str(PROJ_DIR / "paper_results" / "route_b_full_pool_eval.json"))
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()

    with open(args.pool_dataset, "r", encoding="utf-8") as f:
        pool_data = json.load(f)
    pool_samples = [s for s in pool_data["samples"] if s.get("is_complete_sced", True)]
    if args.max_samples > 0:
        pool_samples = pool_samples[:args.max_samples]

    with open(args.reachable_dataset, "r", encoding="utf-8") as f:
        reachable_data = json.load(f)
    reachable_ids = {
        s.get("sample_id")
        for s in reachable_data["samples"]
        if s.get("is_complete_sced") and s.get("sample_id") is not None
    }

    ids = list(range(len(pool_samples)))
    env = SCEDEnvV21(pool_samples, ids, dataset_meta=pool_data)
    trainer = SACTrainer(env._get_obs_dim(), env.action_dim, device=DEVICE)
    trainer.actor.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE, weights_only=False))
    trainer.actor.eval()

    rows = []
    for i, sid in enumerate(ids, 1):
        s = pool_samples[sid]
        cat = sample_category(s, reachable_ids)
        for mode in ("raw_label", "gat_only", "gat_sac"):
            env.reset(sample_idx=sid)
            if mode == "raw_label":
                env.current_power = np.asarray(s["dispatch_68"], dtype=np.float64).copy()
                info = verify_current(env)
            elif mode == "gat_only":
                env.current_power = np.asarray(env.gat_power, dtype=np.float64).copy()
                info = verify_current(env)
            else:
                info = eval_sac(env, trainer, sid)
            rows.append({
                "sid": sid,
                "sample_id": s.get("sample_id"),
                "demand_gw": float(s.get("demand_gw", s["demand_mw"] / 1000.0)),
                "category": cat,
                "mode": mode,
                "passed": bool(info.get("passed", is_pass(info))),
                "converged": bool(info.get("converged", False)),
                "v_violations": int(info.get("v_violations", 999)),
                "n1_violations": int(info.get("n1_violations", 999)),
                "runtime_sec": float(info.get("runtime_sec", 0.0)),
            })
        print(f"[{i}/{len(ids)}] {s.get('sample_id')} {cat}")

    result = {
        "pool_dataset": args.pool_dataset,
        "reachable_dataset": args.reachable_dataset,
        "checkpoint": args.checkpoint,
        "summary": summarize(rows),
        "rows": rows,
    }
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print("Wrote:", args.output)


if __name__ == "__main__":
    main()

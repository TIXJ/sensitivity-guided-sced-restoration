"""Low-cost control baselines for the Route-B correction problem.

These are deployment-time baselines, not retrained ablations. They test whether
the success is mainly due to voltage controls or external-grid voltage support.
The mean voltage correction is fitted only on the master train split.
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

from route_b_split_utils import load_master_split
import train_rlv35_complete as _v35

_v35.MAX_STEPS = 50
_v35.MAX_STEPS_HIGH = 70

from train_rlv35_complete import (
    DEVICE, SCEDEnvV21, SACTrainer, adopted_operation_pass,
    split_samples_stratified,
)


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


def fit_mean_voltage_delta(samples, train_ids, env):
    gen_deltas = []
    ext_deltas = []
    base_gen_vm = np.asarray(env.dcnet.net_base.gen.vm_pu.values, dtype=np.float64)
    base_ext_vm = np.asarray(
        env.dcnet.net_base.ext_grid.vm_pu.values if len(env.dcnet.net_base.ext_grid) else [1.0],
        dtype=np.float64,
    )
    for sid in train_ids:
        s = samples[sid]
        if "rescue_gen_vm_pu" in s and s.get("rescue_gen_vm_pu") is not None:
            gen_deltas.append(np.asarray(s["rescue_gen_vm_pu"], dtype=np.float64) - base_gen_vm)
        if "rescue_ext_grid_vm_pu" in s and s.get("rescue_ext_grid_vm_pu") is not None:
            ext_deltas.append(np.asarray(s["rescue_ext_grid_vm_pu"], dtype=np.float64) - base_ext_vm)
    mean_gen = np.mean(gen_deltas, axis=0) if gen_deltas else np.zeros_like(base_gen_vm)
    mean_ext = np.mean(ext_deltas, axis=0) if ext_deltas else np.zeros_like(base_ext_vm)
    return mean_gen, mean_ext


def run_masked_sac(env, trainer, sid, mask_mode):
    obs = env.reset(sample_idx=sid)
    p_dim = env._n_shift_dirs + env._n_key_gens
    last = {}
    for _ in range(env.max_steps + 1):
        raw = trainer.select_action(obs, deterministic=True)
        lo, hi = env.get_action_bounds()
        action = lo + (raw + 1.0) * 0.5 * (hi - lo)
        if mask_mode == "no_active_power":
            action[:p_dim] = 0.0
        elif mask_mode == "voltage_only_sac":
            action[:p_dim] = 0.0
        elif mask_mode == "no_ext_grid_v":
            action[-1] = 0.0
        obs, _r, done, _trunc, last = env.step(action)
        if mask_mode == "no_ext_grid_v":
            env.current_ext_grid_vm_pu = np.asarray(
                env.dcnet.net_base.ext_grid.vm_pu.values if len(env.dcnet.net_base.ext_grid) else [1.0],
                dtype=np.float64,
            ).copy()
        if done:
            break
    if "n1_violations" not in last:
        last = verify_current(env)
    return last


def summarize(rows):
    out = {}
    for mode in sorted({r["mode"] for r in rows}):
        rs = [r for r in rows if r["mode"] == mode]
        by = defaultdict(lambda: {"n": 0, "passed": 0})
        for r in rs:
            key = f"{float(r['demand_gw']):g}"
            by[key]["n"] += 1
            by[key]["passed"] += int(r["passed"])
        out[mode] = {
            "n": len(rs),
            "passed": int(sum(r["passed"] for r in rs)),
            "pass_rate_pct": round(100.0 * sum(r["passed"] for r in rs) / max(len(rs), 1), 3),
            "by_demand": dict(sorted(by.items(), key=lambda kv: float(kv[0]))),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(PROJ_DIR / "case300_gt_v10_dataset.json"))
    ap.add_argument("--split-file", default=str(PROJ_DIR / "paper_results" / "route_b_master_split.json"))
    ap.add_argument("--checkpoint", default=str(PROJ_DIR / "rl_scratch_sac" / "best_actor.pth"))
    ap.add_argument("--scope", choices=["val", "test"], default="test")
    ap.add_argument("--output", default=str(PROJ_DIR / "paper_results" / "route_b_control_baselines.json"))
    args = ap.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = [s for s in data["samples"] if s.get("is_complete_sced")]
    if pathlib.Path(args.split_file).exists():
        split = load_master_split(args.split_file)
        train_ids, eval_ids = split["train_ids"], split[f"{args.scope}_ids"]
    else:
        train_ids, val_ids, test_ids = split_samples_stratified(samples)
        eval_ids = val_ids if args.scope == "val" else test_ids

    env = SCEDEnvV21(samples, eval_ids, dataset_meta=data)
    mean_gen_delta, mean_ext_delta = fit_mean_voltage_delta(samples, train_ids, env)

    trainer = SACTrainer(env._get_obs_dim(), env.action_dim, device=DEVICE)
    trainer.actor.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE, weights_only=False))
    trainer.actor.eval()

    rows = []
    modes = [
        "gat_only_default_v",
        "fixed_mean_voltage_delta",
        "voltage_only_sac",
        "no_active_power",
        "no_ext_grid_v",
        "gat_sac",
    ]
    for i, sid in enumerate(eval_ids, 1):
        s = samples[sid]
        for mode in modes:
            env.reset(sample_idx=sid)
            if mode == "gat_only_default_v":
                env.current_power = np.asarray(env.gat_power, dtype=np.float64).copy()
                info = verify_current(env)
            elif mode == "fixed_mean_voltage_delta":
                env.current_power = np.asarray(env.gat_power, dtype=np.float64).copy()
                env.current_gen_vm_pu = env.current_gen_vm_pu + mean_gen_delta
                env.current_ext_grid_vm_pu = env.current_ext_grid_vm_pu + mean_ext_delta
                info = verify_current(env)
            elif mode == "gat_sac":
                info = run_masked_sac(env, trainer, sid, "none")
            else:
                info = run_masked_sac(env, trainer, sid, mode)
            rows.append({
                "sid": sid,
                "sample_id": s.get("sample_id"),
                "demand_gw": float(s.get("demand_gw", s["demand_mw"] / 1000.0)),
                "mode": mode,
                "passed": bool(is_pass(info)),
                "converged": bool(info.get("converged", False)),
                "v_violations": int(info.get("v_violations", 999)),
                "n1_violations": int(info.get("n1_violations", 999)),
                "reference_source_violations": int(info.get("reference_source_violations", 999)),
                "reason": info.get("reason"),
            })
        print(f"[{i}/{len(eval_ids)}] {s.get('sample_id')}")

    result = {
        "dataset": args.dataset,
        "split_file": args.split_file,
        "scope": args.scope,
        "notes": "Deployment-time control masks/baselines. Mean voltage delta is fitted on train split rescue evidence only.",
        "summary": summarize(rows),
        "rows": rows,
    }
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print("Wrote:", args.output)


if __name__ == "__main__":
    main()

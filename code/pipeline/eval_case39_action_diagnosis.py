"""IEEE-39 action-group diagnosis for the screened correction benchmark.

This is a deployment-time diagnostic, not a retraining experiment.  It reuses
the trained case-39 SAC policy and masks action groups to show whether the
39-bus result is driven by active-power redispatch, generator-voltage control,
or reference-generator voltage control.
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

import train_rl39_v1 as rl39


def _load_samples(path: pathlib.Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data["samples"] if isinstance(data, dict) and "samples" in data else data
    return data, samples


def _is_pass(info: dict) -> bool:
    return (
        bool(info.get("converged", False))
        and int(info.get("v_violations", 999)) == 0
        and int(info.get("overloaded_lines", 999)) == 0
        and int(info.get("trafo_overloaded", 0)) == 0
        and int(info.get("n1_violations", 999)) == 0
    )


def _reason(info: dict) -> str:
    if _is_pass(info):
        return "full_sced_pass"
    if not info.get("converged", False):
        return "acpf_not_converged"
    if int(info.get("v_violations", 999)) > 0:
        return f"voltage_violation:vv={info.get('v_violations')}:vgap={float(info.get('v_violation_max_pu', 0.0)):.4f}"
    if int(info.get("overloaded_lines", 999)) > 0 or int(info.get("trafo_overloaded", 0)) > 0:
        return "thermal_overload"
    if int(info.get("n1_violations", 999)) > 0:
        return "n1_violation"
    return info.get("reason", "not_full_sced")


def _load_actor(trainer, checkpoint: pathlib.Path):
    ckpt = torch.load(str(checkpoint), map_location=rl39.DEVICE, weights_only=False)
    state = ckpt.get("actor", ckpt.get("actor_state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    trainer.actor.load_state_dict(state)
    trainer.actor.eval()
    return ckpt


def _policy_action(trainer, env, obs, mode: str):
    raw = trainer.select_action(obs, det=True)
    lo, hi = env.get_action_bounds()
    action = lo + (raw + 1.0) * 0.5 * (hi - lo)
    n_gen = rl39.N_GEN
    if mode == "no_active_power":
        action[:n_gen] = 0.0
    elif mode == "no_generator_voltage":
        action[n_gen : 2 * n_gen] = 0.0
    elif mode == "no_reference_voltage":
        action[-1] = 0.0
    elif mode == "voltage_only":
        action[:n_gen] = 0.0
    elif mode == "active_power_only":
        action[n_gen:] = 0.0
    elif mode == "full":
        pass
    else:
        raise ValueError(f"unknown mode: {mode}")
    return raw, action


def _run_variant(env, trainer, sid: int, mode: str):
    obs = env.reset(sample_idx=sid)
    total_reward = 0.0
    last_info = {}
    t0 = time.perf_counter()
    old_ac, old_n1 = rl39.AC_PF_CHECK_EVERY, rl39.N1_CHECK_EVERY_STEPS
    rl39.AC_PF_CHECK_EVERY = 1
    rl39.N1_CHECK_EVERY_STEPS = 1
    try:
        for step in range(env.max_steps + 1):
            raw, action = _policy_action(trainer, env, obs, mode)
            obs, reward, done, _trunc, last_info = env.step(action)
            total_reward += float(np.nan_to_num(reward, nan=-1000.0, neginf=-1000.0, posinf=1000.0))
            if done:
                break
    finally:
        rl39.AC_PF_CHECK_EVERY, rl39.N1_CHECK_EVERY_STEPS = old_ac, old_n1
    runtime = time.perf_counter() - t0
    return {
        "passed": _is_pass(last_info),
        "reason": _reason(last_info),
        "steps": int(step + 1),
        "reward": float(total_reward),
        "runtime_sec": float(runtime),
        "v_violations": int(last_info.get("v_violations", 999)),
        "v_gap": float(last_info.get("v_violation_max_pu", 0.0)),
        "n1_violations": int(last_info.get("n1_violations", 999)),
        "overloaded_lines": int(last_info.get("overloaded_lines", 999)),
    }


def _summarize(rows):
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
            "avg_reward": round(float(np.mean([r["reward"] for r in rs])), 3),
            "avg_steps": round(float(np.mean([r["steps"] for r in rs])), 3),
            "avg_runtime_sec": round(float(np.mean([r["runtime_sec"] for r in rs])), 3),
            "fail_reasons": dict(Counter(r["reason"] for r in rs if not r["passed"]).most_common(8)),
            "by_demand": dict(sorted(by.items(), key=lambda kv: float(kv[0]))),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(PROJ_DIR / "case39_routeb_rescuable_dataset.json"))
    ap.add_argument("--checkpoint", default=str(PROJ_DIR / "rl39_routeb_main" / "best_model.pth"))
    ap.add_argument("--scope", choices=["train", "val", "test"], default="test")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="Optional override for smoke tests; 0 keeps the trained environment default.")
    ap.add_argument(
        "--modes",
        default="full,voltage_only,active_power_only,no_active_power,no_generator_voltage,no_reference_voltage",
        help="Comma-separated action variants.",
    )
    ap.add_argument("--output", default=str(PROJ_DIR / "paper_results" / "case39_action_diagnosis.json"))
    args = ap.parse_args()

    data, samples = _load_samples(pathlib.Path(args.dataset))
    train_ids, val_ids, test_ids = rl39.split_samples(samples)
    ids = {"train": train_ids, "val": val_ids, "test": test_ids}[args.scope]
    if args.max_samples:
        ids = ids[: args.max_samples]

    env = rl39.Env39(samples, ids)
    if args.max_steps:
        env.max_steps = int(args.max_steps)
    trainer = rl39.SACTrainer(env._get_obs_dim(), env.action_dim)
    ckpt = _load_actor(trainer, pathlib.Path(args.checkpoint))

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    rows = []
    for i, sid in enumerate(ids, 1):
        sample = samples[sid]
        for mode in modes:
            result = _run_variant(env, trainer, sid, mode)
            rows.append({
                "sid": int(sid),
                "sample_id": sample.get("id", sample.get("sample_id")),
                "demand_gw": float(sample.get("demand_gw", sample["demand_mw"] / 1000.0)),
                "mode": mode,
                **result,
            })
        print(f"[{i}/{len(ids)}] sid={sid} demand={float(sample['demand_mw'])/1000.0:g}GW")

    result = {
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "scope": args.scope,
        "n_eval": len(ids),
        "checkpoint_meta": {k: v for k, v in ckpt.items() if k != "actor"} if isinstance(ckpt, dict) else {},
        "summary": _summarize(rows),
        "rows": rows,
    }
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print("Wrote:", args.output)


if __name__ == "__main__":
    main()

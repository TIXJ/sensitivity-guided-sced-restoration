"""Evaluate case118 action-component masks with the trained SAC actor."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

import train_rl118_v1 as rl118
from train_rl118_v1 import (
    DEVICE,
    N_VOLTAGE_GROUPS,
    PROJ,
    GaussianActor,
    SCEDEnv118,
    split_samples_stratified,
)


DEFAULT_MODES = (
    "full",
    "no_active_power",
    "active_power_only",
    "no_generator_voltage",
    "no_reference_voltage",
)


def apply_mask(action: np.ndarray, mode: str, p_dim: int) -> np.ndarray:
    masked = np.asarray(action, dtype=np.float64).copy()
    if mode == "no_active_power":
        masked[:p_dim] = 0.0
    elif mode == "active_power_only":
        masked[p_dim:] = 0.0
    elif mode == "no_generator_voltage":
        masked[p_dim:p_dim + N_VOLTAGE_GROUPS] = 0.0
    elif mode == "no_reference_voltage":
        masked[p_dim + N_VOLTAGE_GROUPS:] = 0.0
    elif mode != "full":
        raise ValueError(f"Unsupported mode: {mode}")
    return masked


def evaluate_mode(actor, env, samples, sample_ids, mode: str, max_steps: int) -> tuple[dict, list]:
    passed = 0
    rewards = []
    steps_used = []
    runtimes = []
    reasons = Counter()
    by_demand = defaultdict(lambda: {"n": 0, "passed": 0})
    rows = []
    p_dim = env._n_shift_dirs + env._n_key_gens

    for position, sid in enumerate(sample_ids, 1):
        started = time.perf_counter()
        obs = env.reset(sample_idx=sid)
        info = {}
        reward = 0.0
        for step in range(1, max_steps + 1):
            raw_action = actor.select_action(obs, deterministic=True)
            low, high = env.get_action_bounds()
            action = low + (raw_action + 1.0) * 0.5 * (high - low)
            action = apply_mask(action, mode, p_dim)
            obs, reward, done, _, info = env.step(action)
            if done:
                break

        runtime = time.perf_counter() - started
        ok = bool(
            info.get("converged", False)
            and info.get("v_violations", 999) == 0
            and info.get("overloaded_lines", 999) == 0
            and info.get("n1_violations", 999) == 0
        )
        demand = f"{float(samples[sid]['demand_mw']) / 1000.0:g}"
        by_demand[demand]["n"] += 1
        by_demand[demand]["passed"] += int(ok)
        passed += int(ok)
        rewards.append(float(reward))
        steps_used.append(step)
        runtimes.append(runtime)
        reason = "pass" if ok else str(info.get("reason", "unknown"))
        if not ok:
            reasons[reason] += 1
        rows.append({
            "mode": mode,
            "sid": int(sid),
            "sample_id": samples[sid].get("sample_id", str(sid)),
            "demand_gw": float(samples[sid]["demand_mw"]) / 1000.0,
            "passed": ok,
            "steps": int(step),
            "reward": float(reward),
            "reason": reason,
            "v_violations": int(info.get("v_violations", 999)),
            "n1_violations": int(info.get("n1_violations", 999)),
            "runtime_sec": float(runtime),
        })
        print(
            f"[{mode} {position:02d}/{len(sample_ids)}] "
            f"{samples[sid].get('sample_id', sid)} "
            f"{'OK' if ok else reason} steps={step}"
        )

    total = len(sample_ids)
    return ({
        "n": total,
        "passed": passed,
        "pass_rate_pct": 100.0 * passed / max(total, 1),
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "avg_steps": float(np.mean(steps_used)) if steps_used else 0.0,
        "avg_runtime_sec": float(np.mean(runtimes)) if runtimes else 0.0,
        "fail_reasons": dict(reasons),
        "by_demand": dict(by_demand),
    }, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=str(PROJ / "case118_routeb_rescuable_dataset.json"),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(PROJ / "rl118_routeb_main" / "best_model.pth"),
    )
    parser.add_argument("--scope", choices=["train", "val", "test"], default="test")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument(
        "--output",
        default=str(PROJ / "paper_results" / "case118_action_diagnosis.json"),
    )
    args = parser.parse_args()

    rl118.AC_PF_CHECK_EVERY = 1
    rl118.N1_CHECK_EVERY_STEPS = 1

    dataset_path = Path(args.dataset).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    samples = [sample for sample in data["samples"] if sample.get("is_complete_sced")]
    train_ids, val_ids, test_ids = split_samples_stratified(samples)
    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}[args.scope]
    if args.max_samples > 0:
        split_ids = split_ids[:args.max_samples]

    env = SCEDEnv118(samples, split_ids)
    obs_dim = env._get_obs_dim()
    actor = GaussianActor(obs_dim, env.action_dim).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    unknown = sorted(set(modes) - set(DEFAULT_MODES))
    if unknown:
        raise ValueError(f"Unsupported modes: {unknown}")

    summary = {}
    rows = []
    for mode in modes:
        result, mode_rows = evaluate_mode(
            actor, env, samples, split_ids, mode, max(1, args.max_steps)
        )
        summary[mode] = result
        rows.extend(mode_rows)

    output = {
        "dataset": str(dataset_path),
        "checkpoint": str(checkpoint_path),
        "scope": args.scope,
        "n_eval": len(split_ids),
        "limits_modified": False,
        "checkpoint_meta": {
            "update": checkpoint.get("update"),
            "val_rate": checkpoint.get("val_rate"),
        },
        "summary": summary,
        "rows": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Wrote:", output_path)


if __name__ == "__main__":
    main()

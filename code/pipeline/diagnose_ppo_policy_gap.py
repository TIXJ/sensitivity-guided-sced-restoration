"""Diagnose PPO stochastic-training versus actor-mean deployment behavior."""
import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper_results" / "code"))
sys.path.insert(1, str(ROOT))


def demand_key(sample):
    return round(float(sample.get("demand_gw", float(sample["demand_mw"]) / 1000.0)), 3)


def stratified_probe_ids(samples, ids, per_level, demand_level=None):
    by_level = {}
    for sid in ids:
        by_level.setdefault(demand_key(samples[sid]), []).append(sid)
    levels = sorted(by_level)
    if demand_level is not None:
        levels = [min(levels, key=lambda value: abs(value - demand_level))]
    return [sid for level in levels for sid in by_level[level][:per_level]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="rl_scratch_ppo_lodf277_canonical_v8/final_actor.pth")
    ap.add_argument("--split-file", default="rl_scratch_sac_lodf277_pure_stable_v2/split_ids.json")
    ap.add_argument("--dataset", default="case300_gt_v10_dataset_lodf_corrected_277.json")
    ap.add_argument("--gat-checkpoint", default="gatv10_results_lodf_corrected/gatv10_predictor_best.pth")
    ap.add_argument("--per-level", type=int, default=1)
    ap.add_argument("--demand-level", type=float, default=None)
    ap.add_argument("--output", default="paper_results/ppo_policy_gap_diagnosis.json")
    args = ap.parse_args()

    dataset_path = (ROOT / args.dataset).resolve()
    split_path = (ROOT / args.split_file).resolve()
    checkpoint_path = (ROOT / args.checkpoint).resolve()
    output_path = (ROOT / args.output).resolve()
    os.environ["ROUTE_B_GAT_CHECKPOINT"] = str((ROOT / args.gat_checkpoint).resolve())

    import train_rlv35_complete as v35
    from train_rl_comparison_v3 import PPOGaussianActor
    from train_rlv35_complete import SCEDEnvV21, adopted_operation_pass

    v35.AC_PF_CHECK_EVERY = 5
    v35.N1_CHECK_EVERY_STEPS = 5
    v35.MAX_STEPS = 50
    v35.MAX_STEPS_HIGH = 50
    v35.ENFORCE_REF_SOURCE_LIMIT = False

    with open(dataset_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    samples = [s for s in data["samples"] if s.get("is_complete_sced")]
    with open(split_path, "r", encoding="utf-8") as fh:
        split = json.load(fh)

    train_probe = stratified_probe_ids(samples, split["train_ids"], args.per_level, args.demand_level)
    val_probe = stratified_probe_ids(samples, split["val_ids"], args.per_level, args.demand_level)
    all_ids = train_probe + val_probe
    env = SCEDEnvV21(samples, all_ids, dataset_meta=data)
    actor = PPOGaussianActor(env._get_obs_dim(), env.action_dim).to(v35.DEVICE)
    actor.load_state_dict(torch.load(checkpoint_path, map_location=v35.DEVICE, weights_only=True))
    actor.eval()

    def evaluate(ids, stochastic, seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        passed = 0; rows = []; reasons = Counter()
        for sid in ids:
            obs = env.reset(sample_idx=sid); info = {}
            for step in range(env.max_steps + 1):
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=v35.DEVICE).unsqueeze(0)
                    if stochastic:
                        raw, _, _ = actor.sample(obs_t)
                    else:
                        mean, _ = actor(obs_t); raw = torch.tanh(mean)
                    raw = raw.cpu().numpy()[0]
                lo, hi = env.get_action_bounds()
                action = lo + (raw + 1.0) * 0.5 * (hi - lo)
                obs, _, done, _, info = env.step(action)
                if done:
                    break
            ok = info.get("reason") == "full_sced_pass" or adopted_operation_pass(info)
            passed += int(ok)
            if not ok:
                reasons[info.get("reason", "unknown")] += 1
            rows.append({
                "sid": sid,
                "sample_id": samples[sid].get("sample_id"),
                "demand_gw": demand_key(samples[sid]),
                "passed": bool(ok),
                "steps": step + 1,
                "reason": info.get("reason", "unknown"),
            })
            print(
                f"  {'stochastic' if stochastic else 'mean'} sid={sid} "
                f"gw={demand_key(samples[sid]):g} {'PASS' if ok else 'FAIL'} "
                f"steps={step + 1} reason={info.get('reason', 'unknown')}",
                flush=True,
            )
        return {"passed": passed, "total": len(ids), "reasons": dict(reasons), "rows": rows}

    results = {
        "checkpoint": str(checkpoint_path),
        "train_mean": evaluate(train_probe, False, 42),
        "train_stochastic": evaluate(train_probe, True, 42),
        "val_mean": evaluate(val_probe, False, 42),
        "val_stochastic": evaluate(val_probe, True, 42),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    for key in ("train_mean", "train_stochastic", "val_mean", "val_stochastic"):
        item = results[key]
        print(f"{key}: {item['passed']}/{item['total']} reasons={item['reasons']}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()

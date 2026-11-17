"""Evaluate rl_scratch_sac/best_actor.pth on the held-out test split."""

import json
import pathlib
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

CODE_DIR = pathlib.Path(__file__).resolve().parent
PROJ_DIR = CODE_DIR.parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(1, str(PROJ_DIR))

from train_rlv35_complete import SCEDEnvV21, SACTrainer, split_samples_stratified, DEVICE


def is_pass(info):
    return (
        info.get("reason") == "full_sced_pass"
        or (
            info.get("converged", False)
            and info.get("v_violations", 999) == 0
            and info.get("overloaded_lines", 999) == 0
            and info.get("trafo_overloaded", 999) == 0
            and info.get("n1_violations", 999) == 0
        )
    )


def eval_one(env, trainer, sid):
    obs = env.reset(sample_idx=sid)
    total_r = 0.0
    last_info = {}
    for _ in range(env.max_steps + 1):
        raw = trainer.select_action(obs, deterministic=True)
        lo, hi = env.get_action_bounds()
        action = lo + (raw + 1.0) * 0.5 * (hi - lo)
        obs, r, done, _trunc, last_info = env.step(action)
        total_r += float(r)
        if done:
            break
    return is_pass(last_info), total_r, last_info


def main():
    with open(PROJ_DIR / "case300_gt_v10_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = [s for s in data["samples"] if s.get("is_complete_sced")]

    split_path = PROJ_DIR / "rl_scratch_sac" / "split_ids.json"
    if split_path.exists():
        split = json.load(open(split_path, "r", encoding="utf-8"))
        train_ids, val_ids, test_ids = split["train_ids"], split["val_ids"], split["test_ids"]
    else:
        train_ids, val_ids, test_ids = split_samples_stratified(samples)

    env = SCEDEnvV21(samples, test_ids, dataset_meta=data)
    obs_dim, action_dim = env._get_obs_dim(), env.action_dim
    trainer = SACTrainer(obs_dim, action_dim, device=DEVICE)
    ckpt = PROJ_DIR / "rl_scratch_sac" / "best_actor.pth"
    trainer.actor.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
    trainer.actor.eval()

    passed = 0
    rewards = []
    reasons = Counter()
    by_gw = defaultdict(lambda: {"passed": 0, "n": 0})
    rows = []

    for idx, sid in enumerate(test_ids, 1):
        ok, total_r, info = eval_one(env, trainer, sid)
        s = samples[sid]
        gw = float(s.get("demand_gw", s["demand_mw"] / 1000.0))
        passed += int(ok)
        rewards.append(total_r)
        by_gw[f"{gw:g}"]["n"] += 1
        by_gw[f"{gw:g}"]["passed"] += int(ok)
        if not ok:
            reasons[info.get("reason", "unknown")] += 1
        row = {
            "sid": sid,
            "sample_id": s.get("sample_id"),
            "demand_gw": gw,
            "passed": ok,
            "reward": round(total_r, 3),
            "reason": info.get("reason"),
            "converged": info.get("converged"),
            "v_violations": info.get("v_violations"),
            "v_gap": info.get("v_violation_max_pu"),
            "n1_violations": info.get("n1_violations"),
        }
        rows.append(row)
        print(f"{idx:02d}/{len(test_ids)} {s.get('sample_id')} {gw:g}GW {'OK' if ok else 'FAIL'} {info.get('reason')}")

    summary = {
        "checkpoint": str(ckpt),
        "n_test": len(test_ids),
        "passed": passed,
        "pass_rate_pct": round(passed / max(len(test_ids), 1) * 100.0, 3),
        "avg_reward": round(float(np.mean(rewards)) if rewards else 0.0, 3),
        "reasons": dict(reasons),
        "by_demand": dict(sorted(by_gw.items(), key=lambda kv: float(kv[0]))),
        "rows": rows,
    }
    out = PROJ_DIR / "paper_results" / "best_sac_test_eval.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print("Wrote:", out)


if __name__ == "__main__":
    main()

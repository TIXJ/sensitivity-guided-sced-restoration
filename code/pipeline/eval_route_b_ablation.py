"""Route-B ablation evaluation for SAC variants."""

import json
import argparse
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

_v35.MAX_STEPS = 50
_v35.MAX_STEPS_HIGH = 50

from train_rlv35_complete import (
    DEVICE, SCEDEnvV21, SACTrainer, adopted_operation_pass, project_box_sum,
    split_samples_stratified,
)


VARIANT = "default"


class RouteBAblationEnv(SCEDEnvV21):
    def reset(self, sample_idx=None):
        obs = super().reset(sample_idx=sample_idx)
        if "no-gat" in VARIANT:
            p0 = self.PMAX / (self.PMAX.sum() + 1e-8) * self.target_demand
            p0 = project_box_sum(p0, self.PMIN, self.PMAX * 0.99, self.target_demand)
            self.gat_power = p0.astype(np.float64).copy()
            self.current_power = self.gat_power.copy()
            self.cumulative_delta = np.zeros(self.N, dtype=np.float64)
            obs = self._get_obs()
        return obs

    def step(self, action):
        if "no-voltage" in VARIANT:
            action = np.asarray(action, dtype=np.float64).copy()
            p_dim = self._n_shift_dirs + self._n_key_gens
            if len(action) > p_dim:
                action[p_dim:] = 0.0
        return super().step(action)


def is_pass(info):
    return bool(adopted_operation_pass(info))


def eval_one(env, trainer, sid):
    obs = env.reset(sample_idx=sid)
    t0 = time.perf_counter()
    total_reward = 0.0
    last_info = {}
    steps = 0
    for steps in range(env.max_steps):
        raw = trainer.select_action(obs, deterministic=True)
        lo, hi = env.get_action_bounds()
        action = lo + (raw + 1.0) * 0.5 * (hi - lo)
        obs, r, done, _trunc, last_info = env.step(action)
        total_reward += float(np.nan_to_num(r, nan=-100.0, posinf=1000.0, neginf=-1000.0))
        if done:
            break
    return {
        "passed": is_pass(last_info),
        "reward": total_reward,
        "runtime_sec": time.perf_counter() - t0,
        "steps": steps + 1,
        "info": last_info,
    }


def summarize(rows):
    n = len(rows)
    passed = sum(r["passed"] for r in rows)
    by_demand = defaultdict(lambda: {"passed": 0, "n": 0})
    reasons = Counter(r["reason"] for r in rows if not r["passed"])
    for r in rows:
        key = f"{r['demand_gw']:g}"
        by_demand[key]["n"] += 1
        by_demand[key]["passed"] += int(r["passed"])
    return {
        "n": n,
        "passed": passed,
        "pass_rate_pct": round(100.0 * passed / max(n, 1), 3),
        "avg_reward": round(float(np.mean([r["reward"] for r in rows])), 3),
        "avg_runtime_sec": round(float(np.mean([r["runtime_sec"] for r in rows])), 4),
        "avg_steps": round(float(np.mean([r["steps"] for r in rows])), 3),
        "reasons": dict(reasons),
        "by_demand": dict(sorted(by_demand.items(), key=lambda kv: float(kv[0]))),
    }


def main():
    global VARIANT
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(PROJ_DIR / "case300_gt_v10_dataset.json"))
    ap.add_argument("--split-file", default=str(PROJ_DIR / "rl_scratch_sac" / "split_ids.json"))
    ap.add_argument("--checkpoint", default=str(PROJ_DIR / "rl_scratch_sac" / "best_actor.pth"))
    ap.add_argument("--output", default=str(PROJ_DIR / "paper_results" / "route_b_ablation.json"))
    args = ap.parse_args()
    with open(args.dataset, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = [s for s in data["samples"] if s.get("is_complete_sced")]
    split_path = pathlib.Path(args.split_file)
    if split_path.exists():
        test_ids = json.load(open(split_path, "r", encoding="utf-8"))["test_ids"]
    else:
        _train_ids, _val_ids, test_ids = split_samples_stratified(samples)

    variants = ("default", "no-gat", "no-voltage", "no-gat-no-voltage")
    results = {}
    for variant in variants:
        # Deployment-time diagnosis: mask/remove components from the same
        # trained policy instead of loading separately trained variants.
        ckpt = pathlib.Path(args.checkpoint)
        if not ckpt.exists():
            print(f"{variant}: missing {ckpt}, skipped")
            results[variant] = {"missing_checkpoint": str(ckpt)}
            continue
        VARIANT = variant
        env = RouteBAblationEnv(samples, test_ids, dataset_meta=data)
        obs_dim, action_dim = env._get_obs_dim(), env.action_dim
        trainer = SACTrainer(obs_dim, action_dim, device=DEVICE)
        trainer.actor.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
        trainer.actor.eval()
        rows = []
        print(f"\nEvaluating {variant}: {ckpt}")
        for idx, sid in enumerate(test_ids, 1):
            sample = samples[sid]
            out = eval_one(env, trainer, sid)
            info = out["info"]
            row = {
                "sid": sid,
                "sample_id": sample.get("sample_id"),
                "demand_gw": float(sample.get("demand_gw", sample["demand_mw"] / 1000.0)),
                "passed": bool(out["passed"]),
                "reward": float(out["reward"]),
                "runtime_sec": float(out["runtime_sec"]),
                "steps": int(out["steps"]),
                "reason": info.get("reason"),
                "converged": info.get("converged"),
                "v_violations": info.get("v_violations"),
                "v_gap": info.get("v_violation_max_pu"),
                "n1_violations": info.get("n1_violations"),
                "reference_source_violations": info.get("reference_source_violations"),
            }
            rows.append(row)
            print(f"  {idx:02d}/{len(test_ids)} {sample.get('sample_id')} {'OK' if row['passed'] else 'FAIL'} {row['reason']}")
        results[variant] = {"checkpoint": str(ckpt), "summary": summarize(rows), "rows": rows}

    out = pathlib.Path(args.output)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: v.get("summary", v) for k, v in results.items()}, indent=2))
    print("Wrote:", out)


if __name__ == "__main__":
    main()

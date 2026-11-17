"""Route-B RL algorithm comparison on the shared held-out test split.

Expected checkpoints:
  rl_scratch_sac/best_actor.pth
  rl_scratch_ddpg/best_actor.pth
  rl_scratch_td3/best_actor.pth
  rl_scratch_ppo/best_actor.pth
"""

import argparse
import json
import pathlib
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn

CODE_DIR = pathlib.Path(__file__).resolve().parent
PROJ_DIR = CODE_DIR.parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(1, str(PROJ_DIR))

import train_rlv35_complete as _v35
from train_rlv35_complete import (
    SCEDEnvV21,
    SACTrainer,
    adopted_operation_pass,
    split_samples_stratified,
    DEVICE,
)


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class PPOActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        h = self.net(x)
        return self.mean_head(h), self.log_std.clamp(-5.0, 2.0)

    def deterministic_action(self, x):
        mean, _log_std = self.forward(x)
        return torch.tanh(mean)


def is_pass(info):
    return bool(adopted_operation_pass(info))


def build_actor(algo, obs_dim, act_dim):
    if algo == "SAC":
        trainer = SACTrainer(obs_dim, act_dim, device=DEVICE)
        return trainer.actor, "rl_scratch_sac"
    if algo in {"DDPG", "TD3"}:
        return DeterministicActor(obs_dim, act_dim).to(DEVICE), f"rl_scratch_{algo.lower()}"
    if algo == "PPO":
        return PPOActor(obs_dim, act_dim).to(DEVICE), "rl_scratch_ppo"
    raise ValueError(algo)


def select_action(actor, algo, obs):
    obs = np.nan_to_num(np.asarray(obs, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        if algo == "SAC":
            action = actor.select_action(obs_t, deterministic=True)
        elif algo == "PPO":
            action = actor.deterministic_action(obs_t)
        else:
            action = actor(obs_t)
    action_np = action.detach().cpu().numpy()[0]
    return np.clip(np.nan_to_num(action_np, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)


def eval_one(env, actor, algo, sid, deployment="hierarchical"):
    t0 = time.perf_counter()
    obs = env.reset(sample_idx=sid)
    total_reward = 0.0
    last_info = {}
    steps = 0
    total_actor_steps = 0
    deployment_stage = "full_hybrid"
    for steps in range(env.max_steps):
        raw = select_action(actor, algo, obs)
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

    if deployment == "hierarchical" and not is_pass(last_info):
        obs = env.reset(sample_idx=sid)
        total_reward = 0.0
        deployment_stage = "full_hybrid_fallback"
        for steps in range(env.max_steps):
            raw = select_action(actor, algo, obs)
            lo, hi = env.get_action_bounds()
            action = lo + (raw + 1.0) * 0.5 * (hi - lo)
            obs, r, done, _trunc, last_info = env.step(action)
            total_actor_steps += 1
            total_reward += float(np.nan_to_num(r, nan=-100.0, posinf=1000.0, neginf=-1000.0))
            if done:
                break
    return {
        "passed": is_pass(last_info),
        "reward": float(total_reward),
        "runtime_sec": float(time.perf_counter() - t0),
        "steps": int(total_actor_steps),
        "info": last_info,
        "deployment_stage": deployment_stage,
    }


def summarize(rows):
    n = len(rows)
    passed = sum(r["passed"] for r in rows)
    reasons = Counter(r["reason"] for r in rows if not r["passed"])
    by_demand = defaultdict(lambda: {"passed": 0, "n": 0})
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


def resolve_path(value):
    path = pathlib.Path(value)
    return path if path.is_absolute() else PROJ_DIR / path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="case300_gt_v10_dataset_lodf_corrected_277.json")
    ap.add_argument("--split-file", default="rl_scratch_sac_lodf277_pure_stable_v2/split_ids.json")
    ap.add_argument("--algos", default="SAC,DDPG,TD3")
    ap.add_argument("--sac-checkpoint", default="rl_scratch_sac_lodf277_pure_stable_v2/best_actor.pth")
    ap.add_argument("--ddpg-checkpoint", default="rl_scratch_ddpg_lodf277_final_v8/best_actor.pth")
    ap.add_argument("--td3-checkpoint", default="rl_scratch_td3_lodf277_reproduce/best_actor.pth")
    ap.add_argument("--ppo-checkpoint", default="rl_scratch_ppo_lodf277/best_actor.pth")
    ap.add_argument("--deployment", choices=["hierarchical", "full_hybrid"], default="hierarchical")
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--output", default="paper_results/route_b_rl_compare_lodf277.json")
    args = ap.parse_args()

    _v35.MAX_STEPS = args.max_steps
    _v35.MAX_STEPS_HIGH = args.max_steps
    _v35.ENFORCE_REF_SOURCE_LIMIT = True

    dataset_path = resolve_path(args.dataset)
    split_path = resolve_path(args.split_file)
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = [s for s in data["samples"] if s.get("is_complete_sced")]
    if split_path.exists():
        split = json.load(open(split_path, "r", encoding="utf-8"))
        recorded_dataset = split.get("dataset")
        if recorded_dataset and pathlib.Path(recorded_dataset).resolve() != dataset_path.resolve():
            raise ValueError(f"Split dataset mismatch: split={recorded_dataset}, requested={dataset_path}")
        test_ids = split["test_ids"]
    else:
        _train_ids, _val_ids, test_ids = split_samples_stratified(samples)

    env = SCEDEnvV21(samples, test_ids, dataset_meta=data)
    obs_dim, act_dim = env._get_obs_dim(), env.action_dim
    results = {}

    checkpoint_paths = {
        "SAC": resolve_path(args.sac_checkpoint),
        "DDPG": resolve_path(args.ddpg_checkpoint),
        "TD3": resolve_path(args.td3_checkpoint),
        "PPO": resolve_path(args.ppo_checkpoint),
    }
    algos = [name.strip().upper() for name in args.algos.split(",") if name.strip()]
    for algo in algos:
        if algo not in checkpoint_paths:
            raise ValueError(f"Unsupported algorithm: {algo}")
        actor, _dirname = build_actor(algo, obs_dim, act_dim)
        ckpt = checkpoint_paths[algo]
        if not ckpt.exists():
            print(f"{algo}: missing {ckpt}, skipped")
            results[algo] = {"missing_checkpoint": str(ckpt)}
            continue
        actor.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
        actor.eval()
        rows = []
        print(f"\nEvaluating {algo}: {ckpt}")
        for idx, sid in enumerate(test_ids, 1):
            s = samples[sid]
            out = eval_one(env, actor, algo, sid, deployment=args.deployment)
            info = out["info"]
            row = {
                "sid": sid,
                "sample_id": s.get("sample_id"),
                "demand_gw": float(s.get("demand_gw", s["demand_mw"] / 1000.0)),
                "passed": bool(out["passed"]),
                "reward": out["reward"],
                "runtime_sec": out["runtime_sec"],
                "steps": out["steps"],
                "reason": info.get("reason"),
                "converged": info.get("converged"),
                "v_violations": info.get("v_violations"),
                "v_gap": info.get("v_violation_max_pu"),
                "n1_violations": info.get("n1_violations"),
                "deployment_stage": out["deployment_stage"],
            }
            rows.append(row)
            print(f"  {idx:02d}/{len(test_ids)} {s.get('sample_id')} {'OK' if row['passed'] else 'FAIL'} {row['reason']}")
        results[algo] = {"checkpoint": str(ckpt), "summary": summarize(rows), "rows": rows}

    results["_meta"] = {
        "dataset": str(dataset_path.resolve()),
        "split_file": str(split_path.resolve()),
        "test_samples": len(test_ids),
        "deployment": args.deployment,
        "max_steps": args.max_steps,
        "reference_source_limits_enforced": True,
    }
    out = resolve_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: v.get("summary", v) for k, v in results.items() if k != "_meta"}, indent=2))
    print("Wrote:", out)


if __name__ == "__main__":
    main()

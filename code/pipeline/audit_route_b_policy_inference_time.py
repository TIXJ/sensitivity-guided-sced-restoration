"""Micro-benchmark the learned SAC correction policy against AC-OPF solve time.

The full evaluation time contains a common AC-PF/N-1 verifier.  This script
isolates the learned online decision stage by timing the deployed actor network
and a lightweight action-scaling/projection step, then compares it with the
recorded pandapower AC-OPF restoration solve time.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn

PROJ_DIR = pathlib.Path(__file__).resolve().parents[2]


class ActorMean(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        h = self.trunk(torch.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6))
        return torch.tanh(self.mean_head(h))


def project_box_sum_np(x, lo, hi, target_sum, max_iter=20):
    y = np.clip(np.asarray(x, dtype=np.float64), lo, hi)
    target_sum = float(np.clip(target_sum, float(lo.sum()), float(hi.sum())))
    for _ in range(max_iter):
        err = target_sum - float(y.sum())
        if abs(err) < 1e-7:
            break
        if err > 0:
            free = y < hi - 1e-9
        else:
            free = y > lo + 1e-9
        if not np.any(free):
            break
        y[free] += err / float(np.count_nonzero(free))
        y = np.clip(y, lo, hi)
    return y


def percentile_summary(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean_sec": float(arr.mean()),
        "median_sec": float(np.median(arr)),
        "p95_sec": float(np.percentile(arr, 95)),
        "min_sec": float(arr.min()),
        "max_sec": float(arr.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(PROJ_DIR / "rl_scratch_sac" / "best_actor.pth"))
    ap.add_argument("--acopf-json", default=str(PROJ_DIR / "paper_results" / "route_b_acopf_baseline.json"))
    ap.add_argument("--repeats", type=int, default=5000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=42)
    ap.add_argument("--output", default=str(PROJ_DIR / "paper_results" / "route_b_policy_inference_time.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(args.checkpoint, map_location=device, weights_only=False)
    obs_dim = int(sd["trunk.0.weight"].shape[1])
    action_dim = int(sd["mean_head.weight"].shape[0])
    hidden_dim = int(sd["trunk.0.weight"].shape[0])
    actor = ActorMean(obs_dim, action_dim, hidden_dim).to(device)
    actor.load_state_dict(sd, strict=False)
    actor.eval()

    rng = np.random.default_rng(42)
    obs_one = torch.from_numpy(rng.normal(size=(1, obs_dim)).astype(np.float32)).to(device)
    obs_batch = torch.from_numpy(rng.normal(size=(args.batch_size, obs_dim)).astype(np.float32)).to(device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = actor(obs_one)
        sync()

        one_times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            _ = actor(obs_one)
            sync()
            one_times.append(time.perf_counter() - t0)

        batch_times = []
        for _ in range(max(100, args.repeats // 20)):
            t0 = time.perf_counter()
            _ = actor(obs_batch)
            sync()
            batch_times.append(time.perf_counter() - t0)

    # Lightweight action recovery/projection benchmark. This intentionally
    # excludes AC-PF/N-1 verification and nonlinear optimization.
    pmin = np.zeros(68)
    pmax = np.ones(68) * 700.0
    p0 = np.ones(68) * (16000.0 / 68.0)
    lo = np.ones(action_dim) * -0.02
    hi = np.ones(action_dim) * 0.02
    recovery_times = []
    for _ in range(args.repeats):
        raw = rng.uniform(-1.0, 1.0, size=action_dim)
        t0 = time.perf_counter()
        scaled = lo + (raw + 1.0) * 0.5 * (hi - lo)
        # Approximate the deployed active-power recovery cost: form a small
        # correction and rebalance to generator box constraints.
        delta = np.zeros_like(p0)
        n = min(32, action_dim, len(delta))
        delta[:n] = scaled[:n] * 1000.0
        _ = project_box_sum_np(p0 + delta, pmin, pmax, 16000.0)
        recovery_times.append(time.perf_counter() - t0)

    acopf = {}
    acopf_path = pathlib.Path(args.acopf_json)
    if acopf_path.exists():
        d = json.loads(acopf_path.read_text(encoding="utf-8"))
        acopf = d.get("summary", {}).get("ac_opf_restoration", {})

    one = percentile_summary(one_times)
    batch = percentile_summary(batch_times)
    recovery = percentile_summary(recovery_times)
    learned_decision_mean = one["mean_sec"] + recovery["mean_sec"]
    opf_solve = float(acopf.get("avg_opf_runtime_sec", 0.0) or 0.0)
    opf_total = float(acopf.get("avg_runtime_sec", 0.0) or 0.0)

    result = {
        "device": str(device),
        "checkpoint": args.checkpoint,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dim": hidden_dim,
        "single_sample_actor": one,
        "batch_actor": {
            **batch,
            "batch_size": args.batch_size,
            "mean_sec_per_sample": batch["mean_sec"] / float(args.batch_size),
        },
        "action_recovery_projection": recovery,
        "learned_decision_mean_sec": learned_decision_mean,
        "acopf_avg_opf_solve_sec": opf_solve,
        "acopf_avg_total_sec": opf_total,
        "speedup_vs_acopf_solve": opf_solve / learned_decision_mean if learned_decision_mean > 0 else None,
        "batch_speedup_vs_acopf_solve_per_sample": (
            opf_solve / (batch["mean_sec"] / float(args.batch_size) + recovery["mean_sec"])
            if batch["mean_sec"] > 0
            else None
        ),
    }
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("Wrote:", args.output)


if __name__ == "__main__":
    main()

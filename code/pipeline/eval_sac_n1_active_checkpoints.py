"""Audit deterministic V24 SAC checkpoints on the same held-out split.

This script deliberately imports the V24 environment and actor definition
instead of reconstructing either contract.  It reports individual checkpoint
results and the deterministic common-verifier union across checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
CODE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir", default="rl_sac_n1_active_fixed101_v24"
    )
    parser.add_argument(
        "--dataset", default="case300_gt_n1s0575_active_required_fixed32.json"
    )
    parser.add_argument(
        "--gat-checkpoint",
        default="gatv10_results_n1s0575_candidate/gatv10_predictor_best.pth",
    )
    parser.add_argument(
        "--split-file",
        default="paper_results/case300_n1s0575_active_fixed32_split.json",
    )
    parser.add_argument("--scope", choices=("validation", "test"), default="test")
    parser.add_argument(
        "--output",
        default="paper_results/sac_v24_checkpoint_complementarity_test.json",
    )
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def load_clean_module(args: argparse.Namespace):
    """Import the V24 implementation without leaking this script's CLI args."""
    old_argv = sys.argv
    try:
        sys.argv = [str(CODE_DIR / "train_sac_n1_active_clean.py")]
        sys.path.insert(0, str(CODE_DIR))
        clean = importlib.import_module("train_sac_n1_active_clean")
    finally:
        sys.argv = old_argv

    clean.DATASET_PATH = resolve(args.dataset)
    clean.CHECKPOINT_PATH = resolve(args.gat_checkpoint)
    clean.SPLIT_PATH = resolve(args.split_file)
    os.environ["ROUTE_B_GAT_CHECKPOINT"] = str(clean.CHECKPOINT_PATH)
    return clean


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_name(sample: dict, index: int) -> str:
    return str(
        sample.get("sample_id")
        or sample.get("id")
        or sample.get("name")
        or f"sample_{index:04d}"
    )


def actor_action(actor, obs: np.ndarray, device) -> np.ndarray:
    obs_array = np.nan_to_num(
        np.asarray(obs, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6
    )
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs_array, device=device).unsqueeze(0)
        action = actor.select_action(obs_tensor, deterministic=True)[0]
    result = action.detach().cpu().numpy()
    return np.clip(np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)


def evaluate_actor(clean, actor, env, samples: list[dict], ids: list[int]) -> dict:
    rows = []
    reasons = Counter()
    for position, sid in enumerate(ids, start=1):
        obs = env.reset(sample_idx=sid)
        total_reward = 0.0
        last_info = {}
        for step in range(env.max_steps):
            normalized_action = actor_action(actor, obs, clean.core.DEVICE)
            obs, reward, done, _truncated, last_info = env.step(
                clean.scale_action(env, normalized_action)
            )
            total_reward += float(reward)
            if done:
                break

        passed = clean.is_pass(last_info)
        reason = "pass" if passed else str(last_info.get("reason", "unknown"))
        if not passed:
            reasons[reason] += 1
        row = {
            "position": position,
            "sample_index": sid,
            "sample_id": sample_name(samples[sid], sid),
            "passed": bool(passed),
            "reason": reason,
            "steps": step + 1,
            "reward": float(total_reward),
            "n1_overload_mw": float(last_info.get("n1_overload_mw", np.inf)),
            "n1_violations": int(last_info.get("n1_violations", 0)),
            "voltage_potential_pu": float(
                last_info.get("clean_voltage_potential", np.inf)
            ),
        }
        rows.append(row)
        print(
            f"{position:02d}/{len(ids)} {row['sample_id']} "
            f"{'OK' if passed else 'FAIL'} reason={reason} "
            f"n1={row['n1_overload_mw']:.3f}"
        )

    passed_count = sum(int(row["passed"]) for row in rows)
    return {
        "passed": passed_count,
        "total": len(rows),
        "reasons": dict(reasons),
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    clean = load_clean_module(args)
    dataset_path = resolve(args.dataset)
    split_path = resolve(args.split_file)
    checkpoint_dir = resolve(args.checkpoint_dir)

    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    samples = [sample for sample in data["samples"] if sample.get("is_complete_sced")]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    ids = [
        int(index)
        for index in split["val_ids" if args.scope == "validation" else "test_ids"]
    ]

    env = clean.CleanN1Env(samples, ids, data)
    obs_dim = env._get_obs_dim()
    action_dim = env.action_dim
    p_dim = env._n_shift_dirs + env._n_key_gens
    log_std_max = np.full(
        action_dim, clean.ARGS.voltage_log_std_max, dtype=np.float32
    )
    log_std_max[:p_dim] = clean.ARGS.active_log_std_max
    clean.core.SAC_LOG_STD_MAX_VECTOR = log_std_max

    names = (
        "best_actor.pth",
        "best_train_actor.pth",
        "final_actor.pth",
        "latest_actor.pth",
    )
    checkpoints = [checkpoint_dir / name for name in names if (checkpoint_dir / name).exists()]
    if not checkpoints:
        raise FileNotFoundError(f"No actor checkpoints found in {checkpoint_dir}")

    results = {}
    seen_hashes = set()
    for checkpoint in checkpoints:
        digest = file_sha256(checkpoint)
        if digest in seen_hashes:
            print(f"Skipping duplicate checkpoint: {checkpoint.name}")
            continue
        seen_hashes.add(digest)
        print(f"\n=== {checkpoint.name} ===")
        actor = clean.BranchedGaussianActor(
            obs_dim, p_dim, action_dim - p_dim
        ).to(clean.core.DEVICE)
        state = torch.load(checkpoint, map_location=clean.core.DEVICE, weights_only=True)
        actor.load_state_dict(state)
        actor.eval()
        result = evaluate_actor(clean, actor, env, samples, ids)
        result["checkpoint"] = str(checkpoint)
        result["sha256"] = digest
        results[checkpoint.name] = result

    union_rows = []
    ordered_names = list(results)
    for offset, sid in enumerate(ids):
        passing = [
            name for name in ordered_names if results[name]["rows"][offset]["passed"]
        ]
        best_row = None
        if passing:
            best_row = results[passing[0]]["rows"][offset]
        else:
            candidates = [results[name]["rows"][offset] for name in ordered_names]
            best_row = min(
                candidates,
                key=lambda row: (
                    row["n1_overload_mw"], row["voltage_potential_pu"]
                ),
            )
        union_rows.append(
            {
                "sample_index": sid,
                "sample_id": sample_name(samples[sid], sid),
                "passed": bool(passing),
                "passing_checkpoints": passing,
                "selected_checkpoint": passing[0] if passing else None,
                "best_failed_n1_overload_mw": (
                    None if passing else best_row["n1_overload_mw"]
                ),
            }
        )

    union_passed = sum(int(row["passed"]) for row in union_rows)
    payload = {
        "scope": args.scope,
        "dataset": str(dataset_path),
        "split_file": str(split_path),
        "checkpoint_order": ordered_names,
        "individual": results,
        "deterministic_common_verifier_union": {
            "passed": union_passed,
            "total": len(ids),
            "rows": union_rows,
        },
        "interpretation": (
            "The union is a checkpoint-complementarity diagnostic. Deploying it "
            "would require evaluating the fixed checkpoint sequence and accepting "
            "the first candidate that passes the common verifier."
        ),
    }
    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n=== Summary ===")
    for name, result in results.items():
        print(f"{name}: {result['passed']}/{result['total']}")
    print(f"checkpoint union: {union_passed}/{len(ids)}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()

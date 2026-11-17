"""Audit whether saved rescue_* evidence passes the current RL environment.

This is an offline diagnostic only.  It intentionally injects rescue_dispatch_68
and rescue_*_vm_pu into the environment to check whether the dataset screening
criterion and SCEDEnvV21.full_sced_pass criterion are aligned.  It must not be
used for policy training.
"""

import argparse
import json
import os
import pathlib
import sys
from collections import Counter, defaultdict

import numpy as np

CODE_DIR = pathlib.Path(__file__).resolve().parent
PROJ_DIR = CODE_DIR.parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(1, str(PROJ_DIR))

import train_rlv35_complete as rlv35
from train_rlv35_complete import SCEDEnvV21, adopted_operation_pass, split_samples_stratified


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        default=str(PROJ_DIR / "case300_gt_v10_dataset_lodf_corrected_277.json"),
    )
    ap.add_argument(
        "--gat-checkpoint",
        default=str(PROJ_DIR / "gatv10_results_lodf_corrected" / "gatv10_predictor_best.pth"),
    )
    ap.add_argument(
        "--output",
        default=str(PROJ_DIR / "paper_results" / "rescue_env_alignment_lodf_corrected.json"),
    )
    args = ap.parse_args()
    dataset = pathlib.Path(args.dataset)
    os.environ["ROUTE_B_GAT_CHECKPOINT"] = str(pathlib.Path(args.gat_checkpoint).resolve())
    with open(dataset, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = [s for s in data["samples"] if s.get("is_complete_sced")]
    sample_ids = list(range(len(samples)))

    # Force exact checks for this audit.
    rlv35.AC_PF_CHECK_EVERY = 1
    rlv35.N1_CHECK_EVERY_STEPS = 1

    train_ids, _val_ids, _test_ids = split_samples_stratified(samples)
    env = SCEDEnvV21(samples, train_ids, dataset_meta=data)

    total = 0
    missing = 0
    pass_count = 0
    reasons = Counter()
    by_gw = defaultdict(lambda: [0, 0])
    examples = []

    for sid in sample_ids:
        sample = samples[sid]
        gw = float(sample.get("demand_gw", sample["demand_mw"] / 1000.0))
        total += 1
        by_gw[gw][1] += 1

        if not all(k in sample for k in ("rescue_dispatch_68", "rescue_gen_vm_pu", "rescue_ext_grid_vm_pu")):
            missing += 1
            reasons["missing_rescue_fields"] += 1
            continue

        env.reset(sample_idx=sid)
        env.current_power = np.asarray(sample["rescue_dispatch_68"], dtype=np.float64)
        env.current_gen_vm_pu = np.asarray(sample["rescue_gen_vm_pu"], dtype=np.float64)
        env.current_ext_grid_vm_pu = np.asarray(sample["rescue_ext_grid_vm_pu"], dtype=np.float64)
        env.cumulative_delta = env.current_power - env.gat_power
        env.corr_step = 1

        reward, done, info = env._compute_reward()
        ok = adopted_operation_pass(info)
        if ok:
            pass_count += 1
            by_gw[gw][0] += 1
        else:
            reason = info.get("reason", "unknown")
            reasons[reason] += 1
            if len(examples) < 12:
                examples.append({
                    "sid": sid,
                    "sample_id": sample.get("sample_id"),
                    "gw": gw,
                    "reason": reason,
                    "converged": info.get("converged"),
                    "v_violations": info.get("v_violations"),
                    "v_min": info.get("v_min"),
                    "v_max": info.get("v_max"),
                    "v_gap": info.get("v_violation_max_pu"),
                    "n1_violations": info.get("n1_violations"),
                    "n1_overload_mw": info.get("n1_overload_mw"),
                    "overloaded_lines": info.get("overloaded_lines"),
                    "trafo_overloaded": info.get("trafo_overloaded"),
                    "reference_source_violations": info.get("reference_source_violations"),
                    "scheduled_balance_violations": info.get("scheduled_balance_violations"),
                    "generator_bound_violations": info.get("generator_bound_violations"),
                    "reward": reward,
                })

    summary = {
        "dataset": str(dataset),
        "samples": total,
        "missing_rescue_fields": missing,
        "passed": pass_count,
        "pass_rate_pct": round(pass_count / max(total, 1) * 100.0, 3),
        "reasons": dict(reasons.most_common()),
        "by_demand": {
            f"{gw:g}": {
                "passed": ok,
                "n": n,
                "rate_pct": round(ok / max(n, 1) * 100.0, 3),
            }
            for gw, (ok, n) in sorted(by_gw.items())
        },
        "examples": examples,
    }

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Dataset:", dataset)
    print("Samples:", total)
    print("Missing rescue fields:", missing)
    print(f"Rescue evidence passes current RL env: {pass_count}/{total} ({pass_count/max(total,1)*100:.1f}%)")
    print("Reasons:", dict(reasons.most_common()))
    print("By demand:")
    for gw in sorted(by_gw):
        ok, n = by_gw[gw]
        print(f"  {gw:g}GW: {ok}/{n} ({ok/max(n,1)*100:.1f}%)")
    print("Examples:")
    for ex in examples:
        print(" ", ex)
    print("Wrote:", out)


if __name__ == "__main__":
    main()

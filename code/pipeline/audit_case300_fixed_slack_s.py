#!/usr/bin/env python3
"""Audit the fixed-reference numerical sensitivity used by case300 labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from generate_case300_gt_v10 import Case300Network
from strict_sced_utils import project_box_sum


ROOT = Path(__file__).resolve().parents[2]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old-dataset",
        default="case300_sced_candidates_n1s0575.json",
        help="Optional existing dataset used only for an old/new S comparison.",
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--radius-mw", type=float, default=420.0)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument(
        "--output",
        default="paper_results/case300_fixed_slack_v1_s_audit.json",
    )
    args = parser.parse_args()

    network = Case300Network()
    rng = np.random.default_rng(args.seed)
    demand = float(np.sum(network.P_ref))
    demand_scale = demand / float(network.base_load)

    mae_trials: list[float] = []
    p95_trials: list[float] = []
    max_trials: list[float] = []
    converged = 0
    for _ in range(args.trials):
        direction = rng.standard_normal(network.n_gen)
        direction -= np.mean(direction)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            continue
        radius = float(args.radius_mw) * float(rng.uniform(0.1, 1.0))
        candidate = project_box_sum(
            network.P_ref + radius * direction / norm,
            network.gen_pmin,
            network.gen_pmax,
            demand,
        )
        actual, ok = network._run_dc_approx_pf(candidate, demand_scale=demand_scale)
        if not ok:
            continue
        predicted = network.F_ref + network.S @ (candidate - network.P_ref)
        error = np.abs(actual - predicted)
        mae_trials.append(float(np.mean(error)))
        p95_trials.append(float(np.percentile(error, 95.0)))
        max_trials.append(float(np.max(error)))
        converged += 1

    if converged == 0:
        raise RuntimeError("No AC-PF audit trial converged")

    old_comparison = None
    old_path = resolve(args.old_dataset)
    if old_path.exists():
        old_data = json.loads(old_path.read_text(encoding="utf-8"))
        old_s = np.asarray(
            old_data["linear_security_model"]["S"], dtype=np.float64
        )
        if old_s.shape == network.S.shape:
            difference = network.S - old_s
            old_comparison = {
                "dataset": str(old_path),
                "frobenius_norm": float(np.linalg.norm(difference)),
                "relative_frobenius_norm": float(
                    np.linalg.norm(difference)
                    / max(np.linalg.norm(old_s), 1e-12)
                ),
                "mean_abs_entry_difference": float(np.mean(np.abs(difference))),
                "max_abs_entry_difference": float(np.max(np.abs(difference))),
            }

    payload = {
        "sensitivity_balance": "fixed_ext_grid_reference_source",
        "reference_dispatch_mw": demand,
        "perturbation_mw": 50.0,
        "audit_seed": int(args.seed),
        "requested_trials": int(args.trials),
        "converged_trials": int(converged),
        "balanced_dispatch_radius_mw": float(args.radius_mw),
        "base_flow_affine_error_mw": {
            "trial_mae": summary(mae_trials),
            "trial_p95": summary(p95_trials),
            "trial_max": summary(max_trials),
        },
        "old_s_comparison": old_comparison,
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()

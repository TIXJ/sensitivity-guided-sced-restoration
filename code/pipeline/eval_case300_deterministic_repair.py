"""Evaluate the deterministic repair used during fixed-slack dataset screening.

The evaluator reruns the repair from the GAT prediction.  It never reads the
stored rescue dispatch or voltage settings in the screened dataset.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import sys
import time
from collections import Counter

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "paper_results" / "code"
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dataset", required=True)
    parser.add_argument("--candidate-split", required=True)
    parser.add_argument("--screen-report", required=True)
    parser.add_argument("--gat-checkpoint", required=True)
    parser.add_argument("--scope", choices=("val", "test"), default="test")
    parser.add_argument("--max-delta-mw", type=float, default=400.0)
    parser.add_argument("--projection-margin", type=float, default=0.995)
    parser.add_argument("--projection-iters", type=int, default=12)
    parser.add_argument("--voltage-iters", type=int, default=35)
    parser.add_argument("--vmin", type=float, default=0.933)
    parser.add_argument("--vmax", type=float, default=1.09)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def scheduled_cost(sample: dict, dispatch: np.ndarray) -> float:
    cp2 = np.asarray(sample["sample_cp2"], dtype=np.float64)
    cp1 = np.asarray(sample["sample_cp1"], dtype=np.float64)
    cp0 = np.asarray(sample["sample_cp0"], dtype=np.float64)
    return float(np.sum(cp2 * dispatch**2 + cp1 * dispatch + cp0))


def main() -> None:
    args = parse_args()
    import screen_case300_gat_active_required as screen
    from full_sced_verifier import FullSCEDVerifier
    from generate_case300_gt_v10 import check_n1_security, optimize_voltage_strict

    # The screening helpers intentionally obtain these functions at module scope.
    screen.check_n1_security = check_n1_security
    screen.optimize_voltage_strict = optimize_voltage_strict

    candidate_path = pathlib.Path(args.candidate_dataset).resolve()
    data = json.loads(candidate_path.read_text(encoding="utf-8"))
    samples = [sample for sample in data["samples"] if sample.get("is_complete_sced", False)]
    split = json.loads(pathlib.Path(args.candidate_split).read_text(encoding="utf-8"))
    screen_payload = json.loads(pathlib.Path(args.screen_report).read_text(encoding="utf-8"))
    selected_status = "active_required_and_recoverable"
    selected_ids = {
        int(row["old_index"])
        for row in screen_payload["records"]
        if row.get("partition") == args.scope and row.get("status") == selected_status
    }
    scope_ids = [int(idx) for idx in split[f"{args.scope}_ids"]]
    ids = [idx for idx in scope_ids if idx in selected_ids]
    if args.max_samples > 0:
        ids = ids[: args.max_samples]
    if not ids:
        raise RuntimeError(f"No {selected_status} samples in the requested {args.scope} partition")

    os.environ["ROUTE_B_GAT_CHECKPOINT"] = str(pathlib.Path(args.gat_checkpoint).resolve())
    from train_rlv35_complete import (
        REF_SOURCE_P_MAX_MW,
        REF_SOURCE_P_MIN_MW,
        REF_SOURCE_Q_MAX_MVAR,
        REF_SOURCE_Q_MIN_MVAR,
        SCEDEnvV21,
    )

    env = SCEDEnvV21(
        samples,
        list(range(len(samples))),
        dataset_meta=data,
        action_encoding="physics32fixed",
        action_basis_seed=20260721,
    )
    lower = np.asarray(env.PMIN, dtype=np.float64)
    upper = np.asarray(env.PMAX, dtype=np.float64) * 0.99
    verifier_net = copy.deepcopy(env._ac_net_template)
    if len(verifier_net.ext_grid):
        verifier_net.ext_grid["min_p_mw"] = REF_SOURCE_P_MIN_MW
        verifier_net.ext_grid["max_p_mw"] = REF_SOURCE_P_MAX_MW
        verifier_net.ext_grid["min_q_mvar"] = REF_SOURCE_Q_MIN_MVAR
        verifier_net.ext_grid["max_q_mvar"] = REF_SOURCE_Q_MAX_MVAR
    verifier = FullSCEDVerifier(
        net_template=verifier_net,
        voltage_min_pu=args.vmin,
        voltage_max_pu=args.vmax,
        check_ext_grid_bounds=True,
        check_dispatch_sum_balance=True,
    )

    # Warm up imports, GAT inference, and the fixed basis outside the timing.
    env.reset(ids[0])
    _ = screen.deployed_active_directions(env)

    rows = []
    for position, idx in enumerate(ids, 1):
        sample = samples[idx]
        demand = float(sample["demand_mw"])
        started = time.perf_counter()
        env.reset(idx)
        gat_dispatch = np.asarray(env.gat_power, dtype=np.float64).copy()
        directions = screen.deployed_active_directions(env)
        projected, projection = screen.solve_n1_in_action_span(
            gat_dispatch,
            lower,
            upper,
            directions,
            env.dcnet,
            args.max_delta_mw,
            args.projection_margin,
            args.projection_iters,
        )
        passed = False
        reason = "linear_projection_failed"
        strict = None
        final_dispatch = None
        if projected is not None:
            passed, gen_vm, ext_vm, strict = screen.strict_voltage_result(
                verifier,
                env._ac_net_template,
                projected,
                demand,
                lower,
                upper,
                args.vmin,
                args.vmax,
                args.voltage_iters,
            )
            reason = "deterministic_repair_pass" if passed else "ac_voltage_repair_failed"
            final_dispatch = projected
        elapsed = time.perf_counter() - started
        row = {
            "candidate_index": int(idx),
            "sample_id": sample.get("sample_id", str(idx)),
            "demand_gw": demand / 1000.0,
            "passed": bool(passed),
            "reason": reason,
            "runtime_sec": float(elapsed),
            "projection": projection,
            "strict": strict,
            "active_delta_l1_mw": None if final_dispatch is None else float(np.abs(final_dispatch - gat_dispatch).sum()),
            "scheduled_cost": None if final_dispatch is None else scheduled_cost(sample, final_dispatch),
        }
        rows.append(row)
        print(
            f"[{position:02d}/{len(ids)}] {row['sample_id']} "
            f"{'PASS' if passed else 'FAIL'} time={elapsed:.3f}s",
            flush=True,
        )

    passed_rows = [row for row in rows if row["passed"]]
    payload = {
        "method": "deterministic_action_span_qp_plus_ac_voltage_jacobian",
        "note": "The evaluation reruns the repair from fresh GAT predictions; stored screening rescue actions are not read.",
        "candidate_dataset": str(candidate_path),
        "candidate_split": str(pathlib.Path(args.candidate_split).resolve()),
        "screen_report": str(pathlib.Path(args.screen_report).resolve()),
        "scope": args.scope,
        "selection_status": selected_status,
        "settings": vars(args),
        "summary": {
            "total": len(rows),
            "passed": len(passed_rows),
            "pass_rate_pct": 100.0 * len(passed_rows) / len(rows),
            "avg_runtime_sec": float(np.mean([row["runtime_sec"] for row in rows])),
            "avg_scheduled_cost": float(np.mean([row["scheduled_cost"] for row in passed_rows])) if passed_rows else None,
            "fail_reasons": dict(Counter(row["reason"] for row in rows if not row["passed"])),
        },
        "rows": rows,
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)
    print(f"Wrote: {output}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Screen linear-SCED labels for RL-rescuable AC-PF feasibility.

The input labels remain linear N-1 SCED dispatches.  This script does not turn
them into AC-OPF labels.  It only searches for evidence that a nearby active-
power redispatch exists that:

  1. preserves generator bounds and power balance,
  2. preserves the linear base/N-1 LODF security check,
  3. passes strict AC-PF, optionally with a limited AC N-1 check.

Samples with such an evidence point are marked as RL-rescuable.  The evidence
dispatch is stored for audit only, not as the supervised GAT label.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
PROJ_DIR = CODE_DIR.parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(1, str(PROJ_DIR))

from full_sced_verifier import FullSCEDVerifier
from generate_case300_gt_v10 import Case300Network, check_n1_security
from strict_sced_utils import project_box_sum, zero_sum_unit


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=str(PROJ_DIR / "case300_gt_v10_dataset.json"))
    p.add_argument("--output", default=str(PROJ_DIR / "paper_results" / "case300_rl_rescuable_screen.json"))
    p.add_argument("--rescuable-dataset", default=str(PROJ_DIR / "paper_results" / "case300_rl_rescuable_dataset.json"))
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--attempts", type=int, default=80)
    p.add_argument("--step-mw", type=float, default=80.0)
    p.add_argument("--max-delta-mw", type=float, default=600.0)
    p.add_argument("--voltage-min", type=float, default=0.933)
    p.add_argument("--voltage-max", type=float, default=1.09)
    p.add_argument("--ignore-ext-grid-bounds", action="store_true")
    p.add_argument("--check-ac-n1", action="store_true",
                   help="Also run AC contingency verification for the rescue evidence.")
    p.add_argument("--max-ac-contingencies", type=int, default=None,
                   help="Limit AC N-1 contingencies when --check-ac-n1 is used.")
    p.add_argument("--seed", type=int, default=20260715)
    p.add_argument("--progress-every", type=int, default=5)
    return p.parse_args()


def linear_secure(dispatch: np.ndarray, dcnet: Case300Network) -> tuple[bool, dict]:
    info = check_n1_security(dispatch, dcnet)
    ok = int(info.get("n_violations", 999999)) == 0
    return ok, info


def strict_ac_pass(
    dispatch: np.ndarray,
    demand_mw: float,
    verifier: FullSCEDVerifier,
    check_ac_n1: bool,
    max_ac_contingencies: int | None,
):
    return verifier.verify_dispatch(
        dispatch,
        demand_mw,
        check_base_ac=True,
        check_all_ac_contingencies=bool(check_ac_n1),
        max_contingencies=max_ac_contingencies,
    )


def candidate_directions(P0: np.ndarray, dcnet: Case300Network) -> list[np.ndarray]:
    directions: list[np.ndarray] = []

    def add(vec):
        unit = zero_sum_unit(vec)
        if unit is None:
            return
        for old in directions:
            if abs(float(np.dot(unit, old))) > 0.92:
                return
        directions.append(unit)

    cp1 = np.asarray(dcnet.cp1, dtype=np.float64)
    pmax = np.asarray(dcnet.gen_pmax, dtype=np.float64)
    headroom = pmax - P0
    downroom = P0 - np.asarray(dcnet.gen_pmin, dtype=np.float64)

    add(-(cp1 - cp1.mean()))
    add(cp1 - cp1.mean())
    add(headroom - headroom.mean())
    add(-(downroom - downroom.mean()))
    add(pmax - pmax.mean())
    add(-(pmax - pmax.mean()))

    try:
        sens = np.sum(np.abs(dcnet.S), axis=0)
        add(sens - sens.mean())
        add(-(sens - sens.mean()))
        _u, _s, vh = np.linalg.svd(dcnet.S, full_matrices=False)
        for j in range(min(4, vh.shape[0])):
            add(vh[j, :])
            add(-vh[j, :])
    except Exception:
        pass

    return directions


def make_candidate(
    P0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    demand_mw: float,
    directions: list[np.ndarray],
    rng: np.random.Generator,
    attempt: int,
    step_mw: float,
    max_delta_mw: float,
) -> np.ndarray:
    if attempt == 0:
        return P0.copy()

    if directions and attempt <= 2 * len(directions):
        direction = directions[(attempt - 1) % len(directions)]
        sign = 1.0 if attempt <= len(directions) else -1.0
        magnitude = step_mw * (1.0 + ((attempt - 1) // max(len(directions), 1)))
        delta = sign * magnitude * direction
    else:
        raw = rng.normal(size=len(P0))
        direction = zero_sum_unit(raw)
        if direction is None:
            direction = np.zeros_like(P0)
        # Heavy-tailed but bounded search: many small moves, some larger moves.
        magnitude = min(max_delta_mw, step_mw * rng.gamma(shape=2.0, scale=1.0))
        delta = magnitude * direction

    delta = np.clip(delta, -max_delta_mw, max_delta_mw)
    return project_box_sum(P0 + delta, lower, upper, demand_mw)


def failure_key(failures):
    if not failures:
        return "none"
    text = str(failures[0])
    return text.split(":")[-1].split("=")[0]


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    with open(args.dataset, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = list(data.get("samples", data))
    samples = [s for s in samples if s.get("is_complete_sced", False)]
    if args.max_samples is not None:
        samples = samples[:args.max_samples]
    if not samples:
        raise SystemExit("No complete linear-SCED samples selected")

    dcnet = Case300Network()
    lower = np.asarray(dcnet.gen_pmin, dtype=np.float64)
    upper = np.asarray(dcnet.gen_pmax, dtype=np.float64) * 0.98
    verifier = FullSCEDVerifier(
        voltage_min_pu=args.voltage_min,
        voltage_max_pu=args.voltage_max,
        n1_voltage_min_pu=args.voltage_min,
        n1_voltage_max_pu=args.voltage_max,
        check_ext_grid_bounds=not args.ignore_ext_grid_bounds,
        check_dispatch_sum_balance=True,
    )

    records = []
    rescuable_samples = []
    by_gw = defaultdict(lambda: {"n": 0, "rescuable": 0})
    failure_counts = Counter()
    t0 = time.time()

    print(
        f"RL-rescuable screening {len(samples)} samples, attempts={args.attempts}, "
        f"strict_v=[{args.voltage_min},{args.voltage_max}], ac_n1={args.check_ac_n1}"
    )

    for idx, sample in enumerate(samples, 1):
        ts = time.time()
        sid = sample.get("sample_id", idx - 1)
        demand = float(sample["demand_mw"])
        gw = str(int(round(demand / 1000.0)))
        P0 = np.asarray(sample["dispatch_68"], dtype=np.float64)
        P0 = project_box_sum(P0, lower, upper, demand)
        dirs = candidate_directions(P0, dcnet)

        best_record = None
        found = False
        last_failures = []
        linear_failures = 0
        ac_attempts = 0

        for attempt in range(max(1, args.attempts)):
            Pcand = make_candidate(
                P0, lower, upper, demand, dirs, rng, attempt,
                args.step_mw, args.max_delta_mw,
            )
            lin_ok, lin_info = linear_secure(Pcand, dcnet)
            if not lin_ok:
                linear_failures += 1
                continue

            ac_attempts += 1
            report = strict_ac_pass(
                Pcand, demand, verifier,
                check_ac_n1=args.check_ac_n1,
                max_ac_contingencies=args.max_ac_contingencies,
            )
            last_failures = list(report.failures)
            if report.passed:
                found = True
                best_record = {
                    "attempt": attempt,
                    "dispatch": Pcand.tolist(),
                    "delta_l1_mw": float(np.abs(Pcand - P0).sum()),
                    "delta_linf_mw": float(np.max(np.abs(Pcand - P0))),
                    "linear_n1": lin_info,
                    "strict_metrics": report.metrics,
                }
                break

        by_gw[gw]["n"] += 1
        if found:
            by_gw[gw]["rescuable"] += 1
            out_sample = copy.deepcopy(sample)
            out_sample["rl_rescuable_strict_acpf"] = True
            out_sample["rl_rescue_evidence"] = {
                "attempt": best_record["attempt"],
                "delta_l1_mw": round(best_record["delta_l1_mw"], 6),
                "delta_linf_mw": round(best_record["delta_linf_mw"], 6),
                "strict_metrics": best_record["strict_metrics"],
            }
            # Audit only.  Do not use this as the GAT supervised label.
            out_sample["rescue_dispatch_68"] = best_record["dispatch"]
            rescuable_samples.append(out_sample)
        else:
            failure_counts[failure_key(last_failures)] += 1

        elapsed = time.time() - ts
        records.append({
            "sample_id": sid,
            "demand_mw": demand,
            "rescuable": bool(found),
            "elapsed_sec": round(elapsed, 3),
            "linear_rejected_candidates": linear_failures,
            "strict_ac_attempts": ac_attempts,
            "last_failures": last_failures[:8],
            "rescue_attempt": None if best_record is None else best_record["attempt"],
            "rescue_delta_l1_mw": None if best_record is None else round(best_record["delta_l1_mw"], 6),
            "rescue_delta_linf_mw": None if best_record is None else round(best_record["delta_linf_mw"], 6),
        })

        if idx == 1 or idx % args.progress_every == 0 or idx == len(samples):
            n_ok = len(rescuable_samples)
            print(
                f"[{idx}/{len(samples)}] rescuable={n_ok}/{idx} "
                f"last={sid} {'YES' if found else 'NO'} {elapsed:.1f}s"
            )
            if not found and last_failures:
                print(f"  last failure: {last_failures[0]}")

    elapsed_total = time.time() - t0
    by_gw_out = {
        gw: {
            "n": v["n"],
            "rescuable": v["rescuable"],
            "rate_pct": round(100.0 * v["rescuable"] / v["n"], 2) if v["n"] else 0.0,
        }
        for gw, v in sorted(by_gw.items(), key=lambda kv: int(kv[0]))
    }
    summary = {
        "dataset": str(Path(args.dataset).resolve()),
        "screened_samples": len(samples),
        "rescuable": len(rescuable_samples),
        "not_rescuable": len(samples) - len(rescuable_samples),
        "rescuable_rate_pct": round(100.0 * len(rescuable_samples) / len(samples), 2),
        "attempts": args.attempts,
        "step_mw": args.step_mw,
        "max_delta_mw": args.max_delta_mw,
        "voltage_min": args.voltage_min,
        "voltage_max": args.voltage_max,
        "check_ac_n1": args.check_ac_n1,
        "max_ac_contingencies": args.max_ac_contingencies,
        "elapsed_sec": round(elapsed_total, 3),
        "by_demand_gw": by_gw_out,
        "top_last_failure_types": failure_counts.most_common(20),
    }

    output = {"summary": summary, "records": records}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    rescue_data = dict(data) if isinstance(data, dict) else {"samples": samples}
    rescue_data["samples"] = rescuable_samples
    rescue_data["rl_rescuable_screen_summary"] = summary
    rescue_path = Path(args.rescuable_dataset)
    rescue_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rescue_path, "w", encoding="utf-8") as f:
        json.dump(rescue_data, f, indent=2)

    print("\nSummary")
    print(json.dumps(summary, indent=2))
    print(f"Wrote screen report: {out_path}")
    print(f"Wrote rescuable dataset: {rescue_path}")


if __name__ == "__main__":
    main()

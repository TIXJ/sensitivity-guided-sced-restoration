#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Screen an existing SCED dataset with the shared strict verifier.

This script answers the first v40 question: how many old "complete" labels
remain valid under the strict AC-PF + AC N-1 verifier.
"""

import argparse
import json
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=str(PROJ_DIR / "case300_gt_v10_dataset.json"))
    p.add_argument("--output", default=str(PROJ_DIR / "paper_results" / "case300_gt_v10_strict_screen.json"))
    p.add_argument("--strict-dataset", default=str(PROJ_DIR / "paper_results" / "case300_gt_v10_strict_dataset.json"))
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-contingencies", type=int, default=None,
                   help="For debugging only. Omit for full AC N-1 screening.")
    p.add_argument("--voltage-min", type=float, default=None)
    p.add_argument("--voltage-max", type=float, default=None)
    p.add_argument("--ignore-ext-grid-bounds", action="store_true")
    p.add_argument("--include-incomplete", action="store_true",
                   help="Also screen samples not marked is_complete_sced.")
    p.add_argument("--demand-min-gw", type=float, default=None)
    p.add_argument("--demand-max-gw", type=float, default=None)
    p.add_argument("--progress-every", type=int, default=5)
    return p.parse_args()


def failure_key(failure):
    parts = str(failure).split(":")
    if len(parts) >= 2 and parts[0] == "n1":
        return "n1:" + parts[-1].split("=")[0]
    return parts[-1].split("=")[0]


def main():
    args = parse_args()
    with open(args.dataset, "r", encoding="utf-8") as f:
        data = json.load(f)

    original_samples = data.get("samples", data)
    samples = original_samples
    if not args.include_incomplete:
        samples = [s for s in samples if s.get("is_complete_sced", False)]
    if args.demand_min_gw is not None:
        samples = [s for s in samples if float(s["demand_mw"]) >= args.demand_min_gw * 1000.0]
    if args.demand_max_gw is not None:
        samples = [s for s in samples if float(s["demand_mw"]) <= args.demand_max_gw * 1000.0]
    if args.max_samples is not None:
        samples = samples[:args.max_samples]
    if not samples:
        raise SystemExit("No samples selected for strict screening")

    verifier = FullSCEDVerifier(
        voltage_min_pu=args.voltage_min,
        voltage_max_pu=args.voltage_max,
        check_ext_grid_bounds=not args.ignore_ext_grid_bounds,
    )
    records = []
    strict_samples = []
    failure_counts = Counter()
    by_gw = defaultdict(lambda: {"n": 0, "passed": 0})
    t0 = time.time()

    print(
        f"Strict screening {len(samples)} samples "
        f"(max_contingencies={args.max_contingencies})"
    )

    for idx, sample in enumerate(samples, 1):
        sample_id = sample.get("sample_id", idx - 1)
        demand = float(sample["demand_mw"])
        gw = str(int(round(demand / 1000.0)))
        dispatch = np.asarray(sample["dispatch_68"], dtype=np.float64)
        ts = time.time()

        report = verifier.verify_dispatch(
            dispatch,
            demand,
            check_base_ac=True,
            check_all_ac_contingencies=True,
            max_contingencies=args.max_contingencies,
        )
        elapsed = time.time() - ts

        by_gw[gw]["n"] += 1
        if report.passed:
            by_gw[gw]["passed"] += 1
            strict_sample = dict(sample)
            strict_sample["strict_sced_verified"] = True
            strict_sample["strict_verifier_metrics"] = report.metrics
            strict_samples.append(strict_sample)
        else:
            for failure in report.failures:
                failure_counts[failure_key(failure)] += 1

        records.append({
            "sample_id": sample_id,
            "demand_mw": demand,
            "passed": bool(report.passed),
            "elapsed_sec": round(elapsed, 3),
            "failures": report.failures,
            "metrics": report.metrics,
        })

        if idx == 1 or idx % args.progress_every == 0 or idx == len(samples):
            passed = sum(1 for r in records if r["passed"])
            print(
                f"[{idx}/{len(samples)}] passed={passed}/{idx} "
                f"last={sample_id} {'PASS' if report.passed else 'FAIL'} "
                f"{elapsed:.1f}s"
            )
            if not report.passed and report.failures:
                print(f"  first failure: {report.failures[0]}")

    elapsed_total = time.time() - t0
    passed_total = len(strict_samples)
    by_gw_out = {
        gw: {
            "n": v["n"],
            "passed": v["passed"],
            "rate_pct": round(100.0 * v["passed"] / v["n"], 2) if v["n"] else 0.0,
        }
        for gw, v in sorted(by_gw.items(), key=lambda kv: int(kv[0]))
    }

    summary = {
        "dataset": str(Path(args.dataset).resolve()),
        "screened_samples": len(samples),
        "passed": passed_total,
        "failed": len(samples) - passed_total,
        "pass_rate_pct": round(100.0 * passed_total / len(samples), 2),
        "max_contingencies": args.max_contingencies,
        "demand_min_gw": args.demand_min_gw,
        "demand_max_gw": args.demand_max_gw,
        "voltage_min": args.voltage_min,
        "voltage_max": args.voltage_max,
        "check_ext_grid_bounds": not args.ignore_ext_grid_bounds,
        "elapsed_sec": round(elapsed_total, 3),
        "by_demand_gw": by_gw_out,
        "top_failure_types": failure_counts.most_common(30),
    }

    output = {
        "summary": summary,
        "records": records,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    strict_data = dict(data) if isinstance(data, dict) else {"samples": original_samples}
    strict_data["samples"] = strict_samples
    strict_data["strict_screen_summary"] = summary
    strict_path = Path(args.strict_dataset)
    strict_path.parent.mkdir(parents=True, exist_ok=True)
    with open(strict_path, "w", encoding="utf-8") as f:
        json.dump(strict_data, f, indent=2)

    print("\nSummary")
    print(json.dumps(summary, indent=2))
    print(f"Wrote screen report: {out_path}")
    print(f"Wrote strict dataset: {strict_path}")


if __name__ == "__main__":
    main()

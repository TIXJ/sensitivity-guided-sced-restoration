#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke-test the shared FullSCEDVerifier on a few dataset dispatches.

This is intentionally small and diagnostic. Use it before launching full
evaluation to catch pandapower/case/field compatibility issues quickly.
"""

import argparse
import json
import sys
import time
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
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--max-contingencies", type=int, default=5)
    p.add_argument("--voltage-min", type=float, default=None)
    p.add_argument("--voltage-max", type=float, default=None)
    p.add_argument("--ignore-ext-grid-bounds", action="store_true")
    p.add_argument("--include-incomplete", action="store_true")
    p.add_argument("--output", default=str(PROJ_DIR / "paper_results" / "strict_sced_smoke.json"))
    return p.parse_args()


def short_failures(failures, n=8):
    failures = list(failures)
    if len(failures) <= n:
        return failures
    return failures[:n] + [f"... {len(failures) - n} more"]


def main():
    args = parse_args()
    with open(args.dataset, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("samples", data)
    if not args.include_incomplete:
        samples = [s for s in samples if s.get("is_complete_sced", False)]
    samples = samples[:args.num_samples]
    if not samples:
        raise SystemExit("No samples selected for smoke verification")

    verifier = FullSCEDVerifier(
        voltage_min_pu=args.voltage_min,
        voltage_max_pu=args.voltage_max,
        check_ext_grid_bounds=not args.ignore_ext_grid_bounds,
    )
    records = []
    t0 = time.time()

    print(f"Smoke verifying {len(samples)} samples, max_contingencies={args.max_contingencies}")
    for i, sample in enumerate(samples, 1):
        dispatch = np.asarray(sample["dispatch_68"], dtype=np.float64)
        demand = float(sample["demand_mw"])
        sample_id = sample.get("sample_id", i - 1)
        ts = time.time()
        report = verifier.verify_dispatch(
            dispatch,
            demand,
            check_base_ac=True,
            check_all_ac_contingencies=True,
            max_contingencies=args.max_contingencies,
        )
        elapsed = time.time() - ts
        rec = {
            "sample_id": sample_id,
            "demand_mw": demand,
            "passed": report.passed,
            "elapsed_sec": round(elapsed, 3),
            "failures": report.failures,
            "metrics": report.metrics,
        }
        records.append(rec)
        status = "PASS" if report.passed else "FAIL"
        print(f"[{i}/{len(samples)}] sample={sample_id} demand={demand:.1f} {status} {elapsed:.1f}s")
        if not report.passed:
            for failure in short_failures(report.failures):
                print(f"  - {failure}")

    out = {
        "dataset": str(Path(args.dataset).resolve()),
        "num_samples": len(samples),
        "max_contingencies": args.max_contingencies,
        "voltage_min": args.voltage_min,
        "voltage_max": args.voltage_max,
        "check_ext_grid_bounds": not args.ignore_ext_grid_bounds,
        "elapsed_sec": round(time.time() - t0, 3),
        "records": records,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

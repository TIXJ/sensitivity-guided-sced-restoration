#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a near-boundary linear-SCED dataset for RL training.

The supervised label remains the linear N-1 SCED dispatch.  This script only
filters scenarios so SAC is trained on cases that are not already strict
AC-PF-feasible, but are close enough that active-power redispatch has a
reasonable chance to move them into the feasible region.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import defaultdict
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
    p.add_argument("--output", default=str(PROJ_DIR / "paper_results" / "case300_linear_sced_rl_trainable_dataset.json"))
    p.add_argument("--report", default=str(PROJ_DIR / "paper_results" / "case300_linear_sced_rl_trainable_report.json"))
    p.add_argument("--voltage-min", type=float, default=0.933)
    p.add_argument("--voltage-max", type=float, default=1.09)
    p.add_argument("--demand-min-gw", type=float, default=17.0)
    p.add_argument("--demand-max-gw", type=float, default=19.0)
    p.add_argument("--max-v-violations", type=int, default=12)
    p.add_argument("--max-v-margin-pu", type=float, default=0.04,
                   help="Max distance outside voltage band. Smaller means closer to feasible.")
    p.add_argument("--max-line-loading", type=float, default=100.0)
    p.add_argument("--max-trafo-loading", type=float, default=100.0)
    p.add_argument("--per-gw", type=int, default=30)
    p.add_argument("--include-already-pass", action="store_true",
                   help="Normally excluded because RL should learn correction cases.")
    p.add_argument("--ignore-ext-grid-bounds", action="store_true")
    return p.parse_args()


def margin(vmin, vmax, lo, hi):
    low = max(0.0, lo - float(vmin))
    high = max(0.0, float(vmax) - hi)
    return max(low, high), low, high


def gw_key_from_mw(demand_mw):
    return f"{float(demand_mw) / 1000.0:g}"


def score_record(rec):
    m = rec["metrics"]
    return (
        rec["v_margin_pu"],
        m.get("base_v_violations", 999),
        abs(float(rec["demand_gw"]) - 18.0),
        rec["sample_id"],
    )


def main():
    args = parse_args()
    with open(args.dataset, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = [s for s in data.get("samples", data) if s.get("is_complete_sced", False)]
    verifier = FullSCEDVerifier(
        voltage_min_pu=args.voltage_min,
        voltage_max_pu=args.voltage_max,
        check_ext_grid_bounds=not args.ignore_ext_grid_bounds,
        check_dispatch_sum_balance=True,
    )

    accepted_by_gw = defaultdict(list)
    rejected = []
    t0 = time.time()

    print(
        f"Building RL-trainable linear-SCED dataset from {len(samples)} samples: "
        f"{args.demand_min_gw:g}-{args.demand_max_gw:g}GW, "
        f"v=[{args.voltage_min},{args.voltage_max}]"
    )

    for idx, sample in enumerate(samples, 1):
        demand = float(sample["demand_mw"])
        gw = demand / 1000.0
        sid = sample.get("sample_id", idx - 1)
        if gw < args.demand_min_gw or gw > args.demand_max_gw:
            continue

        dispatch = np.asarray(sample["dispatch_68"], dtype=np.float64)
        report = verifier.verify_dispatch(
            dispatch,
            demand,
            check_base_ac=True,
            check_all_ac_contingencies=False,
        )
        metrics = report.metrics
        reason = None

        if not metrics.get("base_v_min") or not metrics.get("base_v_max"):
            reason = "ac_pf_not_converged"
        else:
            v_margin, v_low_margin, v_high_margin = margin(
                metrics["base_v_min"], metrics["base_v_max"],
                args.voltage_min, args.voltage_max,
            )
            v_viol = int(metrics.get("base_v_violations", 999))
            line_loading = float(metrics.get("base_line_loading_max", 0.0))
            trafo_loading = float(metrics.get("base_trafo_loading_max", 0.0))

            if report.passed and not args.include_already_pass:
                reason = "already_strict_acpf_pass"
            elif v_viol <= 0:
                reason = "no_voltage_violation"
            elif v_viol > args.max_v_violations:
                reason = "too_many_voltage_violations"
            elif v_margin > args.max_v_margin_pu:
                reason = "too_far_from_voltage_band"
            elif line_loading > args.max_line_loading:
                reason = "line_overload"
            elif trafo_loading > args.max_trafo_loading:
                reason = "trafo_overload"
            else:
                rec = {
                    "sample_id": sid,
                    "demand_gw": float(gw),
                    "demand_gw_key": gw_key_from_mw(demand),
                    "passed_raw_strict_acpf": bool(report.passed),
                    "v_margin_pu": round(float(v_margin), 8),
                    "v_low_margin_pu": round(float(v_low_margin), 8),
                    "v_high_margin_pu": round(float(v_high_margin), 8),
                    "metrics": metrics,
                    "failures": list(report.failures),
                }
                accepted_by_gw[gw_key_from_mw(demand)].append(rec)
                continue

        rejected.append({
            "sample_id": sid,
            "demand_gw": float(gw),
            "demand_gw_key": gw_key_from_mw(demand),
            "reason": reason,
            "metrics": metrics,
            "failures": list(report.failures),
        })

    accepted_records = []
    accepted_ids = set()
    for gw, records in sorted(accepted_by_gw.items(), key=lambda kv: float(kv[0])):
        records = sorted(records, key=score_record)[:args.per_gw]
        accepted_records.extend(records)
        accepted_ids.update(r["sample_id"] for r in records)

    out_samples = []
    for sample in samples:
        sid = sample.get("sample_id")
        if sid not in accepted_ids:
            continue
        new_sample = copy.deepcopy(sample)
        rec = next(r for r in accepted_records if r["sample_id"] == sid)
        new_sample["rl_trainable_near_boundary"] = True
        new_sample["raw_strict_acpf_metrics"] = rec["metrics"]
        new_sample["raw_strict_acpf_failures"] = rec["failures"]
        new_sample["raw_strict_acpf_v_margin_pu"] = rec["v_margin_pu"]
        out_samples.append(new_sample)

    elapsed = time.time() - t0
    by_gw = defaultdict(lambda: {"accepted": 0})
    for s in out_samples:
        by_gw[gw_key_from_mw(float(s["demand_mw"]))]["accepted"] += 1
    reject_counts = defaultdict(int)
    for r in rejected:
        reject_counts[str(r["reason"])] += 1

    summary = {
        "source_dataset": str(Path(args.dataset).resolve()),
        "output_dataset": str(Path(args.output).resolve()),
        "source_complete_samples": len(samples),
        "accepted_samples": len(out_samples),
        "voltage_min": args.voltage_min,
        "voltage_max": args.voltage_max,
        "demand_min_gw": args.demand_min_gw,
        "demand_max_gw": args.demand_max_gw,
        "max_v_violations": args.max_v_violations,
        "max_v_margin_pu": args.max_v_margin_pu,
        "per_gw": args.per_gw,
        "by_demand_gw": dict(sorted(by_gw.items(), key=lambda kv: float(kv[0]))),
        "reject_counts": dict(sorted(reject_counts.items())),
        "elapsed_sec": round(elapsed, 3),
    }

    out_data = dict(data) if isinstance(data, dict) else {"samples": samples}
    out_data["samples"] = out_samples
    out_data["rl_trainable_filter_summary"] = summary
    out_data["description"] = (
        "Linear N-1 SCED labels filtered to near-boundary raw strict AC-PF "
        "cases for active-power RL correction. dispatch_68 remains the linear "
        "SCED supervised label."
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    report = {
        "summary": summary,
        "accepted_records": accepted_records,
        "rejected_records": rejected,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote dataset: {out_path}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()

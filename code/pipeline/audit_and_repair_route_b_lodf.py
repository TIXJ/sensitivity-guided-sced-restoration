#!/usr/bin/env python3
"""Audit legacy Route-B labels with a standard DC line-outage LODF model.

Legacy dataset metadata may contain an all-zero LODF matrix. This utility
recomputes LODF from the complete pandapower branch topology, audits every
stored linear-SCED label, and writes corrected metadata only when all audited
labels remain secure under the corrected model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandapower.networks as pn

from generate_case300_gt_v10 import compute_dc_line_lodf


PROJ_DIR = Path(__file__).resolve().parents[2]


def check_dispatch(dispatch, model, tol_mw):
    p = np.asarray(dispatch, dtype=np.float64)
    flows = model["F_ref"] + model["S"] @ (p - model["P_ref"])
    limits = model["limits"]
    base_over = np.maximum(np.abs(flows) - limits, 0.0)
    n_viol = int(np.count_nonzero(base_over > tol_mw))
    max_over = float(base_over.max(initial=0.0))
    worst_kind = "base" if max_over > tol_mw else None
    worst_outage = None
    worst_monitored = int(np.argmax(base_over)) if max_over > tol_mw else None
    worst_flow = float(flows[worst_monitored]) if worst_monitored is not None else None
    worst_limit = float(limits[worst_monitored]) if worst_monitored is not None else None

    for k in model["valid"]:
        post = flows + model["lodf"][:, k] * flows[k]
        post[k] = 0.0
        over = np.maximum(np.abs(post) - limits, 0.0)
        count = int(np.count_nonzero(over > tol_mw))
        local_max = float(over.max(initial=0.0))
        n_viol += count
        if local_max > max_over:
            max_over = local_max
            worst_kind = "n1"
            worst_outage = int(k)
            worst_monitored = int(np.argmax(over))
            worst_flow = float(post[worst_monitored])
            worst_limit = float(limits[worst_monitored])
    return {
        "passed": n_viol == 0,
        "n_violations": n_viol,
        "max_overload_mw": max_over,
        "worst_kind": worst_kind,
        "worst_outage_line": worst_outage,
        "worst_monitored_line": worst_monitored,
        "worst_post_flow_mw": worst_flow,
        "worst_limit_mw": worst_limit,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(PROJ_DIR / "case300_gt_v10_dataset.json"))
    ap.add_argument("--output", default=str(PROJ_DIR / "paper_results" / "route_b_corrected_lodf_audit.json"))
    ap.add_argument("--write-dataset", default="",
                    help="Optional new dataset path; the source file is never overwritten.")
    ap.add_argument(
        "--drop-failing-labels",
        action="store_true",
        help="Write only labels that pass the corrected LODF audit.",
    )
    ap.add_argument("--tol-mw", type=float, default=1e-4)
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    meta = data["linear_security_model"]
    samples = [s for s in data["samples"] if s.get("is_complete_sced", True)]

    print("Computing standard DC line-outage LODF...", flush=True)
    lodf, valid = compute_dc_line_lodf(pn.case300())
    print(f"LODF shape={lodf.shape}, valid={len(valid)}/{lodf.shape[1]}", flush=True)
    model = {
        "P_ref": np.asarray(meta["P_ref"], dtype=np.float64),
        "F_ref": np.asarray(meta["F_ref"], dtype=np.float64),
        "S": np.asarray(meta["S"], dtype=np.float64),
        "limits": np.asarray(meta["effective_limit"], dtype=np.float64),
        "lodf": lodf,
        "valid": valid,
    }

    rows = []
    for i, sample in enumerate(samples, 1):
        row = check_dispatch(sample["dispatch_68"], model, args.tol_mw)
        row.update({
            "index": i - 1,
            "sample_id": sample.get("sample_id"),
            "demand_gw": float(sample.get("demand_gw", sample["demand_mw"] / 1000.0)),
        })
        rows.append(row)
        if i == 1 or i % 20 == 0 or not row["passed"]:
            print(
                f"[{i}/{len(samples)}] passed={sum(r['passed'] for r in rows)}/{i} "
                f"last={row['sample_id']} {'PASS' if row['passed'] else 'FAIL'} "
                f"max_over={row['max_overload_mw']:.4f}MW",
                flush=True,
            )

    passed = int(sum(r["passed"] for r in rows))
    summary = {
        "dataset": str(dataset_path),
        "n": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "pass_rate_pct": 100.0 * passed / max(len(rows), 1),
        "valid_line_outages": int(len(valid)),
        "candidate_lines": int(lodf.shape[1]),
        "max_overload_mw": float(max((r["max_overload_mw"] for r in rows), default=0.0)),
        "metadata_written": False,
    }

    if args.write_dataset:
        out_dataset = Path(args.write_dataset)
        if out_dataset.resolve() == dataset_path.resolve():
            raise RuntimeError("Refusing to overwrite the source dataset")
        if passed != len(rows) and not args.drop_failing_labels:
            print("Refusing to write corrected metadata: one or more labels fail N-1.", flush=True)
        else:
            repaired = dict(data)
            dropped_ids = [r["sample_id"] for r in rows if not r["passed"]]
            if args.drop_failing_labels:
                passing_ids = {r["sample_id"] for r in rows if r["passed"]}
                repaired["samples"] = [
                    s for s in data["samples"] if s.get("sample_id") in passing_ids
                ]
            else:
                repaired["samples"] = list(data["samples"])
            repaired_meta = dict(meta)
            repaired_meta["format"] = "case300_linear_sced_affine_dc_lodf_v2"
            lodf_stored = np.where(np.isfinite(lodf), lodf, 0.0)
            invalid = np.setdiff1d(np.arange(lodf.shape[1]), valid)
            lodf_stored[:, invalid] = 0.0
            repaired_meta["lodf"] = lodf_stored.tolist()
            repaired_meta["valid_contingency_indices"] = valid.tolist()
            repaired["linear_security_model"] = repaired_meta
            repaired["version"] = f"{data.get('version', 'route_b')}_lodf_corrected"
            repaired["lodf_correction_summary"] = {
                "source_dataset": str(dataset_path.resolve()),
                "audited": len(rows),
                "retained": len(repaired["samples"]),
                "dropped_sample_ids": dropped_ids,
                "valid_line_outages": int(len(valid)),
                "tolerance_mw": float(args.tol_mw),
            }
            out_dataset.write_text(json.dumps(repaired, indent=2), encoding="utf-8")
            summary["metadata_written"] = True
            summary["repaired_dataset"] = str(out_dataset)
            summary["retained_samples"] = len(repaired["samples"])
            summary["dropped_sample_ids"] = dropped_ids
            print(f"Wrote corrected dataset: {out_dataset}", flush=True)

    result = {"summary": summary, "failures": [r for r in rows if not r["passed"]], "rows": rows}
    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote audit: {output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a moderate, active-power-required SAC benchmark.

Stored rescue controls are used only to define an offline curation-distance
band. They are removed from the emitted training dataset and never become an
observation, reward input, action target, or policy-training label.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dataset",
        default="case300_gt_n1s0575_active_required_fixed32.json",
    )
    parser.add_argument(
        "--input-split",
        default="paper_results/case300_n1s0575_active_fixed32_split.json",
    )
    parser.add_argument(
        "--output-dataset",
        default="case300_gt_n1s0575_active_required_moderate.json",
    )
    parser.add_argument(
        "--output-split",
        default="paper_results/case300_n1s0575_active_moderate_split.json",
    )
    parser.add_argument(
        "--output-report",
        default="paper_results/case300_n1s0575_active_moderate_report.json",
    )
    parser.add_argument("--max-rescue-linf-mw", type=float, default=100.0)
    parser.add_argument("--max-rescue-l1-mw", type=float, default=1200.0)
    parser.add_argument("--min-rescue-l1-mw", type=float, default=40.0)
    return parser.parse_args()


def resolve(project: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project / path


def main():
    args = parse_args()
    project = Path(__file__).resolve().parents[2]
    input_dataset = resolve(project, args.input_dataset)
    input_split = resolve(project, args.input_split)
    output_dataset = resolve(project, args.output_dataset)
    output_split = resolve(project, args.output_split)
    output_report = resolve(project, args.output_report)

    data = json.loads(input_dataset.read_text(encoding="utf-8"))
    split = json.loads(input_split.read_text(encoding="utf-8"))
    samples = data["samples"]
    partition = {}
    for name in ("train_ids", "val_ids", "test_ids"):
        for idx in split[name]:
            partition[int(idx)] = name

    stripped_fields = (
        "active_required_rescue_dispatch_68",
        "active_required_rescue_gen_vm_pu",
        "active_required_rescue_ext_grid_vm_pu",
        "active_required_rescue_evidence",
    )
    kept_samples = []
    kept_old_indices = []
    metrics = []
    by_demand = Counter()

    for old_idx, sample in enumerate(samples):
        if old_idx not in partition:
            raise RuntimeError(f"Sample index {old_idx} is absent from the input split")
        p0 = np.asarray(sample["gat_start_dispatch_68_audit"], dtype=np.float64)
        rescue = np.asarray(
            sample["active_required_rescue_dispatch_68"], dtype=np.float64
        )
        delta = np.abs(rescue - p0)
        linf = float(np.max(delta))
        l1 = float(np.sum(delta))
        if not (
            args.min_rescue_l1_mw <= l1 <= args.max_rescue_l1_mw
            and linf <= args.max_rescue_linf_mw
        ):
            continue

        clean_sample = copy.deepcopy(sample)
        for field in stripped_fields:
            clean_sample.pop(field, None)
        clean_sample["moderate_active_required_curation"] = True
        kept_samples.append(clean_sample)
        kept_old_indices.append(old_idx)
        metrics.append((l1, linf))
        by_demand[str(sample.get("demand_gw"))] += 1

    old_to_new = {old: new for new, old in enumerate(kept_old_indices)}
    output_split_data = {
        "version": "n1_active_required_moderate_preserved_split_v1",
        "dataset": str(output_dataset),
    }
    for name in ("train_ids", "val_ids", "test_ids"):
        output_split_data[name] = [
            old_to_new[int(idx)] for idx in split[name] if int(idx) in old_to_new
        ]

    if not kept_samples or any(not output_split_data[name] for name in (
        "train_ids", "val_ids", "test_ids"
    )):
        raise RuntimeError("Curation produced an empty dataset partition")

    output_data = copy.deepcopy(data)
    output_data["samples"] = kept_samples
    output_data["n_complete"] = len(kept_samples)
    output_data["moderate_active_required_curation"] = {
        "uses_rescue_only_for_offline_membership_filter": True,
        "rescue_controls_removed_from_output": True,
        "min_rescue_l1_mw": args.min_rescue_l1_mw,
        "max_rescue_l1_mw": args.max_rescue_l1_mw,
        "max_rescue_linf_mw": args.max_rescue_linf_mw,
        "preserves_original_partition_membership": True,
    }

    values_l1 = np.asarray([item[0] for item in metrics], dtype=np.float64)
    values_linf = np.asarray([item[1] for item in metrics], dtype=np.float64)
    report = {
        "input_dataset": str(input_dataset),
        "output_dataset": str(output_dataset),
        "retained": len(kept_samples),
        "split_sizes": {
            name: len(output_split_data[name])
            for name in ("train_ids", "val_ids", "test_ids")
        },
        "by_demand_gw": dict(sorted(by_demand.items())),
        "curation_distance_summary_mw": {
            "l1_min": float(values_l1.min()),
            "l1_mean": float(values_l1.mean()),
            "l1_max": float(values_l1.max()),
            "linf_min": float(values_linf.min()),
            "linf_mean": float(values_linf.mean()),
            "linf_max": float(values_linf.max()),
        },
        "leakage_control": {
            "rescue_dispatch_in_output": False,
            "rescue_voltage_in_output": False,
            "rescue_evidence_in_output": False,
        },
    }

    for path in (output_dataset, output_split, output_report):
        path.parent.mkdir(parents=True, exist_ok=True)
    output_dataset.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    output_split.write_text(json.dumps(output_split_data, indent=2), encoding="utf-8")
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

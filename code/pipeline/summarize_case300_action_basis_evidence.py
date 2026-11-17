"""Summarize the matched physics32, random32, and full68 SAC runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_summary(directory):
    root = Path(directory)
    test = load(root / "test_results.json")
    training = load(root / "training_results.json")
    config = training.get("config", {})
    first_full = None
    for val in training.get("validation", []):
        if val.get("total") and val.get("passed") == val.get("total"):
            first_full = val.get("update")
            break
    return {
        "passed": test.get("passed"),
        "total": test.get("total"),
        "avg_steps": test.get("avg_steps"),
        "first_full_validation_update": first_full,
        "dataset": config.get("dataset"),
        "split_file": config.get("split_file"),
        "train_val_test": config.get("train_val_test"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--physics32-dir", required=True)
    parser.add_argument("--random32-dir", required=True)
    parser.add_argument("--full68-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    physics = test_summary(args.physics32_dir)
    random = test_summary(args.random32_dir)
    full68 = test_summary(args.full68_dir)
    consistency_errors = []
    arms = {"physics32": physics, "random32": random, "full68": full68}
    for name, result in arms.items():
        if result["train_val_test"] != physics["train_val_test"]:
            consistency_errors.append(
                f"{name} split counts differ: "
                f"{result['train_val_test']} versus {physics['train_val_test']}"
            )
        if result["total"] != physics["total"]:
            consistency_errors.append(
                f"{name} test total differs: {result['total']} versus {physics['total']}"
            )
        if Path(result["dataset"] or "").name != Path(physics["dataset"] or "").name:
            consistency_errors.append(
                f"{name} dataset differs: {result['dataset']} versus {physics['dataset']}"
            )
        if Path(result["split_file"] or "").name != Path(physics["split_file"] or "").name:
            consistency_errors.append(
                f"{name} split file differs: "
                f"{result['split_file']} versus {physics['split_file']}"
            )
    if consistency_errors:
        raise ValueError(
            "The action-interface comparison is not matched:\n- "
            + "\n- ".join(consistency_errors)
        )

    payload = {
        "dataset_protocol": {
            "samples": sum(physics["train_val_test"]),
            "train_val_test": physics["train_val_test"],
            "test_samples": physics["total"],
        },
        "action_interface_comparison": {
            "ptdf_lodf_32d": physics,
            "random_balanced_32d": random,
            "generatorwise_68d": full68,
        },
    }
    Path(args.output).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

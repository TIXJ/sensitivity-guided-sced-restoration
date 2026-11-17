"""Create the shared Route-B master split for all 300-bus experiments."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

CODE_DIR = pathlib.Path(__file__).resolve().parent
PROJ_DIR = CODE_DIR.parents[1]
sys.path.insert(0, str(CODE_DIR))

from route_b_split_utils import load_complete_samples, make_stratified_split, split_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(PROJ_DIR / "case300_gt_v10_dataset.json"))
    ap.add_argument("--output", default=str(PROJ_DIR / "paper_results" / "route_b_master_split.json"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-ratio", type=float, default=0.70)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--also-write-rl-split", action="store_true",
                    help="Also write split_ids.json into rl_scratch_sac for legacy evaluators.")
    args = ap.parse_args()

    data, samples = load_complete_samples(args.dataset)
    train_ids, val_ids, test_ids = make_stratified_split(
        samples, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed
    )
    split = {
        "version": "route_b_master_split_v1",
        "dataset": str(pathlib.Path(args.dataset).resolve()),
        "seed": args.seed,
        "sample_filter": "is_complete_sced == true",
        "n_samples": len(samples),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "summary": split_summary(samples, {
            "train_ids": train_ids,
            "val_ids": val_ids,
            "test_ids": test_ids,
        }),
    }
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, indent=2), encoding="utf-8")

    if args.also_write_rl_split:
        legacy_dir = PROJ_DIR / "rl_scratch_sac"
        legacy_dir.mkdir(exist_ok=True)
        (legacy_dir / "split_ids.json").write_text(
            json.dumps({"train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids}, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(split["summary"], indent=2))
    print("Wrote:", out)
    if args.also_write_rl_split:
        print("Wrote:", PROJ_DIR / "rl_scratch_sac" / "split_ids.json")


if __name__ == "__main__":
    main()


"""Shared Route-B split helpers.

The paper experiments should use one master split across GAT, ML baselines,
RL algorithms, and ablations. Splits are stored as sample indices into the
filtered `is_complete_sced` sample list used by the Route-B scripts.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


def demand_key(sample: dict) -> str:
    gw = float(sample.get("demand_gw", float(sample["demand_mw"]) / 1000.0))
    return f"{gw:g}"


def load_complete_samples(dataset_path: str | Path):
    dataset_path = Path(dataset_path)
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    samples = [s for s in data["samples"] if s.get("is_complete_sced", False)]
    return data, samples


def make_stratified_split(samples, train_ratio=0.70, val_ratio=0.15, seed=42):
    rng = random.Random(seed)
    by_demand = defaultdict(list)
    for idx, sample in enumerate(samples):
        by_demand[demand_key(sample)].append(idx)

    train_ids, val_ids, test_ids = [], [], []
    for key in sorted(by_demand, key=lambda x: float(x)):
        ids = list(by_demand[key])
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n_train + n_val >= n and n >= 3:
            n_train = max(1, n - 2)
            n_val = 1
        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train:n_train + n_val])
        test_ids.extend(ids[n_train + n_val:])

    return train_ids, val_ids, test_ids


def split_summary(samples, split):
    out = {}
    for name in ("train_ids", "val_ids", "test_ids"):
        ids = split[name]
        by = defaultdict(int)
        for idx in ids:
            by[demand_key(samples[idx])] += 1
        out[name.replace("_ids", "")] = {
            "n": len(ids),
            "by_demand_gw": dict(sorted(by.items(), key=lambda kv: float(kv[0]))),
        }
    return out


def load_master_split(path: str | Path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        split = json.load(f)
    for key in ("train_ids", "val_ids", "test_ids"):
        if key not in split:
            raise KeyError(f"{path} missing {key}")
        split[key] = [int(x) for x in split[key]]
    return split


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose strict verifier failures for one dataset sample."""

import argparse
import copy
import json
import sys
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
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--voltage-min", type=float, default=None)
    p.add_argument("--voltage-max", type=float, default=None)
    p.add_argument("--ignore-ext-grid-bounds", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.dataset, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = [s for s in data["samples"] if s.get("is_complete_sced", False)]
    sample = samples[args.sample_index]
    dispatch = np.asarray(sample["dispatch_68"], dtype=np.float64)
    demand = float(sample["demand_mw"])

    verifier = FullSCEDVerifier(
        voltage_min_pu=args.voltage_min,
        voltage_max_pu=args.voltage_max,
        check_ext_grid_bounds=not args.ignore_ext_grid_bounds,
    )
    net = verifier._fresh_net(dispatch, demand)
    err = verifier._runpp(net)
    if err:
        print("runpp failed:", err)
        return

    report = verifier.verify_dispatch(
        dispatch,
        demand,
        check_base_ac=True,
        check_all_ac_contingencies=False,
    )

    print("sample_id:", sample.get("sample_id"))
    print("demand_mw:", demand)
    print("gen_sum_mw:", float(dispatch.sum()))
    print("passed:", report.passed)
    print("failures:", report.failures)
    print()

    bus = net.bus.copy()
    res_bus = net.res_bus.copy()
    vmin = bus["min_vm_pu"].values.astype(float) if "min_vm_pu" in bus else np.full(len(bus), 0.9)
    vmax = bus["max_vm_pu"].values.astype(float) if "max_vm_pu" in bus else np.full(len(bus), 1.1)
    vm = res_bus.vm_pu.values.astype(float)
    low_idx = np.where(vm < vmin - 1e-9)[0]
    high_idx = np.where(vm > vmax + 1e-9)[0]
    print("bus limit summary:")
    print("  min_vm unique/count:", sorted((float(x), int((vmin == x).sum())) for x in np.unique(vmin))[:10])
    print("  max_vm unique/count:", sorted((float(x), int((vmax == x).sum())) for x in np.unique(vmax))[:10])
    print("  actual vm min/max:", float(vm.min()), float(vm.max()))
    print("  low violations:", len(low_idx))
    for idx in low_idx[:20]:
        print(f"    bus={idx} vm={vm[idx]:.6f} min={vmin[idx]:.6f}")
    print("  high violations:", len(high_idx))
    for idx in high_idx[:20]:
        print(f"    bus={idx} vm={vm[idx]:.6f} max={vmax[idx]:.6f}")
    print()

    print("ext_grid table/result:")
    cols = [c for c in ["bus", "vm_pu", "min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"] if c in net.ext_grid]
    print(net.ext_grid[cols].to_string(index=True))
    print(net.res_ext_grid.to_string(index=True))
    print()

    relaxed = copy.deepcopy(net)
    relaxed.bus.min_vm_pu = 0.88
    relaxed.bus.max_vm_pu = 1.10
    relaxed_err = verifier._runpp(relaxed)
    print("with voltage bounds reset to 0.88-1.10:")
    print("  runpp error:", relaxed_err)
    if not relaxed_err:
        vm2 = relaxed.res_bus.vm_pu.values.astype(float)
        print("  vm min/max:", float(vm2.min()), float(vm2.max()))
        print("  vv low/high:", int((vm2 < 0.88).sum()), int((vm2 > 1.10).sum()))


if __name__ == "__main__":
    main()

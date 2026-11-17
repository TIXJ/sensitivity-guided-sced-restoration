#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-training reachability screen for P+V RL control.

This script answers the question that must be answered before SAC training:

    Does there exist a nearby active-power redispatch P and voltage setpoint V
    that passes strict AC-PF while preserving linear base/N-1 SCED security?

The original sample dispatch_68 remains the linear N-1 SCED label.  Any
rescue_dispatch_68 / rescue_gen_vm_pu fields written by this script are audit
evidence only, not supervised GAT labels.
"""

from __future__ import annotations

import argparse
import copy
import json
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

from generate_case300_gt_v10 import (
    Case300Network,
    check_n1_security,
    optimize_voltage_strict,
)
from full_sced_verifier import FullSCEDVerifier
from strict_sced_utils import project_box_sum, zero_sum_unit


def load_linear_security_model(dcnet: Case300Network, data: dict) -> None:
    meta = data.get("linear_security_model") if isinstance(data, dict) else None
    if not meta:
        print("[WARN] Dataset has no linear_security_model; screening uses fresh Case300Network")
        return
    dcnet.P_ref = np.asarray(meta["P_ref"], dtype=np.float64)
    dcnet.F_ref = np.asarray(meta["F_ref"], dtype=np.float64)
    dcnet.flow_intercept = np.asarray(meta["flow_intercept"], dtype=np.float64)
    dcnet.S = np.asarray(meta["S"], dtype=np.float64)
    stored_lodf = np.asarray(meta["lodf"], dtype=np.float64)
    if stored_lodf.shape == dcnet.lodf.shape and np.count_nonzero(np.abs(stored_lodf) > 1e-8):
        dcnet.lodf = stored_lodf
        saved_valid = np.asarray(meta.get("valid_contingency_indices", []), dtype=int)
        dcnet.valid_contingency_indices = (
            saved_valid if len(saved_valid)
            else np.flatnonzero(np.isfinite(stored_lodf).all(axis=0)).astype(int)
        )
    else:
        print("[WARN] Stored LODF is degenerate; retaining fresh DC LODF")
    dcnet.flow_limit = np.asarray(meta["flow_limit"], dtype=np.float64)
    dcnet.effective_limit = np.asarray(meta["effective_limit"], dtype=np.float64)
    print(
        "[Screen] Loaded dataset affine model; valid line outages="
        f"{len(dcnet.valid_contingency_indices)}"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=str(PROJ_DIR / "paper_results" / "case300_linear_sced_rl_trainable_dataset.json"))
    p.add_argument("--output", default=str(PROJ_DIR / "paper_results" / "case300_rl_rescuable_pv_screen.json"))
    p.add_argument("--rescuable-dataset", default=str(PROJ_DIR / "paper_results" / "case300_rl_rescuable_pv_dataset.json"))
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--attempts", type=int, default=80)
    p.add_argument("--step-mw", type=float, default=120.0)
    p.add_argument("--max-delta-mw", type=float, default=1200.0)
    p.add_argument("--vmin", type=float, default=0.933)
    p.add_argument("--vmax", type=float, default=1.09)
    p.add_argument("--voltage-iters", type=int, default=25)
    p.add_argument("--include-raw-pass", action="store_true",
                   help="Keep samples that already pass before voltage optimization.")
    p.add_argument("--seed", type=int, default=20260715)
    p.add_argument("--progress-every", type=int, default=5)
    return p.parse_args()


def linear_secure(dispatch: np.ndarray, dcnet: Case300Network) -> tuple[bool, dict]:
    info = check_n1_security(dispatch, dcnet)
    return int(info.get("n_violations", 999999)) == 0, info


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
    pmin = np.asarray(dcnet.gen_pmin, dtype=np.float64)
    headroom = pmax - P0
    downroom = P0 - pmin

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
        for j in range(min(6, vh.shape[0])):
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

    if directions and attempt <= 3 * len(directions):
        direction = directions[(attempt - 1) % len(directions)]
        band = (attempt - 1) // len(directions)
        sign = 1.0 if band % 2 == 0 else -1.0
        magnitude = step_mw * (1.0 + 0.5 * band)
        delta = sign * magnitude * direction
    else:
        direction = zero_sum_unit(rng.normal(size=len(P0)))
        if direction is None:
            direction = np.zeros_like(P0)
        magnitude = min(max_delta_mw, step_mw * rng.gamma(shape=2.0, scale=1.5))
        delta = magnitude * direction

    delta = np.clip(delta, -max_delta_mw, max_delta_mw)
    return project_box_sum(P0 + delta, lower, upper, demand_mw)


def failure_key(ac_result: dict, lin_ok: bool) -> str:
    if not lin_ok:
        return "linear_n1_fail"
    if not ac_result.get("converged", False):
        return "ac_pf_not_converged"
    vv = ac_result.get("vv_total", 999)
    if vv and vv > 0:
        return "voltage_violations"
    if ac_result.get("overloaded_lines"):
        return "line_overload"
    if ac_result.get("overloaded_trafos"):
        return "trafo_overload"
    return "unknown"


def gw_key_from_mw(demand_mw):
    return f"{float(demand_mw) / 1000.0:g}"


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    with open(args.dataset, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = [s for s in data.get("samples", data) if s.get("is_complete_sced", False)]
    if args.max_samples is not None:
        samples = samples[:args.max_samples]
    if not samples:
        raise SystemExit("No complete linear-SCED samples selected")

    dcnet = Case300Network()
    load_linear_security_model(dcnet, data)
    lower = np.asarray(dcnet.gen_pmin, dtype=np.float64)
    upper = np.asarray(dcnet.gen_pmax, dtype=np.float64) * 0.98
    raw_verifier = FullSCEDVerifier(
        net_template=dcnet.net_base,
        voltage_min_pu=args.vmin,
        voltage_max_pu=args.vmax,
        check_ext_grid_bounds=False,
        check_dispatch_sum_balance=True,
    )

    records = []
    rescuable_samples = []
    by_gw = defaultdict(lambda: {"n": 0, "rescuable": 0})
    fail_counts = Counter()
    t0 = time.time()

    print(
        f"P+V reachability screening {len(samples)} samples, attempts={args.attempts}, "
        f"v=[{args.vmin},{args.vmax}], voltage_iters={args.voltage_iters}"
    )

    for idx, sample in enumerate(samples, 1):
        ts = time.time()
        sid = sample.get("sample_id", idx - 1)
        demand = float(sample["demand_mw"])
        gw = gw_key_from_mw(demand)
        P0 = project_box_sum(
            np.asarray(sample["dispatch_68"], dtype=np.float64),
            lower, upper, demand,
        )
        raw_report = raw_verifier.verify_dispatch(
            P0,
            demand,
            check_base_ac=True,
            check_all_ac_contingencies=False,
            gen_vm_pu=sample.get("gen_vm_pu"),
            ext_grid_vm_pu=sample.get("ext_grid_vm_pu"),
        )
        raw_pass = bool(raw_report.passed)
        directions = candidate_directions(P0, dcnet)
        found = None
        last_ac = {}
        last_lin_ok = True
        linear_rejects = 0

        if raw_pass and not args.include_raw_pass:
            by_gw[gw]["n"] += 1
            records.append({
                "sample_id": sid,
                "demand_mw": demand,
                "raw_strict_pass": True,
                "rescuable": False,
                "excluded_reason": "already_raw_strict_pass",
                "elapsed_sec": round(time.time() - ts, 3),
                "linear_rejected_candidates": 0,
                "rescue_attempt": None,
                "rescue_delta_l1_mw": None,
                "rescue_delta_linf_mw": None,
                "raw_failures": raw_report.failures,
                "raw_metrics": raw_report.metrics,
                "last_ac_result": {},
            })
            fail_counts["already_raw_strict_pass"] += 1
            if idx == 1 or idx % args.progress_every == 0 or idx == len(samples):
                ok = len(rescuable_samples)
                print(
                    f"[{idx}/{len(samples)}] rescuable={ok}/{idx} "
                    f"last={sid} RAW-PASS {time.time() - ts:.1f}s"
                )
            continue

        for attempt in range(max(1, args.attempts)):
            Pcand = make_candidate(
                P0, lower, upper, demand, directions, rng, attempt,
                args.step_mw, args.max_delta_mw,
            )
            lin_ok, lin_info = linear_secure(Pcand, dcnet)
            last_lin_ok = lin_ok
            if not lin_ok:
                linear_rejects += 1
                continue

            passed, gen_vm, ext_vm, ac_result = optimize_voltage_strict(
                dcnet.net_base, Pcand, demand, dcnet.gen_pmin, dcnet.gen_pmax,
                vmin_strict=args.vmin, vmax_strict=args.vmax,
                max_iter=args.voltage_iters,
            )
            last_ac = ac_result
            if passed:
                found = {
                    "attempt": attempt,
                    "dispatch": Pcand.tolist(),
                    "gen_vm_pu": np.asarray(gen_vm, dtype=float).tolist(),
                    "ext_grid_vm_pu": np.asarray(ext_vm, dtype=float).tolist(),
                    "delta_l1_mw": float(np.abs(Pcand - P0).sum()),
                    "delta_linf_mw": float(np.max(np.abs(Pcand - P0))),
                    "linear_n1": lin_info,
                    "strict_acpf": ac_result,
                }
                break

        by_gw[gw]["n"] += 1
        if found is not None:
            by_gw[gw]["rescuable"] += 1
            out_sample = copy.deepcopy(sample)
            out_sample["rl_rescuable_pv"] = True
            out_sample["raw_strict_acpf_pass_before_rescue"] = raw_pass
            out_sample["raw_strict_acpf_failures_before_rescue"] = list(raw_report.failures)
            out_sample["raw_strict_acpf_metrics_before_rescue"] = dict(raw_report.metrics)
            out_sample["rl_rescue_pv_evidence"] = {
                "attempt": found["attempt"],
                "delta_l1_mw": round(found["delta_l1_mw"], 6),
                "delta_linf_mw": round(found["delta_linf_mw"], 6),
                "strict_acpf": found["strict_acpf"],
            }
            out_sample["rescue_dispatch_68"] = found["dispatch"]
            out_sample["rescue_gen_vm_pu"] = found["gen_vm_pu"]
            out_sample["rescue_ext_grid_vm_pu"] = found["ext_grid_vm_pu"]
            rescuable_samples.append(out_sample)
        else:
            fail_counts[failure_key(last_ac, last_lin_ok)] += 1

        elapsed = time.time() - ts
        records.append({
            "sample_id": sid,
            "demand_mw": demand,
            "raw_strict_pass": raw_pass,
            "rescuable": found is not None,
            "elapsed_sec": round(elapsed, 3),
            "linear_rejected_candidates": linear_rejects,
            "rescue_attempt": None if found is None else found["attempt"],
            "rescue_delta_l1_mw": None if found is None else round(found["delta_l1_mw"], 6),
            "rescue_delta_linf_mw": None if found is None else round(found["delta_linf_mw"], 6),
            "last_ac_result": last_ac,
            "raw_failures": raw_report.failures,
            "raw_metrics": raw_report.metrics,
        })

        if idx == 1 or idx % args.progress_every == 0 or idx == len(samples):
            ok = len(rescuable_samples)
            print(
                f"[{idx}/{len(samples)}] rescuable={ok}/{idx} "
                f"last={sid} {'YES' if found else 'NO'} {elapsed:.1f}s"
            )
            if found is None and last_ac:
                print(
                    f"  last ac: conv={last_ac.get('converged')} "
                    f"vv={last_ac.get('vv_total')} "
                    f"v=[{last_ac.get('vmin')},{last_ac.get('vmax')}]"
                )

    elapsed_total = time.time() - t0
    by_gw_out = {
        gw: {
            "n": v["n"],
            "rescuable": v["rescuable"],
            "rate_pct": round(100.0 * v["rescuable"] / v["n"], 2) if v["n"] else 0.0,
        }
        for gw, v in sorted(by_gw.items(), key=lambda kv: float(kv[0]))
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
        "vmin": args.vmin,
        "vmax": args.vmax,
        "voltage_iters": args.voltage_iters,
        "include_raw_pass": args.include_raw_pass,
        "elapsed_sec": round(elapsed_total, 3),
        "by_demand_gw": by_gw_out,
        "top_failure_types": fail_counts.most_common(20),
    }

    output = {"summary": summary, "records": records}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    rescue_data = dict(data) if isinstance(data, dict) else {"samples": samples}
    rescue_data["samples"] = rescuable_samples
    rescue_data["rl_rescuable_pv_screen_summary"] = summary
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

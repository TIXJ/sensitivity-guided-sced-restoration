#!/usr/bin/env python3
"""Repair only Route-B labels that fail the corrected standard-DC LODF audit.

The sample IDs and ordering are preserved. Each failed label is re-solved with
its stored generation-cost coefficients and the corrected N-1 model. Existing
AC reachability evidence is then re-verified from the repaired label before a
new dataset is written.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandapower.networks as pn
import gurobipy as gp
from gurobipy import GRB

from audit_and_repair_route_b_lodf import check_dispatch
from full_sced_verifier import FullSCEDVerifier
from generate_case300_gt_v10 import (
    check_n1_security,
    compute_dc_line_lodf,
    correct_n1_violations,
    optimize_voltage_strict,
)
from screen_rl_rescuable_pv_samples import candidate_directions, make_candidate


PROJ_DIR = Path(__file__).resolve().parents[2]


class StoredLinearModel:
    def __init__(self, meta, lodf, valid, net):
        self.P_ref = np.asarray(meta["P_ref"], dtype=np.float64)
        self.F_ref = np.asarray(meta["F_ref"], dtype=np.float64)
        self.S = np.asarray(meta["S"], dtype=np.float64)
        self.flow_intercept = self.F_ref - self.S @ self.P_ref
        self.flow_limit = np.asarray(meta["flow_limit"], dtype=np.float64)
        self.effective_limit = np.asarray(meta["effective_limit"], dtype=np.float64)
        self.lodf = np.asarray(lodf, dtype=np.float64)
        self.valid_contingency_indices = np.asarray(valid, dtype=int)
        self.n_line, self.n_gen = self.S.shape

        self.gen_pmin = np.nan_to_num(
            net.gen.min_p_mw.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )
        self.gen_pmax = net.gen.max_p_mw.to_numpy(dtype=float)
        fallback = np.maximum(
            net.gen.p_mw.to_numpy(dtype=float) * 2.5, self.gen_pmin + 10.0
        )
        bad = ~np.isfinite(self.gen_pmax) | (self.gen_pmax <= self.gen_pmin)
        self.gen_pmax[bad] = fallback[bad]
        self.gen_pmax = np.maximum(self.gen_pmax, self.gen_pmin + 10.0)

    def estimate_base_flows(self, dispatch):
        p = np.asarray(dispatch, dtype=np.float64)
        return self.flow_intercept + self.S @ p

    def estimate_post_contingency_flows(self, dispatch, contingency_idx):
        flows = self.estimate_base_flows(dispatch)
        k = int(contingency_idx)
        return flows + self.lodf[:, k] * flows[k]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=str(PROJ_DIR / "case300_gt_v10_dataset.json"))
    p.add_argument(
        "--output-dataset",
        default=str(PROJ_DIR / "case300_gt_v10_dataset_lodf_corrected.json"),
    )
    p.add_argument(
        "--audit-output",
        default=str(PROJ_DIR / "paper_results" / "route_b_linear_label_repair_audit.json"),
    )
    p.add_argument("--tol-mw", type=float, default=1e-4)
    p.add_argument("--max-iters", type=int, default=20)
    p.add_argument("--rescue-attempts", type=int, default=100)
    p.add_argument("--rescue-step-mw", type=float, default=120.0)
    p.add_argument("--rescue-max-delta-mw", type=float, default=1200.0)
    p.add_argument("--voltage-iters", type=int, default=25)
    p.add_argument("--seed", type=int, default=20260718)
    return p.parse_args()


def verifier_metrics(sample, dispatch, verifier):
    report = verifier.verify_dispatch(
        np.asarray(dispatch, dtype=np.float64),
        float(sample["demand_mw"]),
        check_base_ac=True,
        check_all_ac_contingencies=False,
        gen_vm_pu=sample.get("gen_vm_pu"),
        ext_grid_vm_pu=sample.get("ext_grid_vm_pu"),
    )
    return report


def project_seed_to_linear_security(seed, demand, model, max_iters=20):
    """Minimum-distance projection of an AC-useful seed into linear N-1 security."""
    p = np.asarray(seed, dtype=np.float64).copy()
    for iteration in range(max_iters):
        info = check_n1_security(p, model)
        if info["n_violations"] == 0:
            return p, iteration

        m = gp.Model(f"route_b_rescue_projection_{iteration}")
        m.setParam("OutputFlag", 0)
        m.setParam("TimeLimit", 60)
        x = m.addVars(model.n_gen, name="P")
        for i in range(model.n_gen):
            x[i].lb = float(model.gen_pmin[i])
            x[i].ub = float(model.gen_pmax[i])
        m.setObjective(
            gp.quicksum((x[i] - float(seed[i])) ** 2 for i in range(model.n_gen)),
            GRB.MINIMIZE,
        )
        m.addConstr(gp.quicksum(x[i] for i in range(model.n_gen)) == demand)

        for line in range(model.n_line):
            expr = float(model.flow_intercept[line]) + gp.quicksum(
                float(model.S[line, i]) * x[i] for i in range(model.n_gen)
            )
            limit = float(model.effective_limit[line])
            m.addConstr(expr <= limit)
            m.addConstr(-expr <= limit)

        for item in info.get("qp_constraints", []):
            kind, outage, monitored, sign, limit = item
            if kind != "n1":
                continue
            coeff = sign * (
                model.S[monitored, :] +
                model.lodf[monitored, outage] * model.S[outage, :]
            )
            const = sign * (
                model.flow_intercept[monitored] +
                model.lodf[monitored, outage] * model.flow_intercept[outage]
            )
            expr = const + gp.quicksum(
                float(coeff[i]) * x[i] for i in range(model.n_gen)
            )
            m.addConstr(expr <= float(limit))

        m.optimize()
        if m.status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            return None, iteration
        p = np.asarray([x[i].X for i in range(model.n_gen)], dtype=np.float64)
    return (p, max_iters) if check_n1_security(p, model)["n_violations"] == 0 else (None, max_iters)


def evaluate_voltage_candidate(candidate, sample, model, verifier, audit_model, args):
    linear = check_dispatch(candidate, audit_model, args.tol_mw)
    if not linear["passed"]:
        return None
    passed, gen_vm, ext_vm, ac_result = optimize_voltage_strict(
        verifier.net_template,
        candidate,
        float(sample["demand_mw"]),
        model.gen_pmin,
        model.gen_pmax,
        vmin_strict=0.933,
        vmax_strict=1.09,
        max_iter=args.voltage_iters,
    )
    if not passed:
        return None
    final_report = verifier.verify_dispatch(
        np.asarray(candidate, dtype=np.float64),
        float(sample["demand_mw"]),
        check_base_ac=True,
        check_all_ac_contingencies=False,
        gen_vm_pu=np.asarray(gen_vm, dtype=float),
        ext_grid_vm_pu=np.asarray(ext_vm, dtype=float),
    )
    if not final_report.passed:
        return None
    return {
        "dispatch": np.asarray(candidate, dtype=float),
        "gen_vm_pu": np.asarray(gen_vm, dtype=float),
        "ext_grid_vm_pu": np.asarray(ext_vm, dtype=float),
        "linear": linear,
        "strict_acpf": dict(final_report.metrics),
    }


def find_replacement_rescue(sample, p0, model, verifier, audit_model, args, rng):
    """Find new reachability evidence without using it as a supervised label."""
    demand = float(sample["demand_mw"])
    lower = np.asarray(model.gen_pmin, dtype=np.float64)
    upper = np.asarray(model.gen_pmax, dtype=np.float64) * 0.98
    # Direction construction uses marginal generation costs. Preserve each
    # sample's perturbed SCED cost realization rather than a case-wide default.
    model.cp1 = np.asarray(sample["sample_cp1"], dtype=np.float64)
    seeds = [("repaired_label", np.asarray(p0, dtype=np.float64))]
    old_rescue = sample.get("rescue_dispatch_68")
    if old_rescue is not None:
        projected, projection_iters = project_seed_to_linear_security(
            np.asarray(old_rescue, dtype=np.float64), demand, model
        )
        if projected is not None:
            seeds.insert(0, (f"projected_old_rescue:{projection_iters}", projected))

    attempts_per_seed = max(1, int(np.ceil(args.rescue_attempts / len(seeds))))
    global_attempt = 0
    for seed_name, seed in seeds:
        direct = evaluate_voltage_candidate(
            seed, sample, model, verifier, audit_model, args
        )
        if direct is not None:
            direct.update({"attempt": global_attempt, "seed": seed_name})
            return direct

        directions = candidate_directions(seed, model)
        for local_attempt in range(1, attempts_per_seed + 1):
            global_attempt += 1
            candidate = make_candidate(
                seed,
                lower,
                upper,
                demand,
                directions,
                rng,
                local_attempt,
                args.rescue_step_mw,
                args.rescue_max_delta_mw,
            )
            result = evaluate_voltage_candidate(
                candidate, sample, model, verifier, audit_model, args
            )
            if result is not None:
                result.update({"attempt": global_attempt, "seed": seed_name})
                return result
    return None


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    src = Path(args.dataset)
    data = json.loads(src.read_text(encoding="utf-8"))
    repaired = copy.deepcopy(data)
    samples = repaired["samples"]
    meta = repaired["linear_security_model"]

    net = pn.case300()
    print("Computing corrected standard DC LODF...", flush=True)
    lodf, valid = compute_dc_line_lodf(net)
    model = StoredLinearModel(meta, lodf, valid, net)
    audit_model = {
        "P_ref": model.P_ref,
        "F_ref": model.F_ref,
        "S": model.S,
        "limits": model.effective_limit,
        "lodf": model.lodf,
        "valid": model.valid_contingency_indices,
    }
    verifier = FullSCEDVerifier(
        net_template=net,
        voltage_min_pu=0.933,
        voltage_max_pu=1.09,
        check_ext_grid_bounds=False,
        check_dispatch_sum_balance=True,
    )

    initial_failures = []
    repair_rows = []
    for idx, sample in enumerate(samples):
        before = check_dispatch(sample["dispatch_68"], audit_model, args.tol_mw)
        if before["passed"]:
            continue
        initial_failures.append(sample.get("sample_id", idx))
        p0 = np.asarray(sample["dispatch_68"], dtype=np.float64)
        n1_before = check_n1_security(p0, model)
        coeffs = (
            np.asarray(sample["sample_cp2"], dtype=np.float64),
            np.asarray(sample["sample_cp1"], dtype=np.float64),
            np.asarray(sample.get("sample_cp0", np.zeros(model.n_gen)), dtype=np.float64),
        )
        p_new, iterations, method = correct_n1_violations(
            p0,
            float(sample["demand_mw"]),
            model,
            n1_before,
            max_iters=args.max_iters,
            cost_coeffs=coeffs,
        )
        after = check_dispatch(p_new, audit_model, args.tol_mw)
        if not after["passed"]:
            raise RuntimeError(
                f"Focused repair failed for {sample.get('sample_id')}: {after}"
            )

        raw_report = verifier_metrics(sample, p_new, verifier)
        rescue = sample.get("rescue_dispatch_68")
        rescue_linear = None
        rescue_ac = None
        rescue_replaced = False
        if rescue is not None:
            rescue_linear = check_dispatch(rescue, audit_model, args.tol_mw)
            rescue_report = verifier.verify_dispatch(
                np.asarray(rescue, dtype=np.float64),
                float(sample["demand_mw"]),
                check_base_ac=True,
                check_all_ac_contingencies=False,
                gen_vm_pu=sample.get("rescue_gen_vm_pu"),
                ext_grid_vm_pu=sample.get("rescue_ext_grid_vm_pu"),
            )
            rescue_ac = {
                "passed": bool(rescue_report.passed),
                "failures": list(rescue_report.failures),
                "metrics": dict(rescue_report.metrics),
            }
            if not rescue_linear["passed"] or not rescue_report.passed:
                print(
                    f"Stored rescue invalid for {sample.get('sample_id')} "
                    f"(linear={rescue_linear['passed']}, ac={rescue_report.passed}); "
                    "searching for replacement...",
                    flush=True,
                )
                found = find_replacement_rescue(
                    sample, p_new, model, verifier, audit_model, args, rng
                )
                if found is None:
                    raise RuntimeError(
                        f"No replacement rescue found for {sample.get('sample_id')} "
                        f"after {args.rescue_attempts} attempts."
                    )
                rescue_replaced = True
                rescue = found["dispatch"].tolist()
                sample["rescue_dispatch_68"] = rescue
                sample["rescue_gen_vm_pu"] = found["gen_vm_pu"].tolist()
                sample["rescue_ext_grid_vm_pu"] = found["ext_grid_vm_pu"].tolist()
                rescue_linear = found["linear"]
                rescue_ac = {
                    "passed": True,
                    "failures": [],
                    "metrics": found["strict_acpf"],
                }
                delta_found = found["dispatch"] - p_new
                sample["rl_rescue_pv_evidence"] = {
                    "attempt": int(found["attempt"]),
                    "search_seed": found["seed"],
                    "delta_l1_mw": round(float(np.abs(delta_found).sum()), 6),
                    "delta_linf_mw": round(float(np.max(np.abs(delta_found))), 6),
                    "strict_acpf": found["strict_acpf"],
                    "screening_role": "reachability evidence only; not a GAT label",
                }

        cp2, cp1, cp0 = coeffs
        sample["dispatch_68"] = p_new.tolist()
        sample["gen_sum_mw"] = float(p_new.sum())
        sample["sced_cost_per_h"] = round(float(np.sum(cp2 * p_new**2 + cp1 * p_new + cp0)), 2)
        sample["n1_violations_before_corr"] = int(n1_before["n_violations"])
        sample["n1_violations_after_corr"] = 0
        sample["n1_correction_iters"] = int(iterations)
        sample["n1_correction_method"] = f"corrected-standard-lodf:{method}"
        sample["dc_base_flow_max"] = round(float(np.max(np.abs(model.estimate_base_flows(p_new)))), 2)
        sample["dc_base_overloads"] = 0
        sample["linear_sced_feasible"] = True
        sample["linear_base_pass"] = True
        sample["linear_n1_pass"] = True
        sample["raw_strict_acpf_pass_before_rescue"] = bool(raw_report.passed)
        sample["raw_strict_acpf_failures_before_rescue"] = list(raw_report.failures)
        sample["raw_strict_acpf_metrics_before_rescue"] = dict(raw_report.metrics)
        sample["linear_label_lodf_repaired"] = True
        sample["linear_label_lodf_repair_delta_l1_mw"] = float(np.abs(p_new - p0).sum())
        sample["linear_label_lodf_repair_delta_linf_mw"] = float(np.max(np.abs(p_new - p0)))
        if rescue is not None and "rl_rescue_pv_evidence" in sample and not rescue_replaced:
            delta = np.asarray(rescue, dtype=np.float64) - p_new
            sample["rl_rescue_pv_evidence"]["delta_l1_mw"] = round(float(np.abs(delta).sum()), 6)
            sample["rl_rescue_pv_evidence"]["delta_linf_mw"] = round(float(np.max(np.abs(delta))), 6)

        repair_rows.append({
            "index": idx,
            "sample_id": sample.get("sample_id"),
            "before": before,
            "after": after,
            "iterations": int(iterations),
            "method": method,
            "delta_l1_mw": float(np.abs(p_new - p0).sum()),
            "delta_linf_mw": float(np.max(np.abs(p_new - p0))),
            "raw_ac_pass_after_repair": bool(raw_report.passed),
            "rescue_linear": rescue_linear,
            "rescue_ac": rescue_ac,
            "rescue_replaced": rescue_replaced,
        })
        print(
            f"Repaired {sample.get('sample_id')}: "
            f"L1={repair_rows[-1]['delta_l1_mw']:.3f} MW, "
            f"Linf={repair_rows[-1]['delta_linf_mw']:.3f} MW",
            flush=True,
        )

    final_rows = [
        check_dispatch(sample["dispatch_68"], audit_model, args.tol_mw)
        for sample in samples
    ]
    final_failed = [i for i, row in enumerate(final_rows) if not row["passed"]]
    if final_failed:
        raise RuntimeError(f"Final corrected audit still has failures at indices {final_failed}")

    lodf_stored = np.where(np.isfinite(lodf), lodf, 0.0)
    invalid = np.setdiff1d(np.arange(lodf.shape[1]), valid)
    lodf_stored[:, invalid] = 0.0
    meta["format"] = "case300_linear_sced_affine_dc_lodf_v2"
    meta["lodf"] = lodf_stored.tolist()
    meta["valid_contingency_indices"] = np.asarray(valid, dtype=int).tolist()
    repaired["version"] = f"{data.get('version', 'route_b')}_lodf_corrected"
    repaired["lodf_correction_summary"] = {
        "source_dataset": str(src.resolve()),
        "initial_failed_labels": initial_failures,
        "repaired_labels": len(repair_rows),
        "final_passed": len(samples),
        "valid_line_outages": int(len(valid)),
    }

    out = Path(args.output_dataset)
    if out.resolve() == src.resolve():
        raise RuntimeError("Refusing to overwrite the source dataset")
    out.write_text(json.dumps(repaired, indent=2), encoding="utf-8")
    audit = {
        "summary": repaired["lodf_correction_summary"],
        "repairs": repair_rows,
    }
    audit_path = Path(args.audit_output)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit["summary"], indent=2))
    print(f"Wrote corrected dataset: {out}")
    print(f"Wrote repair audit: {audit_path}")


if __name__ == "__main__":
    main()

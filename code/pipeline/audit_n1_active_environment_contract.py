#!/usr/bin/env python3
"""Audit the contract between fixed-basis screening and the RL environment."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="case300_gt_n1s0575_active_required_fixed32.json")
    parser.add_argument("--split-file", default="paper_results/case300_n1s0575_active_fixed32_split.json")
    parser.add_argument("--gat-checkpoint", default="gatv10_results_n1s0575_candidate/gatv10_predictor_best.pth")
    parser.add_argument("--scope", choices=["train", "val", "test"], default="test")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--residual-scale", type=float, default=20.0)
    parser.add_argument("--cumulative-scale-mult", type=float, default=20.0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--voltage-range", type=float, default=0.10)
    parser.add_argument(
        "--scripted-replay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replay each stored rescue through the deployed action decoder.",
    )
    parser.add_argument("--output", default="paper_results/n1_active_environment_contract.json")
    return parser.parse_args()


def main():
    args = parse_args()
    project = Path(__file__).resolve().parents[2]
    dataset_path = (project / args.dataset).resolve()
    split_path = (project / args.split_file).resolve()
    checkpoint_path = (project / args.gat_checkpoint).resolve()
    output_path = (project / args.output).resolve()

    os.environ["ROUTE_B_GAT_CHECKPOINT"] = str(checkpoint_path)
    import train_rlv35_complete as core

    core.UNIFORM_RESIDUAL_MW = float(args.residual_scale)
    core.UNIFORM_CUMULATIVE_MULT = float(args.cumulative_scale_mult)
    core.MAX_STEPS = int(args.max_steps)

    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    samples = [sample for sample in data["samples"] if sample.get("is_complete_sced")]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    ids = [int(idx) for idx in split[f"{args.scope}_ids"]]
    if args.max_samples > 0:
        ids = ids[:args.max_samples]

    env = core.SCEDEnvV21(
        samples,
        ids,
        dataset_meta=data,
        action_encoding="physics32fixed",
        action_basis_seed=20260721,
        voltage_action_mode="absolute_residual",
        voltage_absolute_range_pu=float(args.voltage_range),
        observation_mode="normalized_compact",
        generator_upper_margin=0.99,
        include_bus_voltage_feedback=True,
        initial_acpf_on_reset=True,
    )
    rows = []
    for position, sample_idx in enumerate(ids, 1):
        sample = samples[sample_idx]
        env.reset(sample_idx=sample_idx)
        env._compute_shift_directions()
        env._select_key_generators()

        stored_gat = np.asarray(sample["gat_start_dispatch_68_audit"], dtype=np.float64)
        rescue_p = np.asarray(sample["active_required_rescue_dispatch_68"], dtype=np.float64)
        rescue_gen_vm = np.asarray(sample["active_required_rescue_gen_vm_pu"], dtype=np.float64)
        rescue_ext_vm = np.asarray(sample["active_required_rescue_ext_grid_vm_pu"], dtype=np.float64)
        target_delta = rescue_p - env.gat_power

        columns = [direction for direction in np.asarray(env._shift_directions)]
        for gen_idx in np.asarray(env._key_gen_indices, dtype=int):
            column = np.full(env.N, -1.0 / float(env.N - 1), dtype=np.float64)
            column[gen_idx] = 1.0
            columns.append(column)
        basis = np.asarray(columns, dtype=np.float64).T
        coefficients, *_ = np.linalg.lstsq(basis, target_delta, rcond=None)
        raw_delta = basis @ coefficients
        cumulative_limit = float(args.residual_scale) * float(args.cumulative_scale_mult)
        raw_delta = np.clip(raw_delta, -cumulative_limit, cumulative_limit)
        reconstructed_p = core.project_box_sum(
            env.gat_power + raw_delta,
            env.PMIN,
            env.PMAX * 0.99,
            env.target_demand,
        )

        env.current_power = rescue_p.copy()
        env.cumulative_delta = rescue_p - env.gat_power
        env.current_gen_vm_pu = rescue_gen_vm.copy()
        env.current_ext_grid_vm_pu = rescue_ext_vm.copy()
        ac = env._run_ac_pf_check()
        n1_overload_mw, n1_violations = env._compute_n1_overload_margin()
        direct_pass = (
            bool(ac.get("converged", False))
            and int(ac.get("v_violations", 999)) == 0
            and int(ac.get("overloaded_lines", 999)) == 0
            and int(ac.get("trafo_overloaded", 999)) == 0
            and int(ac.get("reference_source_violations", 999)) == 0
            and int(ac.get("scheduled_balance_violations", 999)) == 0
            and int(ac.get("generator_bound_violations", 999)) == 0
            and int(n1_violations) == 0
        )
        scripted_pass = None
        scripted_steps = None
        scripted_final_linf_mw = None
        scripted_reason = None
        if args.scripted_replay:
            env.reset(sample_idx=sample_idx)
            env._compute_shift_directions()
            env._select_key_generators()
            n_script_steps = max(
                1,
                int(np.ceil(np.max(np.abs(coefficients)) / float(args.residual_scale))),
            )
            n_script_steps = min(n_script_steps, int(args.max_steps))
            physical_action = np.zeros(env.action_dim, dtype=np.float64)
            physical_action[:basis.shape[1]] = coefficients / float(n_script_steps)
            p_dim = env._n_shift_dirs + env._n_key_gens
            physical_action[p_dim:p_dim + env.N] = rescue_gen_vm - env.base_gen_vm_pu
            physical_action[-1] = float(rescue_ext_vm[0] - env.base_ext_grid_vm_pu[0])
            last_info = {}
            for scripted_steps in range(1, n_script_steps + 1):
                _obs, _reward, terminated, _truncated, last_info = env.step(physical_action)
                if terminated:
                    break
            scripted_pass = bool(
                last_info.get("reason") == "full_sced_pass"
                or core.adopted_operation_pass(last_info)
            )
            scripted_final_linf_mw = float(np.max(np.abs(env.current_power - rescue_p)))
            scripted_reason = last_info.get("reason", "unknown")
        row = {
            "sample_id": sample.get("sample_id", sample_idx),
            "gat_reproduction_linf_mw": float(np.max(np.abs(env.gat_power - stored_gat))),
            "stored_rescue_direct_pass": bool(direct_pass),
            "stored_rescue_n1_violations": int(n1_violations),
            "stored_rescue_n1_overload_mw": float(n1_overload_mw),
            "stored_rescue_ac_reason": None if ac.get("converged", False) else ac.get("error", "not_converged"),
            "voltage_residual_linf_pu": float(np.max(np.abs(rescue_gen_vm - env.base_gen_vm_pu))),
            "voltage_action_range_ok": bool(
                np.max(np.abs(rescue_gen_vm - env.base_gen_vm_pu)) <= args.voltage_range + 1e-9
                and np.max(np.abs(rescue_ext_vm - env.base_ext_grid_vm_pu)) <= args.voltage_range + 1e-9
            ),
            "active_span_reconstruction_l1_mw": float(np.sum(np.abs(reconstructed_p - rescue_p))),
            "active_span_reconstruction_linf_mw": float(np.max(np.abs(reconstructed_p - rescue_p))),
            "coefficient_linf_mw": float(np.max(np.abs(coefficients))),
            "coefficient_horizon_ok": bool(
                np.max(np.abs(coefficients)) <= args.residual_scale * args.max_steps + 1e-9
            ),
            "scripted_replay_pass": scripted_pass,
            "scripted_replay_steps": scripted_steps,
            "scripted_final_linf_mw": scripted_final_linf_mw,
            "scripted_final_reason": scripted_reason,
        }
        rows.append(row)
        print(
            f"[{position}/{len(ids)}] {row['sample_id']} "
            f"direct={'OK' if direct_pass else 'FAIL'} "
            f"gat_err={row['gat_reproduction_linf_mw']:.3g} "
            f"span_linf={row['active_span_reconstruction_linf_mw']:.2f}MW "
            f"v_range={'OK' if row['voltage_action_range_ok'] else 'FAIL'} "
            f"scripted={scripted_pass if args.scripted_replay else 'skip'}"
        )

    summary = {
        "dataset": str(dataset_path),
        "scope": args.scope,
        "n": len(rows),
        "direct_passed": sum(row["stored_rescue_direct_pass"] for row in rows),
        "gat_exact": sum(row["gat_reproduction_linf_mw"] <= 1e-4 for row in rows),
        "voltage_range_ok": sum(row["voltage_action_range_ok"] for row in rows),
        "span_linf_le_1mw": sum(row["active_span_reconstruction_linf_mw"] <= 1.0 for row in rows),
        "coefficient_horizon_ok": sum(row["coefficient_horizon_ok"] for row in rows),
        "scripted_replay_passed": (
            sum(bool(row["scripted_replay_pass"]) for row in rows)
            if args.scripted_replay else None
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"summary": summary, "records": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Wrote:", output_path)


if __name__ == "__main__":
    main()

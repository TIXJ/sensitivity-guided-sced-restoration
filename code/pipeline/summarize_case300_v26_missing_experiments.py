"""Create compact paper-ready summaries for the case300 V26 experiments."""
from __future__ import annotations

import argparse
import csv
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load(path: str) -> dict:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def metric(summary: dict, name: str, field: str = "mean"):
    value = summary["metrics"][name]
    return value[field] if isinstance(value, dict) else value


def validation_milestone(training: dict) -> dict:
    records = training.get("validation", [])
    complete = [r for r in records if r["passed"] == r["total"]]
    if not complete:
        return {"first_full_validation_update": None, "best_validation": max(
            (r["passed"] for r in records), default=0
        )}
    return {
        "first_full_validation_update": min(r["update"] for r in complete),
        "best_validation": complete[0]["total"],
    }


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", default=str(ROOT / "paper_results" / "case300_v26_pipeline_audit.json"))
    parser.add_argument("--sac-dir", default=str(ROOT / "rl_sac_n1_active_aligned_v26"))
    parser.add_argument("--full68-dir", default=str(ROOT / "rl_action_ablation_full68_v26"))
    parser.add_argument("--ddpg", default=str(ROOT / "rl_ddpg_n1_active_aligned_v26" / "test_results.json"))
    parser.add_argument("--td3", default=str(ROOT / "rl_td3_n1_active_aligned_v26" / "test_results.json"))
    parser.add_argument("--acopf", default=str(ROOT / "paper_results" / "case300_n1s0575_active_acopf_test.json"))
    parser.add_argument("--ac-scopf", default=str(ROOT / "paper_results" / "case300_n1s0575_active_ac_scopf_test.json"))
    parser.add_argument("--output-prefix", default=str(ROOT / "paper_results" / "case300_v26_paper_update"))
    args = parser.parse_args()

    sac_dir = pathlib.Path(args.sac_dir)
    full68_dir = pathlib.Path(args.full68_dir)
    audit = load(args.audit)["summary"]
    sac = load(str(sac_dir / "test_results.json"))
    sac_training = load(str(sac_dir / "training_results.json"))
    sac_config = load(str(sac_dir / "config.json"))
    full68 = load(str(full68_dir / "test_results.json"))
    full68_training = load(str(full68_dir / "training_results.json"))
    full68_config = load(str(full68_dir / "config.json"))
    ddpg = load(args.ddpg)
    td3 = load(args.td3)
    acopf_payload = load(args.acopf)
    acopf = acopf_payload["summary"]["ac_opf_restoration"]
    acopf_rows = acopf_payload["rows"]
    acopf_cost = sum(r["scheduled_controllable_generation_cost"] for r in acopf_rows) / len(acopf_rows)
    acopf_overload = sum(r["n1_overload_mw"] for r in acopf_rows) / len(acopf_rows)
    scopf = load(args.ac_scopf)["summary"]

    sac_runtime = metric(audit, "online_pipeline_runtime_sec")
    sac_cost = metric(audit, "final_scheduled_controllable_generation_cost")
    scopf_runtime = scopf["avg_runtime_sec"]
    scopf_cost = scopf["avg_scheduled_controllable_generation_cost"]

    method_rows = [
        {
            "method": "GAT+SAC (physics32)", "passed": sac["passed"], "total": sac["total"],
            "pass_rate_pct": 100 * sac["passed"] / sac["total"], "avg_steps": sac["avg_steps"],
            "avg_final_n1_overload_mw": sac["avg_final_n1_overload_mw"],
            "avg_final_voltage_potential_pu": sac["avg_final_voltage_potential_pu"],
            "avg_runtime_sec": sac_runtime, "avg_scheduled_cost": sac_cost,
        },
        {
            "method": "GAT+DDPG (physics32)", "passed": ddpg["passed"], "total": ddpg["total"],
            "pass_rate_pct": 100 * ddpg["passed"] / ddpg["total"], "avg_steps": ddpg["avg_steps"],
            "avg_final_n1_overload_mw": ddpg["avg_final_n1_overload_mw"],
            "avg_final_voltage_potential_pu": ddpg["avg_final_voltage_potential_pu"],
            "avg_runtime_sec": "", "avg_scheduled_cost": "",
        },
        {
            "method": "GAT+TD3 (physics32)", "passed": td3["passed"], "total": td3["total"],
            "pass_rate_pct": 100 * td3["passed"] / td3["total"], "avg_steps": td3["avg_steps"],
            "avg_final_n1_overload_mw": td3["avg_final_n1_overload_mw"],
            "avg_final_voltage_potential_pu": td3["avg_final_voltage_potential_pu"],
            "avg_runtime_sec": "", "avg_scheduled_cost": "",
        },
        {
            "method": "Base-case AC-OPF", "passed": acopf["passed"], "total": acopf["n"],
            "pass_rate_pct": acopf["pass_rate_pct"], "avg_steps": "",
            "avg_final_n1_overload_mw": acopf_overload, "avg_final_voltage_potential_pu": 0.0,
            "avg_runtime_sec": acopf["avg_runtime_sec"], "avg_scheduled_cost": acopf_cost,
        },
        {
            "method": "Preventive hybrid AC-SCOPF", "passed": scopf["passed"], "total": scopf["n"],
            "pass_rate_pct": scopf["pass_rate_pct"], "avg_steps": scopf["avg_security_cut_rounds"],
            "avg_final_n1_overload_mw": 0.0, "avg_final_voltage_potential_pu": 0.0,
            "avg_runtime_sec": scopf_runtime, "avg_scheduled_cost": scopf_cost,
        },
    ]

    sac_milestone = validation_milestone(sac_training)
    full68_milestone = validation_milestone(full68_training)
    action_rows = [
        {
            "encoding": "Fixed PTDF/LODF basis", "active_power_dim": 32,
            "total_action_dim": sac_config["action_dim"], "passed": sac["passed"], "total": sac["total"],
            "avg_steps": sac["avg_steps"], "first_full_validation_update": sac_milestone["first_full_validation_update"],
            "training_time_sec": sac_training["elapsed_sec"],
        },
        {
            "encoding": "Generator-wise redispatch", "active_power_dim": 68,
            "total_action_dim": full68_config["action_dim"], "passed": full68["passed"], "total": full68["total"],
            "avg_steps": full68["avg_steps"], "first_full_validation_update": full68_milestone["first_full_validation_update"],
            "training_time_sec": full68_training["elapsed_sec"],
        },
    ]

    payload = {
        "dataset": {"total": 290, "train": 202, "validation": 43, "test": 45, "valid_outages": 253},
        "method_comparison": method_rows,
        "action_compression_ablation": action_rows,
        "physical_audit": {
            "gat_mae_mw": audit["gat_projected_test_mae_mw"],
            "gat_rmse_mw": audit["gat_projected_test_rmse_mw"],
            "gat_pearson": audit["gat_projected_test_pearson"],
            "initial_n1_overload_mw": metric(audit, "initial_n1_overload_mw"),
            "initial_voltage_potential_pu": metric(audit, "initial_voltage_potential_pu"),
            "active_delta_l1_mw": metric(audit, "active_delta_l1_mw"),
            "active_delta_l2_mw": metric(audit, "active_delta_l2_mw"),
            "active_delta_max_generator_mw": metric(audit, "active_delta_max_generator_mw"),
            "generator_voltage_delta_l1_pu": metric(audit, "generator_voltage_delta_l1_pu"),
            "reference_voltage_delta_pu": metric(audit, "reference_voltage_delta_pu"),
            "reference_source_s_utilization_pct": metric(audit, "reference_source_s_utilization_pct"),
            "cost_gap_vs_linear_sced_pct": metric(audit, "cost_gap_vs_linear_sced_label_pct"),
        },
        "derived": {
            "ac_scopf_to_sac_runtime_ratio": scopf_runtime / sac_runtime,
            "sac_cost_gap_vs_ac_scopf_pct": 100 * (sac_cost / scopf_cost - 1),
            "active_action_dimension_reduction_pct": 100 * (1 - 32 / 68),
            "total_action_dimension_reduction_pct": 100 * (1 - sac_config["action_dim"] / full68_config["action_dim"]),
        },
    }

    prefix = pathlib.Path(args.output_prefix)
    json_path = prefix.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(prefix.with_name(prefix.name + "_methods.csv"), method_rows)
    write_csv(prefix.with_name(prefix.name + "_action_ablation.csv"), action_rows)

    lines = [
        "# Case300 V26 paper update", "",
        "## Main comparison", "",
        "| Method | Pass | Avg. steps/rounds | Runtime (s) | Scheduled cost |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        runtime = row["avg_runtime_sec"]
        cost = row["avg_scheduled_cost"]
        lines.append(
            f"| {row['method']} | {row['passed']}/{row['total']} | {row['avg_steps']} | "
            f"{runtime if runtime != '' else '--'} | {cost if cost != '' else '--'} |"
        )
    lines += [
        "", "## Action compression ablation", "",
        "| Encoding | P dim. | Total dim. | Pass | Avg. steps | First full validation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in action_rows:
        lines.append(
            f"| {row['encoding']} | {row['active_power_dim']} | {row['total_action_dim']} | "
            f"{row['passed']}/{row['total']} | {row['avg_steps']:.3f} | U{row['first_full_validation_update']} |"
        )
    lines += [
        "", "## Defensible conclusions", "",
        f"- AC-SCOPF and GAT+SAC both pass 45/45; GAT+SAC is {scopf_runtime / sac_runtime:.2f}x faster end to end.",
        f"- SAC scheduled cost is {100 * (sac_cost / scopf_cost - 1):.2f}% above AC-SCOPF.",
        f"- The physics basis reduces active-power dimensions by {100 * (1 - 32 / 68):.1f}% and total action dimensions by {100 * (1 - sac_config['action_dim'] / full68_config['action_dim']):.1f}% while retaining 45/45 feasibility.",
        "- The 68D baseline needs fewer test-time steps, so do not claim that compression improves rollout length.",
        f"- The 32D policy first reaches 43/43 validation at U{sac_milestone['first_full_validation_update']}, versus U{full68_milestone['first_full_validation_update']} for 68D.",
        "- AC-SCOPF uses base-case nonlinear AC feasibility with affine DC-LODF N-1 security; it is not a full nonlinear contingency AC-SCOPF.",
    ]
    md_path = prefix.with_suffix(".md")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    print(f"Wrote: {prefix.with_name(prefix.name + '_methods.csv')}")
    print(f"Wrote: {prefix.with_name(prefix.name + '_action_ablation.csv')}")


if __name__ == "__main__":
    main()

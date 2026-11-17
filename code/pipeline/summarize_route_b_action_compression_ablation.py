"""Summarize retrained Route-B active-power action-compression ablations."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = {
    "PTDF/LODF 32-D": ROOT / "rl_action_ablation_physics32",
    "Full-generator 68-D": ROOT / "rl_action_ablation_full68",
    "Matched nonphysical 32-D": ROOT / "rl_action_ablation_matched32",
}


def first_threshold_update(history: list[dict], threshold: float) -> int | None:
    for row in history:
        total = int(row.get("total", 0))
        passed = int(row.get("passed", 0))
        if total and passed / total >= threshold:
            return int(row["update"])
    return None


def parse_runs(specs: list[str]) -> dict[str, Path]:
    if not specs:
        return DEFAULT_RUNS
    runs = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Run must be LABEL=DIR, got: {spec}")
        label, path = spec.split("=", 1)
        run_dir = Path(path)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        runs[label] = run_dir
    return runs


def load_row(label: str, run_dir: Path) -> dict:
    rewards_path = run_dir / "rewards.json"
    test_path = run_dir / "test_results.json"
    if not rewards_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing rewards.json or test_results.json in {run_dir}"
        )
    rewards = json.loads(rewards_path.read_text(encoding="utf-8"))
    test = json.loads(test_path.read_text(encoding="utf-8"))
    val_history = rewards.get("val_history", [])
    total = int(test["total"])
    passed = int(test["passed"])
    return {
        "variant": label,
        "action_encoding": test["action_encoding"],
        "active_action_dim": int(test["active_action_dim"]),
        "total_action_dim": int(test["total_action_dim"]),
        "actor_parameter_count": int(test.get("actor_parameter_count", 0)),
        "n1_limit_scale": float(test["n1_limit_scale"]),
        "test_passed": passed,
        "test_total": total,
        "test_pass_rate_pct": 100.0 * passed / max(total, 1),
        "n1_passed": int(test.get("n1_passed", 0)),
        "avg_steps": float(test.get("avg_steps", 0.0)),
        "avg_reward": float(test.get("avg_reward", 0.0)),
        "avg_n1_overload_mw": float(test.get("avg_n1_overload_mw", 0.0)),
        "max_n1_overload_mw": float(test.get("max_n1_overload_mw", 0.0)),
        "avg_runtime_sec": float(test.get("avg_runtime_sec", 0.0)),
        "training_runtime_sec": float(test.get("training_runtime_sec", 0.0)),
        "first_val_80pct_update": first_threshold_update(val_history, 0.80),
        "first_val_90pct_update": first_threshold_update(val_history, 0.90),
        "best_val_passed": int(rewards.get("best_val_pass", -1)),
        "best_val_total": int(rewards.get("best_val_total", 0)),
        "run_dir": str(run_dir.resolve()),
    }


def plot_rows(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    colors = ["#3C5488", "#E64B35", "#00A087"]
    labels = [row["variant"] for row in rows]
    pass_rates = [row["test_pass_rate_pct"] for row in rows]
    steps = [row["avg_steps"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.25))
    bars = axes[0].bar(range(len(rows)), pass_rates, color=colors[: len(rows)])
    axes[0].set_ylabel("Strict AC + linear N-1 pass rate (%)")
    axes[0].set_ylim(0, 108)
    axes[0].set_xticks(range(len(rows)), labels, rotation=15, ha="right")
    axes[0].bar_label(bars, fmt="%.1f", padding=2, fontsize=8)

    bars = axes[1].bar(range(len(rows)), steps, color=colors[: len(rows)])
    axes[1].set_ylabel("Average correction steps")
    axes[1].set_xticks(range(len(rows)), labels, rotation=15, ha="right")
    ymax = max(steps, default=1.0)
    axes[1].set_ylim(0, max(1.0, 1.18 * ymax))
    axes[1].bar_label(bars, fmt="%.1f", padding=2, fontsize=8)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run",
        action="append",
        default=[],
        help="Optional LABEL=DIR specification; repeat once per variant.",
    )
    ap.add_argument(
        "--output-json",
        default=str(ROOT / "paper_results" / "route_b_action_compression_ablation.json"),
    )
    ap.add_argument(
        "--output-csv",
        default=str(ROOT / "paper_results" / "route_b_action_compression_ablation.csv"),
    )
    ap.add_argument(
        "--plot",
        default=str(
            ROOT
            / "GAT-SAC-SCED-Framework"
            / "chapter4_plots"
            / "routeb_fig9_action_compression.pdf"
        ),
    )
    args = ap.parse_args()

    runs = parse_runs(args.run)
    rows = [load_row(label, path) for label, path in runs.items()]
    scales = {round(row["n1_limit_scale"], 12) for row in rows}
    if len(scales) != 1:
        raise ValueError(f"Runs use inconsistent N-1 limit scales: {sorted(scales)}")

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "experiment": "retrained_action_compression_ablation",
                "n1_limit_scale": rows[0]["n1_limit_scale"],
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_rows(rows, Path(args.plot))
    print(json.dumps(rows, indent=2))
    print("Wrote:", output_json)
    print("Wrote:", output_csv)
    print("Wrote:", args.plot)


if __name__ == "__main__":
    main()

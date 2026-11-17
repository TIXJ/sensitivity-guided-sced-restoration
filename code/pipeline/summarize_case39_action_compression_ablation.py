"""Summarize the retrained case39 PTDF/LODF action-compression ablation."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNS = [
    ("PTDF/LODF 6-D", ROOT / "rl39_action_ablation_physics6"),
    ("Full-generator 9-D", ROOT / "rl39_action_ablation_full9"),
    ("Matched random 6-D", ROOT / "rl39_action_ablation_matched6"),
]


def first_val_update(history, threshold):
    for row in history:
        total = int(row.get("total", 0))
        if total and int(row.get("passed", 0)) / total >= threshold:
            return int(row["update"])
    return None


def load_row(label, run_dir):
    result_path = run_dir / "main_experiment_result.json"
    history_path = run_dir / "training_history.json"
    if not result_path.exists() or not history_path.exists():
        raise FileNotFoundError(f"Missing case39 ablation output in {run_dir}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    val_history = history.get("val_history", [])
    return {
        "variant": label,
        "action_encoding": result["action_encoding"],
        "active_action_dim": int(result["active_action_dim"]),
        "total_action_dim": int(result["action_dim"]),
        "actor_parameter_count": int(result["actor_parameter_count"]),
        "test_passed": int(result["test_pass"]),
        "test_total": int(result["test_total"]),
        "test_pass_rate_pct": float(result["test_pass_rate_pct"]),
        "avg_test_steps": float(result["avg_test_steps"]),
        "avg_test_runtime_sec": float(result["avg_test_runtime_sec"]),
        "first_val_80pct_update": first_val_update(val_history, 0.80),
        "first_val_90pct_update": first_val_update(val_history, 0.90),
        "training_runtime_sec": float(result["training_runtime_sec"]),
        "run_dir": str(run_dir.resolve()),
    }


def plot(rows, output):
    import matplotlib.pyplot as plt

    colors = ["#3C5488", "#E64B35", "#00A087"]
    labels = [row["variant"] for row in rows]
    rates = [row["test_pass_rate_pct"] for row in rows]
    convergence = [
        row["first_val_90pct_update"]
        if row["first_val_90pct_update"] is not None else 100
        for row in rows
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    bars = axes[0].bar(range(3), rates, color=colors)
    axes[0].set_ylabel("Held-out pass rate (%)")
    axes[0].set_ylim(0, 108)
    axes[0].bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    bars = axes[1].bar(range(3), convergence, color=colors)
    axes[1].set_ylabel("First update reaching 90% validation")
    axes[1].set_ylim(0, max(convergence) * 1.18 if convergence else 1)
    axes[1].bar_label(bars, fmt="%.0f", padding=2, fontsize=8)
    for axis in axes:
        axis.set_xticks(range(3), labels, rotation=14, ha="right")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
        axis.set_axisbelow(True)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    rows = [load_row(label, run_dir) for label, run_dir in RUNS]
    json_path = ROOT / "paper_results" / "case39_action_compression_ablation.json"
    csv_path = ROOT / "paper_results" / "case39_action_compression_ablation.csv"
    figure_path = (
        ROOT / "GAT-SAC-SCED-Framework" / "chapter4_plots"
        / "case39_action_compression_ablation.pdf"
    )
    json_path.write_text(
        json.dumps({"experiment": "case39_retrained_action_compression", "rows": rows}, indent=2),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, figure_path)
    print(json.dumps(rows, indent=2))
    print("Wrote:", json_path)
    print("Wrote:", csv_path)
    print("Wrote:", figure_path)


if __name__ == "__main__":
    main()

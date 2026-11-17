"""Summarize Route-B main experiments for IEEE 39-bus and 118-bus cases."""
import json
from pathlib import Path


ROOT = Path(r".")
OUT = ROOT / "paper_results" / "route_b_smallcase_main_results.json"


def load_result(case_name, rel_path):
    path = ROOT / rel_path
    if not path.exists():
        return {"case": case_name, "status": "missing", "path": str(path)}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["status"] = "ok"
    data["path"] = str(path)
    return data


def main():
    rows = [
        load_result("case39", r"rl39_routeb_main\main_experiment_result.json"),
        load_result("case118", r"rl118_routeb_main\main_experiment_result.json"),
    ]
    summary = {"experiments": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()

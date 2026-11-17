"""Build B2 from saved measurements only; never import physical experiment code."""
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas


REV = Path(__file__).resolve().parents[1]
ROOT = REV.parents[1]
PDF = REV / "revision_figures/runtime_cost_feasibility.pdf"
EVIDENCE = REV / "_audit/runtime_cost_plot_evidence.json"
SOURCES = {
    "sac27": ROOT / "paper_results/case300_fixed_slack_v1_pipeline_audit.json",
    "repair27": ROOT / "paper_results/case300_fixed_slack_v1_deterministic_repair_test.json",
    "preventive27": ROOT / "paper_results/case300_fixed_slack_v1_scopf.json",
    "acopf27": ROOT / "paper_results/case300_fixed_slack_v1_acopf.json",
    "final_dual": REV / "experiment_results/dual_verified_once_v2/dual.json",
    "final_protocol": REV / "experiment_results/dual_verified_once_v2/protocol.json",
}
FONT_FILES = {
    "TNR": Path("external/Windows/Fonts/times.ttf"),
    "TNR-Bold": Path("external/Windows/Fonts/timesbd.ttf"),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_number(row, field):
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Missing or nonfinite {field} for {row.get('sample_id')}")
    return float(value)


def aggregate_method(rows, time_fields, cost_field, expected_repeats=None):
    """Average measured stages within a run, repeats within a case, then cases."""
    if not rows:
        raise ValueError("No measured rows")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["sample_id"]].append(row)
    cases = []
    for sid, records in sorted(grouped.items()):
        actual = [row.get("repeat") for row in records]
        expected = [None] if expected_repeats is None else list(expected_repeats)
        if len(actual) != len(expected) or set(actual) != set(expected):
            raise ValueError(f"Missing or duplicate repeat for {sid}: {actual}")
        if any(type(row.get("passed")) is not bool for row in records):
            raise ValueError(f"Missing acceptance flag for {sid}")
        if len({row["passed"] for row in records}) != 1:
            raise ValueError(f"Acceptance changes across repeats for {sid}")
        measurements = []
        for row in sorted(records, key=lambda item: item.get("repeat", 0)):
            stages = {field: finite_number(row, field) for field in time_fields}
            if any(value < 0 for value in stages.values()):
                raise ValueError(f"Negative elapsed time for {sid}")
            measurements.append({
                "repeat": row.get("repeat"), "passed": row["passed"],
                "measured_time_stages_sec": stages,
                "end_to_end_time_sec": sum(stages.values()),
                "scheduled_cost": finite_number(row, cost_field),
            })
        cases.append({
            "sample_id": sid, "passed": records[0]["passed"],
            "mean_time_sec": mean(row["end_to_end_time_sec"] for row in measurements),
            "mean_scheduled_cost": mean(row["scheduled_cost"] for row in measurements),
            "measurements": measurements,
        })
    return {
        "case_count": len(cases),
        "accepted_count": sum(row["passed"] for row in cases),
        "mean_time_sec": mean(row["mean_time_sec"] for row in cases),
        "mean_scheduled_cost": mean(row["mean_scheduled_cost"] for row in cases),
        "case_measurements": cases,
    }


def unique_rows(rows):
    indexed = {row["sample_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate sample_id in historical records")
    return indexed


def build_evidence():
    snapshots = {name: path.read_bytes() for name, path in SOURCES.items()}
    payloads = {name: json.loads(blob.decode("utf-8-sig")) for name, blob in snapshots.items()}
    hashes = {name: hashlib.sha256(blob).hexdigest() for name, blob in snapshots.items()}
    if payloads["final_dual"]["protocol_sha256"] != hashes["final_protocol"]:
        raise ValueError("Final records do not match the frozen protocol")
    if payloads["final_dual"].get("complete") is not True:
        raise ValueError("Final paired measurement is incomplete")
    spec = [
        ("sac27", "SAC", ("online_pipeline_runtime_sec", "fresh_final_verification_runtime_sec"),
         "final_scheduled_controllable_generation_cost", 27),
        ("repair27", "Repair", ("runtime_sec",), "scheduled_cost", 27),
        ("preventive27", "Prev. opt.", ("runtime_sec",), "scheduled_controllable_generation_cost", 27),
        ("acopf27", "AC-OPF", ("runtime_sec",), "scheduled_controllable_generation_cost", 0),
    ]
    ids = set(unique_rows(payloads["sac27"]["rows"]))
    if len(ids) != 27:
        raise ValueError("Historical SAC population must contain 27 cases")
    historical = []
    for source, label, time_fields, cost_field, accepted in spec:
        rows = payloads[source]["rows"]
        if set(unique_rows(rows)) != ids:
            raise ValueError(f"Historical case IDs do not match: {source}")
        summary = aggregate_method(rows, time_fields, cost_field)
        if summary["accepted_count"] != accepted:
            raise ValueError(f"Unexpected acceptance: {source}")
        summary.update(source=source, label=label, time_fields=list(time_fields), cost_field=cost_field)
        historical.append(summary)

    raw = payloads["final_dual"]["rows"]
    by_key = {(row["repeat"], row["sample_id"], row["arm"]): row for row in raw}
    repeats = tuple(sorted({row["repeat"] for row in raw}))
    all_ids = sorted({row["sample_id"] for row in raw})
    if repeats != (0, 1, 2) or len(all_ids) != 75 or len(by_key) != len(raw) or len(raw) != 450:
        raise ValueError("Expected 75 unique cases, two arms and three complete repeats")
    accepted_by_repeat = []
    for repeat in repeats:
        arm_ids = []
        for arm in ("direct", "selected"):
            keys = [(repeat, sid, arm) for sid in all_ids]
            if not all(key in by_key for key in keys):
                raise ValueError("Missing final case/arm/repeat measurement")
            arm_ids.append({sid for sid in all_ids if by_key[repeat, sid, arm]["passed"]})
        if arm_ids[0] != arm_ids[1] or len(arm_ids[0]) != 60:
            raise ValueError("Final workflows must accept the same 60 cases")
        accepted_by_repeat.append(arm_ids[0])
    if any(accepted != accepted_by_repeat[0] for accepted in accepted_by_repeat):
        raise ValueError("Common acceptance population changes across repeats")
    common = sorted(accepted_by_repeat[0])
    final = []
    for arm, label in (("direct", "Direct"), ("selected", "Gated")):
        rows = [by_key[repeat, sid, arm] for sid in common for repeat in repeats]
        summary = aggregate_method(rows, ("runtime_sec",), "scheduled_cost", repeats)
        summary.update(source="final_dual", arm=arm, label=label,
                       time_fields=["runtime_sec"], cost_field="scheduled_cost")
        final.append(summary)
    evidence = {
        "schema_version": 1, "analysis": "B2 only; no physical experiments, estimation or interpolation",
        "builder": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)},
        "sources": {name: {"path": str(SOURCES[name]), "sha256": hashes[name]} for name in SOURCES},
        "figure": {"path": str(PDF), "width_in": 3.35, "height_in": 3.4,
                   "font_family": "Times New Roman", "minimum_font_size_pt": 8,
                   "font_files": {name: {"path": str(path), "sha256": sha256(path)}
                                  for name, path in FONT_FILES.items()},
                   "vector_only": True, "numeric_point_annotations": False},
        "cost_definition": "Scheduled controllable-generation cost, excluding reference-source exchange pricing",
        "panel_a": {
            "population": "Historical screened 27-case test benchmark", "case_ids": sorted(ids),
            "join_key": "sample_id; never join historical sources by row order or sid alone",
            "aggregation": "Arithmetic mean of the 27 measured per-case time/cost pairs for each method",
            "stage_handling": {
                "SAC": "Add online_pipeline_runtime_sec and fresh_final_verification_runtime_sec exactly once; 0.675 s alone excludes the independent final check. Do not add GAT, reset or rollout subtimers again.",
                "baselines": "runtime_sec is the recorded end-to-end elapsed time and already includes required final checks; do not add verification_sec or solver subtimers.",
                "acceptance": "SAC, repair and preventive optimization accept 27/27; AC-OPF accepts 0/27. Its observed output cost is not a feasible accepted-dispatch cost.",
                "provenance": "Retained historical measurements, not a new controlled remeasurement; individual hardware/load equivalence is not inferred from paths or final-workflow environment statements.",
            }, "methods": historical,
        },
        "panel_b": {
            "population": "Final dual-start workflows on their identical 60 common accepted historical cases",
            "case_ids": common, "repeat_ids": list(repeats),
            "join_key": ["repeat", "sample_id", "arm"],
            "aggregation": "Average three measured repeats within each case, then give each of the 60 cases equal weight",
            "stage_handling": {
                "time": "Use runtime_sec once, including prediction, initial checks, any policy/optimization, required acceptance checks and context-fingerprint work; resident model setup/loading is excluded.",
                "verification": "A checked unchanged optimizer candidate can reuse its single-use acceptance receipt; this is already inside runtime_sec. Do not add final_check_sec or optimizer_sec.",
                "repeats": "Three rotating serial repeats; 180 measurements per arm represent 60 cases, not 180 independent cases.",
                "population_restriction": "Exclude the other 15 cases from both coordinates; the published 75-case overall means 1.912/1.725 s are not used.",
            }, "methods": final,
        },
        "no_crosspanel_comparison": True,
        "crosspanel_scope": "Panels have different populations, method variants, measurement dates and protocols. No pooled analysis, cross-panel speed ratio or common-hardware inference is made.",
    }
    return evidence


INK = HexColor("#202020")
GREEN = HexColor("#18745E")
BLUE = HexColor("#28659A")
RED = HexColor("#A54646")
GRAY = HexColor("#606060")


def label(canvas, x, y, text, size=8, bold=False, align="left"):
    canvas.setFillColor(INK)
    canvas.setFont("TNR-Bold" if bold else "TNR", size)
    getattr(canvas, {"left": "drawString", "center": "drawCentredString", "right": "drawRightString"}[align])(x, y, text)


def marker(canvas, x, y, shape, color, filled=True, radius=2.7):
    canvas.setStrokeColor(color)
    canvas.setFillColor(color if filled else HexColor("#FFFFFF"))
    canvas.setLineWidth(0.85)
    if shape == "circle":
        canvas.circle(x, y, radius, stroke=1, fill=1)
    elif shape == "square":
        canvas.rect(x-radius, y-radius, 2*radius, 2*radius, stroke=1, fill=1)
    else:
        path = canvas.beginPath()
        vertices = [(0, radius+0.4), (-radius-0.3, -radius), (radius+0.3, -radius)] if shape == "triangle" else [(0, radius+0.6), (-radius-0.6, 0), (0, -radius-0.6), (radius+0.6, 0)]
        path.moveTo(x+vertices[0][0], y+vertices[0][1])
        for dx, dy in vertices[1:]:
            path.lineTo(x+dx, y+dy)
        path.close()
        canvas.drawPath(path, stroke=1, fill=1)


def axes(canvas, bottom, xlim, ylim, xticks, yticks):
    left, width, height = 46, 182, 55
    project = lambda t, cost: (left+(t-xlim[0])*width/(xlim[1]-xlim[0]),
                              bottom+(cost/1000-ylim[0])*height/(ylim[1]-ylim[0]))
    canvas.setLineWidth(0.4)
    for tick in yticks:
        y = bottom+(tick-ylim[0])*height/(ylim[1]-ylim[0])
        canvas.setStrokeColor(HexColor("#DDDDDD"))
        canvas.line(left, y, left+width, y)
        label(canvas, left-5, y-2.5, f"{tick:g}", align="right")
    canvas.setStrokeColor(INK)
    canvas.line(left, bottom, left, bottom+height)
    canvas.line(left, bottom, left+width, bottom)
    for tick in xticks:
        x = left+(tick-xlim[0])*width/(xlim[1]-xlim[0])
        canvas.line(x, bottom, x, bottom-3)
        label(canvas, x, bottom-12, f"{tick:g}", align="center")
    label(canvas, left+width/2, bottom-25, "End-to-end time (s)", align="center")
    return project


def draw_pdf(evidence):
    for name, path in FONT_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f"Times New Roman font not found: {path}")
        pdfmetrics.registerFont(TTFont(name, str(path)))
    canvas = Canvas(str(PDF), pagesize=(3.35*72, 3.4*72), invariant=1, pageCompression=1)
    canvas.setTitle("Runtime, scheduled cost and acceptance: separate historical benchmarks")
    canvas.setAuthor("B2 saved-record analysis")
    canvas.saveState()
    canvas.translate(10, 122)
    canvas.rotate(90)
    label(canvas, 0, 0, "Scheduled cost (10^3 cost units per dispatch interval)", align="center")
    canvas.restoreState()
    label(canvas, 46, 234, "(a) Historical 27 cases: accepted / total", 8.5, True)
    project = axes(canvas, 166, (0, 4), (430, 450), (0, 1, 2, 3, 4), (430, 440, 450))
    styles = [("circle", GREEN, True), ("square", BLUE, True),
              ("triangle", RED, True), ("diamond", GRAY, False)]
    for index, (item, style) in enumerate(zip(evidence["panel_a"]["methods"], styles)):
        marker(canvas, *project(item["mean_time_sec"], item["mean_scheduled_cost"]), *style)
        x, y = (46, 130) if index == 0 else (145, 130) if index == 1 else (46, 118) if index == 2 else (145, 118)
        marker(canvas, x, y+2, *style, radius=2.4)
        label(canvas, x+8, y, f"{item['label']} ({item['accepted_count']}/27)")
    label(canvas, 46, 103, "(b) Final workflows: 60 common cases", 8.5, True)
    project = axes(canvas, 36, (0.9, 1.4), (455.2, 455.6), (0.9, 1.1, 1.3), (455.2, 455.4, 455.6))
    for item, style in zip(evidence["panel_b"]["methods"], [("circle", BLUE, True), ("square", GREEN, True)]):
        x, y = project(item["mean_time_sec"], item["mean_scheduled_cost"])
        marker(canvas, x, y, *style)
        if item["arm"] == "direct":
            label(canvas, x+6, y-8, "Direct (60/60)")
        else:
            label(canvas, x+6, y+5, "Gated (60/60)")
    canvas.showPage()
    canvas.save()


def main():
    evidence = build_evidence()
    draw_pdf(evidence)
    evidence["figure"]["sha256"] = sha256(PDF)
    for source in evidence["sources"].values():
        if sha256(source["path"]) != source["sha256"]:
            raise RuntimeError(f"Source changed during the build: {source['path']}")
    EVIDENCE.write_text(json.dumps(evidence, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    for panel in ("panel_a", "panel_b"):
        for item in evidence[panel]["methods"]:
            print(f"{panel}: {item['label']}: {item['accepted_count']}/{item['case_count']}; "
                  f"time={item['mean_time_sec']:.12f} s; cost={item['mean_scheduled_cost']:.12f}")
    print(f"PDF: {PDF}\nEvidence: {EVIDENCE}")


if __name__ == "__main__":
    main()

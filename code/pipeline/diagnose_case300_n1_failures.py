#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose full N-1 failures from the latest case300-SCED v41 audit.

The goal is to separate credible post-contingency violations from topology
events such as islanding or disconnected buses, which should not be mixed into
SCED N-1 acceptance unless the study explicitly models emergency controls.
"""

import argparse
import copy
import json
import re
import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
PROJ_DIR = CODE_DIR.parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(1, str(PROJ_DIR))

from generate_case300_v41_acopf_labels import apply_case_profile, prepare_case300_for_opf


FAIL_RE = re.compile(r"n1(?:_(line|trafo)_(\d+)|:(line|trafo):(\d+):)(.*)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--audit", default=str(PROJ_DIR / "paper_results" / "case300_sced_v41_acopf_audit.json"))
    p.add_argument("--record", type=int, default=0)
    p.add_argument("--output", default=str(PROJ_DIR / "paper_results" / "case300_sced_n1_failure_diagnosis.json"))
    return p.parse_args()


def parse_failures(failures):
    out = []
    seen = set()
    for failure in failures:
        m = FAIL_RE.match(failure)
        if not m:
            continue
        typ = m.group(1) or m.group(3)
        idx = int(m.group(2) or m.group(4))
        key = (typ, idx)
        if key in seen:
            continue
        seen.add(key)
        out.append({"type": typ, "idx": idx, "failure": failure})
    return out


def connected_components_after_outage(net, typ, idx):
    import networkx as nx
    import pandapower.topology as top

    net_c = copy.deepcopy(net)
    if typ == "line":
        net_c.line.at[idx, "in_service"] = False
    elif typ == "trafo":
        net_c.trafo.at[idx, "in_service"] = False
    graph = top.create_nxgraph(net_c, respect_switches=True, include_trafos=True)
    comps = [sorted(int(x) for x in comp) for comp in nx.connected_components(graph)]
    comps.sort(key=len, reverse=True)
    return {
        "n_components": len(comps),
        "component_sizes": [len(c) for c in comps[:10]],
        "small_components": comps[1:6],
    }


def element_info(net, typ, idx):
    if typ == "line":
        row = net.line.loc[idx]
        return {
            "from_bus": int(row.from_bus),
            "to_bus": int(row.to_bus),
            "length_km": float(row.length_km),
            "max_i_ka": float(row.max_i_ka),
            "parallel": int(row.parallel) if "parallel" in row else None,
        }
    row = net.trafo.loc[idx]
    return {
        "hv_bus": int(row.hv_bus),
        "lv_bus": int(row.lv_bus),
        "sn_mva": float(row.sn_mva),
        "vn_hv_kv": float(row.vn_hv_kv),
        "vn_lv_kv": float(row.vn_lv_kv),
    }


def main():
    args = parse_args()
    from pandapower.networks import case300

    with open(args.audit, "r", encoding="utf-8") as f:
        audit = json.load(f)
    rec = audit["records"][args.record]
    failures = rec.get("strict", {}).get("failures", [])
    demand = float(rec.get("demand_gw", 16.0)) * 1000.0

    net = case300()
    apply_case_profile(net, audit.get("config", {}).get("case_profile", "case300-sced"))
    prepare_case300_for_opf(net, demand)

    rows = []
    for item in parse_failures(failures):
        typ = item["type"]
        idx = item["idx"]
        row = dict(item)
        try:
            row.update(element_info(net, typ, idx))
            row.update(connected_components_after_outage(net, typ, idx))
        except Exception as exc:
            row["diagnosis_error"] = f"{type(exc).__name__}:{exc}"
        rows.append(row)

    by_components = {}
    for r in rows:
        key = str(r.get("n_components", "unknown"))
        by_components[key] = by_components.get(key, 0) + 1

    payload = {
        "audit": args.audit,
        "record": args.record,
        "n_failed_elements_in_failure_list": len(rows),
        "by_components": by_components,
        "failed_elements": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps({
        "n_failed_elements_in_failure_list": len(rows),
        "by_components": by_components,
        "output": str(out),
    }, indent=2))
    for r in rows[:30]:
        print(r)


if __name__ == "__main__":
    main()

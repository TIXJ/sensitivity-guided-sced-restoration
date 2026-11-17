#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build an explicit credible N-1 contingency set for case300-SCED.

The default rule excludes islanding outages. Optionally, when an audit file is
provided, it also removes contingencies that failed the AC-PF screening in that
audit. The resulting JSON is an explicit whitelist for final AC-PF verification.
"""

import argparse
import json
import re
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
PROJ_DIR = CODE_DIR.parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(1, str(PROJ_DIR))

from generate_case300_v41_acopf_labels import apply_case_profile, prepare_case300_for_opf

FAIL_RE = re.compile(r"n1(?:_(line|trafo)_(\d+)|:(line|trafo):(\d+):)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--demand-mw", type=float, default=16000.0)
    p.add_argument("--case-profile", default="case300-sced", choices=["raw", "case300-sced"])
    p.add_argument("--audit", default="")
    p.add_argument("--record", type=int, default=0,
                   help="Audit record index to inspect. Use -1 for all records.")
    p.add_argument("--drop-audit-failures", action="store_true")
    p.add_argument("--output", default=str(PROJ_DIR / "paper_results" / "case300_sced_credible_n1.json"))
    return p.parse_args()


def islanding_keys(net):
    import copy
    import networkx as nx
    import pandapower.topology as top

    keys = set()
    for typ, table in (("line", net.line), ("trafo", net.trafo)):
        for idx in table.index:
            net_c = copy.deepcopy(net)
            if typ == "line":
                net_c.line.at[idx, "in_service"] = False
            else:
                net_c.trafo.at[idx, "in_service"] = False
            graph = top.create_nxgraph(net_c, respect_switches=True, include_trafos=True)
            if nx.number_connected_components(graph) > 1:
                keys.add((typ, int(idx)))
    return keys


def audit_failure_keys(path, record):
    if not path:
        return set()
    with open(path, "r", encoding="utf-8") as f:
        audit = json.load(f)
    records = audit["records"] if int(record) < 0 else [audit["records"][record]]
    keys = set()
    fallback_failures = []
    for rec in records:
        strict = rec.get("strict", {})
        failed_contingencies = strict.get("metrics", {}).get("n1_failed_contingencies", [])
        for item in failed_contingencies:
            if "type" in item and "idx" in item:
                keys.add((str(item["type"]), int(item["idx"])))
        fallback_failures.extend(strict.get("failures", []))
    if keys:
        return keys
    for failure in fallback_failures:
        m = FAIL_RE.match(failure)
        if m:
            typ = m.group(1) or m.group(3)
            idx = int(m.group(2) or m.group(4))
            keys.add((typ, idx))
    return keys


def load_base_contingencies_from_audit(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        audit = json.load(f)
    cont_path = audit.get("config", {}).get("contingency_file")
    if not cont_path:
        return None
    p = Path(cont_path)
    if not p.is_absolute():
        p = PROJ_DIR / p
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    contingencies = data.get("contingencies", data)
    return [(str(item["type"]), int(item["idx"])) for item in contingencies]


def main():
    args = parse_args()
    from pandapower.networks import case300

    net = case300()
    apply_case_profile(net, args.case_profile)
    prepare_case300_for_opf(net, args.demand_mw)

    base_keys = load_base_contingencies_from_audit(args.audit)
    all_keys = base_keys or [("line", int(i)) for i in net.line.index] + [("trafo", int(i)) for i in net.trafo.index]
    island = islanding_keys(net)
    audit_fails = audit_failure_keys(args.audit, args.record) if args.drop_audit_failures else set()
    dropped = island | audit_fails
    contingencies = [{"type": typ, "idx": idx} for typ, idx in all_keys if (typ, idx) not in dropped]
    audit_config = {}
    if args.audit:
        with open(args.audit, "r", encoding="utf-8") as f:
            audit_config = json.load(f).get("config", {})

    payload = {
        "case_profile": args.case_profile,
        "demand_mw": args.demand_mw,
        "rule": {
            "drop_islanding": True,
            "drop_audit_failures": bool(args.drop_audit_failures),
            "audit": args.audit or None,
            "record": args.record if args.audit else None,
            "record_meaning": "all records" if int(args.record) < 0 else "single record",
            "base_contingency_file": audit_config.get("contingency_file"),
        },
        "counts": {
            "all": len(all_keys),
            "islanding_dropped": len(island),
            "audit_failures_dropped": len(audit_fails),
            "kept": len(contingencies),
        },
        "contingencies": contingencies,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload["counts"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

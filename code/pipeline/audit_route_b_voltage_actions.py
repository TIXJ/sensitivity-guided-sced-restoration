"""Voltage/reactive-power action audit for the 300-bus correction benchmark.

Reports how much the trained GAT+SAC policy moves generator voltage setpoints
and the reference-source voltage setpoint, and how those moves affect AC-PF
voltage limits and generator reactive-power margins.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import Counter

import numpy as np
import torch

CODE_DIR = pathlib.Path(__file__).resolve().parent
PROJ_DIR = CODE_DIR.parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(1, str(PROJ_DIR))

from audit_route_b_ext_grid import apply_state_to_net
from eval_route_b_pipeline import is_pass, verify_current
from train_rlv35_complete import (
    DEVICE,
    EXT_VM_MAX,
    EXT_VM_MIN,
    GEN_VM_MAX,
    GEN_VM_MIN,
    SCEDEnvV21,
    SACTrainer,
    split_samples_stratified,
)


def _load_samples(path: pathlib.Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data["samples"] if isinstance(data, dict) and "samples" in data else data
    samples = [s for s in samples if s.get("is_complete_sced", True)]
    return data, samples


def _load_split(path: pathlib.Path, samples):
    if path.exists():
        split = json.loads(path.read_text(encoding="utf-8"))
        return split["train_ids"], split["val_ids"], split["test_ids"]
    return split_samples_stratified(samples)


def _load_actor(trainer, checkpoint: pathlib.Path):
    ckpt = torch.load(str(checkpoint), map_location=DEVICE, weights_only=False)
    state = ckpt.get("actor", ckpt.get("actor_state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    trainer.actor.load_state_dict(state)
    trainer.actor.eval()


def _run_ac_details(env):
    import pandapower as pp

    net = apply_state_to_net(env)
    err = None
    t0 = time.perf_counter()
    try:
        pp.runpp(net, calculate_voltage_angles=True, enforce_q_lims=True, numba=False, max_iteration=50)
    except Exception as exc:
        err = repr(exc)
    runtime = time.perf_counter() - t0
    converged = bool(getattr(net, "converged", False)) and err is None
    out = {"converged": converged, "runtime_sec": runtime, "error": err}
    if not converged:
        return out
    vm = net.res_bus.vm_pu.values
    low = np.maximum(net.bus.min_vm_pu.values - vm, 0.0)
    high = np.maximum(vm - net.bus.max_vm_pu.values, 0.0)
    out.update({
        "v_min": float(vm.min()),
        "v_max": float(vm.max()),
        "v_violations": int((low > 0.0).sum() + (high > 0.0).sum()),
        "v_low": int((low > 0.0).sum()),
        "v_high": int((high > 0.0).sum()),
        "v_gap_max": float(max(low.max(initial=0.0), high.max(initial=0.0))),
        "line_overloaded": int((net.res_line.loading_percent.values > 100.0).sum()),
        "trafo_overloaded": int((net.res_trafo.loading_percent.values > 100.0).sum()) if len(net.trafo) else 0,
        "ext_grid_p_mw": float(net.res_ext_grid.p_mw.iloc[0]) if len(net.ext_grid) else None,
        "ext_grid_q_mvar": float(net.res_ext_grid.q_mvar.iloc[0]) if len(net.ext_grid) else None,
    })
    if hasattr(net, "res_gen") and len(net.res_gen):
        q = net.res_gen.q_mvar.values.astype(float)
        qmin = net.gen.min_q_mvar.values.astype(float) if "min_q_mvar" in net.gen else np.full(len(q), np.nan)
        qmax = net.gen.max_q_mvar.values.astype(float) if "max_q_mvar" in net.gen else np.full(len(q), np.nan)
        out.update({
            "gen_q_min": float(np.nanmin(q)),
            "gen_q_max": float(np.nanmax(q)),
            "gen_q_abs_max": float(np.nanmax(np.abs(q))),
            "gen_q_min_margin_min": float(np.nanmin(q - qmin)),
            "gen_q_max_margin_min": float(np.nanmin(qmax - q)),
            "gen_q_at_min_count": int(np.nansum((q - qmin) <= 1e-3)),
            "gen_q_at_max_count": int(np.nansum((qmax - q) <= 1e-3)),
        })
    return out


def _eval_gat_only(env, sid):
    env.reset(sample_idx=sid)
    env.current_power = np.asarray(env.gat_power, dtype=np.float64).copy()
    base_gen = np.asarray(env.current_gen_vm_pu, dtype=np.float64).copy()
    base_ext = np.asarray(env.current_ext_grid_vm_pu, dtype=np.float64).copy()
    ac = _run_ac_details(env)
    verification = verify_current(env)
    ac.update({k: v for k, v in verification.items() if k not in ac})
    ac["passed"] = bool(is_pass(verification))
    return ac, base_gen, base_ext


def _eval_sac(env, trainer, sid, deployment, max_steps_override=0):
    obs = env.reset(sample_idx=sid)
    base_gen = np.asarray(env.current_gen_vm_pu, dtype=np.float64).copy()
    base_ext = np.asarray(env.current_ext_grid_vm_pu, dtype=np.float64).copy()
    last = {}
    raw_action_l1 = 0.0
    p_dim = env._n_shift_dirs + env._n_key_gens
    max_steps = int(max_steps_override) if max_steps_override else env.max_steps
    # Match the final hierarchical deployment: exhaust voltage-only SAC
    # correction before enabling the active-power action components.
    first_steps = max_steps
    for step in range(first_steps):
        raw = trainer.select_action(obs, deterministic=True)
        lo, hi = env.get_action_bounds()
        action = lo + (raw + 1.0) * 0.5 * (hi - lo)
        if deployment == "hierarchical":
            action[:p_dim] = 0.0
        raw_action_l1 += float(np.abs(action).sum())
        obs, _r, done, _trunc, last = env.step(action)
        if done:
            break
    if deployment == "hierarchical" and not is_pass(last):
        obs = env.reset(sample_idx=sid)
        raw_action_l1 = 0.0
        for step in range(max_steps):
            raw = trainer.select_action(obs, deterministic=True)
            lo, hi = env.get_action_bounds()
            action = lo + (raw + 1.0) * 0.5 * (hi - lo)
            raw_action_l1 += float(np.abs(action).sum())
            obs, _r, done, _trunc, last = env.step(action)
            if done:
                break
    ac = _run_ac_details(env)
    ac.update({k: v for k, v in last.items() if k not in ac})
    ac["passed"] = bool(is_pass(last))
    gen_delta = np.asarray(env.current_gen_vm_pu, dtype=np.float64) - base_gen
    ext_delta = np.asarray(env.current_ext_grid_vm_pu, dtype=np.float64) - base_ext
    ac.update({
        "steps": int(step + 1),
        "action_l1_total": raw_action_l1,
        "gen_vm_delta_mean_abs": float(np.mean(np.abs(gen_delta))),
        "gen_vm_delta_max_abs": float(np.max(np.abs(gen_delta))),
        "gen_vm_delta_min": float(np.min(gen_delta)),
        "gen_vm_delta_max": float(np.max(gen_delta)),
        "gen_vm_touch_minmax_count": int(np.sum((env.current_gen_vm_pu <= GEN_VM_MIN + 1e-6) | (env.current_gen_vm_pu >= GEN_VM_MAX - 1e-6))),
        "ref_vm_delta": float(ext_delta[0]) if len(ext_delta) else 0.0,
        "ref_vm_touch_minmax": bool(len(env.current_ext_grid_vm_pu) and ((env.current_ext_grid_vm_pu[0] <= EXT_VM_MIN + 1e-6) or (env.current_ext_grid_vm_pu[0] >= EXT_VM_MAX - 1e-6))),
        "active_action_dim": int(p_dim),
        "voltage_action_dim": int(env._n_voltage_actions),
    })
    return ac


def _mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def _max(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.max(vals)) if vals else None


def _summary(rows):
    out = {}
    for mode in sorted({r["mode"] for r in rows}):
        rs = [r for r in rows if r["mode"] == mode]
        out[mode] = {
            "n": len(rs),
            "passed": int(sum(r["passed"] for r in rs)),
            "pass_rate_pct": round(100.0 * sum(r["passed"] for r in rs) / max(len(rs), 1), 3),
            "v_violations_mean": _mean(rs, "v_violations"),
            "v_gap_max_mean": _mean(rs, "v_gap_max"),
            "v_min_mean": _mean(rs, "v_min"),
            "v_min_min": float(np.min([r["v_min"] for r in rs if r.get("v_min") is not None])) if any(r.get("v_min") is not None for r in rs) else None,
            "v_max_mean": _mean(rs, "v_max"),
            "gen_vm_delta_mean_abs": _mean(rs, "gen_vm_delta_mean_abs"),
            "gen_vm_delta_max_abs": _max(rs, "gen_vm_delta_max_abs"),
            "gen_vm_touch_minmax_total": int(sum(r.get("gen_vm_touch_minmax_count", 0) for r in rs)),
            "ref_vm_delta_mean": _mean(rs, "ref_vm_delta"),
            "ref_vm_delta_max_abs": float(np.max(np.abs([r["ref_vm_delta"] for r in rs if r.get("ref_vm_delta") is not None]))) if any(r.get("ref_vm_delta") is not None for r in rs) else None,
            "ref_vm_touch_minmax_count": int(sum(bool(r.get("ref_vm_touch_minmax", False)) for r in rs)),
            "gen_q_min_margin_min": float(np.min([r["gen_q_min_margin_min"] for r in rs if r.get("gen_q_min_margin_min") is not None])) if any(r.get("gen_q_min_margin_min") is not None for r in rs) else None,
            "gen_q_max_margin_min": float(np.min([r["gen_q_max_margin_min"] for r in rs if r.get("gen_q_max_margin_min") is not None])) if any(r.get("gen_q_max_margin_min") is not None for r in rs) else None,
            "gen_q_at_min_total": int(sum(r.get("gen_q_at_min_count", 0) for r in rs)),
            "gen_q_at_max_total": int(sum(r.get("gen_q_at_max_count", 0) for r in rs)),
            "fail_reasons": dict(Counter(r.get("reason", "fail") for r in rs if not r["passed"]).most_common(8)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(PROJ_DIR / "case300_gt_v10_dataset.json"))
    ap.add_argument("--split-file", default=str(PROJ_DIR / "rl_scratch_sac" / "split_ids.json"))
    ap.add_argument("--checkpoint", default=str(PROJ_DIR / "rl_scratch_sac" / "best_actor.pth"))
    ap.add_argument("--scope", choices=["val", "test"], default="test")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="Optional smoke-test override; 0 keeps the trained environment default.")
    ap.add_argument("--deployment", choices=["hierarchical", "full_hybrid"], default="hierarchical")
    ap.add_argument("--output", default=str(PROJ_DIR / "paper_results" / "route_b_voltage_action_audit.json"))
    args = ap.parse_args()

    data, samples = _load_samples(pathlib.Path(args.dataset))
    _train, val_ids, test_ids = _load_split(pathlib.Path(args.split_file), samples)
    ids = val_ids if args.scope == "val" else test_ids
    if args.max_samples:
        ids = ids[: args.max_samples]

    env = SCEDEnvV21(samples, ids, dataset_meta=data if isinstance(data, dict) else None)
    trainer = SACTrainer(env._get_obs_dim(), env.action_dim, device=DEVICE)
    _load_actor(trainer, pathlib.Path(args.checkpoint))

    rows = []
    for i, sid in enumerate(ids, 1):
        sample = samples[sid]
        gat, _base_gen, _base_ext = _eval_gat_only(env, sid)
        rows.append({
            "sid": sid,
            "sample_id": sample.get("sample_id"),
            "demand_gw": float(sample.get("demand_gw", sample["demand_mw"] / 1000.0)),
            "mode": "gat_only",
            "reason": "pass" if gat["passed"] else ("acpf_not_converged" if not gat.get("converged") else "voltage_violation"),
            **gat,
        })
        sac = _eval_sac(env, trainer, sid, args.deployment, max_steps_override=args.max_steps)
        rows.append({
            "sid": sid,
            "sample_id": sample.get("sample_id"),
            "demand_gw": float(sample.get("demand_gw", sample["demand_mw"] / 1000.0)),
            "mode": "gat_sac",
            "reason": "pass" if sac["passed"] else ("acpf_not_converged" if not sac.get("converged") else "voltage_violation"),
            **sac,
        })
        print(f"[{i}/{len(ids)}] {sample.get('sample_id')} gat={'OK' if gat['passed'] else 'FAIL'} sac={'OK' if sac['passed'] else 'FAIL'}")

    result = {
        "dataset": args.dataset,
        "split_file": args.split_file,
        "checkpoint": args.checkpoint,
        "scope": args.scope,
        "deployment": args.deployment,
        "n_eval": len(ids),
        "summary": _summary(rows),
        "rows": rows,
    }
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print("Wrote:", args.output)


if __name__ == "__main__":
    main()

"""Per-sample physical, accuracy, cost, and runtime audit for V26 GAT+SAC."""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
import time
from collections import Counter

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
ALIGNED_PATH = ROOT / "paper_results" / "code" / "train_sac_n1_active_aligned.py"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=str(ROOT / "case300_gt_n1s0575_active_required_fixed32.json"))
    p.add_argument("--split-file", default=str(ROOT / "paper_results" / "case300_n1s0575_active_fixed32_split.json"))
    p.add_argument("--gat-checkpoint", default=str(ROOT / "gatv10_results_n1s0575_candidate" / "gatv10_predictor_best.pth"))
    p.add_argument("--actor-checkpoint", default=str(ROOT / "rl_sac_n1_active_aligned_v26" / "best_actor.pth"))
    p.add_argument("--timing-repeats", type=int, default=30)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--output", default=str(ROOT / "paper_results" / "case300_v26_pipeline_audit.json"))
    return p.parse_args()


def load_aligned(args):
    old_argv = sys.argv[:]
    sys.argv = [
        str(ALIGNED_PATH),
        "--dataset", str(args.dataset),
        "--split-file", str(args.split_file),
        "--gat-checkpoint", str(args.gat_checkpoint),
        "--output-dir", str(ROOT / "paper_results" / ".audit_v26_runtime"),
    ]
    try:
        spec = importlib.util.spec_from_file_location("_v26_aligned_audit", ALIGNED_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = old_argv


def sync_device():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def generation_cost(sample, p):
    p = np.asarray(p, dtype=np.float64)
    cp2 = np.asarray(sample.get("sample_cp2", np.zeros_like(p)), dtype=np.float64)
    cp1 = np.asarray(sample.get("sample_cp1", np.ones_like(p) * 30.0), dtype=np.float64)
    cp0 = np.asarray(sample.get("sample_cp0", np.zeros_like(p)), dtype=np.float64)
    return float(np.sum(cp2 * p * p + cp1 * p + cp0))


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(a)), "std": float(np.std(a)),
        "min": float(np.min(a)), "max": float(np.max(a)),
        "median": float(np.median(a)),
    }


def main():
    args = parse_args()
    aligned = load_aligned(args)

    class TimedEnv(aligned.CleanN1Env):
        def __init__(self, *a, **kw):
            self.clear_timers()
            super().__init__(*a, **kw)

        def clear_timers(self):
            self.audit_ac_sec = 0.0
            self.audit_ac_calls = 0
            self.audit_n1_sec = 0.0
            self.audit_n1_calls = 0

        def _run_ac_pf_check(self):
            t0 = time.perf_counter()
            out = super()._run_ac_pf_check()
            self.audit_ac_sec += time.perf_counter() - t0
            self.audit_ac_calls += 1
            return out

        def _vectorized_n1_state(self):
            t0 = time.perf_counter()
            out = super()._vectorized_n1_state()
            self.audit_n1_sec += time.perf_counter() - t0
            self.audit_n1_calls += 1
            return out

    data = json.loads(pathlib.Path(args.dataset).read_text(encoding="utf-8"))
    samples = [s for s in data["samples"] if s.get("is_complete_sced", True)]
    split = json.loads(pathlib.Path(args.split_file).read_text(encoding="utf-8"))
    test_ids = [int(i) for i in split["test_ids"]]
    if args.max_samples > 0:
        test_ids = test_ids[:args.max_samples]
    env = TimedEnv(samples, test_ids, data)
    env.clear_timers()
    obs = env.reset(sample_idx=test_ids[0])
    p_dim = env._n_shift_dirs + env._n_key_gens
    aligned.core.SAC_LOG_STD_MAX_VECTOR = np.concatenate([
        np.full(p_dim, aligned.ARGS.active_log_std_max, dtype=np.float32),
        np.full(env.action_dim - p_dim, aligned.ARGS.voltage_log_std_max, dtype=np.float32),
    ])
    actor = aligned.BranchedGaussianActor(len(obs), p_dim, env.action_dim - p_dim).to(aligned.core.DEVICE)
    state = torch.load(args.actor_checkpoint, map_location=aligned.core.DEVICE, weights_only=True)
    actor.load_state_dict(state)
    actor.eval()

    # Warm up kernels and model dispatch before collecting timings.
    for _ in range(5):
        _ = actor.select_action(torch.as_tensor(obs, dtype=torch.float32, device=aligned.core.DEVICE).unsqueeze(0), deterministic=True)
        _ = env._predict_gat(float(samples[test_ids[0]]["demand_mw"]))
    sync_device()

    rows = []
    flat_pred, flat_label = [], []
    reasons = Counter()
    for position, sid in enumerate(test_ids, 1):
        sample = samples[sid]
        env.clear_timers()
        sync_device()
        reset_t0 = time.perf_counter()
        obs = env.reset(sample_idx=sid)
        sync_device()
        reset_sec = time.perf_counter() - reset_t0
        initial_p = np.asarray(env.current_power, dtype=np.float64).copy()
        label_p = np.asarray(sample["dispatch_68"], dtype=np.float64)
        initial_n1 = env._vectorized_n1_state().copy()
        initial_ac = env._run_ac_pf_check()
        initial_voltage = float(initial_ac.get("v_violation_mag_pu", 0.0)) + float(initial_ac.get("v_violation_max_pu", 0.0))
        base_gen_vm = np.asarray(env.base_gen_vm_pu, dtype=np.float64).copy()
        base_ext_vm = float(env.base_ext_grid_vm_pu[0])

        # Pure GAT+projection microbenchmark, separate from environment setup.
        gat_times = []
        for _ in range(max(1, args.timing_repeats)):
            sync_device()
            t0 = time.perf_counter()
            prediction = env._predict_gat(float(sample["demand_mw"]))
            _ = aligned.core.project_box_sum(
                prediction, env.PMIN, env.PMAX * 0.99, float(sample["demand_mw"])
            )
            sync_device()
            gat_times.append(time.perf_counter() - t0)

        env.clear_timers()
        actor_sec = 0.0
        step_sec = 0.0
        last_info = {}
        actions = []
        rollout_t0 = time.perf_counter()
        for step in range(env.max_steps):
            tensor = torch.as_tensor(obs, dtype=torch.float32, device=aligned.core.DEVICE).unsqueeze(0)
            sync_device()
            t0 = time.perf_counter()
            action = actor.select_action(tensor, deterministic=True).squeeze(0).cpu().numpy()
            sync_device()
            actor_sec += time.perf_counter() - t0
            actions.append(action.tolist())
            t0 = time.perf_counter()
            obs, reward, done, _truncated, last_info = env.step(aligned.scale_action(env, action))
            step_sec += time.perf_counter() - t0
            if done:
                break
        rollout_sec = time.perf_counter() - rollout_t0
        rollout_ac_sec = env.audit_ac_sec
        rollout_ac_calls = env.audit_ac_calls
        rollout_n1_sec = env.audit_n1_sec
        rollout_n1_calls = env.audit_n1_calls
        final_p = np.asarray(env.current_power, dtype=np.float64).copy()
        final_gen_vm = np.asarray(env.current_gen_vm_pu, dtype=np.float64).copy()
        final_ext_vm = float(env.current_ext_grid_vm_pu[0])

        # Fresh final audit, independent of the terminal info dictionary.
        verify_t0 = time.perf_counter()
        final_ac = env._run_ac_pf_check()
        env._aligned_n1_cache_key = None
        final_n1 = env._vectorized_n1_state().copy()
        verify_sec = time.perf_counter() - verify_t0
        success = bool(aligned.is_pass({**last_info, **final_ac, "n1_violations": final_n1["violation_count"], "n1_overload_mw": final_n1["total_overload_mw"]}))
        reason = "full_sced_pass" if success else str(last_info.get("reason", "unknown"))
        reasons[reason] += 1

        delta_p = final_p - initial_p
        delta_vm = final_gen_vm - base_gen_vm
        gat_cost = generation_cost(sample, initial_p)
        label_cost = generation_cost(sample, label_p)
        final_cost = generation_cost(sample, final_p)
        row = {
            "sid": sid,
            "sample_id": sample.get("sample_id", str(sid)),
            "demand_gw": float(sample.get("demand_gw", sample["demand_mw"] / 1000.0)),
            "passed": success,
            "reason": reason,
            "steps": step + 1,
            "gat_mae_mw": float(np.mean(np.abs(initial_p - label_p))),
            "gat_rmse_mw": float(np.sqrt(np.mean((initial_p - label_p) ** 2))),
            "initial_n1_overload_mw": float(initial_n1["total_overload_mw"]),
            "initial_n1_violations": int(initial_n1["violation_count"]),
            "initial_voltage_potential_pu": initial_voltage,
            "initial_voltage_violations": int(initial_ac.get("v_violations", 999)),
            "final_n1_overload_mw": float(final_n1["total_overload_mw"]),
            "final_n1_violations": int(final_n1["violation_count"]),
            "final_voltage_potential_pu": float(final_ac.get("v_violation_mag_pu", 0.0)) + float(final_ac.get("v_violation_max_pu", 0.0)),
            "final_voltage_violations": int(final_ac.get("v_violations", 999)),
            "active_delta_l1_mw": float(np.linalg.norm(delta_p, 1)),
            "active_delta_l2_mw": float(np.linalg.norm(delta_p, 2)),
            "active_delta_max_generator_mw": float(np.max(np.abs(delta_p))),
            "generator_voltage_delta_l1_pu": float(np.linalg.norm(delta_vm, 1)),
            "generator_voltage_delta_max_pu": float(np.max(np.abs(delta_vm))),
            "generator_voltage_touch_count": int(np.count_nonzero((final_gen_vm <= aligned.core.GEN_VM_MIN + 1e-8) | (final_gen_vm >= aligned.core.GEN_VM_MAX - 1e-8))),
            "reference_voltage_delta_pu": final_ext_vm - base_ext_vm,
            "reference_source_p_mw": float(final_ac.get("reference_source_p_mw", np.nan)),
            "reference_source_q_mvar": float(final_ac.get("reference_source_q_mvar", np.nan)),
            "reference_source_s_mva": float(final_ac.get("reference_source_s_mva", np.nan)),
            "reference_source_s_utilization_pct": float(final_ac.get("reference_source_s_utilization_pct", np.nan)),
            "projected_gat_cost": gat_cost,
            "linear_sced_label_cost": label_cost,
            "final_scheduled_controllable_generation_cost": final_cost,
            "cost_gap_vs_projected_gat_pct": 100.0 * (final_cost - gat_cost) / max(abs(gat_cost), 1e-9),
            "cost_gap_vs_linear_sced_label_pct": 100.0 * (final_cost - label_cost) / max(abs(label_cost), 1e-9),
            "gat_projection_runtime_sec": float(np.mean(gat_times)),
            "reset_runtime_sec": reset_sec,
            "actor_runtime_sec": actor_sec,
            "actor_runtime_per_step_sec": actor_sec / (step + 1),
            "environment_step_runtime_sec": step_sec,
            "rollout_runtime_sec": rollout_sec,
            "online_pipeline_runtime_sec": reset_sec + rollout_sec,
            "fresh_final_verification_runtime_sec": verify_sec,
            "rollout_ac_pf_runtime_sec": rollout_ac_sec,
            "rollout_ac_pf_calls": rollout_ac_calls,
            "rollout_n1_runtime_sec": rollout_n1_sec,
            "rollout_n1_calls": rollout_n1_calls,
            "final_dispatch_mw": final_p.tolist(),
            "final_generator_vm_pu": final_gen_vm.tolist(),
            "final_reference_vm_pu": final_ext_vm,
            "deterministic_actions": actions,
        }
        rows.append(row)
        flat_pred.extend(initial_p.tolist())
        flat_label.extend(label_p.tolist())
        print(f"[{position:02d}/{len(test_ids)}] {row['sample_id']} {'PASS' if success else 'FAIL'} steps={step + 1} time={row['online_pipeline_runtime_sec']:.3f}s", flush=True)

    pred = np.asarray(flat_pred, dtype=np.float64)
    label = np.asarray(flat_label, dtype=np.float64)
    metric_keys = [
        "initial_n1_overload_mw", "initial_n1_violations", "initial_voltage_potential_pu",
        "initial_voltage_violations", "active_delta_l1_mw", "active_delta_l2_mw",
        "active_delta_max_generator_mw", "generator_voltage_delta_l1_pu",
        "generator_voltage_delta_max_pu", "generator_voltage_touch_count",
        "reference_voltage_delta_pu", "reference_source_p_mw", "reference_source_q_mvar",
        "reference_source_s_mva", "reference_source_s_utilization_pct",
        "projected_gat_cost", "linear_sced_label_cost", "final_scheduled_controllable_generation_cost",
        "cost_gap_vs_projected_gat_pct", "cost_gap_vs_linear_sced_label_pct",
        "gat_projection_runtime_sec", "reset_runtime_sec", "actor_runtime_sec",
        "actor_runtime_per_step_sec", "environment_step_runtime_sec", "rollout_runtime_sec",
        "online_pipeline_runtime_sec", "fresh_final_verification_runtime_sec",
        "rollout_ac_pf_runtime_sec", "rollout_n1_runtime_sec", "steps",
    ]
    summary = {
        "total": len(rows),
        "passed": int(sum(r["passed"] for r in rows)),
        "reasons": dict(reasons),
        "gat_projected_test_mae_mw": float(np.mean(np.abs(pred - label))),
        "gat_projected_test_rmse_mw": float(np.sqrt(np.mean((pred - label) ** 2))),
        "gat_projected_test_pearson": float(np.corrcoef(pred, label)[0, 1]),
        "metrics": {key: stats([r[key] for r in rows]) for key in metric_keys},
        "reference_source_limits": {
            "p_mw": [aligned.core.REF_SOURCE_P_MIN_MW, aligned.core.REF_SOURCE_P_MAX_MW],
            "q_mvar": [aligned.core.REF_SOURCE_Q_MIN_MVAR, aligned.core.REF_SOURCE_Q_MAX_MVAR],
            "s_mva_max": aligned.core.REF_SOURCE_S_MAX_MVA,
        },
    }
    payload = {
        "dataset": str(pathlib.Path(args.dataset).resolve()),
        "split_file": str(pathlib.Path(args.split_file).resolve()),
        "gat_checkpoint": str(pathlib.Path(args.gat_checkpoint).resolve()),
        "actor_checkpoint": str(pathlib.Path(args.actor_checkpoint).resolve()),
        "timing_repeats": args.timing_repeats,
        "summary": summary,
        "rows": rows,
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote: {output}", flush=True)


if __name__ == "__main__":
    main()

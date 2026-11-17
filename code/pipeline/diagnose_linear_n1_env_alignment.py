"""Audit current RL linear N-1 checker against dataset labels and rescue points."""

import json
import pathlib
import sys
from collections import Counter

import numpy as np

CODE_DIR = pathlib.Path(__file__).resolve().parent
PROJ_DIR = CODE_DIR.parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(1, str(PROJ_DIR))

import train_rlv35_complete as rlv35
from train_rlv35_complete import SCEDEnvV21, split_samples_stratified


def check_dispatch(env, sid, dispatch, gen_vm=None, ext_vm=None):
    env.reset(sample_idx=sid)
    env.current_power = np.asarray(dispatch, dtype=np.float64)
    if gen_vm is not None:
        env.current_gen_vm_pu = np.asarray(gen_vm, dtype=np.float64)
    if ext_vm is not None:
        env.current_ext_grid_vm_pu = np.asarray(ext_vm, dtype=np.float64)
    env.corr_step = 1
    n1_overload_mw, n1_violations = env._compute_n1_overload_margin()
    return {
        "n1_violations": int(n1_violations),
        "n1_overload_mw": float(n1_overload_mw),
    }


def main():
    dataset = PROJ_DIR / "case300_gt_v10_dataset.json"
    with open(dataset, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = [s for s in data["samples"] if s.get("is_complete_sced")]

    rlv35.AC_PF_CHECK_EVERY = 1
    rlv35.N1_CHECK_EVERY_STEPS = 1

    train_ids, _val_ids, _test_ids = split_samples_stratified(samples)
    env = SCEDEnvV21(samples, train_ids, dataset_meta=data)

    label_ok = 0
    rescue_ok = 0
    label_reasons = Counter()
    rescue_reasons = Counter()
    examples = []

    for sid, sample in enumerate(samples):
        label = check_dispatch(env, sid, sample["dispatch_68"])
        label_pass = label["n1_violations"] == 0
        if label_pass:
            label_ok += 1
        else:
            label_reasons["label_n1_fail"] += 1

        rescue = None
        rescue_pass = False
        if "rescue_dispatch_68" in sample:
            rescue = check_dispatch(
                env,
                sid,
                sample["rescue_dispatch_68"],
                sample.get("rescue_gen_vm_pu"),
                sample.get("rescue_ext_grid_vm_pu"),
            )
            rescue_pass = rescue["n1_violations"] == 0
            if rescue_pass:
                rescue_ok += 1
            else:
                rescue_reasons["rescue_n1_fail"] += 1

        if len(examples) < 12 and (not label_pass or not rescue_pass):
            examples.append({
                "sid": sid,
                "sample_id": sample.get("sample_id"),
                "demand_gw": sample.get("demand_gw"),
                "dataset_linear_n1_pass": sample.get("linear_n1_pass"),
                "dataset_n1_after_corr": sample.get("n1_violations_after_corr"),
                "label": label,
                "rescue": rescue,
            })

    out = {
        "dataset": str(dataset),
        "samples": len(samples),
        "label_n1_pass_current_env": label_ok,
        "label_n1_pass_rate_pct": round(label_ok / max(len(samples), 1) * 100, 3),
        "rescue_n1_pass_current_env": rescue_ok,
        "rescue_n1_pass_rate_pct": round(rescue_ok / max(len(samples), 1) * 100, 3),
        "label_reasons": dict(label_reasons),
        "rescue_reasons": dict(rescue_reasons),
        "examples": examples,
    }

    out_path = PROJ_DIR / "paper_results" / "linear_n1_env_alignment_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()

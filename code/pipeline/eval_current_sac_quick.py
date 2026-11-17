"""Quick deterministic evaluation for the current SAC checkpoint."""

import json
import pathlib
import sys
from collections import Counter, defaultdict

import torch

CODE_DIR = pathlib.Path(__file__).resolve().parent
PROJ_DIR = CODE_DIR.parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(1, str(PROJ_DIR))

from train_rlv35_complete import SCEDEnvV21, SACTrainer, split_samples_stratified, DEVICE


def run_one(env, trainer, sid):
    obs = env.reset(sample_idx=sid)
    last = {}
    total_r = 0.0
    for _ in range(env.max_steps + 1):
        raw = trainer.select_action(obs, deterministic=True)
        lo, hi = env.get_action_bounds()
        action = lo + (raw + 1.0) * 0.5 * (hi - lo)
        obs, r, done, _trunc, info = env.step(action)
        total_r += float(r)
        last = info
        if done:
            break
    ok = (
        last.get("reason") == "full_sced_pass"
        or (
            last.get("converged", False)
            and last.get("v_violations", 999) == 0
            and last.get("overloaded_lines", 999) == 0
            and last.get("trafo_overloaded", 999) == 0
            and last.get("n1_violations", 999) == 0
        )
    )
    return ok, total_r, last


def main():
    with open(PROJ_DIR / "case300_gt_v10_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = [s for s in data["samples"] if s.get("is_complete_sced")]
    train_ids, val_ids, test_ids = split_samples_stratified(samples)
    sample_ids = test_ids[:20]

    env = SCEDEnvV21(samples, sample_ids, dataset_meta=data)
    od, ad = env._get_obs_dim(), env.action_dim
    trainer = SACTrainer(od, ad, device=DEVICE)
    ckpt = PROJ_DIR / "rl_scratch_sac" / "best_actor.pth"
    trainer.actor.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
    trainer.actor.eval()

    ok_count = 0
    reasons = Counter()
    by_gw = defaultdict(lambda: [0, 0])
    rows = []
    for sid in sample_ids:
        ok, total_r, info = run_one(env, trainer, sid)
        gw = float(samples[sid].get("demand_gw", samples[sid]["demand_mw"] / 1000.0))
        by_gw[gw][1] += 1
        if ok:
            ok_count += 1
            by_gw[gw][0] += 1
        else:
            reasons[info.get("reason", "unknown")] += 1
        rows.append({
            "sid": sid,
            "sample_id": samples[sid].get("sample_id"),
            "gw": gw,
            "ok": ok,
            "reward": round(total_r, 3),
            "reason": info.get("reason"),
            "v_violations": info.get("v_violations"),
            "v_gap": info.get("v_violation_max_pu"),
            "n1_violations": info.get("n1_violations"),
        })
        print(f"{len(rows):02d}/{len(sample_ids)} sid={sid} {gw:g}GW {'OK' if ok else 'FAIL'} reason={info.get('reason')}")

    out = {
        "checkpoint": str(ckpt),
        "n_eval": len(sample_ids),
        "passed": ok_count,
        "pass_rate_pct": round(ok_count / max(len(sample_ids), 1) * 100.0, 3),
        "reasons": dict(reasons),
        "by_demand": {f"{k:g}": {"passed": v[0], "n": v[1]} for k, v in sorted(by_gw.items())},
        "rows": rows,
    }
    out_path = PROJ_DIR / "paper_results" / "current_sac_quick_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()

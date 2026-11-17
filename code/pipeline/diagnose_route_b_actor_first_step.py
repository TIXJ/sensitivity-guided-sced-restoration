"""Compare actor checkpoints on one deterministic correction step per demand."""
import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch

CODE_DIR = pathlib.Path(__file__).resolve().parent
PROJ_DIR = CODE_DIR.parents[1]
sys.path.insert(0, str(CODE_DIR))

from train_rlv35_complete import DEVICE, SACTrainer, SCEDEnvV21


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(PROJ_DIR / "case300_gt_v10_dataset_lodf_corrected_277.json"))
    ap.add_argument("--split", default=str(PROJ_DIR / "rl_scratch_sac_lodf_corrected_final" / "split_ids.json"))
    ap.add_argument("--gat-checkpoint", default=str(PROJ_DIR / "gatv10_results_lodf_corrected" / "gatv10_predictor_best.pth"))
    ap.add_argument("--checkpoints", nargs="+", required=True)
    args = ap.parse_args()

    os.environ["ROUTE_B_GAT_CHECKPOINT"] = str(pathlib.Path(args.gat_checkpoint).resolve())
    data = json.load(open(args.dataset, "r", encoding="utf-8"))
    samples = [s for s in data["samples"] if s.get("is_complete_sced")]
    split = json.load(open(args.split, "r", encoding="utf-8"))
    val_ids = split["val_ids"]
    by_gw = {}
    for sid in val_ids:
        gw = round(float(samples[sid].get("demand_gw", samples[sid]["demand_mw"] / 1000.0)), 3)
        by_gw.setdefault(gw, sid)
    probe_ids = [by_gw[gw] for gw in sorted(by_gw)]

    env = SCEDEnvV21(samples, probe_ids, dataset_meta=data)
    obs_dim, action_dim = env._get_obs_dim(), env.action_dim
    p_dim = env._n_shift_dirs + env._n_key_gens

    for checkpoint in args.checkpoints:
        trainer = SACTrainer(obs_dim, action_dim, device=DEVICE)
        trainer.actor.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
        trainer.actor.eval()
        print(f"\nCHECKPOINT {checkpoint}")
        for sid in probe_ids:
            obs = env.reset(sample_idx=sid)
            raw = trainer.select_action(obs, deterministic=True)
            low, high = env.get_action_bounds()
            action = low + (raw + 1.0) * 0.5 * (high - low)
            _obs, _reward, _done, _truncated, info = env.step(action)
            print(
                f"{samples[sid].get('sample_id', sid)} "
                f"{samples[sid]['demand_mw']/1000.0:g}GW "
                f"praw={np.mean(np.abs(raw[:p_dim])):.4f} "
                f"vraw={np.mean(np.abs(raw[p_dim:])):.4f} "
                f"conv={int(bool(info.get('converged', False)))} "
                f"vv={info.get('v_violations', 999)} "
                f"vgap={info.get('v_violation_max_pu', 999.0):.4f} "
                f"n1={info.get('n1_violations', 999)} "
                f"reason={info.get('reason')}"
            )


if __name__ == "__main__":
    main()

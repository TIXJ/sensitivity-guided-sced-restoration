"""
Full paper experiment pipeline — everything from scratch, no checkpoint confusion.

Step 1: Train SAC from scratch (100 updates) → rl_scratch_sac/
Step 2: Label Gen vs Gurobi comparison (no training needed)
Step 3: Lookup baseline (no training needed)
Step 4: Fixed SAC eval (P2 mean-action + P3 K=3 retry, proper N-1)
Step 5: Cost gap analysis

One model, two eval modes. Clean.
"""
import subprocess, sys, json
from pathlib import Path

PROJ_DIR = Path(r".")
PYTHON = r"external/Anaconda\envs\lct-GPU\python.exe"


def run(name, script_args, timeout=7200):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    result = subprocess.run(
        [PYTHON] + script_args,
        cwd=str(PROJ_DIR), capture_output=True, text=True, timeout=timeout,
        encoding='utf-8', errors='replace'
    )
    out = result.stdout or ''
    print(out[-2000:] if len(out) > 2000 else out)
    if result.returncode != 0:
        print(f"  FAILED (rc={result.returncode})")
        print(result.stderr[-1000:] if result.stderr else "")
    return result.returncode


# ===== Step 1: Train SAC from scratch (skip if checkpoint exists) =====
ckpt = PROJ_DIR / 'rl_scratch_sac' / 'best_actor.pth'
if ckpt.exists():
    print(f"\nStep 1: SKIP — checkpoint already exists: {ckpt}")
else:
    run("Step 1: Train SAC from scratch (100 updates, ~2.1h)",
        ["_train_scratch_full.py", "SAC"])

# ===== Step 2: Label Gen vs Gurobi =====
run("Step 2: Label Gen vs Gurobi comparison",
    ["paper_results/code/label_gen_vs_gurobi.py"])

# ===== Step 3: Lookup baseline =====
run("Step 3: Lookup baseline",
    ["paper_results/code/lookup_baseline.py"])

# ===== Step 4: Fixed SAC eval (P2 + P3, proper N-1) =====
run("Step 4: SAC eval with N-1 fix (P2 mean + P3 retry)",
    ["paper_results/code/fixed_eval_sac_p3.py"])

# ===== Step 4b: Fixed ALL algos eval (Table 5: SAC/TD3/PPO/DDPG) =====
run("Step 4b: ALL RL algorithms with N-1 fix (Table 5)",
    ["paper_results/code/fixed_eval_all_algo.py"])

# ===== Step 5: Cost gap =====
run("Step 5: Cost gap analysis",
    ["paper_results/code/cost_gap.py"])

print(f"\n{'='*60}")
print("ALL DONE. Results in paper_results/")
print(f"{'='*60}")
for f in sorted(PROJ_DIR.glob('paper_results/*.json')):
    print(f"  {f.name}")

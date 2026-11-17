"""
Launch all four RL algorithms in parallel from scratch.
Each runs 100 updates on GPU, saves to rl_scratch_{algo}/.
Uses the FIXED train_rlv35_complete.py with real N-1 verification.

Usage: python paper_results/code/train_all_parallel.py
"""
import subprocess, sys, time, os
from pathlib import Path

PROJ_DIR = Path(r".")
PYTHON = r"external/Anaconda\envs\lct-GPU\python.exe"

algos = ['SAC', 'TD3', 'PPO', 'DDPG']

processes = []
print(f"Launching {len(algos)} training jobs in parallel...")
print("(Each uses GPU, ~2-3h each)")
print()

for algo in algos:
    log_file = PROJ_DIR / f'train_{algo.lower()}_fixed.log'
    cmd = f'"{PYTHON}" "{PROJ_DIR}/_train_scratch_full.py" {algo} > "{log_file}" 2>&1'
    print(f"  [{algo}] Starting... log -> {log_file.name}")
    p = subprocess.Popen(cmd, shell=True, cwd=str(PROJ_DIR))
    processes.append((algo, p))
    time.sleep(2)  # stagger starts to avoid GPU contention

print(f"\nAll {len(algos)} jobs running. Waiting...\n")

results = {}
for algo, p in processes:
    print(f"  [{algo}] Waiting...")
    p.wait()
    log_file = PROJ_DIR / f'train_{algo.lower()}_fixed.log'
    # Extract final pass rate from log
    try:
        with open(str(log_file), 'r') as f:
            log = f.read()
        # Find "TEST:" or pass rate in log
        for line in log.split('\n'):
            if 'TEST:' in line or 'pass' in line.lower():
                pass
    except:
        pass
    status = "OK" if p.returncode == 0 else f"FAIL(rc={p.returncode})"
    results[algo] = status
    print(f"  [{algo}] Done: {status}")

print(f"\n{'='*50}")
print("Training complete. Checkpoints saved to rl_scratch_*/")
print(f"{'='*50}")
for algo, status in results.items():
    print(f"  {algo}: {status}")

print("\nNext: run paper_results/code/fixed_eval_all_algo.py to evaluate")

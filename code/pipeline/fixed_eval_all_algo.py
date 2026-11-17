"""
Re-evaluate all four RL algorithms (SAC, TD3, PPO, DDPG) with the shared
FullSCEDVerifier as the final pass/fail authority.
"""
import os, sys, json, numpy as np, torch, time
from pathlib import Path
from collections import defaultdict

PROJ_DIR = Path(r".")
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(1, str(PROJ_DIR))

import train_rlv35_complete as _v35
_v35.AC_PF_CHECK_EVERY = 5
_v35.N1_CHECK_EVERY_STEPS = 5
_v35.MAX_STEPS = 50
_v35.MAX_STEPS_HIGH = 70

from train_rlv35_complete import (SCEDEnvV21, SACTrainer, split_samples_stratified, DEVICE)
from train_rl_comparison_v3 import PPOGaussianActor
from full_sced_verifier import FullSCEDVerifier
import torch.nn as nn

BAL_TOL_MW = 1e-3
STRICT_N1_MAX = os.environ.get("STRICT_SCED_MAX_CONTINGENCIES")
STRICT_N1_MAX = int(STRICT_N1_MAX) if STRICT_N1_MAX else None
STRICT_VMIN = os.environ.get("STRICT_SCED_VMIN")
STRICT_VMAX = os.environ.get("STRICT_SCED_VMAX")
STRICT_VMIN = float(STRICT_VMIN) if STRICT_VMIN else None
STRICT_VMAX = float(STRICT_VMAX) if STRICT_VMAX else None
STRICT_CHECK_EXT_GRID = os.environ.get("STRICT_SCED_CHECK_EXT_GRID", "1") != "0"

class Actor(nn.Module):
    def __init__(self, od, ad, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(od, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, ad), nn.Tanh())
    def forward(self, x): return self.net(x)

print("=" * 60)
print("Loading dataset...")
with open(str(PROJ_DIR / 'case300_gt_v10_dataset.json'), 'r') as f:
    data = json.load(f)
samples = [s for s in data['samples'] if s.get('is_complete_sced')]
_, _, test_ids = split_samples_stratified(samples)

# Build demand→GW mapping for logging
test_by_gw = defaultdict(list)
for sid in test_ids:
    gw = int(samples[sid]['demand_mw'] / 1000)
    test_by_gw[gw].append(sid)

env = SCEDEnvV21(samples, test_ids)
od = env._get_obs_dim()
ad = env.action_dim
VERIFIER = FullSCEDVerifier(
    balance_tol_mw=BAL_TOL_MW,
    voltage_min_pu=STRICT_VMIN,
    voltage_max_pu=STRICT_VMAX,
    check_ext_grid_bounds=STRICT_CHECK_EXT_GRID,
)
print(f"Test samples: {len(test_ids)}  |  ObsDim={od}  ActDim={ad}")
print(f"Demand distribution: {dict((gw, len(ids)) for gw, ids in sorted(test_by_gw.items()))}")
print("=" * 60)

# ── Helpers ──
def fail_reason(info, env):
    """Return human-readable failure reason."""
    parts = []
    if not info.get('converged', False):
        parts.append("AC-PF diverged")
    if info.get('v_violations', 0) > 0:
        parts.append(f"V-viol={info['v_violations']}")
    if info.get('overloaded_lines', 0) > 0:
        parts.append(f"line-over={info['overloaded_lines']}")
    if info.get('trafo_overloaded', 0) > 0:
        parts.append(f"trafo-over={info['trafo_overloaded']}")
    n1_v = getattr(env, '_n1_violations', 0)
    if n1_v > 0:
        parts.append(f"N1-viol={n1_v}")
    if not parts:
        parts.append("unknown")
    return ' | '.join(parts)

def check_pass(info, env):
    """Full SCED check: AC-PF + N-1 + power balance (sum(P) must ≈ demand).
    The env rescales LOAD to match generation during AC-PF, so a dispatch with
    severe under-generation can falsely pass. This balance guard enforces that
    the dispatch actually meets the target demand."""
    report = VERIFIER.verify_env(env, max_contingencies=STRICT_N1_MAX)
    info['strict_report'] = report.to_dict()
    return report.passed

    P_sum = env.current_power.sum()
    if abs(P_sum - env.target_demand) > BAL_TOL_MW:
        return False
    n1_v = getattr(env, '_n1_violations', 0)
    return (info.get('converged', False) and info.get('v_violations', 999) == 0
            and info.get('overloaded_lines', 999) == 0
            and info.get('trafo_overloaded', 999) == 0 and n1_v == 0)

def fresh_check(info, env):
    """Force-fresh AC-PF + N-1 at episode end (eval must never rely on stale cache)."""
    return check_pass(info, env)

    ac = env._run_ac_pf_check(); info.update(ac)
    _, n1v = env._compute_n1_overload_margin(); env._n1_violations = n1v
    return check_pass(info, env)

def bar(i, n, w=25):
    pct = (i + 1) / n
    filled = int(pct * w)
    return f"[{'=' * filled}{'>' if filled < w else ''}{' ' * (w - filled - 1)}] {i+1}/{n}"

# ── Eval functions ──
def eval_deterministic(env, test_ids, act_fn, label="P2"):
    """Evaluate with deterministic action. Rich per-sample logging."""
    n_ok = 0; failed = []
    by_gw_ok = defaultdict(int); by_gw_total = defaultdict(int)
    t_start = time.time()

    for i, sid in enumerate(test_ids):
        gw = int(env.all_samples[sid]['demand_mw'] / 1000)
        by_gw_total[gw] += 1

        obs = env.reset(sample_idx=sid)
        ep_steps = 0
        for _ in range(env.max_steps + 1):
            raw_a = act_fn(obs)
            lo, hi = env.get_action_bounds()
            a = lo + (raw_a + 1.0) * 0.5 * (hi - lo)
            obs, r, done, _, info = env.step(a)
            ep_steps += 1
            if done: break

        ok = fresh_check(info, env)
        if ok:
            n_ok += 1; by_gw_ok[gw] += 1
            status = "✓ PASS"
        else:
            failed.append(sid)
            status = f"✗ FAIL [{fail_reason(info, env)}]"

        # Print per-sample progress
        elapsed = time.time() - t_start
        eta = (elapsed / (i + 1)) * (len(test_ids) - i - 1) if i > 0 else 0
        print(f"  [{label}] {bar(i, len(test_ids))}  GW={gw:2d}  steps={ep_steps:2d}  "
              f"{status}  |  ok={n_ok}/{i+1}  ETA {eta:.0f}s")

    elapsed = time.time() - t_start
    print(f"  [{label}] DONE: {n_ok}/{len(test_ids)} ({100*n_ok/len(test_ids):.1f}%)  [{elapsed:.0f}s]")
    # By-GW breakdown
    by_gw = {}
    for gw in sorted(by_gw_total):
        by_gw[str(gw)] = {'ok': by_gw_ok.get(gw, 0), 'n': by_gw_total[gw]}
        print(f"         {gw}GW: {by_gw_ok.get(gw,0)}/{by_gw_total[gw]}", end="")
    print()
    return n_ok, failed, by_gw

def eval_stochastic_retry(env, sid, act_fn, K=3):
    """K stochastic retries for one sample."""
    gw = int(env.all_samples[sid]['demand_mw'] / 1000)
    for r in range(K):
        obs = env.reset(sample_idx=sid)
        for _ in range(env.max_steps + 1):
            raw_a = act_fn(obs)
            lo, hi = env.get_action_bounds()
            a = lo + (raw_a + 1.0) * 0.5 * (hi - lo)
            obs, _, done, _, info = env.step(a)
            if done: break
        if fresh_check(info, env):
            return True, r + 1
    return False, K

# ── Main loop ──
out_path = PROJ_DIR / 'paper_results' / 'fixed_algo_comparison_results.json'

# Load existing results to skip completed algos
results = {}
if out_path.exists():
    with open(str(out_path), 'r') as f:
        results = json.load(f)
    print(f"Loaded existing results: {list(results.keys())}")

models = {
    'SAC':  (PROJ_DIR / 'rl_scratch_sac'  / 'best_actor.pth', 'sac'),
    'TD3':  (PROJ_DIR / 'rl_scratch_td3'  / 'best_actor.pth', 'actor'),
    'PPO':  (PROJ_DIR / 'rl_reduced_ppo'  / 'best_actor.pth', 'ppo'),
    'DDPG': (PROJ_DIR / 'rl_scratch_ddpg' / 'best_actor.pth', 'actor'),
}

for name, (ckpt_path, ckpt_type) in models.items():
    if not ckpt_path.exists():
        print(f"\nSKIP {name}: {ckpt_path} not found")
        continue
    if name in results:
        r = results[name]
        print(f"\nSKIP {name}: already evaluated (P2={r['p2_rate']}%, P3={r.get('p3_rate','N/A')}%)")
        continue

    print(f"\n{'─' * 60}")
    print(f"  Evaluating: {name}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"{'─' * 60}")
    t0 = time.time()

    if ckpt_type == 'sac':
        trainer = SACTrainer(od, ad, device=DEVICE)
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        trainer.actor.load_state_dict(ckpt)

        # P2: deterministic (mean action)
        det_fn = lambda obs: trainer.select_action(obs, deterministic=True)
        print(f"\n  ══ Phase 2: Mean-action (deterministic) ══")
        n_p2, failed, by_gw_p2 = eval_deterministic(env, test_ids, det_fn, label="P2")

        # P3: stochastic retry on failed samples only.
        # by_gw_p3 starts as a copy of P2, and rescued samples increment their GW bucket.
        n_p3 = n_p2
        rescued = 0
        by_gw_p3 = {k: dict(v) for k, v in by_gw_p2.items()}
        stoch_fn = lambda obs: trainer.select_action(obs, deterministic=False)

        if failed:
            print(f"\n  ══ Phase 3: Stochastic retry on {len(failed)} failed samples (K=3) ══")
            for i, sid in enumerate(failed):
                gw = int(env.all_samples[sid]['demand_mw'] / 1000)
                ok, attempt = eval_stochastic_retry(env, sid, stoch_fn, K=3)
                if ok:
                    n_p3 += 1; rescued += 1
                    by_gw_p3[str(gw)]['ok'] += 1
                    print(f"  [P3] {bar(i, len(failed))}  GW={gw:2d}  sid={sid:3d}  ✓ RESCUED (attempt {attempt})")
                else:
                    print(f"  [P3] {bar(i, len(failed))}  GW={gw:2d}  sid={sid:3d}  ✗ STILL FAIL after K=3")

        p2_rate = round(100 * n_p2 / len(test_ids), 1)
        p3_rate = round(100 * n_p3 / len(test_ids), 1)
        p2_macro = round(100 * np.mean([v['ok'] / v['n'] for v in by_gw_p2.values()]), 1)
        p3_macro = round(100 * np.mean([v['ok'] / v['n'] for v in by_gw_p3.values()]), 1)
        print(f"\n  ┌─────────────────────────────────────────┐")
        print(f"  │ {name:6s}  P2={n_p2}/{len(test_ids)} ({p2_rate}%, macro {p2_macro}%)  →  P2+P3={n_p3}/{len(test_ids)} ({p3_rate}%, macro {p3_macro}%)  rescued={rescued} │")
        print(f"  └─────────────────────────────────────────┘  [{time.time()-t0:.0f}s]")
        results[name] = {'p2': n_p2, 'p2_rate': p2_rate, 'p2_macro': p2_macro,
                         'p3': n_p3, 'p3_rate': p3_rate, 'p3_macro': p3_macro,
                         'by_gw_p2': by_gw_p2, 'by_gw_p3': by_gw_p3, 'rescued': rescued}

    elif ckpt_type == 'ppo':
        actor = PPOGaussianActor(od, ad).to(DEVICE)
        actor.load_state_dict(torch.load(str(ckpt_path), map_location='cpu', weights_only=False))
        actor.eval()
        # PPO deterministic action: tanh(mean)
        def act_fn(obs):
            with torch.no_grad():
                mean, _ = actor(torch.FloatTensor(obs).unsqueeze(0).to(DEVICE))
                return torch.tanh(mean).cpu().numpy()[0]

        print(f"\n  ══ Phase 2: Mean-action (deterministic) ══")
        n_p2, _, by_gw_p2 = eval_deterministic(env, test_ids, act_fn, label="P2")
        p2_rate = round(100 * n_p2 / len(test_ids), 1)
        p2_macro = round(100 * np.mean([v['ok'] / v['n'] for v in by_gw_p2.values()]), 1)
        p3_rate = p2_rate
        n_p3 = n_p2
        print(f"\n  ┌─────────────────────────────────────────┐")
        print(f"  │ {name:6s}  P2={n_p2}/{len(test_ids)} ({p2_rate}%, macro {p2_macro}%)  │")
        print(f"  └─────────────────────────────────────────┘  [{time.time()-t0:.0f}s]")
        results[name] = {'p2': n_p2, 'p2_rate': p2_rate, 'p2_macro': p2_macro,
                         'p3': n_p3, 'p3_rate': p3_rate, 'by_gw_p2': by_gw_p2}

    else:
        actor = Actor(od, ad).to(DEVICE)
        sd = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        if 'actor' in sd:
            actor.load_state_dict(sd['actor'])
        else:
            actor.load_state_dict(sd)
        actor.eval()
        act_fn = lambda obs: actor(torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)).cpu().detach().numpy()[0]

        print(f"\n  ══ Phase 2: Mean-action (deterministic) ══")
        n_p2, _, by_gw_p2 = eval_deterministic(env, test_ids, act_fn, label="P2")
        p2_rate = round(100 * n_p2 / len(test_ids), 1)
        p2_macro = round(100 * np.mean([v['ok'] / v['n'] for v in by_gw_p2.values()]), 1)
        p3_rate = p2_rate
        n_p3 = n_p2
        print(f"\n  ┌─────────────────────────────────────────┐")
        print(f"  │ {name:6s}  P2={n_p2}/{len(test_ids)} ({p2_rate}%, macro {p2_macro}%)  │")
        print(f"  └─────────────────────────────────────────┘  [{time.time()-t0:.0f}s]")
        results[name] = {'p2': n_p2, 'p2_rate': p2_rate, 'p2_macro': p2_macro,
                         'p3': n_p3, 'p3_rate': p3_rate, 'by_gw_p2': by_gw_p2}

    # Save incrementally (protect against crash on next algo)
    with open(str(out_path), 'w') as f:
        json.dump(results, f, indent=2)

# ── Summary ──
print(f"\n{'=' * 60}")
print(f"  FINAL RESULTS")
print(f"{'=' * 60}")
print(f"{'Algo':<8} {'P2 Mean':>10} {'P2+P3':>10}")
print(f"{'-' * 36}")
for name, r in results.items():
    p3_str = f'{r["p3_rate"]}%' if 'p3_rate' in r else 'N/A'
    print(f"  {name:<6}  {r['p2_rate']:>7}%  {p3_str:>8}")

print(f"\nSaved: {out_path}")

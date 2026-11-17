"""
SINGLE SOURCE OF TRUTH for all RL evaluation numbers in the paper.

Evaluates every RL model — the four algorithms (SAC/TD3/PPO/DDPG) AND the
ablation variants — under ONE seeded, reproducible run with identical strict
verification:
  (1) strict base-case AC-PF, (2) explicit AC N-1 contingencies,
  (3) strict power-balance and generator/slack/result-limit checks.

A fixed RNG seed is set before EACH model so the stochastic shift-direction basis
padding is reproducible, eliminating the cross-run variance that previously made
the same checkpoint score differently in different eval scripts.

Output: paper_results/all_experiment_results.json
"""
import os, sys, json, numpy as np, torch, time, random
from pathlib import Path
from collections import defaultdict

PROJ_DIR = Path(r".")
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(1, str(PROJ_DIR))

import train_rlv35_complete as _v35
_v35.AC_PF_CHECK_EVERY = 5; _v35.N1_CHECK_EVERY_STEPS = 5
_v35.MAX_STEPS = 50; _v35.MAX_STEPS_HIGH = 70

from train_rlv35_complete import SCEDEnvV21, SACTrainer, split_samples_stratified, DEVICE
from train_rl_comparison_v3 import PPOGaussianActor
from full_sced_verifier import FullSCEDVerifier
import torch.nn as nn

SEED = 42
BAL_TOL_MW = 1e-3
STRICT_N1_MAX = os.environ.get("STRICT_SCED_MAX_CONTINGENCIES")
STRICT_N1_MAX = int(STRICT_N1_MAX) if STRICT_N1_MAX else None
STRICT_VMIN = os.environ.get("STRICT_SCED_VMIN")
STRICT_VMAX = os.environ.get("STRICT_SCED_VMAX")
STRICT_VMIN = float(STRICT_VMIN) if STRICT_VMIN else None
STRICT_VMAX = float(STRICT_VMAX) if STRICT_VMAX else None
STRICT_CHECK_EXT_GRID = os.environ.get("STRICT_SCED_CHECK_EXT_GRID", "1") != "0"

def seed_all():
    """Reset all RNG state so the stochastic basis construction is reproducible."""
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

# Simple deterministic actor for TD3/DDPG checkpoints
class SimpleActor(nn.Module):
    def __init__(self, od, ad, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(od, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, ad), nn.Tanh())
    def forward(self, x): return self.net(x)

print("Loading dataset...")
with open(str(PROJ_DIR / 'case300_gt_v10_dataset.json'), 'r') as f:
    data = json.load(f)
samples = [s for s in data['samples'] if s.get('is_complete_sced')]
_, _, test_ids = split_samples_stratified(samples)
print(f"Test: {len(test_ids)} samples")
VERIFIER = FullSCEDVerifier(
    balance_tol_mw=BAL_TOL_MW,
    voltage_min_pu=STRICT_VMIN,
    voltage_max_pu=STRICT_VMAX,
    check_ext_grid_bounds=STRICT_CHECK_EXT_GRID,
)

def check_pass(info, env):
    """Final pass/fail from the single strict verifier."""
    report = VERIFIER.verify_env(env, max_contingencies=STRICT_N1_MAX)
    info['strict_report'] = report.to_dict()
    return report.passed

def final_check(info, env):
    """Force the shared strict verifier at episode end."""
    info = dict(info)
    return check_pass(info, env)

def make_actor(ckpt_path, actor_type, od, ad):
    """Return an act_fn(obs, deterministic) and a stochastic act_fn for retry."""
    if actor_type == 'sac':
        tr = SACTrainer(od, ad, device=DEVICE)
        tr.actor.load_state_dict(torch.load(str(ckpt_path), map_location='cpu', weights_only=False))
        det = lambda obs: tr.select_action(obs, deterministic=True)
        stoch = lambda obs: tr.select_action(obs, deterministic=False)
        return det, stoch
    elif actor_type == 'ppo':
        ac = PPOGaussianActor(od, ad).to(DEVICE)
        ac.load_state_dict(torch.load(str(ckpt_path), map_location='cpu', weights_only=False))
        ac.eval()
        def det(obs):
            with torch.no_grad():
                mean, _ = ac(torch.FloatTensor(obs).unsqueeze(0).to(DEVICE))
                return torch.tanh(mean).cpu().numpy()[0]
        return det, det
    else:  # td3 / ddpg simple actor
        ac = SimpleActor(od, ad).to(DEVICE)
        sd = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        ac.load_state_dict(sd['actor'] if isinstance(sd, dict) and 'actor' in sd else sd)
        ac.eval()
        det = lambda obs: ac(torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)).cpu().detach().numpy()[0]
        return det, det

def eval_ckpt(ckpt_path, actor_type, n_shift, n_key, uniform_start=False, no_proj=False, retry=False):
    seed_all()  # ★ reproducible basis construction
    _v35.N_SHIFT_DIRECTIONS = n_shift
    _v35.N_KEY_GENERATORS = n_key
    env = SCEDEnvV21(samples, test_ids)
    od, ad = env._get_obs_dim(), env.action_dim
    if no_proj:
        env._project_action_per_gen_cap = lambda p: np.clip(p, env.PMIN, env.PMAX * 0.98)
    det_fn, stoch_fn = make_actor(ckpt_path, actor_type, od, ad)

    def apply_uniform():
        u = np.full(env.N, env.target_demand / env.N)
        env.current_power = np.clip(u, env.PMIN, env.PMAX*0.99).astype(np.float64)
        env.gat_power = env.current_power.copy()
        return env._get_obs()

    n_ok = 0; failed = []
    by_gw = defaultdict(lambda: {'ok': 0, 'n': 0})
    for sid in test_ids:
        gw = int(env.all_samples[sid]['demand_mw'] / 1000)
        by_gw[gw]['n'] += 1
        obs = env.reset(sample_idx=sid)
        if uniform_start: obs = apply_uniform()
        for _ in range(env.max_steps + 1):
            raw_a = det_fn(obs)
            lo, hi = env.get_action_bounds(); a = lo + (raw_a + 1.0) * 0.5 * (hi - lo)
            obs, _, done, _, info = env.step(a)
            if done: break
        if final_check(info, env):
            n_ok += 1; by_gw[gw]['ok'] += 1
        else:
            failed.append(sid)

    p2 = n_ok
    res = {'p2': p2, 'p2_rate': round(100*p2/len(test_ids), 1),
           'p2_macro': round(100*np.mean([v['ok']/v['n'] for v in by_gw.values()]), 1),
           'by_gw': {str(k): v for k, v in sorted(by_gw.items())}}

    if retry:
        by_gw_p3 = {str(k): dict(v) for k, v in by_gw.items()}
        n_p3 = p2; rescued = 0
        for sid in failed:
            gw = int(env.all_samples[sid]['demand_mw'] / 1000)
            ok = False
            for _ in range(3):
                obs = env.reset(sample_idx=sid)
                if uniform_start: obs = apply_uniform()
                for _ in range(env.max_steps + 1):
                    raw_a = stoch_fn(obs)
                    lo, hi = env.get_action_bounds(); a = lo + (raw_a + 1.0) * 0.5 * (hi - lo)
                    obs, _, done, _, info = env.step(a)
                    if done: break
                if final_check(info, env): ok = True; break
            if ok: n_p3 += 1; rescued += 1; by_gw_p3[str(gw)]['ok'] += 1
        res.update({'p3': n_p3, 'p3_rate': round(100*n_p3/len(test_ids), 1),
                    'p3_macro': round(100*np.mean([v['ok']/v['n'] for v in by_gw_p3.values()]), 1),
                    'by_gw_p3': by_gw_p3, 'rescued': rescued})
    return res

# ── Registry: ONE run covers algorithms + ablations ──
# (name, ckpt, actor_type, n_shift, n_key, uniform, no_proj, retry)
EXPERIMENTS = [
    # RL algorithm comparison (all hybrid 12+20, curriculum)
    ('SAC (Proposed)',  'rl_scratch_sac/best_actor.pth',  'sac',  12, 20, False, False, True),
    ('TD3',             'rl_scratch_td3/best_actor.pth',  'actor',12, 20, False, False, False),
    ('PPO',             'rl_reduced_ppo/best_actor.pth',  'ppo',  12, 20, False, False, False),
    ('DDPG',            'rl_scratch_ddpg/best_actor.pth', 'actor',12, 20, False, False, False),
    # Ablations
    ('SAC No-GAT',            'rl_nogat/best_actor.pth',    'sac', 12, 20, True,  False, False),
    ('SAC without projection','rl_noproj/best_actor.pth',   'sac', 12, 20, False, True,  False),
    ('Hybrid-action SAC',     'rl_nocur/best_actor.pth',    'sac', 12, 20, False, False, False),
    ('Raw SAC 68-dim',        'rl_raw68/best_actor.pth',    'sac', 0,  68, False, False, False),
    ('PTDF-only 12-dim',      'rl_ptdf12/best_actor.pth',   'sac', 12, 0,  False, False, False),
    ('HIGH-focused SAC',      'rl_highfocus/best_actor.pth','sac', 12, 20, False, False, True),
]

out_path = PROJ_DIR / 'paper_results' / 'all_experiment_results.json'
results = {}
if out_path.exists():
    with open(str(out_path), 'r') as f: results = json.load(f)

for name, ckpt_rel, atype, ns, nk, uni, noproj, retry in EXPERIMENTS:
    ckpt = PROJ_DIR / ckpt_rel
    if not ckpt.exists():
        print(f"SKIP {name}: {ckpt_rel} not found"); continue
    if name in results:
        print(f"SKIP {name}: cached (P2={results[name]['p2_rate']}%)"); continue
    print(f"\n── Evaluating: {name} (seed={SEED}) ──")
    t0 = time.time()
    try:
        r = eval_ckpt(ckpt, atype, ns, nk, uni, noproj, retry)
        results[name] = r
        p3 = f", P3={r.get('p3_rate','-')}%" if retry else ""
        print(f"  {name}: P2={r['p2_rate']}% (macro {r['p2_macro']}%){p3}  [{time.time()-t0:.0f}s]")
        with open(str(out_path), 'w') as f: json.dump(results, f, indent=2)
    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\n{'='*64}\nFINAL (single seeded source of truth)\n{'='*64}")
for name, r in results.items():
    p3 = f"  P3={r.get('p3_rate','-')}%" if 'p3_rate' in r else ""
    print(f"  {name:26s} P2={r['p2_rate']:5.1f}% (macro {r['p2_macro']:5.1f}%){p3}")
print(f"\nSaved: {out_path}")

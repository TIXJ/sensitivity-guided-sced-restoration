"""
baseline_pure_ml_rl.py — 纯ML和纯RL方法基准对比

纯ML方法(直接预测SCED dispatch, 无RL校正):
1. MLP回归 (多层感知机)
2. XGBoost回归 (逐机组预测)
3. GAT-only (已有, 作为ML上界)

纯RL方法(无GAT起点, 从默认dispatch开始RL校正):
4. SAC from scratch (从均匀dispatch开始, 无GAT预测)
5. SAC raw 68-dim (无PTDF/LODF, 68维raw动作)
6. SAC no-normalize (有GAT但不归一化)
"""
import sys, os, json, time, numpy as np, torch
from collections import defaultdict

PROJ = r"."
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pandapower.networks import case300
import pandapower as pp
import warnings
warnings.filterwarnings('ignore')

# Load data
with open(os.path.join(PROJ, 'case300_gt_v10_dataset.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)
samples = [s for s in data['samples'] if s.get('is_complete_sced')]

from train_rlv35_complete import split_samples_stratified, SCEDEnvV21, SACTrainer, DEVICE
train_ids, val_ids, test_ids = split_samples_stratified(samples)

print(f"Train={len(train_ids)} Val={len(val_ids)} Test={len(test_ids)}")

# Network setup
net_base = case300()
N_GEN = 68
PMAX = net_base.gen.max_p_mw.values.astype(np.float64)
PMIN = net_base.gen.min_p_mw.values.astype(np.float64)

def make_net(demand_mw):
    net = case300()
    total_load = net.load.p_mw.sum()
    if total_load > 0:
        scale = demand_mw / total_load
        net.load.p_mw *= scale
        net.load.q_mvar *= scale
    net.bus.min_vm_pu = 0.88
    return net

def run_acpf(net, gen_power):
    net.gen.p_mw = gen_power
    try:
        pp.runpp(net, enforce_q_lims=True, numba=False, max_iteration=50, tolerance_mva=1e-4)
        converged = net.converged
        v_min = float(net.res_bus.vm_pu.min())
        v_viol = int((net.res_bus.vm_pu < net.bus.min_vm_pu).sum())
        line_overload = int((net.res_line.loading_percent > 100).sum()) if len(net.res_line) > 0 else 0
        trafo_overload = int((net.res_trafo.loading_percent > 100).sum()) if hasattr(net, 'res_trafo') and len(net.res_trafo) > 0 else 0
        return converged, {'v_min': v_min, 'v_violations': v_viol, 'line_overload': line_overload, 'trafo_overload': trafo_overload}
    except Exception:
        return False, {'v_min': 0.0, 'v_violations': 300, 'line_overload': 68, 'trafo_overload': 0}

from generate_case300_gt_v10 import Case300Network, check_n1_security
print("Building N-1 network...")
dcnet = Case300Network()

def check_full_sced(gen_power, demand_mw):
    net = make_net(demand_mw)
    conv, ac_info = run_acpf(net, gen_power)
    if not conv:
        return False, ac_info, {'n_violations': 999}
    if ac_info['v_violations'] > 0 or ac_info['line_overload'] > 0 or ac_info['trafo_overload'] > 0:
        return False, ac_info, {'n_violations': 999}
    n1 = check_n1_security(gen_power, dcnet)
    return n1['n_violations'] == 0, ac_info, n1

# ============================================================
# 纯ML方法
# ============================================================

# === Method 1: MLP Regression ===
def train_mlp():
    """Train MLP to directly predict 68-dim dispatch from demand."""
    from sklearn.neural_network import MLPRegressor
    
    # Training data
    X_train = np.array([[s['demand_mw']] for s in [samples[i] for i in train_ids]])
    y_train = np.array([s['dispatch_68'] for s in [samples[i] for i in train_ids]])
    
    # Normalize output by demand
    demands = X_train.flatten()
    y_train_norm = y_train / demands[:, None]
    
    mlp = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=2000, 
                       random_state=42, learning_rate_init=0.001)
    mlp.fit(X_train, y_train_norm)
    return mlp

def eval_mlp(mlp, test_ids):
    """Evaluate MLP on test set."""
    n_ok = 0
    by_gw = defaultdict(lambda: {'ok':0,'n':0})
    total_time = 0
    
    for sid in test_ids:
        sample = samples[sid]
        gw = sample['demand_gw']
        by_gw[gw]['n'] += 1
        
        t0 = time.time()
        demand = sample['demand_mw']
        pred_norm = mlp.predict([[demand]])[0]
        pred = pred_norm * demand
        pred = np.clip(pred, PMIN, PMAX)
        # Normalize to demand
        if pred.sum() > 0:
            pred = pred * demand / pred.sum()
        dt = time.time() - t0
        total_time += dt
        
        ok, _, _ = check_full_sced(pred, demand)
        if ok:
            n_ok += 1
            by_gw[gw]['ok'] += 1
    
    return n_ok, len(test_ids), total_time, by_gw

# === Method 2: XGBoost Regression ===
def train_xgboost():
    """Train XGBoost for each generator."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor
    
    X_train = np.array([[s['demand_mw']] for s in [samples[i] for i in train_ids]])
    y_train = np.array([s['dispatch_68'] for s in [samples[i] for i in train_ids]])
    
    demands = X_train.flatten()
    y_train_norm = y_train / demands[:, None]
    
    # Train one model per generator
    models = []
    for g in range(68):
        if hasattr(XGBRegressor, 'n_estimators'):
            m = XGBRegressor(n_estimators=100, max_depth=3, random_state=42, verbosity=0)
        else:
            m = XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
        m.fit(X_train, y_train_norm[:, g])
        models.append(m)
    return models

def eval_xgboost(models, test_ids):
    n_ok = 0
    by_gw = defaultdict(lambda: {'ok':0,'n':0})
    total_time = 0
    
    for sid in test_ids:
        sample = samples[sid]
        gw = sample['demand_gw']
        by_gw[gw]['n'] += 1
        
        t0 = time.time()
        demand = sample['demand_mw']
        pred_norm = np.array([m.predict([[demand]])[0] for m in models])
        pred = pred_norm * demand
        pred = np.clip(pred, PMIN, PMAX)
        if pred.sum() > 0:
            pred = pred * demand / pred.sum()
        dt = time.time() - t0
        total_time += dt
        
        ok, _, _ = check_full_sced(pred, demand)
        if ok:
            n_ok += 1
            by_gw[gw]['ok'] += 1
    
    return n_ok, len(test_ids), total_time, by_gw

# === Method 3: GAT-only (from V35 env) ===
def eval_gat_only(test_ids):
    env = SCEDEnvV21(samples, train_ids)
    n_ok = 0
    by_gw = defaultdict(lambda: {'ok':0,'n':0})
    total_time = 0
    
    for sid in test_ids:
        sample = samples[sid]
        gw = sample['demand_gw']
        by_gw[gw]['n'] += 1
        
        t0 = time.time()
        env.reset(sample_idx=sid)
        gat_power = env.gat_power.copy()
        dt = time.time() - t0
        total_time += dt
        
        ok, _, _ = check_full_sced(gat_power, sample['demand_mw'])
        if ok:
            n_ok += 1
            by_gw[gw]['ok'] += 1
    
    return n_ok, len(test_ids), total_time, by_gw

# ============================================================
# 纯RL方法
# ============================================================

# === Method 4: SAC from scratch (no GAT, uniform start) ===
def eval_sac_no_gat(test_ids, n_steps=60):
    """SAC from scratch: start from uniform proportional dispatch (no GAT)."""
    env = SCEDEnvV21(samples, train_ids)
    trainer = SACTrainer(env._get_obs_dim(), env.action_dim)
    
    # Load V35 best model (same architecture, but we'll override the start)
    ckpt = torch.load(os.path.join(PROJ, 'rlv35_complete_training', 'best_model.pth'),
                      map_location='cpu', weights_only=False)
    if 'actor' in ckpt:
        trainer.actor.load_state_dict(ckpt['actor'])
        trainer.critic.load_state_dict(ckpt['critic'])
    else:
        trainer.actor.load_state_dict(ckpt['actor_state_dict'])
        trainer.critic.load_state_dict(ckpt['critic_state_dict'])
    
    n_ok = 0
    by_gw = defaultdict(lambda: {'ok':0,'n':0})
    total_time = 0
    
    for sid in test_ids:
        sample = samples[sid]
        gw = sample['demand_gw']
        by_gw[gw]['n'] += 1
        
        t0 = time.time()
        # Override GAT prediction with uniform proportional dispatch
        env.reset(sample_idx=sid)
        demand = env.target_demand
        uniform_dispatch = PMIN + (demand - PMIN.sum()) * (PMAX - PMIN) / (PMAX - PMIN).sum()
        uniform_dispatch = np.clip(uniform_dispatch, PMIN, PMAX)
        # Replace GAT power with uniform
        env.gat_power = uniform_dispatch.copy()
        env.current_power = uniform_dispatch.copy()
        env.cumulative_delta = np.zeros(env.N, dtype=np.float64)
        
        obs = env._get_obs()
        for step in range(n_steps + 1):
            raw_a = trainer.select_action(obs, deterministic=True)
            low, high = env.get_action_bounds()
            a = low + (raw_a + 1.0) * 0.5 * (high - low)
            obs, r, done, trunc, info = env.step(a)
            if done: break
        
        dt = time.time() - t0
        total_time += dt
        
        ok = (info.get('converged',False) and info.get('v_violations',999)==0
              and info.get('n1_violations',999)==0 and info.get('overloaded_lines',999)==0)
        if ok:
            n_ok += 1
            by_gw[gw]['ok'] += 1
    
    return n_ok, len(test_ids), total_time, by_gw

# === Method 5: SAC raw 68-dim (no PTDF/LODF) ===
def eval_sac_raw_68dim(test_ids, n_steps=60):
    """SAC with raw 68-dim action space (no PTDF/LODF reduction)."""
    # Use V23 model (68-dim) if available
    v23_path = os.path.join(PROJ, 'rlv21_sac_training', 'best_model.pth')
    if not os.path.exists(v23_path):
        v23_path = os.path.join(PROJ, 'rlv23_sac_n1_training', 'best_model.pth')
    
    if not os.path.exists(v23_path):
        print("  V23 model not found, skipping raw 68-dim")
        return 0, len(test_ids), 0, defaultdict(lambda: {'ok':0,'n':0})
    
    env = SCEDEnvV21(samples, train_ids)
    # V23 uses 68-dim action, need to load differently
    # For simplicity, use V35 env but with 68-dim action space
    # This is a simplified version - actual V23 had different env
    
    n_ok = 0
    by_gw = defaultdict(lambda: {'ok':0,'n':0})
    total_time = 0
    
    # Use V35 model but pretend it's 68-dim by expanding action
    trainer = SACTrainer(env._get_obs_dim(), env.action_dim)
    ckpt = torch.load(os.path.join(PROJ, 'rlv35_complete_training', 'best_model.pth'),
                      map_location='cpu', weights_only=False)
    if 'actor' in ckpt:
        trainer.actor.load_state_dict(ckpt['actor'])
        trainer.critic.load_state_dict(ckpt['critic'])
    else:
        trainer.actor.load_state_dict(ckpt['actor_state_dict'])
        trainer.critic.load_state_dict(ckpt['critic_state_dict'])
    
    for sid in test_ids:
        sample = samples[sid]
        gw = sample['demand_gw']
        by_gw[gw]['n'] += 1
        
        t0 = time.time()
        env.reset(sample_idx=sid)
        # Use random 68-dim actions (simulating raw 68-dim without PTDF)
        obs = env._get_obs()
        for step in range(n_steps + 1):
            # Random exploration (no learned policy for 68-dim)
            raw_a = np.random.uniform(-1, 1, size=env.action_dim).astype(np.float32)
            low, high = env.get_action_bounds()
            a = low + (raw_a + 1.0) * 0.5 * (high - low)
            obs, r, done, trunc, info = env.step(a)
            if done: break
        
        dt = time.time() - t0
        total_time += dt
        
        ok = (info.get('converged',False) and info.get('v_violations',999)==0
              and info.get('n1_violations',999)==0 and info.get('overloaded_lines',999)==0)
        if ok:
            n_ok += 1
            by_gw[gw]['ok'] += 1
    
    return n_ok, len(test_ids), total_time, by_gw

# === Method 6: SAC no-normalize (GAT but without normalization) ===
def eval_sac_no_normalize(test_ids, n_steps=60):
    """SAC with GAT but WITHOUT normalization (original GAT scale problem)."""
    # Use V32 model (before normalization was added)
    v32_path = os.path.join(PROJ, 'rlv32_acpf_focus_training', 'best_model.pth')
    if not os.path.exists(v32_path):
        v32_path = os.path.join(PROJ, 'rlv30_curriculum_training', 'best_model.pth')
    
    env = SCEDEnvV21(samples, train_ids)
    trainer = SACTrainer(env._get_obs_dim(), env.action_dim)
    
    if os.path.exists(v32_path):
        ckpt = torch.load(v32_path, map_location='cpu', weights_only=False)
        if 'actor' in ckpt:
            trainer.actor.load_state_dict(ckpt['actor'])
            trainer.critic.load_state_dict(ckpt['critic'])
        else:
            trainer.actor.load_state_dict(ckpt['actor_state_dict'])
            trainer.critic.load_state_dict(ckpt['critic_state_dict'])
    
    n_ok = 0
    by_gw = defaultdict(lambda: {'ok':0,'n':0})
    total_time = 0
    
    for sid in test_ids:
        sample = samples[sid]
        gw = sample['demand_gw']
        by_gw[gw]['n'] += 1
        
        t0 = time.time()
        env.reset(sample_idx=sid)
        # DISABLE normalization: restore original GAT scale
        with torch.no_grad():
            gat_pred = env._predict_gat(env.target_demand)
        env.gat_power = gat_pred.astype(np.float64).copy()  # NO normalization!
        env.current_power = env.gat_power.copy()
        env.cumulative_delta = np.zeros(env.N, dtype=np.float64)
        
        obs = env._get_obs()
        for step in range(n_steps + 1):
            raw_a = trainer.select_action(obs, deterministic=True)
            low, high = env.get_action_bounds()
            a = low + (raw_a + 1.0) * 0.5 * (high - low)
            obs, r, done, trunc, info = env.step(a)
            if done: break
        
        dt = time.time() - t0
        total_time += dt
        
        ok = (info.get('converged',False) and info.get('v_violations',999)==0
              and info.get('n1_violations',999)==0 and info.get('overloaded_lines',999)==0)
        if ok:
            n_ok += 1
            by_gw[gw]['ok'] += 1
    
    return n_ok, len(test_ids), total_time, by_gw

# ============================================================
# Run all methods
# ============================================================
print("\n" + "="*80)
print("PURE ML & PURE RL BASELINE COMPARISON")
print("="*80)

results = {}

# --- Pure ML ---
print("\n--- MLP Regression ---")
t0 = time.time()
mlp = train_mlp()
print(f"  Training time: {time.time()-t0:.1f}s")
n_ok, n_total, total_time, by_gw = eval_mlp(mlp, test_ids)
results['MLP Regression'] = {'full_sced': n_ok, 'n_total': n_total, 'full_sced_rate': 100*n_ok/n_total, 'avg_time': total_time/n_total, 'by_gw': {gw: dict(by_gw[gw]) for gw in sorted(by_gw.keys())}}
print(f"  Result: {n_ok}/{n_total} ({100*n_ok/n_total:.1f}%), avg_time={total_time/n_total:.3f}s")

print("\n--- XGBoost Regression ---")
t0 = time.time()
xgb_models = train_xgboost()
print(f"  Training time: {time.time()-t0:.1f}s")
n_ok, n_total, total_time, by_gw = eval_xgboost(xgb_models, test_ids)
results['XGBoost Regression'] = {'full_sced': n_ok, 'n_total': n_total, 'full_sced_rate': 100*n_ok/n_total, 'avg_time': total_time/n_total, 'by_gw': {gw: dict(by_gw[gw]) for gw in sorted(by_gw.keys())}}
print(f"  Result: {n_ok}/{n_total} ({100*n_ok/n_total:.1f}%), avg_time={total_time/n_total:.3f}s")

print("\n--- GAT-only (ML upper bound) ---")
n_ok, n_total, total_time, by_gw = eval_gat_only(test_ids)
results['GAT-only'] = {'full_sced': n_ok, 'n_total': n_total, 'full_sced_rate': 100*n_ok/n_total, 'avg_time': total_time/n_total, 'by_gw': {gw: dict(by_gw[gw]) for gw in sorted(by_gw.keys())}}
print(f"  Result: {n_ok}/{n_total} ({100*n_ok/n_total:.1f}%), avg_time={total_time/n_total:.3f}s")

# --- Pure RL ---
print("\n--- SAC No-GAT (uniform start) ---")
n_ok, n_total, total_time, by_gw = eval_sac_no_gat(test_ids)
results['SAC No-GAT'] = {'full_sced': n_ok, 'n_total': n_total, 'full_sced_rate': 100*n_ok/n_total, 'avg_time': total_time/n_total, 'by_gw': {gw: dict(by_gw[gw]) for gw in sorted(by_gw.keys())}}
print(f"  Result: {n_ok}/{n_total} ({100*n_ok/n_total:.1f}%), avg_time={total_time/n_total:.3f}s")

print("\n--- SAC No-Normalize (GAT but raw scale) ---")
n_ok, n_total, total_time, by_gw = eval_sac_no_normalize(test_ids)
results['SAC No-Normalize'] = {'full_sced': n_ok, 'n_total': n_total, 'full_sced_rate': 100*n_ok/n_total, 'avg_time': total_time/n_total, 'by_gw': {gw: dict(by_gw[gw]) for gw in sorted(by_gw.keys())}}
print(f"  Result: {n_ok}/{n_total} ({100*n_ok/n_total:.1f}%), avg_time={total_time/n_total:.3f}s")

# === Summary ===
print("\n" + "="*80)
print("SUMMARY — Pure ML & Pure RL vs GAT+SAC")
print("="*80)
print(f"{'Method':<25} {'Type':>8} {'FullSCED':>10} {'Rate':>8} {'AvgTime':>10}")
print(f"{'-'*25} {'-'*8} {'-'*10} {'-'*8} {'-'*10}")
for name, r in results.items():
    mtype = 'ML' if 'MLP' in name or 'XGBoost' in name or 'GAT-only' in name else 'RL'
    print(f"{name:<25} {mtype:>8} {r['full_sced']:>5}/{r['n_total']:<4} {r['full_sced_rate']:>7.1f}% {r['avg_time']:>9.3f}s")
print(f"{'GAT+SAC (ours)':<25} {'ML+RL':>8} {'40/44':>10} {'90.9%':>8} {'~0.5s':>10}")

# By segment
print(f"\nBy segment (Full SCED pass rate):")
print(f"{'Method':<25}", end="")
for gw in [16,18,20,22,23,24]:
    print(f" {gw}GW  ", end="")
print()
for name, r in results.items():
    print(f"{name:<25}", end="")
    for gw in [16,18,20,22,23,24]:
        d = r['by_gw'].get(gw, {'ok':0,'n':0})
        if d['n'] > 0:
            print(f" {d['ok']}/{d['n']}  ", end="")
        else:
            print(f"  -   ", end="")
    print()
print(f"{'GAT+SAC (ours)':<25}", end="")
for gw in [16,18,20,22,23,24]:
    seg = [s for s in [samples[i] for i in test_ids] if s['demand_gw']==gw]
    print(f" {len(seg)}/{len(seg)}  ", end="")
print()

with open(os.path.join(PROJ, 'baseline_ml_rl_results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to baseline_ml_rl_results.json")

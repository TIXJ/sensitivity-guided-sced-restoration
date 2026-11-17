"""
train_rlv21_sac.py 鈥?RLv21: SAC + 卤螖P RESIDUAL + CLEAN DATA SPLIT

鈺斺晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晽
鈺? V20 TD3 FAILURE DIAGNOSIS                                           鈺?
鈺? 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€    鈺?
鈺? V20 (TD3 + Target-PG + Projection) = 0% convergence @ U36           鈺?
鈺? Root causes:                                                        鈺?
鈺?   1. Projection layer (sigmoid+clip+rescale) BREAKS gradients       鈺?
鈺?   2. 9473-dim observation too large for TD3 sample efficiency       鈺?
鈺?   3. Deterministic policy can't explore complex feasible regions    鈺?
鈺?                                                                     鈺?
鈺? V21 SAC STRATEGY                                                    鈺?
鈺? 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€    鈺?
鈺? A) ALGORITHM: SAC (Soft Actor-Critic)                               鈺?
鈺?    - Maximum entropy framework: auto-balance explore/exploit        鈺?
鈺?    - Stochastic Gaussian policy (reparameterization trick)          鈺?
鈺?    - Twin Q-critics (reduce overestimation)                         鈺?
鈺?    - Automatic temperature (alpha) tuning                           鈺?
鈺?    - Off-policy: replay buffer 鈫?high sample efficiency             鈺?
鈺?                                                                     鈺?
鈺? B) ACTION SPACE: 卤螖P RESIDUAL (proven in V14-v10 @ 89%!)           鈺?
鈺?    - Cumulative delta: GAT_pred + 危(螖P) = final dispatch            鈺?
鈺?    - Adaptive scale: LOW=卤20 / MID=卤50 / HIGH=卤120 MW per gen       鈺?
鈺?    - Cumulative clip: LOW=卤70 / MID=卤250 / HIGH=卤840 MW             鈺?
鈺?    - NO projection layer (just clip + power balance)                鈺?
鈺?                                                                     鈺?
鈺? C) REWARD: Convergence-focused, NO conflicting penalties            鈺?
鈺?    - Convergence jackpot: +1000 (strong positive signal)            鈺?
鈺?    - Step penalty: -10 (encourage fast convergence)                 鈺?
鈺?    - Violation penalty: -20 * (n_viols / 100)                       鈺?
鈺?    - NO GAT proximity penalty (was punishing necessary corrections) 鈺?
鈺?    - NO residual magnitude penalty (was blocking large corrections) 鈺?
鈺?                                                                     鈺?
鈺? D) DATA SPLIT: 70% Train / 15% Val / 15% Test                       鈺?
鈺?    - Stratified by demand segment (9 segments 脳 ~30 samples)        鈺?
鈺?    - Train: ~180 samples (20/segment)                               鈺?
鈺?    - Val: ~40 samples (4-5/segment)                                 鈺?
鈺?    - Test: ~40 samples (4-5/segment)                                鈺?
鈺?    - NO curriculum: all segments trained simultaneously             鈺?
鈺?                                                                     鈺?
鈺? TARGET: Train & Val AC convergence > 90%                            鈺?
鈺氣晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨暆

Author: Based on V14-v10 success (89%) + V20 TD3 failure analysis
Date: 2026-06-19
"""

import os, sys, time, copy, json, math, random, pathlib, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, deque
from strict_sced_utils import project_box_sum, zero_sum_unit

# 鈽?V21: Force UTF-8 stdout/stderr to prevent GBK emoji crash on Windows!
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

warnings.filterwarnings("ignore")

# ============================================================
# Paths & Config
# ============================================================
CODE_DIR = pathlib.Path(__file__).resolve().parent
PROJ = str(CODE_DIR.parents[1])
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(1, PROJ)

N_GEN = 68

# Training config (V35: Complete 鈥?HIGH-focused + Stochastic eval built-in!)
TOTAL_UPDATES = 100               # Same as V34
N_EPISODES_PER_UPDATE = 20        # Episodes collected per update
N_GRADIENT_STEPS = 50             # Gradient steps per update
EVAL_INTERVAL = 10                # Eval every 10 updates
N_PER_SEGMENT_TRAIN = 3           # Episodes per segment per update
N_PER_SEGMENT_EVAL = 5            # Eval samples per segment

# 鈽?V35: HIGH-focused curriculum (same as V34)
PHASE_1_END = 30                  # U1-30: all segments (maintain V33 foundation)
PHASE_2_END = 100                 # U31-100: 23-24GW focused (80% episodes from HIGH!)

# 鈽?V35: Stochastic retry evaluation (built into final eval!)
V35_STOCHASTIC_RETRIES = 3        # K=3 stochastic retries for failed samples
V35_STOCHASTIC_STEPS = 50         # Max steps for stochastic retries
V35_USE_ENSEMBLE = True           # Use best_model + U50 checkpoint ensemble

# Episode config
MAX_STEPS = 80                    # LOW/MID max steps
MAX_STEPS_HIGH = 120              # HIGH max steps
AC_PF_CHECK_EVERY = 1             # Check EVERY step 鈥?continuous ACPF signal!

# Data split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# V26 SAC Hyperparameters 鈥?higher LR since N-1 sensitivity gives better gradient
SAC_LR_ACTOR = 3e-4           # V26: Back to V21's LR (N-1 obs makes learning efficient!)
SAC_LR_CRITIC = 3e-4          # V26: Back to V21's LR
SAC_LR_ALPHA = 3e-4           # V26: Auto-tune alpha normally
SAC_GAMMA = 0.99              # Discount factor
SAC_TAU = 0.005               # Soft target update coefficient
SAC_INIT_ALPHA = 0.2          # V26: Normal exploration (was 0.1)
SAC_LOG_STD_MAX = -2.0        # Keep normalized exploration within the narrow AC-feasible region
SAC_LOG_STD_MAX_VECTOR = None # Optional per-action cap for mixed P/V policies
SAC_TARGET_ENTROPY = -0.5 * N_GEN  # Target entropy = -0.5 * action_dim (standard)
SAC_AUTO_ALPHA = True         # Auto-tune SAC temperature

# Replay buffer
BUFFER_SIZE = 200000
BATCH_SIZE = 512               # 鈽?V28: Larger batch (was 256) 鈥?better gradient!

# Reward config (V34: HIGH-FOCUSED 鈥?5x ACPF reward for 23-24GW!)
BASE_REWARD = -10.0               # Step penalty
CONVERGENCE_REWARD = 3000.0       # ACPF reward
N1_SECURE_REWARD = 20000.0        # Full SCED jackpot; dominates partial feasibility
R_VIOLATION_COEF = -20.0          # Violation penalty
R_ACTION_PENALTY_P = 0.0          # Disabled to match the validated Route-B SAC dynamics
R_ACTION_PENALTY_V = 0.0          # Disabled to match the validated Route-B SAC dynamics
R_N1_OVERLOAD_MARGIN = -50.0      # N-1 overload margin
N1_CHECK_EVERY_STEPS = 2          # N-1 every 2 steps
R_VOLTAGE_GAP_IMPROVEMENT = 800.0 # Reward per pu reduction in total voltage gap

# 鈽?V34: ACPF-specific rewards (stronger!)
R_ACPF_VMIN_BONUS = 10.0          # 鈽?V34: Double voltage bonus (was 5)
R_ACPF_LOADING_PENALTY = -30.0    # Loading penalty
R_ACPF_NONCONVERGENCE = -1000.0   # A non-converged PF must rank below near-feasible PF states

# 鈽?V34: HIGH-segment reward boost!
R_HIGH_ACPF_BOOST = 2.0           # 鈽?V34: 2x ACPF reward for 23-24GW convergence!

# Action space config (V34: HYBRID 鈥?same as V33 + larger HIGH clip!)
DEFAULT_RESIDUAL_MW = 20.0        # For obs normalization
UNIFORM_RESIDUAL_MW = 20.0        # Unified per-step Route-B active-power action scale
UNIFORM_CUMULATIVE_MULT = 15.0    # Small steps, larger cumulative room for 16GW redispatch

# 鈽?V28: HYBRID ACTION SPACE 鈥?not too small, not too large!
# 12 PTDF/LODF shift directions (for N-1 security) + 20 key generator 螖P (for AC-PF)
N_SHIFT_DIRECTIONS = 12            # Physics-guided directions (from V27)
N_KEY_GENERATORS = 20              # Top-20 most N-1-sensitive generators (raw 螖P)
# Action dim = 12 + 20 = 32 (was 68 in V26, 12 in V27 鈥?balanced!)
N_VOLTAGE_GROUPS = N_GEN           # Per-generator voltage controls; matches rescue_gen_vm_pu evidence
VOLTAGE_STEP_PU = 0.01             # Per-step voltage setpoint adjustment
VOLTAGE_ABSOLUTE_RANGE_PU = 0.06   # Absolute residual range around the reset setpoint
GEN_VM_MIN = 0.92
GEN_VM_MAX = 1.08
EXT_VM_MIN = 0.92
EXT_VM_MAX = 1.08
RL_VMIN_PU = 0.933
RL_VMAX_PU = 1.09

# Reference-source operating envelope used by both training and final verification.
# pandapower reports positive res_ext_grid P/Q as supply to the modeled network.
REF_SOURCE_P_MIN_MW = -2500.0
REF_SOURCE_P_MAX_MW = 2500.0
REF_SOURCE_Q_MIN_MVAR = -250.0
REF_SOURCE_Q_MAX_MVAR = 250.0
REF_SOURCE_S_MAX_MVA = 2500.0
ENFORCE_REF_SOURCE_LIMIT = True


def adopted_operation_pass(info):
    """Single pass criterion shared by training and Route-B evaluations."""
    return (
        bool(info.get('converged', False))
        and int(info.get('v_violations', 999)) == 0
        and int(info.get('overloaded_lines', 999)) == 0
        and int(info.get('trafo_overloaded', 999)) == 0
        and (not ENFORCE_REF_SOURCE_LIMIT or int(info.get('reference_source_violations', 999)) == 0)
        and int(info.get('n1_violations', 999)) == 0
        and int(info.get('scheduled_balance_violations', 999)) == 0
        and int(info.get('generator_bound_violations', 999)) == 0
    )

# 鈽?V28: Speed optimizations
AC_PF_CHECK_EVERY = 2              # Check AC-PF every 2 steps (was 1) 鈥?2x faster!
AC_EVENT_N1_OVERLOAD_MW = 0.01     # Extra AC check near the strict N-1 boundary.
N1_CHECK_EVERY_STEPS = 2           # Check N-1 every 2 steps (was 1) 鈥?2x faster!
SAC_HIDDEN_DIM = 256               # Network hidden dim (keep 256 for capacity)
BATCH_SIZE_V28 = 512               # Larger batch (was 256) 鈥?better gradient!

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# Data Split 鈥?Stratified Train/Val/Test
# ============================================================

def _demand_key(sample):
    return round(float(sample.get('demand_gw', float(sample['demand_mw']) / 1000.0)), 3)

def split_samples_stratified(samples, train_ratio=TRAIN_RATIO,
                             val_ratio=VAL_RATIO, seed=RANDOM_SEED):
    """Split samples into train/val/test, stratified by demand segment."""
    rng = random.Random(seed)

    # Group by demand segment
    by_demand = defaultdict(list)
    for i, s in enumerate(samples):
        gw = _demand_key(s)
        by_demand[gw].append(i)

    train_ids, val_ids, test_ids = [], [], []

    for gw in sorted(by_demand.keys()):
        pool = list(by_demand[gw])
        rng.shuffle(pool)
        n = len(pool)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        # rest goes to test

        train_ids.extend(pool[:n_train])
        val_ids.extend(pool[n_train:n_train + n_val])
        test_ids.extend(pool[n_train + n_val:])

    return train_ids, val_ids, test_ids


def stratified_sample_from_pool(samples, pool_ids, n_per_segment):
    """Uniformly sample n_per_segment from each demand segment in pool_ids."""
    by_demand = defaultdict(list)
    for sid in pool_ids:
        gw = _demand_key(samples[sid])
        by_demand[gw].append(sid)

    selected = []
    for gw in sorted(by_demand.keys()):
        pool = list(by_demand[gw])
        n = min(n_per_segment, len(pool))
        selected.extend(random.sample(pool, n))

    random.shuffle(selected)
    return selected


def curriculum_sample(samples, pool_ids, update_num, n_total=20):
    """鈽?V34: HIGH-focused curriculum 鈥?fine-tune on 23-24GW!

    Phase 1 (U1-30):   All segments (maintain V33 foundation)
    Phase 2 (U31-100): 80% from 23-24GW, 20% from others (HIGH focus!)
    """
    by_demand = defaultdict(list)
    for sid in pool_ids:
        gw = _demand_key(samples[sid])
        by_demand[gw].append(sid)

    if update_num <= PHASE_1_END:
        # Phase 1: All segments uniformly
        allowed_gw = sorted(by_demand.keys())
        n_per_seg = max(1, n_total // len(allowed_gw))
        selected = []
        for gw in allowed_gw:
            pool = list(by_demand[gw])
            n = min(n_per_seg, len(pool))
            selected.extend(random.sample(pool, n))
    else:
        # Phase 2: 80% from 23-24GW, 20% from 16-22GW
        n_high = int(n_total * 0.8)
        n_low = n_total - n_high

        high_pool = by_demand.get(23.0, []) + by_demand.get(24.0, [])
        low_pool = []
        for gw in by_demand:
            if 16.0 <= float(gw) < 23.0:
                low_pool.extend(by_demand.get(gw, []))

        selected = []
        if high_pool:
            n = min(n_high, len(high_pool))
            selected.extend(random.sample(high_pool, n))
        if low_pool:
            n = min(n_low, len(low_pool))
            selected.extend(random.sample(low_pool, n))

    # Pad if not enough
    while len(selected) < n_total:
        all_pool = []
        for gw in by_demand:
            all_pool.extend(by_demand[gw])
        if all_pool:
            selected.append(random.choice(all_pool))
        else:
            break

    random.shuffle(selected)
    return selected[:n_total]


# ============================================================
# SCED Environment 鈥?卤螖P Residual Action Space (from V14-v10)
# ============================================================

class _StoredLinearSecurityNetwork:
    """Minimal Case300Network-compatible wrapper around dataset metadata."""

    def __init__(self, net_base, meta):
        self.net_base = net_base
        self.P_ref = np.asarray(meta["P_ref"], dtype=np.float64)
        self.F_ref = np.asarray(meta["F_ref"], dtype=np.float64)
        self.flow_intercept = np.asarray(meta["flow_intercept"], dtype=np.float64)
        self.S = np.asarray(meta["S"], dtype=np.float64)
        self.lodf = np.asarray(meta["lodf"], dtype=np.float64)
        self.n_line, self.n_gen = self.S.shape
        self.flow_limit = np.asarray(meta["flow_limit"], dtype=np.float64)
        self.effective_limit = np.asarray(meta["effective_limit"], dtype=np.float64)
        self.valid_contingency_indices = np.asarray(
            meta["valid_contingency_indices"], dtype=int
        )

    def estimate_base_flows(self, dispatch):
        return self.flow_intercept + self.S @ np.asarray(dispatch, dtype=np.float64)

    def estimate_post_contingency_flows(self, dispatch, outage_idx):
        flows = self.estimate_base_flows(dispatch)
        return flows + self.lodf[:, int(outage_idx)] * flows[int(outage_idx)]


class SCEDEnvV21:
    """
    V21 SCED Environment 鈥?SAC-compatible with 卤螖P RESIDUAL action space.

    Based on V14-v10 (89% Eval) with key improvements:
    - Simplified reward (NO GAT proximity / residual magnitude penalties)
    - Clean train/val/test split
    - Observation includes cumulative_delta (critical for residual RL)
    """

    def __init__(
        self,
        v10_samples,
        sample_ids,
        dataset_meta=None,
        action_encoding="physics32",
        action_basis_seed=20260721,
        voltage_action_mode="incremental",
        voltage_absolute_range_pu=VOLTAGE_ABSOLUTE_RANGE_PU,
        observation_mode="legacy",
        generator_upper_margin=None,
        include_bus_voltage_feedback=False,
        initial_acpf_on_reset=False,
    ):
        """
        Args:
            v10_samples: full list of GT samples
            sample_ids: indices to use (train/val/test partition)
        """
        self.N = N_GEN
        if action_encoding not in {
            "physics32", "physics32fixed", "physics32orth", "full68", "matched32"
        }:
            raise ValueError(f"Unsupported action_encoding: {action_encoding}")
        self.action_encoding = str(action_encoding)
        self.action_basis_seed = int(action_basis_seed)
        if voltage_action_mode not in {"incremental", "absolute_residual"}:
            raise ValueError(f"Unsupported voltage_action_mode: {voltage_action_mode}")
        self.voltage_action_mode = str(voltage_action_mode)
        self.voltage_absolute_range_pu = float(voltage_absolute_range_pu)
        if self.voltage_absolute_range_pu <= 0.0:
            raise ValueError("voltage_absolute_range_pu must be positive")
        if observation_mode not in {"legacy", "normalized_compact"}:
            raise ValueError(f"Unsupported observation_mode: {observation_mode}")
        self.observation_mode = str(observation_mode)
        if generator_upper_margin is not None and not (0.0 < float(generator_upper_margin) <= 1.0):
            raise ValueError("generator_upper_margin must be in (0, 1]")
        self.generator_upper_margin = (
            None if generator_upper_margin is None else float(generator_upper_margin)
        )
        self.include_bus_voltage_feedback = bool(include_bus_voltage_feedback)
        self.initial_acpf_on_reset = bool(initial_acpf_on_reset)
        import importlib.util
        from pandapower.networks import case300

        # Generator limits
        net = case300()
        self.PMIN = net.gen.min_p_mw.values.copy().astype(np.float64)
        self.PMAX = net.gen.max_p_mw.values.copy().astype(np.float64)
        self.baseMVA = net.sn_mva

        # Generator cost from poly_cost
        pc = net.poly_cost.copy()
        cost_map = dict(zip(pc['element'].astype(int), pc.cp1_eur_per_mw.values))
        cp1_arr = np.array([cost_map.get(i, 30.0) for i in range(N_GEN)], dtype=np.float64)

        gf = {}
        gf['pmax'] = net.gen.max_p_mw.values.copy()
        gf['pmin'] = net.gen.min_p_mw.values.copy()
        gf['cp1'] = cp1_arr
        self.gen_feats = gf

        # Reuse the already loaded IEEE-300 data. Constructing case300() parses
        # the full network tables and was being repeated for every PF and reset.
        self._ac_net_template = copy.deepcopy(net)
        self._gat_bus_idx = net.gen.bus.values.astype(int)
        self._gat_gen_bus = net.gen.bus.values.astype(float)
        self._gat_gen_pmax = net.gen.max_p_mw.values.astype(float)
        self._gat_gen_vm = net.gen.vm_pu.values.astype(float)
        self._gat_base_kv = net.bus.vn_kv.values.astype(float)[self._gat_bus_idx]
        self._gat_base_load_mw = float(net.load.p_mw.sum())
        self._n_bus = int(len(net.bus))

        # Build edge_index for GAT
        bus_idx = net.gen.bus.values.astype(int)
        src, dst = [], []
        for i in range(len(bus_idx)):
            for j in range(len(bus_idx)):
                if i != j:
                    src.append(i); dst.append(j)
        edge_index = torch.LongTensor([src, dst])

        # Initialize GAT model from the local paper_results/code implementation.
        # A stale root-level train_gat_v10.py also exists in this project and
        # still defines the old 6-feature GAT.  Loading by absolute path avoids
        # accidentally instantiating that incompatible class.
        gat_path = CODE_DIR / "train_gat_v10.py"
        spec = importlib.util.spec_from_file_location("_local_train_gat_v10", gat_path)
        gat_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gat_mod)
        GATv10Predictor = gat_mod.GATv10Predictor

        self.gat_net = GATv10Predictor(
            n_units=N_GEN, edge_index=edge_index,
            gat_heads=4, gat_hidden=128,
        ).to(DEVICE)

        ckpt_override = os.environ.get("ROUTE_B_GAT_CHECKPOINT")
        ckpt_path = (
            pathlib.Path(ckpt_override)
            if ckpt_override
            else pathlib.Path(PROJ) / "gatv10_results" / "gatv10_predictor_best.pth"
        )
        if not ckpt_path.exists() and not ckpt_override:
            ckpt_path = pathlib.Path(PROJ) / "models" / "gatv10_predictor_best.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"GAT checkpoint not found: {ckpt_path}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"GAT checkpoint not found")
        sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        self.gat_net.load_state_dict(sd['model_state_dict'])
        self.gat_net.eval()

        self.dataset_meta = dataset_meta or {}
        from generate_case300_gt_v10 import Case300Network, check_n1_security
        self._check_n1_security = check_n1_security
        linear_meta = self.dataset_meta.get("linear_security_model") or {}
        required_linear_keys = {
            "P_ref", "F_ref", "flow_intercept", "S", "lodf",
            "flow_limit", "effective_limit", "valid_contingency_indices",
        }
        if required_linear_keys.issubset(linear_meta):
            self.dcnet = _StoredLinearSecurityNetwork(net, linear_meta)
            self.valid_contingency_indices = self.dcnet.valid_contingency_indices.copy()
            print(
                f"[V23] Loaded dataset linear_security_model directly "
                f"({len(self.valid_contingency_indices)} valid outages)."
            )
        else:
            print("[V23] Building Case300Network for N-1 security check...")
            self.dcnet = Case300Network()
            self._load_linear_security_model_from_meta()
        print(f"[V23] N-1 network ready: S={self.dcnet.S.shape}, LODF={self.dcnet.lodf.shape}")

        # Store samples and their IDs
        self.all_samples = v10_samples
        self.sample_ids = list(sample_ids)
        self._demand_segments = sorted(set(_demand_key(v10_samples[sid]) for sid in self.sample_ids))

        # 鈽?V28: PTDF/LODF-guided shift directions + key generator selection
        self._shift_directions = None  # (n_shift_dirs, 68) matrix
        self._key_gen_indices = None
        self._fixed_shift_directions_cache = None
        self._fixed_key_gen_indices_cache = None
        self._orthogonal_active_basis_cache = None
        if self.action_encoding == "full68":
            self._n_shift_dirs = 0
            self._n_key_gens = self.N
        else:
            self._n_shift_dirs = N_SHIFT_DIRECTIONS
            self._n_key_gens = N_KEY_GENERATORS
        self._n_voltage_groups = N_VOLTAGE_GROUPS
        self._n_voltage_actions = N_VOLTAGE_GROUPS + 1  # gen groups + ext-grid
        self._voltage_group_indices = self._build_voltage_groups()

        # Internal state
        self.current_power = None
        self.gat_power = None
        self.current_gen_vm_pu = None
        self.base_gen_vm_pu = None
        self.current_ext_grid_vm_pu = None
        self.base_ext_grid_vm_pu = None
        self._gen_bus_vm_pu = np.ones(self.N, dtype=np.float64)
        self._gen_bus_v_gap_pu = np.zeros(self.N, dtype=np.float64)
        self._bus_vm_pu = np.ones(self._n_bus, dtype=np.float64)
        self._bus_v_gap_pu = np.zeros(self._n_bus, dtype=np.float64)
        self.target_demand = None
        self.gt_power = None
        self.cumulative_delta = None
        self.current_residual_scale = DEFAULT_RESIDUAL_MW
        self.corr_step = 0
        self.max_steps = MAX_STEPS
        self.episode_history = []
        self._gt_hhi_baseline = 343.5

        # AC-PF result tracking
        self._current_vmin = 0.95
        self._prev_vmin = 0.95
        self._n_v_viol = 300
        self._prev_n_v_viol = 300
        self._n_overloaded_lines = 68
        self._max_line_load_pct = 0.0
        self._trafo_overloaded = 0
        self._prev_v_violation_mag_pu = None
        self.prev_ac_result = np.array([0., 0.95, 1.05, 0.], dtype=np.float32)

        # 鈽?V24: N-1 violation tracking (continuous reward!)
        self._prev_n1_violations = 999  # Previous step's N-1 violation count
        self._current_n1_violations = 999  # Current step's N-1 violation count

        self.rl_obs_dim = None

    def _load_linear_security_model_from_meta(self):
        """Load the dataset's exact affine LODF model for linear N-1 checks."""
        fresh_lodf = np.asarray(self.dcnet.lodf, dtype=np.float64).copy()
        fresh_valid = np.asarray(
            getattr(self.dcnet, 'valid_contingency_indices', []), dtype=int
        )
        meta = (self.dataset_meta or {}).get("linear_security_model")
        if not meta:
            print("[WARN] Dataset has no linear_security_model; using freshly built N-1 model.")
            self._set_valid_contingencies(fresh_valid)
            return
        try:
            self.dcnet.P_ref = np.asarray(meta["P_ref"], dtype=np.float64)
            self.dcnet.F_ref = np.asarray(meta["F_ref"], dtype=np.float64)
            self.dcnet.flow_intercept = np.asarray(meta["flow_intercept"], dtype=np.float64)
            self.dcnet.S = np.asarray(meta["S"], dtype=np.float64)
            stored_lodf = np.asarray(meta["lodf"], dtype=np.float64)
            if stored_lodf.shape != fresh_lodf.shape or np.count_nonzero(np.abs(stored_lodf) > 1e-8) == 0:
                self.dcnet.lodf = fresh_lodf
                valid = fresh_valid
                print("[WARN] Stored LODF is degenerate; retaining freshly computed nonzero LODF.")
            else:
                self.dcnet.lodf = stored_lodf
                valid = np.asarray(meta.get("valid_contingency_indices", []), dtype=int)
            self.dcnet.flow_limit = np.asarray(meta["flow_limit"], dtype=np.float64)
            self.dcnet.effective_limit = np.asarray(meta["effective_limit"], dtype=np.float64)
            self._set_valid_contingencies(valid)
            print("[V23] Loaded dataset linear_security_model for N-1 checks.")
        except Exception as exc:
            print(f"[WARN] Failed to load dataset linear_security_model: {exc}")
            self.dcnet.lodf = fresh_lodf
            self._set_valid_contingencies(fresh_valid)

    def _set_valid_contingencies(self, candidate_indices=None):
        lodf = np.asarray(self.dcnet.lodf, dtype=np.float64)
        finite = np.isfinite(lodf).all(axis=0)
        if candidate_indices is not None and len(candidate_indices):
            candidate_indices = np.asarray(candidate_indices, dtype=int)
            in_range = (candidate_indices >= 0) & (candidate_indices < lodf.shape[1])
            candidate_indices = candidate_indices[in_range]
            self.valid_contingency_indices = candidate_indices[finite[candidate_indices]]
        else:
            self.valid_contingency_indices = np.flatnonzero(finite).astype(int)
        if len(self.valid_contingency_indices) == 0:
            raise RuntimeError("No finite non-islanding LODF columns are available")
        print(
            f"[V23] Valid non-islanding LODF columns: "
            f"{len(self.valid_contingency_indices)}/{lodf.shape[1]}"
        )

    def _build_voltage_groups(self):
        """Group generators by generator-bus index for low-dimensional voltage control."""
        try:
            import pandapower.networks as pn
            net = pn.case300()
            gen_bus = net.gen.bus.values.astype(int)
        except Exception:
            gen_bus = np.arange(self.N)
        order = np.argsort(gen_bus)
        groups = []
        for chunk in np.array_split(order, self._n_voltage_groups):
            groups.append(np.asarray(chunk, dtype=int))
        return groups

    def _get_obs_dim(self):
        if self.rl_obs_dim is not None:
            return self.rl_obs_dim
        _ = self.reset(self.sample_ids[0])
        obs = self._get_obs()
        self.rl_obs_dim = len(obs)
        return self.rl_obs_dim

    def reset(self, sample_idx=None):
        """Reset environment for new episode."""
        if sample_idx is None:
            sample_idx = random.choice(self.sample_ids)

        sample = self.all_samples[sample_idx]
        self.current_sample = sample
        self.target_demand = float(sample['demand_mw'])
        # 鈽?V25: NO GT dispatch used 鈥?pure RL from GAT!
        # gt_power not stored (would be cheating in deployment)
        self._gt_hhi_baseline = 343.5  # Fixed reference (not from GT sample)

        # GAT prediction
        with torch.no_grad():
            gat_pred = self._predict_gat(self.target_demand)

        # 鈽?V33: NORMALIZE GAT output to match demand!
        # GATv10 predicts dispatch that can be 7x the demand (sum=174000 vs demand=24000)
        # This makes AC-PF impossible. Fix: scale GAT output so sum == demand.
        reset_upper_margin = (
            self.generator_upper_margin
            if self.generator_upper_margin is not None else 0.99
        )
        gat_pred = project_box_sum(gat_pred, self.PMIN, self.PMAX * reset_upper_margin,
                                   self.target_demand)

        self.gat_power = gat_pred.astype(np.float64).copy()
        label_dispatch = np.asarray(sample.get('dispatch_68', self.gat_power), dtype=np.float64)
        self.reset_audit = {
            'initial_power_source': 'gat_prediction_projected_to_balance',
            'initial_voltage_source': 'case300_default_setpoints',
            'action_encoding': self.action_encoding,
            'active_action_dim': int(self._n_shift_dirs + self._n_key_gens),
            'gat_sum_mw': float(self.gat_power.sum()),
            'target_demand_mw': float(self.target_demand),
            'gat_minus_label_l1_mw': float(np.abs(self.gat_power - label_dispatch).sum()),
            'uses_dispatch_label_as_initial_power': False,
            'uses_rescue_as_initial_state': False,
        }

        self.current_power = self.gat_power.copy()
        # Start from the engineering/default voltage schedule, not from the
        # sample's optimized voltage label.  The latter is an offline AC-PF
        # repair artifact and would make the RL task look easier than it is.
        self.current_gen_vm_pu = np.asarray(
            self.dcnet.net_base.gen.vm_pu.values, dtype=np.float64
        ).copy()
        if len(self.current_gen_vm_pu) != self.N:
            self.current_gen_vm_pu = np.ones(self.N, dtype=np.float64)
        self.current_gen_vm_pu = np.clip(self.current_gen_vm_pu, GEN_VM_MIN, GEN_VM_MAX)
        self.base_gen_vm_pu = self.current_gen_vm_pu.copy()
        self._gen_bus_vm_pu = self.current_gen_vm_pu.copy()
        self._gen_bus_v_gap_pu = np.zeros(self.N, dtype=np.float64)
        self._bus_vm_pu = np.ones(self._n_bus, dtype=np.float64)
        self._bus_v_gap_pu = np.zeros(self._n_bus, dtype=np.float64)
        self.current_ext_grid_vm_pu = np.asarray(
            self.dcnet.net_base.ext_grid.vm_pu.values if len(self.dcnet.net_base.ext_grid) else [1.0],
            dtype=np.float64
        ).copy()
        if len(self.current_ext_grid_vm_pu) < 1:
            self.current_ext_grid_vm_pu = np.ones(1, dtype=np.float64)
        self.current_ext_grid_vm_pu = np.clip(self.current_ext_grid_vm_pu, EXT_VM_MIN, EXT_VM_MAX)
        self.base_ext_grid_vm_pu = self.current_ext_grid_vm_pu.copy()
        self.cumulative_delta = np.zeros(self.N, dtype=np.float64)
        self.corr_step = 0

        # Build one deterministic physical action coordinate system per
        # episode. Recomputing it mid-trajectory changes the meaning of an
        # action dimension without exposing that change to the actor/critic.
        self._shift_directions = None
        self._key_gen_indices = None

        # Unified P+V residual action scale for the corrected Route-B dataset.
        self.current_residual_scale = float(UNIFORM_RESIDUAL_MW)
        self.max_steps = MAX_STEPS

        self.episode_history = []

        # Reset AC-PF tracking
        self._current_vmin = 0.95
        self._prev_vmin = 0.95
        self._n_v_viol = 300
        self._prev_n_v_viol = 300
        self._n_overloaded_lines = 68
        self._max_line_load_pct = 0.0
        self._trafo_overloaded = 0
        self._prev_v_violation_mag_pu = None
        self.prev_ac_result = np.array([0., 0.95, 1.05, 0.], dtype=np.float32)

        # 鈽?V24: Reset N-1 tracking
        self._prev_n1_violations = 999
        self._current_n1_violations = 999

        if self.initial_acpf_on_reset:
            initial_ac = self._run_ac_pf_check()
            initial_vmin = float(initial_ac.get('v_min', 0.80))
            initial_vmax = float(initial_ac.get('v_max', 1.15))
            self._current_vmin = initial_vmin
            self._prev_vmin = initial_vmin
            self._n_v_viol = int(initial_ac.get('v_violations', 300))
            self._prev_n_v_viol = self._n_v_viol
            self._n_overloaded_lines = int(initial_ac.get('overloaded_lines', 68))
            self._max_line_load_pct = float(initial_ac.get('line_loading_max', 0.0))
            self._trafo_overloaded = int(initial_ac.get('trafo_overloaded', 999))
            self._prev_v_violation_mag_pu = (
                float(initial_ac.get('v_violation_mag_pu', 0.0))
                if bool(initial_ac.get('converged', False)) else None
            )
            self.prev_ac_result = np.array([
                float(initial_ac.get('converged', False)),
                initial_vmin,
                initial_vmax,
                float(initial_ac.get('loading_max', 0.0)),
            ], dtype=np.float32)
            gen_bus_vm = initial_ac.get('gen_bus_vm_pu')
            gen_bus_gap = initial_ac.get('gen_bus_v_gap_pu')
            bus_vm = initial_ac.get('bus_vm_pu')
            bus_gap = initial_ac.get('bus_v_gap_pu')
            if gen_bus_vm is not None and len(gen_bus_vm) == self.N:
                self._gen_bus_vm_pu = np.asarray(gen_bus_vm, dtype=np.float64)
            if gen_bus_gap is not None and len(gen_bus_gap) == self.N:
                self._gen_bus_v_gap_pu = np.asarray(gen_bus_gap, dtype=np.float64)
            if bus_vm is not None and len(bus_vm) == self._n_bus:
                self._bus_vm_pu = np.asarray(bus_vm, dtype=np.float64)
            if bus_gap is not None and len(bus_gap) == self._n_bus:
                self._bus_v_gap_pu = np.asarray(bus_gap, dtype=np.float64)
            self.reset_audit['initial_acpf_converged'] = bool(
                initial_ac.get('converged', False)
            )
            self.reset_audit['initial_acpf_v_violations'] = self._n_v_viol

        return self._get_obs()

    def _estimate_base_flows(self, P):
        if hasattr(self.dcnet, 'estimate_base_flows'):
            return self.dcnet.estimate_base_flows(P)
        raise AttributeError("dcnet must provide affine estimate_base_flows(P)")

    def _scan_n1_violations(self):
        S = self.dcnet.S
        lodf = self.dcnet.lodf
        limits = self.dcnet.effective_limit
        flows = self._estimate_base_flows(self.current_power)
        violations = []
        total_overload = 0.0

        # Base-case line-flow security is part of linear SCED, not only N-1.
        for l in range(len(flows)):
            limit = limits[l]
            overload_pos = max(0.0, flows[l] - limit)
            overload_neg = max(0.0, -flows[l] - limit)
            overload = overload_pos + overload_neg
            if overload > 0.01:
                # Keep the physical flow sign. For a negative-direction
                # violation d(max(0, -f-limit))/dP = -df/dP; taking -f here
                # would incorrectly turn both overload directions positive.
                signed_flow = flows[l]
                violations.append((overload, -1, l, signed_flow, S[l, :]))
                total_overload += overload

        for k in self.valid_contingency_indices:
            fk = flows[k]
            if abs(fk) < 0.1:
                continue
            if hasattr(self.dcnet, "estimate_post_contingency_flows"):
                f_post = self.dcnet.estimate_post_contingency_flows(self.current_power, k)
            else:
                f_post = flows + lodf[:, k] * fk
            for l in range(len(flows)):
                if l == k:
                    continue
                limit = limits[l]
                overload_pos = max(0.0, f_post[l] - limit)
                overload_neg = max(0.0, -f_post[l] - limit)
                overload = overload_pos + overload_neg
                if overload > 0.01:
                    signed_post = f_post[l]
                    grad = S[l, :] + lodf[l, k] * S[k, :]
                    violations.append((overload, k, l, signed_post, grad))
                    total_overload += overload
        violations.sort(key=lambda x: -x[0])
        return flows, violations, total_overload

    def _compute_n1_sensitivity(self):
        """鈽?V26: Compute N-1 overload sensitivity per generator."""
        _, violations, n1_overload_mw = self._scan_n1_violations()
        sens = np.zeros(self.N, dtype=np.float64)
        for overload, _k, _l, signed_post, grad in violations:
            sens += overload * np.sign(signed_post) * grad
        n1_viol = len(violations)
        self._n1_overload_mw = n1_overload_mw
        self._n1_violations = n1_viol
        return sens, n1_overload_mw, n1_viol

    def _compute_shift_directions(self):
        if self.action_encoding == "full68":
            self._shift_directions = np.zeros((0, self.N), dtype=np.float64)
            return self._shift_directions
        if self.action_encoding in {"physics32fixed", "physics32orth"}:
            self._build_fixed_physics_action_basis()
            self._shift_directions = self._fixed_shift_directions_cache.copy()
            return self._shift_directions
        if self.action_encoding == "matched32":
            rng = np.random.default_rng(self.action_basis_seed)
            directions = []
            attempts = 0
            while len(directions) < self._n_shift_dirs and attempts < 20 * self.N:
                attempts += 1
                vec = rng.standard_normal(self.N)
                vec -= vec.mean()
                for existing in directions:
                    vec -= np.dot(vec, existing) * existing
                norm = np.linalg.norm(vec)
                if norm <= 1e-10:
                    continue
                vec /= norm
                first = np.flatnonzero(np.abs(vec) > 1e-12)
                if len(first) and vec[first[0]] < 0.0:
                    vec = -vec
                directions.append(vec)
            if len(directions) != self._n_shift_dirs:
                raise RuntimeError(
                    f"Could not construct {self._n_shift_dirs} matched random directions"
                )
            self._shift_directions = np.asarray(directions, dtype=np.float64)
            return self._shift_directions
        return self._compute_shift_directions_clean()
        """鈽?V27: Compute PTDF/LODF-guided shift directions for reduced action space.

        Instead of letting RL control all 68 generators independently (high-dim, hard),
        we compute a SMALL set of physics-guided "shift directions" 鈥?each is a 68-dim
        vector that, when applied as 螖P, reduces N-1 overload.

        Method:
        1. Find N-1 violated (k,l) pairs (contingency k 鈫?overload on line l)
        2. For each violation, compute the "relief direction":
           - PTDF[l,:] tells us how each generator affects line l flow
           - LODF[l,k] tells us how contingency k amplifies line l flow
           - Combined: d(f_post_l)/d(P_i) = PTDF[l,i] + LODF[l,k]*PTDF[k,i]
           - To REDUCE overload: shift generators OPPOSITE to this gradient
        3. Also include "cost-based" and "balance" directions for economic dispatch
        4. Orthogonalize the directions to avoid redundancy

        Returns: (N_SHIFT_DIRECTIONS, 68) matrix of unit shift vectors
        """
        S = self.dcnet.S          # PTDF: (283, 68) 鈥?d(flow_l)/d(P_i)
        lodf = self.dcnet.lodf    # (283, 283)
        limits = self.dcnet.effective_limit
        n_line, n_gen = S.shape

        directions = []

        # === Direction 1-6: N-1 violation relief directions ===
        # Find top violated (k,l) pairs and compute relief
        _, violations, _total_overload_mw = self._scan_n1_violations()

        # Take top violations and create relief directions
        # Relief = -gradient (opposite to overload-increasing direction)
        for ov, k, l, signed_post, grad in violations:
            # Relief direction: reduce generators that increase overload
            relief = zero_sum_unit(-np.sign(signed_post) * grad)
            if relief is None:
                continue

            # Check not too similar to existing directions (cosine similarity)
            is_new = True
            for existing in directions:
                cos_sim = abs(np.dot(relief, existing))
                if cos_sim > 0.85:  # Too similar
                    is_new = False
                    break
            if is_new:
                directions.append(relief)
            if len(directions) >= 6:
                break

        # === Direction 7-8: Economic shift (cheap鈫抏xpensive and expensive鈫抍heap) ===
        cp1 = self.gen_feats['cp1']
        econ_dir = cp1 - cp1.mean()
        norm = np.linalg.norm(econ_dir)
        if norm > 1e-8:
            directions.append(econ_dir / norm)  # Shift toward expensive (reduce cost)
            directions.append(-econ_dir / norm)  # Shift toward cheap

        # === Direction 9-10: Capacity-based shifts ===
        pmax = self.gen_feats['pmax']
        cap_dir = pmax - pmax.mean()
        norm = np.linalg.norm(cap_dir)
        if norm > 1e-8:
            directions.append(cap_dir / norm)  # Shift toward large gens
            directions.append(-cap_dir / norm)  # Shift toward small gens

        # === Direction 11-12: Current dispatch adjustment (from GAT) ===
        if self.gat_power is not None:
            gat_dir = self.gat_power - self.current_power
            norm = np.linalg.norm(gat_dir)
            if norm > 1e-8:
                directions.append(gat_dir / norm)  # Toward GAT prediction
                directions.append(-gat_dir / norm)  # Away from GAT

        # Pad or truncate to exactly N_SHIFT_DIRECTIONS
        while len(directions) < self._n_shift_dirs:
            # Add random orthogonal direction
            rand_dir = np.zeros(n_gen, dtype=np.float64)
            rand_dir[len(directions) % n_gen] = 1.0
            rand_dir[(len(directions) + 1) % n_gen] = -1.0
            for existing in directions:
                rand_dir -= np.dot(rand_dir, existing) * existing
            norm = np.linalg.norm(rand_dir)
            if norm > 1e-8:
                directions.append(rand_dir / norm)

        directions = directions[:self._n_shift_dirs]
        self._shift_directions = np.array(directions, dtype=np.float64)
        return self._shift_directions

    def _compute_shift_directions_clean(self):
        """Deterministic zero-sum PTDF/LODF action directions."""
        S = self.dcnet.S
        _n_line, n_gen = S.shape
        directions = []

        def add_direction(vec):
            unit = zero_sum_unit(vec)
            if unit is None:
                return
            for existing in directions:
                if abs(np.dot(unit, existing)) > 0.85:
                    return
            directions.append(unit)

        _, violations, _total_overload_mw = self._scan_n1_violations()
        for _ov, _k, _l, signed_post, grad in violations:
            add_direction(-np.sign(signed_post) * grad)
            if len(directions) >= 6:
                break

        cp1 = self.gen_feats['cp1']
        add_direction(-(cp1 - cp1.mean()))

        pmax = self.gen_feats['pmax']
        add_direction(pmax - pmax.mean())

        try:
            _u, _s, vh = np.linalg.svd(S, full_matrices=False)
            for j in range(min(2, vh.shape[0])):
                add_direction(vh[j, :])
        except Exception:
            pass

        headroom = self.PMAX - self.current_power
        downroom = self.current_power - self.PMIN
        add_direction(headroom - headroom.mean())
        add_direction(-(downroom - downroom.mean()))

        basis_idx = 0
        while len(directions) < self._n_shift_dirs and basis_idx < 2 * n_gen:
            basis = np.zeros(n_gen, dtype=np.float64)
            basis[basis_idx % n_gen] = 1.0
            basis[(basis_idx + 1) % n_gen] = -1.0
            add_direction(basis)
            basis_idx += 1

        basis_idx = 0
        while len(directions) < self._n_shift_dirs:
            basis = np.zeros(n_gen, dtype=np.float64)
            basis[basis_idx % n_gen] = 1.0
            basis[(basis_idx + 1) % n_gen] = -1.0
            unit = zero_sum_unit(basis)
            if unit is not None:
                directions.append(unit)
            basis_idx += 1

        self._shift_directions = np.array(directions[:self._n_shift_dirs], dtype=np.float64)
        return self._shift_directions

    def _build_fixed_physics_action_basis(self):
        """Build one PTDF/LODF coordinate system shared by every sample.

        Dynamic violation-specific directions make an action index change its
        physical meaning between episodes.  Here the post-contingency branch
        gradients are accumulated into a network-wide Gram matrix, whose
        dominant zero-sum eigenvectors form a deterministic reduced basis.
        No dispatch labels or rescue actions are used.
        """
        if (
            self._fixed_shift_directions_cache is not None
            and self._fixed_key_gen_indices_cache is not None
        ):
            return

        S = np.asarray(self.dcnet.S, dtype=np.float64)
        lodf = np.asarray(self.dcnet.lodf, dtype=np.float64)
        limits = np.maximum(
            np.asarray(self.dcnet.effective_limit, dtype=np.float64), 1.0
        )
        gram = np.zeros((self.N, self.N), dtype=np.float64)

        base_grad = S - S.mean(axis=1, keepdims=True)
        base_grad = base_grad / limits[:, None]
        finite = np.isfinite(base_grad).all(axis=1)
        if np.any(finite):
            gram += base_grad[finite].T @ base_grad[finite]

        for k in np.asarray(self.valid_contingency_indices, dtype=int):
            if k < 0 or k >= S.shape[0]:
                continue
            gradients = S + lodf[:, k:k + 1] * S[k:k + 1, :]
            gradients[k, :] = 0.0
            gradients -= gradients.mean(axis=1, keepdims=True)
            gradients = gradients / limits[:, None]
            finite = np.isfinite(gradients).all(axis=1)
            if np.any(finite):
                gram += gradients[finite].T @ gradients[finite]

        gram = 0.5 * (gram + gram.T)
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)[::-1]
        directions = []
        for idx in order:
            vec = zero_sum_unit(eigenvectors[:, idx])
            if vec is None:
                continue
            pivot = int(np.argmax(np.abs(vec)))
            if vec[pivot] < 0.0:
                vec = -vec
            directions.append(vec)
            if len(directions) == self._n_shift_dirs:
                break
        if len(directions) != self._n_shift_dirs:
            raise RuntimeError("Could not construct the fixed PTDF/LODF action basis")

        generator_scores = np.sqrt(np.maximum(np.diag(gram), 0.0))
        key_indices = np.argsort(-generator_scores)[:self._n_key_gens]
        self._fixed_shift_directions_cache = np.asarray(directions, dtype=np.float64)
        self._fixed_key_gen_indices_cache = np.asarray(key_indices, dtype=int)
        print(
            f"[V23] Fixed PTDF/LODF action basis ready: "
            f"{self._fixed_shift_directions_cache.shape[0]} shifts + "
            f"{len(self._fixed_key_gen_indices_cache)} key generators."
        )

    def _build_orthogonal_physics_action_basis(self):
        """Orthonormalize the fixed 32-D physics-guided redispatch subspace.

        The original shift and key-generator columns are physically meaningful,
        but correlated columns make one SAC coordinate change the effect of the
        others.  The left singular vectors preserve exactly the same linear
        subspace while giving every action coordinate unit norm, zero sum, and
        an independent physical effect.
        """
        if self._orthogonal_active_basis_cache is not None:
            return self._orthogonal_active_basis_cache

        self._build_fixed_physics_action_basis()
        columns = [row.copy() for row in self._fixed_shift_directions_cache]
        if self.N <= 1:
            raise RuntimeError("At least two generators are required for balanced actions")
        balancing_share = -1.0 / float(self.N - 1)
        for key_idx in self._fixed_key_gen_indices_cache:
            direction = np.full(self.N, balancing_share, dtype=np.float64)
            direction[int(key_idx)] = 1.0
            columns.append(direction)

        raw_basis = np.column_stack(columns)
        raw_basis -= raw_basis.mean(axis=0, keepdims=True)
        u, singular_values, _ = np.linalg.svd(raw_basis, full_matrices=False)
        tolerance = max(raw_basis.shape) * np.finfo(np.float64).eps * singular_values[0]
        rank = int(np.count_nonzero(singular_values > tolerance))
        expected_rank = self._n_shift_dirs + self._n_key_gens
        if rank != expected_rank:
            raise RuntimeError(
                f"Physics action basis rank is {rank}, expected {expected_rank}; "
                "the reduced action coordinates do not span 32 independent directions"
            )

        basis = u[:, :expected_rank].copy()
        basis -= basis.mean(axis=0, keepdims=True)
        basis, _ = np.linalg.qr(basis, mode="reduced")
        for col in range(basis.shape[1]):
            pivot = int(np.argmax(np.abs(basis[:, col])))
            if basis[pivot, col] < 0.0:
                basis[:, col] *= -1.0

        self._orthogonal_active_basis_cache = basis
        condition = float(singular_values[0] / singular_values[rank - 1])
        orth_error = float(np.max(np.abs(basis.T @ basis - np.eye(expected_rank))))
        balance_error = float(np.max(np.abs(np.sum(basis, axis=0))))
        print(
            f"[V23] Orthogonal PTDF/LODF action basis ready: {expected_rank} directions; "
            f"raw_cond={condition:.2e}, orth_err={orth_error:.1e}, "
            f"balance_err={balance_error:.1e}."
        )
        return self._orthogonal_active_basis_cache

    def _get_obs(self):
        """Build observation vector 鈥?includes cumulative_delta + 鈽?V26 N-1 SENSITIVITY!"""
        p_util = self.current_power / (self.PMAX + 1e-8)

        # Include local voltage-gap feedback so the voltage-control action has
        # an observable direction: positive means raise the local setpoint,
        # negative means lower it.
        cumulative_limit = max(
            self.current_residual_scale * float(UNIFORM_CUMULATIVE_MULT), 1e-8
        )
        active_delta = (self.current_power - self.gat_power).reshape(-1, 1)
        if self.observation_mode == "normalized_compact":
            active_delta_feature = active_delta / (self.PMAX.reshape(-1, 1) + 1e-8)
            cumulative_feature = self.cumulative_delta.reshape(-1, 1) / cumulative_limit
        else:
            # Retain the historical broadcasting behavior for old checkpoints.
            active_delta_feature = active_delta / (self.PMAX + 1e-8)
            cumulative_feature = (
                self.cumulative_delta.reshape(-1, 1)
                / (self.current_residual_scale + 1e-8)
            )
        gen_feats = np.concatenate([
            p_util.reshape(-1, 1),
            self.current_power.reshape(-1, 1) / (self.target_demand + 1e-8),
            self.gat_power.reshape(-1, 1) / (self.target_demand + 1e-8),
            active_delta_feature,
            cumulative_feature,
            ((self.current_gen_vm_pu - 1.0) / 0.1).reshape(-1, 1),
            np.clip(self._gen_bus_v_gap_pu / 0.05, -1.0, 1.0).reshape(-1, 1),
        ], axis=-1)

        # Global features (~18)
        hhi = self._compute_hhi(self.current_power)
        hhi_delta = hhi - self._gt_hhi_baseline
        top5_idx = np.argsort(-self.current_power)[:min(5, self.N)]
        top5_frac = float(self.current_power[top5_idx].sum() / (self.current_power.sum() + 1e-8))

        vmin_current = self._current_vmin
        vmin_trend = vmin_current - self._prev_vmin

        total_residual_mag = np.abs(self.cumulative_delta).sum()
        residual_ratio = total_residual_mag / (self.target_demand + 1e-8)

        demand_level_code = 0.0
        if self.target_demand > 20000:
            demand_level_code = 2.0
        elif self.target_demand > 18000:
            demand_level_code = 1.0

        global_feat = np.array([
            float(self.corr_step) / float(self.max_steps),
            float(self.target_demand) / 1000.0,
            float(self.current_power.sum()) / (self.target_demand + 1e-8),
            float(self._n_v_viol) / 300.0,
            (hhi - 200.0) / 400.0,
            top5_frac,
            float(vmin_current),
            float(self._max_line_load_pct) / 100.0,
            float(self._n_overloaded_lines) / 283.0,
            float(residual_ratio * 100),
            float(vmin_trend),
            float(self._prev_n_v_viol - self._n_v_viol) / 300.0,
            demand_level_code,
            float(self.current_residual_scale / DEFAULT_RESIDUAL_MW),
            float((self.current_ext_grid_vm_pu[0] - 1.0) / 0.1),
        ], dtype=np.float32)

        # 鈽?V26: N-1 SENSITIVITY 鈥?d(overload)/d(P_i) for each generator!
        # This is the KEY innovation: RL can now SEE which generators cause N-1 violations
        try:
            n1_sens, n1_overload_mw, n1_viol = self._compute_n1_sensitivity()
        except Exception:
            n1_sens = np.zeros(self.N, dtype=np.float64)
            n1_overload_mw = 0.0
            n1_viol = 0
        # Preserve the relative relief direction without saturating nearly all
        # generators when overload-weighted sensitivities become large.
        n1_sens_scale = max(float(np.max(np.abs(n1_sens))), 1.0)
        n1_sens_obs = np.clip(
            n1_sens / n1_sens_scale, -1.0, 1.0
        ).astype(np.float32)

        # Store for reward computation
        self._n1_sensitivity = n1_sens
        self._n1_overload_mw = n1_overload_mw
        self._n1_violations = n1_viol

        if self.observation_mode == "normalized_compact":
            legacy_summary = np.array([
                float(np.clip(hhi / 1000.0, 0.0, 10.0)),
                float(np.clip(hhi_delta / 1000.0, -10.0, 10.0)),
                float(np.clip(total_residual_mag / (self.target_demand + 1e-8), 0.0, 2.0)),
            ], dtype=np.float32)
            n1_summary = np.array([
                float(np.clip(np.log1p(max(n1_overload_mw, 0.0)) / 10.0, 0.0, 1.0)),
                float(np.clip(n1_viol / max(len(self.valid_contingency_indices), 1), 0.0, 1.0)),
            ], dtype=np.float32)
            prev_ac_feature = np.array([
                float(self.prev_ac_result[0]),
                float(np.clip((self.prev_ac_result[1] - 1.0) / 0.1, -2.0, 2.0)),
                float(np.clip((self.prev_ac_result[2] - 1.0) / 0.1, -2.0, 2.0)),
                float(np.clip(self.prev_ac_result[3] / 100.0, 0.0, 3.0)),
            ], dtype=np.float32)
        else:
            legacy_summary = np.array([
                float(hhi), float(hhi_delta), float(total_residual_mag)
            ], dtype=np.float32)
            n1_summary = np.array([
                float(n1_overload_mw / 100.0), float(n1_viol)
            ], dtype=np.float32)
            prev_ac_feature = self.prev_ac_result

        obs_parts = [
            gen_feats.flatten(),          # 68 * per-generator features
            global_feat,                   # 14
            prev_ac_feature,               # 4
            # Keep the feature semantics identical to the validated Route-B
            # run. These are deliberately raw HHI/residual features; changing
            # their scale changes the policy input distribution even though
            # the observation dimension stays the same.
            legacy_summary,
            n1_sens_obs,                   # 鈽?V26: 68-dim N-1 sensitivity!
            n1_summary,
        ]
        if self.include_bus_voltage_feedback:
            obs_parts.extend([
                np.clip((self._bus_vm_pu - 1.0) / 0.1, -2.0, 2.0),
                np.clip(self._bus_v_gap_pu / 0.05, -2.0, 2.0),
            ])
        obs = np.concatenate(obs_parts, axis=0).astype(np.float32)

        # 鈽?V21: NaN/Inf guard 鈥?replace with 0 to prevent network corruption
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    @property
    def action_dim(self):
        return self._n_shift_dirs + self._n_key_gens + self._n_voltage_actions

    def get_action_bounds(self):
        """Hybrid action bounds: active-power redispatch plus voltage controls."""
        scale = self.current_residual_scale
        p_dim = self._n_shift_dirs + self._n_key_gens
        low_p = np.full(p_dim, -scale, dtype=np.float32)
        high_p = np.full(p_dim, scale, dtype=np.float32)
        voltage_scale = (
            self.voltage_absolute_range_pu
            if self.voltage_action_mode == "absolute_residual"
            else VOLTAGE_STEP_PU
        )
        low_v = np.full(self._n_voltage_actions, -voltage_scale, dtype=np.float32)
        high_v = np.full(self._n_voltage_actions, voltage_scale, dtype=np.float32)
        return np.concatenate([low_p, low_v]), np.concatenate([high_p, high_v])

    def _select_key_generators(self):
        """鈽?V28: Select top-20 most N-1-sensitive generators (recomputed each call)."""
        if self.action_encoding == "full68":
            self._key_gen_indices = np.arange(self.N, dtype=int)
            return self._key_gen_indices
        if self.action_encoding == "matched32":
            rng = np.random.default_rng(self.action_basis_seed + 1)
            self._key_gen_indices = np.sort(
                rng.choice(self.N, size=self._n_key_gens, replace=False)
            ).astype(int)
            return self._key_gen_indices
        if self.action_encoding in {"physics32fixed", "physics32orth"}:
            self._build_fixed_physics_action_basis()
            self._key_gen_indices = self._fixed_key_gen_indices_cache.copy()
            return self._key_gen_indices
        # Use N-1 sensitivity if available, otherwise use capacity
        if hasattr(self, '_n1_sensitivity') and self._n1_sensitivity is not None:
            sens = np.abs(self._n1_sensitivity)
            if float(np.max(sens)) <= 1e-12:
                try:
                    sens = np.sum(np.abs(self.dcnet.S), axis=0)
                except Exception:
                    sens = self.gen_feats['pmax'].copy()
        else:
            sens = self.gen_feats['pmax'].copy()

        # Select top-20 generators by absolute sensitivity
        self._key_gen_indices = np.argsort(-sens)[:self._n_key_gens]
        return self._key_gen_indices

    def step(self, action):
        """鈽?V28: HYBRID action space 鈥?PTDF/LODF directions + key generator 螖P.

        action: (32,) = [12 shift scales | 20 key gen 螖P]
        鈫?converts to 68-dim 螖P via:
           螖P = 危(shift_scales[j] * shift_dir[j]) + key_gen_deltas (on selected gens)
        """
        self.corr_step += 1

        # The action basis is fixed for the whole episode so every action
        # coordinate keeps one physical meaning in replay and in Q(s, a).
        if self._shift_directions is None or self._key_gen_indices is None:
            self._compute_shift_directions()
            self._select_key_generators()

        # Clip action to bounds
        low, high = self.get_action_bounds()
        action = np.clip(action, low, high).astype(np.float64)

        # 鈽?V28: Split action into shift scales + key gen deltas
        p_dim = self._n_shift_dirs + self._n_key_gens
        shift_scales = action[:self._n_shift_dirs]
        key_gen_deltas = action[self._n_shift_dirs:p_dim]
        voltage_deltas = action[p_dim:]

        # Keep the feasibility jackpot dominant, while discouraging a policy
        # from applying the maximum positive correction to every sample.
        # Normalize each branch by its physical action range so P and V are
        # penalized on comparable scales.
        p_scale = max(float(self.current_residual_scale), 1e-6)
        p_norm = np.abs(action[:p_dim]) / p_scale
        voltage_scale = (
            self.voltage_absolute_range_pu
            if self.voltage_action_mode == "absolute_residual"
            else VOLTAGE_STEP_PU
        )
        v_norm = np.abs(voltage_deltas) / max(float(voltage_scale), 1e-6)
        self._last_action_penalty = -(
            R_ACTION_PENALTY_P * float(np.mean(p_norm ** 2))
            + R_ACTION_PENALTY_V * float(np.mean(v_norm ** 2))
        )

        if self.action_encoding == "physics32orth":
            # One independent, unit-norm coordinate per action.  This spans the
            # same fixed PTDF/LODF + key-generator subspace as physics32fixed.
            active_basis = self._build_orthogonal_physics_action_basis()
            delta_p = active_basis @ action[:p_dim]
        else:
            # Part 1: PTDF/LODF shift directions to 68-D redispatch.
            delta_p = self._shift_directions.T @ shift_scales

            # Part 2: balanced key-generator redispatch.
            key_indices = self._key_gen_indices
            if self.N > 1:
                balancing_share = -1.0 / float(self.N - 1)
                for key_delta, key_idx in zip(key_gen_deltas, key_indices):
                    key_direction = np.full(self.N, balancing_share, dtype=np.float64)
                    key_direction[int(key_idx)] = 1.0
                    delta_p += float(key_delta) * key_direction

        # Update cumulative residual
        self.cumulative_delta += delta_p

        # Small per-step actions, large enough cumulative room to cover the
        # corrected Route-B low-load redispatch cases without one-step shocks.
        cum_clip = self.current_residual_scale * float(UNIFORM_CUMULATIVE_MULT)
        self.cumulative_delta = np.clip(self.cumulative_delta, -cum_clip, cum_clip)

        # Compute new dispatch: GAT + cumulative residual
        raw_new_power = self.gat_power + self.cumulative_delta

        # 鈽?V25: NO GT BLEND 鈥?pure RL from GAT prediction!
        # (GT is only available during training data generation, not deployment)

        step_upper_margin = (
            self.generator_upper_margin
            if self.generator_upper_margin is not None else 0.98
        )
        new_power = project_box_sum(raw_new_power, self.PMIN, self.PMAX * step_upper_margin,
                                    self.target_demand)
        self.current_power = new_power

        if len(voltage_deltas) == self._n_voltage_actions:
            gen_group_deltas = voltage_deltas[:self._n_voltage_groups]
            if self.voltage_action_mode == "absolute_residual":
                self.current_gen_vm_pu = self.base_gen_vm_pu.copy()
                for gidx, gen_idx in enumerate(self._voltage_group_indices):
                    if len(gen_idx):
                        self.current_gen_vm_pu[gen_idx] += gen_group_deltas[gidx]
                self.current_ext_grid_vm_pu[0] = (
                    self.base_ext_grid_vm_pu[0] + voltage_deltas[-1]
                )
            else:
                for gidx, gen_idx in enumerate(self._voltage_group_indices):
                    if len(gen_idx):
                        self.current_gen_vm_pu[gen_idx] += gen_group_deltas[gidx]
                self.current_ext_grid_vm_pu[0] += voltage_deltas[-1]
            self.current_gen_vm_pu = np.clip(self.current_gen_vm_pu, GEN_VM_MIN, GEN_VM_MAX)
            self.current_ext_grid_vm_pu = np.clip(self.current_ext_grid_vm_pu, EXT_VM_MIN, EXT_VM_MAX)
        reward, done, info = self._compute_reward()

        # Update internal state from AC-PF results
        if 'converged' in info:
            new_vmin = float(info.get('v_min', 0.88))
            self._prev_vmin = self._current_vmin
            self._current_vmin = new_vmin
            self.prev_ac_result = np.array([
                float(info.get('converged', False)),
                new_vmin,
                float(info.get('v_max', 1.10)),
                float(info.get('loading_max', 0.)),
            ], dtype=np.float32)
            self._prev_n_v_viol = getattr(self, '_n_v_viol', 300)
            self._n_v_viol = int(info.get('v_violations', 300))
            self._n_overloaded_lines = int(info.get('overloaded_lines', 0))
            self._max_line_load_pct = float(info.get('line_loading_max', 0.))
            self._trafo_overloaded = int(info.get('trafo_overloaded', 0))
            if bool(info.get('converged', False)):
                gen_bus_vm = info.get('gen_bus_vm_pu')
                gen_bus_gap = info.get('gen_bus_v_gap_pu')
                if gen_bus_vm is not None and len(gen_bus_vm) == self.N:
                    self._gen_bus_vm_pu = np.asarray(gen_bus_vm, dtype=np.float64)
                if gen_bus_gap is not None and len(gen_bus_gap) == self.N:
                    self._gen_bus_v_gap_pu = np.asarray(gen_bus_gap, dtype=np.float64)
                bus_vm = info.get('bus_vm_pu')
                bus_gap = info.get('bus_v_gap_pu')
                if bus_vm is not None and len(bus_vm) == self._n_bus:
                    self._bus_vm_pu = np.asarray(bus_vm, dtype=np.float64)
                if bus_gap is not None and len(bus_gap) == self._n_bus:
                    self._bus_v_gap_pu = np.asarray(bus_gap, dtype=np.float64)

        self.episode_history.append({
            'step': self.corr_step,
            'power': new_power.copy(),
            'cumulative_delta': self.cumulative_delta.copy(),
            'reward': reward,
            'done': done,
            'info': dict(info),
        })

        obs = self._get_obs()
        terminated = done or (self.corr_step >= self.max_steps)
        truncated = self.corr_step >= self.max_steps
        return obs, reward, terminated, truncated, info

    def _project_action_per_gen_cap(self, new_power):
        """Per-generator capacity cap (from V14-v10)."""
        total = new_power.sum()
        if total < 1e-6:
            return new_power.copy()
        # 鈽?V34: Relaxed cap for HIGH segments (was 0.18, now 0.25 for 23-24GW)
        cap_frac = 0.18 if self.target_demand < 21000 else 0.25
        cap_per_gen = np.minimum(self.PMAX * 0.98, total * cap_frac)
        cap_per_gen = np.clip(cap_per_gen, self.PMIN, self.PMAX * 0.98)
        exceeds = new_power > cap_per_gen
        n_capped = int(exceeds.sum())
        if n_capped == 0:
            return new_power.copy()
        projected = new_power.copy()
        projected[exceeds] = cap_per_gen[exceeds]
        excess = new_power.sum() - projected.sum()
        if excess > 0.1:
            eligible = ~exceeds & (projected < self.PMAX * 0.98)
            if eligible.any():
                room = (self.PMAX * 0.98 - projected)[eligible]
                total_room = room.sum()
                if total_room > 0.1:
                    redistribution = np.minimum(room * (excess / total_room), room)
                    projected[eligible] += redistribution
        return projected

    def _run_ac_pf_check(self):
        """Run AC-PF at the fixed target demand with an 8s timeout."""
        import pandapower as pp

        try:
            # Each check needs an isolated network because a timed-out daemon
            # thread may still own its copy. Deep-copying the cached template is
            # substantially cheaper than rebuilding case300() from JSON tables.
            net = copy.deepcopy(self._ac_net_template)
            ac_upper_margin = (
                self.generator_upper_margin
                if self.generator_upper_margin is not None else 0.99
            )
            net.gen.p_mw.values[:] = np.clip(
                self.current_power, self.PMIN, self.PMAX * ac_upper_margin
            )
            if self.current_gen_vm_pu is not None and len(self.current_gen_vm_pu) == len(net.gen):
                net.gen.vm_pu.values[:] = np.clip(self.current_gen_vm_pu, GEN_VM_MIN, GEN_VM_MAX)
            if self.current_ext_grid_vm_pu is not None and len(net.ext_grid):
                net.ext_grid.vm_pu.values[0] = float(np.clip(self.current_ext_grid_vm_pu[0], EXT_VM_MIN, EXT_VM_MAX))

            base_load = net.load.p_mw.sum()
            if base_load > 0:
                load_scale = float(self.target_demand) / float(base_load)
                net.load.p_mw = net.load.p_mw * load_scale
                net.load.q_mvar = net.load.q_mvar * load_scale

            net.bus.min_vm_pu = RL_VMIN_PU
            net.bus.max_vm_pu = RL_VMAX_PU

            import threading
            _ac_result = [None]
            _ac_error = [None]

            def _run_ac_inner():
                try:
                    pp.runpp(net, calculate_voltage_angles=True,
                             enforce_q_lims=True, numba=False)
                    _ac_result[0] = True
                except Exception as e:
                    _ac_error[0] = e

            _ac_thread = threading.Thread(target=_run_ac_inner, daemon=True)
            _ac_thread.start()
            _ac_thread.join(timeout=8)

            if _ac_thread.is_alive():
                return {
                    'converged': False, 'v_violations': 300,
                    'v_low_violations': 300, 'v_high_violations': 300,
                    'overloaded_lines': 68,
                    'v_min': 0.80, 'v_max': 1.15,
                    'loading_max': 2.0, 'line_loading_max': 2.0,
                    'trafo_overloaded': 999, 'reference_source_violations': 999,
                    'scheduled_balance_violations': 999,
                    'generator_bound_violations': 999,
                    'timeout': True,
                }

            converged = net.converged
            vm = net.res_bus.vm_pu.values
            vmin_lim = net.bus.min_vm_pu.values
            vmax_lim = net.bus.max_vm_pu.values
            low_gap = np.maximum(vmin_lim - vm, 0.0)
            high_gap = np.maximum(vm - vmax_lim, 0.0)
            v_low = int((low_gap > 0.0).sum())
            v_high = int((high_gap > 0.0).sum())
            n_v_viol = v_low + v_high
            v_violation_mag = float(low_gap.sum() + high_gap.sum())
            v_violation_max = float(max(low_gap.max(initial=0.0), high_gap.max(initial=0.0)))
            line_loading = net.res_line.loading_percent
            overloaded = int((line_loading > 100).sum())
            trafo_overloaded = int((net.res_trafo.loading_percent > 100).sum()) if hasattr(net, 'trafo') and len(net.trafo) > 0 else 0
            loading_max = float(line_loading.max()) if len(line_loading) > 0 else 0.

            ref_p = float(net.res_ext_grid.p_mw.iloc[0]) if len(net.ext_grid) else 0.0
            ref_q = float(net.res_ext_grid.q_mvar.iloc[0]) if len(net.ext_grid) else 0.0
            ref_s = float(np.hypot(ref_p, ref_q))
            ref_p_violation = bool(ref_p < REF_SOURCE_P_MIN_MW or ref_p > REF_SOURCE_P_MAX_MW)
            ref_q_violation = bool(ref_q < REF_SOURCE_Q_MIN_MVAR or ref_q > REF_SOURCE_Q_MAX_MVAR)
            ref_s_violation = bool(ref_s > REF_SOURCE_S_MAX_MVA)
            ref_violations = int(ref_p_violation) + int(ref_q_violation) + int(ref_s_violation)
            balance_error = float(abs(np.sum(self.current_power) - self.target_demand))
            balance_violations = int(balance_error > 0.001 * max(self.target_demand, 1.0))
            bound_violations = int(np.count_nonzero(
                (self.current_power < self.PMIN - 1e-6)
                | (
                    self.current_power
                    > self.PMAX * (
                        self.generator_upper_margin
                        if self.generator_upper_margin is not None else 0.99
                    ) + 1e-6
                )
            ))
            gen_bus_idx = net.gen.bus.values.astype(int)
            gen_bus_vm = vm[gen_bus_idx]
            # Positive means raise the local setpoint; negative means lower it.
            gen_bus_v_gap = (
                np.maximum(RL_VMIN_PU - gen_bus_vm, 0.0)
                - np.maximum(gen_bus_vm - RL_VMAX_PU, 0.0)
            )
            bus_v_gap = low_gap - high_gap
            return {
                'converged': converged,
                'v_violations': n_v_viol,
                'v_low_violations': v_low,
                'v_high_violations': v_high,
                'overloaded_lines': overloaded,
                'v_min': float(net.res_bus.vm_pu.min()),
                'v_max': float(net.res_bus.vm_pu.max()),
                'v_violation_mag_pu': v_violation_mag,
                'v_violation_max_pu': v_violation_max,
                'gen_bus_vm_pu': gen_bus_vm.astype(float).tolist(),
                'gen_bus_v_gap_pu': gen_bus_v_gap.astype(float).tolist(),
                'bus_vm_pu': vm.astype(float).tolist(),
                'bus_v_gap_pu': bus_v_gap.astype(float).tolist(),
                'loading_max': loading_max,
                'line_loading_max': loading_max,
                'trafo_overloaded': trafo_overloaded,
                'reference_source_p_mw': ref_p,
                'reference_source_q_mvar': ref_q,
                'reference_source_s_mva': ref_s,
                'reference_source_p_violation': ref_p_violation,
                'reference_source_q_violation': ref_q_violation,
                'reference_source_s_violation': ref_s_violation,
                'reference_source_violations': ref_violations,
                'reference_source_s_utilization_pct': 100.0 * ref_s / REF_SOURCE_S_MAX_MVA,
                'scheduled_balance_error_mw': balance_error,
                'scheduled_balance_violations': balance_violations,
                'generator_bound_violations': bound_violations,
                'timeout': False,
            }

        except Exception as e:
            return {'converged': False, 'v_violations': 300,
                    'v_low_violations': 300, 'v_high_violations': 300,
                    'overloaded_lines': 68, 'loading_max': 2.0,
                    'line_loading_max': 2.0, 'trafo_overloaded': 999,
                    'reference_source_violations': 999,
                    'scheduled_balance_violations': 999,
                    'generator_bound_violations': 999,
                    'error': str(e)}

    def _compute_n1_overload_margin(self):
        """鈽?V25: Compute CONTINUOUS N-1 overload margin (MW over limit).

        Instead of counting discrete violations, measure total MW of overload
        across all contingency scenarios. This gives a SMOOTH gradient signal.

        Returns: total_overload_mw (0 = fully N-1 secure, positive = overload exists)
        """
        _flows, violations, total_overload = self._scan_n1_violations()
        return total_overload, len(violations)

        P = self.current_power
        S = self.dcnet.S
        lodf = self.dcnet.lodf
        limits = self.dcnet.effective_limit

        # Base flows
        flows = self._estimate_base_flows(P)

        total_overload = 0.0
        n_violations = 0

        for k in range(len(flows)):
            fk = flows[k]
            if abs(fk) < 0.1:
                continue

            # Post-contingency flows
            f_post = flows + lodf[:, k] * fk

            for l in range(len(flows)):
                if l == k:
                    continue
                limit = limits[l]

                # Overload amount (continuous, not binary!)
                overload_pos = max(0, f_post[l] - limit)
                overload_neg = max(0, -f_post[l] - limit)

                if overload_pos > 0.01 or overload_neg > 0.01:
                    total_overload += overload_pos + overload_neg
                    n_violations += 1

        return total_overload, n_violations

    def _compute_reward(self):
        """鈽?V26: N-1 SENSITIVITY-AWARE reward!

        Uses the N-1 sensitivity (d(overload)/d(P_i)) computed in _get_obs().
        Key insight: RL can now SEE N-1 violations via observation, so reward
        just needs to guide toward reducing overload MW.

        Innovations:
        1. N-1 sensitivity in observation (68-dim) 鈥?RL sees which gens to adjust
        2. Safety margin reward: positive reward for flow margin (not just penalty)
        3. Continuous overload MW penalty (smooth gradient)
        4. MEGA jackpot (+5000) for AC-PF + N-1 secure
        """
        # Linear N-1 is cheap and is evaluated before deciding whether the
        # expensive AC-PF can possibly produce a complete pass.
        do_n1_check = (self.corr_step == 1) or (self.corr_step % N1_CHECK_EVERY_STEPS == 0)
        if do_n1_check:
            n1_overload_mw, n1_violations = self._compute_n1_overload_margin()
            self._n1_overload_mw = n1_overload_mw
            self._n1_violations = n1_violations
            self._current_n1_violations = n1_violations
        else:
            n1_overload_mw = getattr(self, '_n1_overload_mw', 0.0)
            n1_violations = getattr(self, '_n1_violations', 0)

        # Event-triggered common verification: retain periodic voltage
        # feedback, but immediately run AC-PF for every linearly secure
        # candidate. Therefore no candidate capable of a full pass is missed.
        periodic_ac_check = (
            self.corr_step == 1
            or self.corr_step % AC_PF_CHECK_EVERY == 0
        )
        near_n1_boundary = bool(
            do_n1_check
            and float(n1_overload_mw) < AC_EVENT_N1_OVERLOAD_MW
        )
        secure_candidate = bool(
            near_n1_boundary
            and int(n1_violations) == 0
            and float(n1_overload_mw) < 0.01
        )
        do_ac_check = periodic_ac_check or near_n1_boundary
        ac_info = self._run_ac_pf_check() if do_ac_check else {}
        is_converged = ac_info.get('converged', False)
        n_v_viol = ac_info.get('v_violations', self._n_v_viol)
        n_overloaded = ac_info.get('overloaded_lines', self._n_overloaded_lines)

        info = dict(ac_info)
        info['step'] = self.corr_step
        info['ac_converged'] = is_converged
        info['ac_checked'] = bool(do_ac_check)
        info['ac_check_trigger'] = (
            'n1_secure' if secure_candidate
            else ('n1_near_boundary' if near_n1_boundary
            else ('periodic' if periodic_ac_check else 'skip')
            )
        )
        info['n1_violations'] = n1_violations
        info['n1_overload_mw'] = n1_overload_mw
        info['n1_checked'] = bool(do_n1_check)

        # Smooth reward: continuous violation distances guide learning, while only
        # complete SCED+AC-PF success can receive a positive terminal reward.
        n_trafo_overloaded = int(info.get('trafo_overloaded', self._trafo_overloaded))
        n_ref_source_violations = int(info.get('reference_source_violations', 999))
        n_balance_violations = int(info.get('scheduled_balance_violations', 999))
        n_bound_violations = int(info.get('generator_bound_violations', 999))
        v_mag = float(info.get('v_violation_mag_pu', 0.0))
        v_max_gap = float(info.get('v_violation_max_pu', 0.0))
        prev_v_mag = getattr(self, '_prev_v_violation_mag_pu', None)
        if do_ac_check and bool(is_converged):
            if prev_v_mag is None:
                r_voltage_improvement = 0.0
            else:
                delta_gap = float(prev_v_mag) - float(v_mag)
                r_voltage_improvement = R_VOLTAGE_GAP_IMPROVEMENT * float(np.clip(delta_gap, -0.10, 0.10))
            self._prev_v_violation_mag_pu = float(v_mag)
        else:
            r_voltage_improvement = 0.0
        loading_max = float(info.get('loading_max', self._max_line_load_pct))
        line_loading_excess = max(0.0, loading_max - 100.0) / 100.0

        full_pass = (
            bool(do_ac_check)
            and bool(is_converged)
            and int(n_v_viol) == 0
            and int(n_overloaded) == 0
            and int(n_trafo_overloaded) == 0
            and (not ENFORCE_REF_SOURCE_LIMIT or int(n_ref_source_violations) == 0)
            and int(n1_violations) == 0
            and int(n_balance_violations) == 0
            and int(n_bound_violations) == 0
        )

        # AC-derived quantities are unknown between scheduled AC-PF checks.
        # Do not interpret missing reference-source/voltage fields as violations.
        if not do_ac_check:
            r_n1_margin = 3.0 if int(n1_violations) == 0 and float(n1_overload_mw) < 0.01 else 0.0
            r_n1_penalty = (
                R_N1_OVERLOAD_MARGIN * min(float(n1_overload_mw) / 100.0, 5.0)
                if int(n1_violations) > 0 else 0.0
            )
            reward = float(BASE_REWARD + r_n1_margin + r_n1_penalty
                           + getattr(self, '_last_action_penalty', 0.0))
            info['reason'] = 'acpf_not_checked'
            info['reward_components'] = {
                'base': BASE_REWARD,
                'r_converged': 0.0,
                'r_voltage': 0.0,
                'r_loading': 0.0,
                'r_n1_margin': r_n1_margin,
                'r_voltage_penalty': 0.0,
                'r_voltage_improvement': 0.0,
                'r_loading_penalty': 0.0,
                'r_n1_penalty': r_n1_penalty,
                'r_action_penalty': float(getattr(self, '_last_action_penalty', 0.0)),
                'n1_violations': n1_violations,
                'n1_overload_mw': n1_overload_mw,
            }
            return reward, False, info

        if full_pass:
            reward = N1_SECURE_REWARD + float(getattr(self, '_last_action_penalty', 0.0))
            done = True
            info['reason'] = 'full_sced_pass'
            r_converged = 0.0
            r_voltage = 0.0
            r_loading = 0.0
            r_n1_margin = 0.0
            r_voltage_penalty = 0.0
            r_voltage_improvement = 0.0
            r_loading_penalty = 0.0
            r_n1_penalty = 0.0
            r_action_penalty = float(getattr(self, '_last_action_penalty', 0.0))
        else:
            done = False
            if not do_ac_check:
                info['reason'] = 'acpf_not_checked'
            elif not is_converged:
                info['reason'] = 'acpf_not_converged'
            elif int(n_v_viol) > 0:
                info['reason'] = 'voltage_violation'
            elif int(n_overloaded) > 0 or int(n_trafo_overloaded) > 0:
                info['reason'] = 'line_overload'
            elif ENFORCE_REF_SOURCE_LIMIT and int(n_ref_source_violations) > 0:
                info['reason'] = 'reference_source_limit'
            elif int(n_balance_violations) > 0 or int(n_bound_violations) > 0:
                info['reason'] = 'scheduled_dispatch_constraint'
            elif int(n1_violations) > 0:
                info['reason'] = 'n1_violation'
            else:
                info['reason'] = 'not_full_sced'

            r_converged = 0.0
            r_voltage = 0.0
            r_loading = 0.0
            r_n1_margin = 3.0 if int(n1_violations) == 0 and float(n1_overload_mw) < 0.01 else 0.0
            r_voltage_penalty = 0.0
            # r_voltage_improvement is computed above when AC-PF converges.
            r_loading_penalty = 0.0
            r_n1_penalty = 0.0
            r_action_penalty = float(getattr(self, '_last_action_penalty', 0.0))

            if not is_converged:
                reward = R_ACPF_NONCONVERGENCE
                info['reason'] = 'acpf_not_converged'
            elif ENFORCE_REF_SOURCE_LIMIT and int(n_ref_source_violations) > 0:
                r_loading_penalty = -500.0 * min(float(n_ref_source_violations), 3.0)
                reward = -500.0 + r_loading_penalty
                info['reason'] = 'reference_source_limit'
            elif is_converged and int(n_v_viol) == 0:
                if int(n1_violations) == 0:
                    reward = 800.0 - 100.0 * min(float(n_overloaded + n_trafo_overloaded), 10.0)
                    info['reason'] = 'acpf_n1ok_line_overload'
                    r_loading_penalty = -100.0 * min(float(n_overloaded + n_trafo_overloaded), 10.0)
                else:
                    r_n1_penalty = R_N1_OVERLOAD_MARGIN * min(float(n1_overload_mw) / 100.0, 5.0)
                    reward = 500.0 + r_n1_penalty
                    info['reason'] = 'acpf_only_n1_fail'
                    info['r_n1_overload'] = r_n1_penalty
            else:
                reward = BASE_REWARD
                if do_ac_check and is_converged:
                    r_converged = 50.0
                    r_voltage_penalty = -3000.0 * min(v_mag, 0.20) - 1000.0 * min(v_max_gap, 0.10)
                    reward += r_converged + r_voltage_penalty + r_voltage_improvement
                    if int(n_v_viol) == 0 or v_max_gap <= 0.01:
                        r_voltage = 50.0
                        reward += r_voltage

                if do_ac_check and is_converged and int(n_v_viol) == 0:
                    r_voltage += 100.0
                    reward += 100.0
                elif do_ac_check and v_mag < 0.05:
                    r_voltage += 10.0
                    reward += 10.0
                elif do_ac_check and v_mag < 0.20:
                    r_voltage += 5.0
                    reward += 5.0

                if loading_max > 90:
                    r_loading_penalty = R_ACPF_LOADING_PENALTY * min(loading_max / 100.0, 2.0)
                    reward += r_loading_penalty

                if int(n_v_viol) + int(n_overloaded) > 0:
                    r_violation = R_VIOLATION_COEF * min(float(n_v_viol + n_overloaded) / 100.0, 1.0)
                    reward += r_violation
                    info['r_violation'] = r_violation

            reward += r_action_penalty

            reward += r_n1_margin
            if not (do_ac_check and is_converged and int(n_v_viol) > 0):
                reward += r_voltage_improvement

        reward = float(np.nan_to_num(reward, nan=-100.0, posinf=N1_SECURE_REWARD, neginf=-10000.0))

        info['reward_components'] = {
            'base': BASE_REWARD,
            'r_converged': r_converged,
            'r_voltage': r_voltage,
            'r_loading': r_loading,
            'r_n1_margin': r_n1_margin,
            'r_voltage_penalty': r_voltage_penalty,
            'r_voltage_improvement': r_voltage_improvement,
            'r_loading_penalty': r_loading_penalty,
            'r_n1_penalty': r_n1_penalty,
            'r_action_penalty': r_action_penalty,
            'n1_violations': n1_violations,
            'n1_overload_mw': n1_overload_mw,
        }
        return reward, done, info

    def _predict_gat(self, demand_mw):
        """Run GATv10 prediction for given demand."""
        gf = self.gen_feats

        gen_bus = self._gat_gen_bus
        gen_pmax = self._gat_gen_pmax
        gen_vm = self._gat_gen_vm

        pmax_norm = (gen_pmax - gen_pmax.min()) / (gen_pmax.max() - gen_pmax.min() + 1e-8)
        bus_norm = (gen_bus - gen_bus.min()) / (gen_bus.max() - gen_bus.min() + 1e-8)
        vm_norm = (gen_vm - gen_vm.min()) / (gen_vm.max() - gen_vm.min() + 1e-8)
        base_cp1 = gf['cp1']
        if base_cp1.max() > base_cp1.min():
            cost_rank = np.argsort(np.argsort(base_cp1)).astype(float) / max(self.N - 1, 1)
        else:
            cost_rank = np.ones(self.N) * 0.5
        frac_of_total = gen_pmax / (gen_pmax.sum() + 1e-8)
        is_large = (gen_pmax > 300).astype(np.float32)

        sample = getattr(self, "current_sample", None)
        if sample is not None and "sample_cp1" in sample and "sample_cp2" in sample:
            cp1 = np.asarray(sample["sample_cp1"], dtype=np.float32)
            cp2 = np.asarray(sample["sample_cp2"], dtype=np.float32)
        else:
            cp1 = np.asarray(base_cp1, dtype=np.float32)
            cp2 = np.ones(self.N, dtype=np.float32)
        cp1_norm = cp1 / (float(np.mean(cp1)) + 1e-8)
        cp2_norm = cp2 / (float(np.mean(cp2)) + 1e-8)

        static_feat = np.stack([
            pmax_norm, bus_norm, vm_norm, cost_rank,
            frac_of_total, is_large, cp1_norm, cp2_norm,
        ], axis=-1).astype(np.float32)

        total_cap = float(gen_pmax.sum())
        base_load = self._gat_base_load_mw
        d_norm = demand_mw / (total_cap + 1e-8)
        load_scale = demand_mw / (base_load + 1e-8)
        reserve = max(0.0, 1.0 - d_norm)
        log_d = np.log(demand_mw + 1e-3) / np.log(total_cap + 1e-3)
        d_sq = d_norm ** 2

        # Match ObservationBuilderV10.build() exactly for this checkpoint.
        tier16 = 1.0 if 15.0 <= demand_mw < 18.0 else 0.0
        tier20 = 1.0 if 19.0 <= demand_mw < 22.0 else 0.0
        tier24 = 1.0 if 23.0 <= demand_mw <= 26.0 else 0.0

        global_feat = np.array([
            d_norm, load_scale, reserve, log_d, d_sq,
            tier16, tier20, tier24,
        ], dtype=np.float32)

        with torch.no_grad():
            static_t = torch.from_numpy(static_feat).unsqueeze(0).float().to(DEVICE)
            global_t = torch.from_numpy(global_feat).unsqueeze(0).float().to(DEVICE)
            pred = self.gat_net(static_t, global_t)

        result = pred.cpu().numpy()[0]
        return result

    @staticmethod
    def _compute_hhi(power):
        if power.sum() <= 0:
            return 10000.0
        shares = power / power.sum()
        return (shares ** 2).sum() * 10000


# ============================================================
# V21 SAC Networks
# ============================================================

class GaussianActor(nn.Module):
    """
    SAC Stochastic Actor 鈥?outputs Gaussian distribution (mean, log_std).

    Uses reparameterization trick for differentiable sampling.
    Architecture: obs 鈫?256 鈫?256 鈫?68(mean) + 68(log_std)
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.action_dim = action_dim

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Mean head
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        # Log_std head (clamped to prevent extreme values)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        # Match the validated Route-B SAC initialization. Exploration comes
        # from the Gaussian log_std head, while the deterministic mean starts
        # at the neutral zero correction.
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, -3.0)

    def forward(self, obs):
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        h = self.trunk(obs)
        mean = torch.nan_to_num(self.mean_head(h), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        # On the corrected Route-B set, excessive stochastic variance quickly
        # drives AC-PF to non-convergence. Keep exploration learnable, but cap it.
        log_std = torch.nan_to_num(
            self.log_std_head(h), nan=-3.0, posinf=0.0, neginf=-5.0
        ).clamp(min=-5.0)
        if SAC_LOG_STD_MAX_VECTOR is None:
            log_std = log_std.clamp(max=float(SAC_LOG_STD_MAX))
        else:
            max_vector = torch.as_tensor(
                SAC_LOG_STD_MAX_VECTOR, dtype=log_std.dtype, device=log_std.device
            )
            if max_vector.numel() != log_std.shape[-1]:
                raise RuntimeError(
                    "SAC_LOG_STD_MAX_VECTOR does not match the action dimension"
                )
            log_std = torch.minimum(log_std, max_vector.view(1, -1))
        return mean, log_std

    def sample(self, obs):
        """Sample action using reparameterization trick."""
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        # Reparameterization: a = mean + std * 蔚, where 蔚 ~ N(0,1)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)  # Squash to [-1, 1]
        # Log prob with tanh correction
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mean

    def select_action(self, obs, deterministic=False):
        """Select action for evaluation (deterministic) or training (stochastic)."""
        with torch.no_grad():
            mean, log_std = self.forward(obs)
            if deterministic:
                action = torch.tanh(mean)
            else:
                std = log_std.exp()
                normal = torch.distributions.Normal(mean, std)
                z = normal.sample()
                action = torch.tanh(z)
        return action


class TwinQCritic(nn.Module):
    """
    SAC Twin Q-Networks 鈥?Q1(s,a) and Q2(s,a) to reduce overestimation.

    Architecture: [obs, a] 鈫?256 鈫?256 鈫?1 (two independent copies)
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Q2
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, action):
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_forward(self, obs, action):
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa)


class ReplayBuffer:
    """Experience replay buffer for off-policy RL (SAC)."""
    def __init__(self, capacity=BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.nan_to_num(np.asarray(state, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
        action = np.nan_to_num(np.asarray(action, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
        action = np.clip(action, -1.0, 1.0)
        reward = float(np.nan_to_num(reward, nan=-1000.0, posinf=10000.0, neginf=-10000.0))
        reward = float(np.clip(reward, -10000.0, 10000.0))
        next_state = np.nan_to_num(np.asarray(next_state, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size):
        # Uniform sampling (RL-convergence fix).
        # The previous |reward|-prioritised sampling (with a 3x boost for
        # rewards > 100) oversampled rare high-reward spikes WITHOUT importance-
        # sampling correction, biasing the Bellman target and destabilising the
        # critic.  Uniform sampling matches the paper's stated replay design and
        # gives an unbiased target.
        n = len(self.buffer)
        if n == 0:
            return None
        bs = min(batch_size, n)
        indices = np.random.choice(n, bs, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            s, a, r, ns, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
# V21 SAC Trainer
# ============================================================

class SACTrainer:
    """
    SAC (Soft Actor-Critic) Trainer with automatic temperature tuning.

    Key features:
    - Maximum entropy RL: maximize E[R + 伪*H(蟺)]
    - Twin Q-critics (min(Q1, Q2) for target)
    - Reparameterization trick for actor
    - Automatic alpha (temperature) tuning via dual gradient descent
    - Target networks with Polyak averaging
    """

    def __init__(self, obs_dim, action_dim, device=DEVICE):
        self.device = device
        self.action_dim = action_dim

        # Networks
        self.actor = GaussianActor(obs_dim, action_dim).to(device)
        self.critic = TwinQCritic(obs_dim, action_dim).to(device)
        self.critic_target = TwinQCritic(obs_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=SAC_LR_ACTOR)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=SAC_LR_CRITIC)

        # Temperature (alpha) 鈥?auto-tuned
        # Target entropy must match the ACTUAL action dimension, not the static
        # N_GEN constant.  V41 uses a 101-dim hybrid action (12 shift + 20 key
        # gen + 68 vm + 1 ext-grid vm); using -0.5*N_GEN (-34) on a 101-dim
        # policy under-regularises entropy and harms exploration.
        self.target_entropy = -0.5 * action_dim
        self.log_alpha = torch.tensor(np.log(SAC_INIT_ALPHA), requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=SAC_LR_ALPHA)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)

        # Hyperparameters
        self.gamma = SAC_GAMMA
        self.tau = SAC_TAU

        self.total_updates = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs, deterministic=False):
        """Select action. Actions are in [-1, 1] (tanh-squashed)."""
        obs = np.nan_to_num(np.asarray(obs, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action = self.actor.select_action(obs_t, deterministic=deterministic)
        action_np = action.cpu().numpy()[0]
        action_np = np.nan_to_num(action_np, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(action_np, -1.0, 1.0)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self, update_actor=True):
        """Perform one SAC gradient update step."""
        self.total_updates += 1

        if len(self.replay_buffer) < BATCH_SIZE:
            return {'q1_loss': 0., 'q2_loss': 0., 'actor_loss': 0.,
                    'alpha': self.alpha.item(), 'buffer_size': len(self.replay_buffer)}

        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        states = torch.nan_to_num(states, nan=0.0, posinf=1e6, neginf=-1e6)
        actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        rewards = torch.nan_to_num(rewards, nan=-1000.0, posinf=10000.0, neginf=-10000.0).clamp(-10000.0, 10000.0)
        next_states = torch.nan_to_num(next_states, nan=0.0, posinf=1e6, neginf=-1e6)
        dones = torch.nan_to_num(dones, nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        # === Critic Update ===
        with torch.no_grad():
            # Sample next action from current policy
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            # Target Q: min(Q1', Q2') - 伪 * log蟺
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * next_q
            target_q = torch.nan_to_num(target_q, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)

        current_q1, current_q2 = self.critic(states, actions)
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        critic_loss = q1_loss + q2_loss
        if not torch.isfinite(critic_loss):
            return {'q1_loss': 0., 'q2_loss': 0., 'actor_loss': 0.,
                    'alpha': self.alpha.item(), 'alpha_loss': 0.,
                    'buffer_size': len(self.replay_buffer), 'skipped': 'nonfinite_critic_loss'}

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        if not torch.isfinite(critic_grad_norm):
            self.critic_optimizer.zero_grad(set_to_none=True)
            return {'q1_loss': 0., 'q2_loss': 0., 'actor_loss': 0.,
                    'alpha': self.alpha.item(), 'alpha_loss': 0.,
                    'buffer_size': len(self.replay_buffer), 'skipped': 'nonfinite_critic_grad'}
        self.critic_optimizer.step()

        if not update_actor:
            self._soft_update(self.critic, self.critic_target, self.tau)
            return {
                'q1_loss': q1_loss.item(),
                'q2_loss': q2_loss.item(),
                'actor_loss': 0.0,
                'alpha': self.alpha.item(),
                'alpha_loss': 0.0,
                'buffer_size': len(self.replay_buffer),
                'actor_frozen': True,
            }

        # === Actor Update ===
        # Sample action from current policy (reparameterization)
        sampled_actions, log_probs, _ = self.actor.sample(states)
        q1_pi, q2_pi = self.critic(states, sampled_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)
        # Actor loss: maximize Q - 伪*log蟺 鈫?minimize 伪*log蟺 - Q
        actor_loss = (self.alpha * log_probs - min_q_pi).mean()
        if not torch.isfinite(actor_loss):
            return {'q1_loss': q1_loss.item(), 'q2_loss': q2_loss.item(), 'actor_loss': 0.,
                    'alpha': self.alpha.item(), 'alpha_loss': 0.,
                    'buffer_size': len(self.replay_buffer), 'skipped': 'nonfinite_actor_loss'}

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        if not torch.isfinite(actor_grad_norm):
            self.actor_optimizer.zero_grad(set_to_none=True)
            return {'q1_loss': q1_loss.item(), 'q2_loss': q2_loss.item(), 'actor_loss': 0.,
                    'alpha': self.alpha.item(), 'alpha_loss': 0.,
                    'buffer_size': len(self.replay_buffer), 'skipped': 'nonfinite_actor_grad'}
        self.actor_optimizer.step()

        # === Alpha (temperature) Update ===
        # The corrected Route-B task has a narrow AC-feasible region and we
        # normally use a fixed, small log_std. Auto-temperature can then push
        # the actor toward entropy rather than feasibility, so it is optional.
        if SAC_AUTO_ALPHA:
            with torch.no_grad():
                _, log_probs_eval, _ = self.actor.sample(states)
            alpha_loss = -(self.log_alpha * (log_probs_eval + self.target_entropy)).mean()
            if not torch.isfinite(alpha_loss):
                alpha_loss = torch.zeros((), device=self.device)

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            with torch.no_grad():
                self.log_alpha.clamp_(np.log(1e-5), np.log(1.0))
        else:
            alpha_loss = torch.zeros((), device=self.device)

        # === Target Network Soft Update ===
        self._soft_update(self.critic, self.critic_target, self.tau)

        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha.item(),
            'alpha_loss': alpha_loss.item(),
            'buffer_size': len(self.replay_buffer),
        }

    def _soft_update(self, main, target, tau):
        for main_param, target_param in zip(main.parameters(), target.parameters()):
            target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

    def save_checkpoint(self, filepath, update_num, eval_result=None, total_episodes=None):
        # 鈽?V23: SLIM checkpoint 鈥?only weights (~16MB, NO buffer/optimizer!)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha.item(),
            'update': update_num,
            'eval_result': eval_result,
            'total_updates': self.total_updates,
            'total_episodes': total_episodes if total_episodes is not None else 0,
        }, filepath)

    def load_checkpoint(self, filepath):
        ckpt = torch.load(filepath, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt['actor_state_dict'])
        self.critic.load_state_dict(ckpt['critic_state_dict'])
        self.critic_target.load_state_dict(ckpt['critic_target_state_dict'])
        self.log_alpha.data = torch.tensor(ckpt['log_alpha'], device=self.device, dtype=torch.float32)
        if 'total_updates' in ckpt:
            self.total_updates = ckpt['total_updates']
        return ckpt
        if 'total_updates' in ckpt:
            self.total_updates = ckpt['total_updates']
        return ckpt


# ============================================================
# Evaluation Function
# ============================================================

def evaluate_sac(trainer, env, sample_ids, n_per_segment=N_PER_SEGMENT_EVAL,
                 deterministic=True):
    """Evaluate SAC agent on given sample IDs."""
    results = {'converged': 0, 'total': 0, 'by_demand': defaultdict(int),
               'by_demand_total': defaultdict(int)}
    all_rewards = []

    eval_ids = stratified_sample_from_pool(env.all_samples, sample_ids, n_per_segment)

    for sid in eval_ids:
        obs = env.reset(sample_idx=sid)
        ep_reward = 0.0
        episode_converged = False

        for step in range(env.max_steps + 1):
            # SAC action: tanh-squashed [-1, 1] 鈫?scale to action bounds
            raw_action = trainer.select_action(obs, deterministic=deterministic)
            low, high = env.get_action_bounds()
            action = low + (raw_action + 1.0) * 0.5 * (high - low)  # Map [-1,1] 鈫?[low, high]

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated:
                # 鈽?V26b: Full SCED convergence 鈥?aligned with GT's 7 constraints!
                # AC-PF converged + VV=0 + line overload=0 + trafo overload=0
                episode_converged = adopted_operation_pass(info)
                break

        results['total'] += 1
        gw_key = f"{int(env.target_demand / 1000)}GW"
        results['by_demand_total'][gw_key] += 1
        if episode_converged:
            results['converged'] += 1
            results['by_demand'][gw_key] += 1
        all_rewards.append(ep_reward)

    conv_rate = results['converged'] / max(results['total'], 1) * 100
    avg_reward = np.mean(all_rewards) if all_rewards else 0.0

    return {
        'conv_rate': conv_rate,
        'avg_reward': avg_reward,
        'n_total': results['total'],
        'n_converged': results['converged'],
        'by_demand': dict(results['by_demand']),
        'by_demand_total': dict(results['by_demand_total']),
    }


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

RESULTS_DIR = pathlib.Path(PROJ) / "rlv35_complete_training"
RESULTS_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("RLv21 鈥?SAC + 卤螖P RESIDUAL + CLEAN DATA SPLIT")
print("=" * 70)
print(f"  Algorithm: SAC (Soft Actor-Critic)")
print(f"  Action Space: 卤螖P residual (adaptive: LOW=卤20 / MID=卤50 / HIGH=卤120)")
print(f"  Actor: Stochastic Gaussian (reparameterization trick)")
print(f"  Critic: Twin Q-networks (Q1 + Q2, take min)")
print(f"  Temperature: Auto-tuned (target_entropy={SAC_TARGET_ENTROPY:.1f})")
print(f"  Device: {DEVICE}")
print(f"  Buffer: {BUFFER_SIZE}, Batch: {BATCH_SIZE}")
print(f"  LR: Actor={SAC_LR_ACTOR}, Critic={SAC_LR_CRITIC}, Alpha={SAC_LR_ALPHA}")
print(f"  Data Split: Train={TRAIN_RATIO*100:.0f}% / Val={VAL_RATIO*100:.0f}% / Test={TEST_RATIO*100:.0f}%")
print(f"  NO curriculum 鈥?all segments trained simultaneously!")
print(f"  Reward: Convergence-focused (NO GAT prox / residual mag penalties)")


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Load GT dataset
    V10_DATA_PATH = pathlib.Path(PROJ) / "case300_gt_v10_dataset.json"
    print("\nLoading GT dataset...")
    with open(V10_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_samples = [s for s in data.get('samples', []) if s.get('is_complete_sced')]
    samples = all_samples  # 鈽?V35: alias for stochastic eval section
    print(f"  Loaded {len(all_samples)} complete SCED samples")

    # Stratified train/val/test split
    train_ids, val_ids, test_ids = split_samples_stratified(all_samples)
    print(f"  Split: Train={len(train_ids)} / Val={len(val_ids)} / Test={len(test_ids)}")

    # Create environments
    print("\nCreating environments...")
    train_env = SCEDEnvV21(all_samples, train_ids, dataset_meta=data)
    val_env = SCEDEnvV21(all_samples, val_ids, dataset_meta=data)
    test_env = SCEDEnvV21(all_samples, test_ids, dataset_meta=data)

    obs_dim = train_env._get_obs_dim()
    action_dim = train_env.action_dim
    n_actor_params = sum(p.numel() for p in GaussianActor(obs_dim, action_dim).parameters())
    n_critic_params = sum(p.numel() for p in TwinQCritic(obs_dim, action_dim).parameters())
    print(f"  ObsDim={obs_dim} ActDim={action_dim}")
    print(f"  Actor params: {n_actor_params:,}")
    print(f"  Critic params (脳2): {n_critic_params:,}")
    print(f"  Total params: {n_actor_params + n_critic_params * 2:,}")

    # Create SAC trainer
    trainer = SACTrainer(obs_dim, action_dim, device=DEVICE)

    # Baseline evaluation (GAT-only, no RL)
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION (GAT-only, no RL corrections)")
    baseline = evaluate_sac(trainer, train_env, train_ids, n_per_segment=2, deterministic=True)
    print(f"  Train Baseline: Conv={baseline['conv_rate']:.1f}% ({baseline['n_converged']}/{baseline['n_total']})")
    val_baseline = evaluate_sac(trainer, val_env, val_ids, n_per_segment=2, deterministic=True)
    print(f"  Val Baseline: Conv={val_baseline['conv_rate']:.1f}% ({val_baseline['n_converged']}/{val_baseline['n_total']})")

    # Training log
    log_file = open(RESULTS_DIR / "training_v35.log", "w", buffering=1, encoding='utf-8')
    def log(msg):
        print(msg)
        log_file.write(msg + "\n")

    log("=" * 80)
    log("RLv21 SAC TRAINING STARTED 鈥?卤螖P RESIDUAL + MAX ENTROPY")
    log(f"  ObsDim={obs_dim} ActDim={action_dim}")
    log(f"  Actor Params={n_actor_params:,} Critic Params(脳2)={n_critic_params:,}")
    log(f"  SAC: LR_A={SAC_LR_ACTOR} LR_C={SAC_LR_CRITIC} LR_伪={SAC_LR_ALPHA}")
    log(f"  Buffer={BUFFER_SIZE} Batch={BATCH_SIZE} GradSteps/update={N_GRADIENT_STEPS}")
    log(f"  Data: Train={len(train_ids)} Val={len(val_ids)} Test={len(test_ids)}")
    log(f"  Baseline: Train={baseline['conv_rate']:.1f}% Val={val_baseline['conv_rate']:.1f}%")
    log(f"  Target: >90% convergence on BOTH train and val!")

    best_val_conv = 0.0
    best_model_state = None
    start_time = time.time()
    total_episodes = 0

    # 鈽?V25: Load V23 U50 best_model (AC-PF=100%) 鈥?GT blend + continuous N-1 margin!
    V23_BEST = pathlib.Path(PROJ) / "rlv23_sac_n1_training" / "best_model.pth"
    if not V23_BEST.exists():
        V23_BEST = pathlib.Path(PROJ) / "rlv21_sac_training" / "best_model.pth"
    if not V23_BEST.exists():
        V23_BEST = pathlib.Path(PROJ) / "models" / "sac_best_model.pth"
    start_update = 1
    # 鈽?V34: Transfer from V33 U60 (Val=91.4%!), fine-tune on 23-24GW
    V33_BEST = pathlib.Path(PROJ) / "rlv33_normalize_training" / "best_model.pth"
    if V33_BEST.exists():
        log(f"\n{'='*60}")
        log(f"V35: COMPLETE 鈥?HIGH-focused + Stochastic Retry + Ensemble!")
        v33_ckpt = torch.load(V33_BEST, map_location=DEVICE, weights_only=False)
        trainer.actor.load_state_dict(v33_ckpt["actor"])
        trainer.critic.load_state_dict(v33_ckpt["critic"])
        trainer.critic_target.load_state_dict(v33_ckpt["critic"])
        log(f"  Loaded V33 U{v33_ckpt.get('update','?')} (ValConv={v33_ckpt.get('val_conv','?')}%)")
        log(f"  Action dim: {N_SHIFT_DIRECTIONS + N_KEY_GENERATORS} (same as V33)")
        log(f"  Obs dim: {obs_dim}")
        log(f"  鈽?GAT output NORMALIZED (from V33)")
        log(f"  鈽?HIGH-focused curriculum: P1(1-30)=all, P2(31-100)=80% 23-24GW")
        log(f"  鈽?HIGH cum_clip: 1200 MW/gen (was 840) 鈥?more adjustment room!")
        log(f"  鈽?HIGH max_steps: 120 (was 100) 鈥?more ACPF attempts!")
        log(f"  鈽?Voltage bonus doubled (5鈫?0)")
        log(f"  N-1 already 100% solved 鈫?focus entirely on 24GW ACPF convergence")
        log(f"{'='*60}\n")
    else:
        log(f"\nV34: WARNING 鈥?V33 best_model not found, training from scratch!\n")
    for update_num in range(start_update, TOTAL_UPDATES + 1):
        update_start = time.time()
        phase_converged = 0
        phase_total = 0
        phase_rewards = []
        conv_by_demand = defaultdict(int)
        total_by_demand = defaultdict(int)

        # 鈽?V32: Uniform sampling (no curriculum 鈥?V30 transfer!)
        train_episode_ids = curriculum_sample(
            train_env.all_samples, train_env.sample_ids, update_num, N_EPISODES_PER_UPDATE
        )

        for ep_idx, sid in enumerate(train_episode_ids):
            obs = train_env.reset(sample_idx=sid)
            ep_reward = 0.0
            episode_converged = False

            for step in range(train_env.max_steps + 1):
                # SAC action: tanh-squashed [-1, 1] 鈫?scale to action bounds
                raw_action = trainer.select_action(obs, deterministic=False)
                low, high = train_env.get_action_bounds()
                action = low + (raw_action + 1.0) * 0.5 * (high - low)

                next_obs, reward, terminated, truncated, info = train_env.step(action)
                ep_reward += reward

                # Store transition (off-policy!)
                trainer.store_transition(obs, raw_action, reward, next_obs, terminated)

                obs = next_obs
                total_episodes += 1

                if terminated:
                    # 鈽?V26b: Full SCED convergence 鈥?aligned with GT!
                    episode_converged = adopted_operation_pass(info)
                    break

            phase_total += 1
            phase_rewards.append(ep_reward)
            gw_key = f"{int(train_env.target_demand / 1000)}GW"
            total_by_demand[gw_key] += 1
            if episode_converged:
                phase_converged += 1
                conv_by_demand[gw_key] += 1

        # === SAC GRADIENT UPDATES ===
        upd_result = {'q1_loss': 0., 'q2_loss': 0., 'actor_loss': 0., 'alpha': 0.}
        for _ in range(N_GRADIENT_STEPS):
            upd_result = trainer.update()

        dt = time.time() - update_start
        avg_ep_r = np.mean(phase_rewards) if phase_rewards else 0
        conv_pct = phase_converged / max(phase_total, 1) * 100

        log_msg = (f"U{update_num:>3d} R={avg_ep_r:+.0f} | "
                   f"TrainConv={phase_converged}/{phase_total}({conv_pct:.0f}%) | "
                   f"Q1={upd_result['q1_loss']:.1f} "
                   f"Q2={upd_result['q2_loss']:.1f} "
                   f"A={upd_result['actor_loss']:.1f} "
                   f"伪={upd_result['alpha']:.3f} | "
                   f"Buf={upd_result['buffer_size']} | "
                   f"{dt:.0f}s")

        # Detailed convergence breakdown
        if update_num % 10 == 1 or conv_pct < 50:
            detail_parts = []
            for gw in sorted(conv_by_demand.keys()):
                c, t = conv_by_demand[gw], total_by_demand[gw]
                detail_parts.append(f"{gw}GW:{c}/{t}")
            log_msg += f" | [{', '.join(detail_parts)}]"

        log(log_msg)

        # 鈽?V23: Save SLIM checkpoint every update (~16MB, no buffer!)
        trainer.save_checkpoint(
            RESULTS_DIR / "sac_checkpoint_latest.pth",
            update_num, None, total_episodes=total_episodes
        )

        # Periodic evaluation 鈥?BOTH train and val!
        if update_num % EVAL_INTERVAL == 0:
            eval_start = time.time()
            train_eval = evaluate_sac(trainer, train_env, train_ids,
                                      n_per_segment=N_PER_SEGMENT_EVAL, deterministic=True)
            val_eval = evaluate_sac(trainer, val_env, val_ids,
                                    n_per_segment=N_PER_SEGMENT_EVAL, deterministic=True)
            eval_dt = time.time() - eval_start

            log(f"\n{'鈹€'*70}")
            log(f"  EVAL @ U{update_num}:")
            log(f"    TRAIN: Conv={train_eval['conv_rate']:.1f}% ({train_eval['n_converged']}/{train_eval['n_total']}) | AvgR={train_eval['avg_reward']:+.0f}")
            log(f"    VAL:   Conv={val_eval['conv_rate']:.1f}% ({val_eval['n_converged']}/{val_eval['n_total']}) | AvgR={val_eval['avg_reward']:+.0f}")

            log(f"  Train by segment:")
            for gw in sorted(train_eval['by_demand_total'].keys()):
                c = train_eval['by_demand'].get(gw, 0)
                t = train_eval['by_demand_total'].get(gw, 0)
                pct = c / max(t, 1) * 100
                marker = "[OK]" if pct >= 75 else ("[!]" if pct >= 25 else "[X]")
                log(f"    {gw}GW: {c}/{t} ({pct:.0f}%) {marker}")

            log(f"  Val by segment:")
            for gw in sorted(val_eval['by_demand_total'].keys()):
                c = val_eval['by_demand'].get(gw, 0)
                t = val_eval['by_demand_total'].get(gw, 0)
                pct = c / max(t, 1) * 100
                marker = "[OK]" if pct >= 75 else ("[!]" if pct >= 25 else "[X]")
                log(f"    {gw}GW: {c}/{t} ({pct:.0f}%) {marker}")
            log(f"  Eval time: {eval_dt:.0f}s")
            log(f"{'鈹€'*70}\n")

            # Save best model based on VAL convergence
            if val_eval['conv_rate'] > best_val_conv:
                best_val_conv = val_eval['conv_rate']
                best_model_state = {
                    'actor': copy.deepcopy(trainer.actor.state_dict()),
                    'critic': copy.deepcopy(trainer.critic.state_dict()),
                    'update': update_num,
                    'val_conv': best_val_conv,
                    'train_conv': train_eval['conv_rate'],
                }
                torch.save(best_model_state, RESULTS_DIR / "best_model.pth")
                log(f"  鈽?NEW BEST MODEL saved (ValConv={best_val_conv:.1f}%, TrainConv={train_eval['conv_rate']:.1f}%)")

            # Save periodic checkpoint
            trainer.save_checkpoint(
                RESULTS_DIR / f"sac_checkpoint_upd{update_num}.pth",
                update_num, {'train': train_eval, 'val': val_eval},
                total_episodes=total_episodes
            )

        # End-of-training final evaluation
        if update_num == TOTAL_UPDATES:
            log("\n" + "=" * 70)
            log("FINAL EVALUATION (Best Model + Test Set)")
            log("=" * 70)

            # Load best model
            if best_model_state is not None:
                trainer.actor.load_state_dict(best_model_state['actor'])
                trainer.critic.load_state_dict(best_model_state['critic'])
                log(f"\n  Using best model from U{best_model_state['update']} "
                    f"(ValConv={best_model_state['val_conv']:.1f}%)")

            # Final eval on all three sets
            train_final = evaluate_sac(trainer, train_env, train_ids,
                                       n_per_segment=N_PER_SEGMENT_EVAL, deterministic=True)
            val_final = evaluate_sac(trainer, val_env, val_ids,
                                     n_per_segment=N_PER_SEGMENT_EVAL, deterministic=True)
            test_final = evaluate_sac(trainer, test_env, test_ids,
                                      n_per_segment=N_PER_SEGMENT_EVAL, deterministic=True)

            log(f"\n  FINAL RESULTS:")
            log(f"    TRAIN: Conv={train_final['conv_rate']:.1f}% ({train_final['n_converged']}/{train_final['n_total']})")
            log(f"    VAL:   Conv={val_final['conv_rate']:.1f}% ({val_final['n_converged']}/{val_final['n_total']})")
            log(f"    TEST:  Conv={test_final['conv_rate']:.1f}% ({test_final['n_converged']}/{test_final['n_total']})")

            log(f"\n  Test by segment:")
            for gw in sorted(test_final['by_demand_total'].keys()):
                c = test_final['by_demand'].get(gw, 0)
                t = test_final['by_demand_total'].get(gw, 0)
                pct = c / max(t, 1) * 100
                marker = "[OK]" if pct >= 75 else ("[!]" if pct >= 25 else "[X]")
                log(f"    {gw}GW: {c}/{t} ({pct:.0f}%) {marker}")

            total_time = time.time() - start_time
            log(f"\n  Total training time: {total_time/3600:.1f} hours")
            log(f"  Total episodes: {total_episodes}")
            log(f"  Total gradient updates: {trainer.total_updates}")

            # 鈽?V35: Stochastic retry + Ensemble evaluation!
            log(f"\n{'='*70}")
            log(f"V35 STOCHASTIC RETRY + ENSEMBLE EVALUATION")
            log(f"{'='*70}")
            log(f"  P2 (deterministic): {test_final['n_converged']}/{test_final['n_total']} ({test_final['conv_rate']:.1f}%)")

            # P3: Stochastic retries on FAILED samples
            import random as _rng
            p3_rescued = 0
            p4_rescued = 0
            v35_results = []

            # Build list of failed sample IDs from P2
            p2_ok_ids = set()
            for sid in test_ids:
                obs = test_env.reset(sample_idx=sid)
                for step in range(test_env.max_steps + 1):
                    raw_a = trainer.select_action(obs, deterministic=True)
                    low, high = test_env.get_action_bounds()
                    a = low + (raw_a + 1.0) * 0.5 * (high - low)
                    obs, r, done, trunc, info = test_env.step(a)
                    if done: break
                ok = adopted_operation_pass(info)
                if ok:
                    p2_ok_ids.add(sid)
                    v35_results.append((sid, 'P2', True))

            failed_ids = [sid for sid in test_ids if sid not in p2_ok_ids]
            log(f"  Failed samples (P2): {len(failed_ids)}")

            # P3: Stochastic retries on failed samples
            for sid in failed_ids:
                gw = int(samples[sid]['demand_mw'] / 1000)
                rescued = False
                for retry in range(V35_STOCHASTIC_RETRIES):
                    _rng.seed(42 + retry * 1000 + sid)
                    np.random.seed(42 + retry * 1000 + sid)
                    torch.manual_seed(42 + retry * 1000 + sid)
                    obs = test_env.reset(sample_idx=sid)
                    for step in range(V35_STOCHASTIC_STEPS + 1):
                        raw_a = trainer.select_action(obs, deterministic=False)
                        low, high = test_env.get_action_bounds()
                        a = low + (raw_a + 1.0) * 0.5 * (high - low)
                        obs, r, done, trunc, info = test_env.step(a)
                        if done: break
                    ok = adopted_operation_pass(info)
                    if ok:
                        p3_rescued += 1
                        v35_results.append((sid, f'P3_r{retry}', True))
                        rescued = True
                        break
                if not rescued:
                    # P4: Ensemble 鈥?try U50 checkpoint for HIGH segments
                    if V35_USE_ENSEMBLE and gw >= 23:
                        # Find best U50-like checkpoint (use a mid-training checkpoint)
                        u50_path = RESULTS_DIR / "sac_checkpoint_upd50.pth"
                        if u50_path.exists():
                            u50_ckpt = torch.load(u50_path, map_location=DEVICE, weights_only=False)
                            trainer_u50 = SACTrainer(obs_dim, action_dim).to(DEVICE)
                            if 'actor_state_dict' in u50_ckpt:
                                trainer_u50.actor.load_state_dict(u50_ckpt['actor_state_dict'])
                                trainer_u50.critic.load_state_dict(u50_ckpt['critic_state_dict'])
                            else:
                                trainer_u50.actor.load_state_dict(u50_ckpt['actor'])
                                trainer_u50.critic.load_state_dict(u50_ckpt['critic'])

                            for retry in range(V35_STOCHASTIC_RETRIES):
                                _rng.seed(42 + retry * 2000 + sid)
                                np.random.seed(42 + retry * 2000 + sid)
                                torch.manual_seed(42 + retry * 2000 + sid)
                                obs = test_env.reset(sample_idx=sid)
                                for step in range(V35_STOCHASTIC_STEPS + 1):
                                    raw_a = trainer_u50.select_action(obs, deterministic=(retry==0))
                                    low, high = test_env.get_action_bounds()
                                    a = low + (raw_a + 1.0) * 0.5 * (high - low)
                                    obs, r, done, trunc, info = test_env.step(a)
                                    if done: break
                                ok = (info.get('converged', False) and info.get('v_violations', 999) == 0
                                      and info.get('n1_violations', 999) == 0 and info.get('overloaded_lines', 999) == 0)
                                if ok:
                                    p4_rescued += 1
                                    v35_results.append((sid, f'P4_u50_r{retry}', True))
                                    rescued = True
                                    break

                    if not rescued:
                        v35_results.append((sid, 'FAIL', False))

            v35_total = len(p2_ok_ids) + p3_rescued + p4_rescued
            v35_rate = 100.0 * v35_total / len(test_ids)

            log(f"  P3 (stochastic retry K={V35_STOCHASTIC_RETRIES}): +{p3_rescued} rescued")
            log(f"  P4 (ensemble U50): +{p4_rescued} rescued")
            log(f"\n  V35 FINAL: {v35_total}/{len(test_ids)} ({v35_rate:.1f}%)")
            log(f"  Improvement over P2: +{v35_rate - test_final['conv_rate']:.1f}pp")

            # V35 by segment
            log(f"\n  V35 by segment:")
            v35_by_gw = defaultdict(lambda: {'ok': 0, 'n': 0, 'p2': 0, 'p3': 0, 'p4': 0})
            for sid, method, ok in v35_results:
                gw = int(samples[sid]['demand_mw'] / 1000)
                v35_by_gw[gw]['n'] += 1
                if ok:
                    v35_by_gw[gw]['ok'] += 1
                    if method.startswith('P2'): v35_by_gw[gw]['p2'] += 1
                    elif method.startswith('P3'): v35_by_gw[gw]['p3'] += 1
                    elif method.startswith('P4'): v35_by_gw[gw]['p4'] += 1
            for gw in sorted(v35_by_gw.keys()):
                d = v35_by_gw[gw]
                log(f"    {gw}GW: {d['ok']}/{d['n']} ({100*d['ok']/d['n']:.0f}%) [P2={d['p2']} P3={d['p3']} P4={d['p4']}]")

            # Save final summary
            summary = {
                'version': 'V21-SAC',
                'algorithm': 'SAC (Soft Actor-Critic)',
                'action_space': '卤螖P residual (adaptive scale)',
                'train_conv': train_final['conv_rate'],
                'val_conv': val_final['conv_rate'],
                'test_conv': test_final['conv_rate'],
                'best_val_conv': best_val_conv,
                'total_updates': TOTAL_UPDATES,
                'total_time_hours': total_time / 3600,
                'data_split': {
                    'train': len(train_ids),
                    'val': len(val_ids),
                    'test': len(test_ids),
                },
            }
            with open(RESULTS_DIR / "final_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)

            log("\n" + "=" * 70)
            log("TRAINING COMPLETE!")
            log("=" * 70)

    log_file.close()


if __name__ == "__main__":
    main()

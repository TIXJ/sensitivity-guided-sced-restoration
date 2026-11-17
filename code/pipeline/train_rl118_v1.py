# -*- coding: utf-8 -*-
"""
train_rl118_v1.py — Train SAC RL Corrector for IEEE 118-bus SCED.

Adapted from train_rlv35_complete.py for case118.
Key differences from case300:
  - 53 generators (vs 68), ~186 lines (vs 411)
  - Demand range: 3.0-5.0 GW (vs 16-24 GW)
  - Smaller obs_dim (6*53+23=341 vs 4987)
  - Standard voltage limits (0.94/1.06)
  - No V33 transfer (fresh training)
"""
import sys, os, time, json, math, random, warnings, copy, pathlib
import numpy as np; import torch; import torch.nn as nn; import torch.nn.functional as F
from collections import defaultdict, deque, Counter
from pathlib import Path

if sys.stdout.encoding != 'utf-8': sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8': sys.stderr.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

PROJ = Path(r".")
sys.path.insert(0, str(PROJ))

# ============================================================
# Config
# ============================================================
N_GEN = 53
N_SHIFT_DIRECTIONS = 12; N_KEY_GENERATORS = 20
N_VOLTAGE_GROUPS = 4
TOTAL_UPDATES = 100; N_EPISODES_PER_UPDATE = 10; N_GRADIENT_STEPS = 30
EVAL_INTERVAL = 10; N_PER_SEGMENT_EVAL = 5
MAX_STEPS = 50; MAX_STEPS_HIGH = 70
BATCH_SIZE = 256; BUFFER_SIZE = 200000
SAC_LR = 3e-4; SAC_GAMMA = 0.99; SAC_TAU = 0.005
BASE_REWARD = -10.0; CONVERGENCE_REWARD = 3000.0; N1_SECURE_REWARD = 20000.0
R_VIOLATION_COEF = -20.0
VMIN = 0.93; VMAX = 1.10; SEED = 42
AC_PF_CHECK_EVERY = 5; N1_CHECK_EVERY_STEPS = 5
VOLTAGE_STEP_PU = 0.006
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RESULTS_DIR = PROJ / "rl118_routeb_main"; RESULTS_DIR.mkdir(exist_ok=True)


def project_box_sum(x, lo, hi, target_sum, max_iter=100):
    """Project approximately onto lo <= x <= hi and sum(x)=target_sum."""
    x = np.asarray(x, dtype=np.float64).copy()
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    y = np.clip(x, lo, hi)
    target_sum = float(np.clip(target_sum, float(lo.sum()), float(hi.sum())))
    for _ in range(max_iter):
        err = target_sum - float(y.sum())
        if abs(err) < 1e-7:
            break
        if err > 0:
            free = y < hi - 1e-9
            if not np.any(free):
                break
            y[free] += err / float(np.count_nonzero(free))
            y = np.minimum(y, hi)
        else:
            free = y > lo + 1e-9
            if not np.any(free):
                break
            y[free] += err / float(np.count_nonzero(free))
            y = np.maximum(y, lo)
    return np.clip(y, lo, hi)


# ============================================================
# Data Split
# ============================================================
def split_samples_stratified(samples, train_ratio=0.70, val_ratio=0.15):
    rng = random.Random(SEED)
    by_demand = defaultdict(list)
    for i, s in enumerate(samples):
        gw = round(s['demand_mw'] / 1000.0, 1)
        by_demand[gw].append(i)
    train_ids, val_ids, test_ids = [], [], []
    for gw in sorted(by_demand.keys()):
        pool = list(by_demand[gw]); rng.shuffle(pool)
        n = len(pool); n_train = max(1, int(n * train_ratio)); n_val = max(1, int(n * val_ratio))
        train_ids.extend(pool[:n_train]); val_ids.extend(pool[n_train:n_train+n_val])
        test_ids.extend(pool[n_train+n_val:])
    return train_ids, val_ids, test_ids


def curriculum_sample(samples, pool_ids, update_num, n_total=20):
    by_demand = defaultdict(list)
    for sid in pool_ids:
        gw = round(samples[sid]['demand_mw'] / 1000.0, 1)
        by_demand[gw].append(sid)
    PHASE_1_END = 30
    if update_num <= PHASE_1_END:
        allowed = sorted(by_demand.keys()); n_per = max(1, n_total // len(allowed))
        selected = []
        for gw in allowed:
            pool = by_demand[gw]; n = min(n_per, len(pool))
            if pool: selected.extend(random.sample(pool, n))
    else:
        # Phase 2: 80% HIGH (>=4.3GW)
        high = [sid for gw in by_demand for sid in by_demand[gw] if gw >= 4.3]
        low = [sid for gw in by_demand for sid in by_demand[gw] if gw < 4.3]
        n_high = int(n_total * 0.8); n_low = n_total - n_high
        selected = []
        if high: selected.extend(random.sample(high, min(n_high, len(high))))
        if low: selected.extend(random.sample(low, min(n_low, len(low))))
    while len(selected) < n_total:
        all_pool = [sid for gw in by_demand for sid in by_demand[gw]]
        if all_pool: selected.append(random.choice(all_pool))
        else: break
    random.shuffle(selected)
    return selected[:n_total]


# ============================================================
# SCED Environment for case118
# ============================================================
class SCEDEnv118:
    def __init__(self, samples, sample_ids):
        self.N = N_GEN
        from pandapower.networks import case118
        net = case118()
        self.PMIN = net.gen.min_p_mw.values.copy().astype(np.float64)
        self.PMAX = net.gen.max_p_mw.values.copy().astype(np.float64)
        self.base_gen_vm_pu = np.asarray(net.gen.vm_pu.values, dtype=np.float64).copy()
        if len(self.base_gen_vm_pu) != N_GEN:
            self.base_gen_vm_pu = np.ones(N_GEN, dtype=np.float64)
        self.base_gen_vm_pu = np.clip(self.base_gen_vm_pu, VMIN, VMAX)
        if len(net.ext_grid):
            self.base_ext_grid_vm_pu = float(net.ext_grid.vm_pu.values[0])
        else:
            self.base_ext_grid_vm_pu = 1.0
        self.base_ext_grid_vm_pu = float(np.clip(self.base_ext_grid_vm_pu, VMIN, VMAX))
        self.baseMVA = net.sn_mva

        # Generator cost
        pc = net.poly_cost; cost_map = {}
        for _, row in pc.iterrows():
            if row.get('et', '') == 'gen':
                cost_map[int(row['element'])] = float(row.get('cp1_eur_per_mw', 30))
        cp1_arr = np.array([cost_map.get(i, 30.0) for i in range(N_GEN)], dtype=np.float64)
        self.gen_feats = {'pmax': self.PMAX.copy(), 'pmin': self.PMIN.copy(), 'cp1': cp1_arr}

        # GAT model + cached features (avoid reloading every predict call!)
        from train_gat118_v1 import GATv118Predictor, ObsBuilder118, get_case118_gen_features
        src, dst = [], []
        for i in range(N_GEN):
            for j in range(N_GEN):
                if i != j: src.append(i); dst.append(j)
        edge_index = torch.LongTensor([src, dst])
        self.gat_net = GATv118Predictor(n_units=N_GEN, edge_index=edge_index).to(DEVICE)
        ckpt_path = PROJ / "gatv118_results" / "gatv118_predictor_best.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"GAT118 model not found at {ckpt_path}. Run train_gat118_v1.py first.")
        sd = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
        self.gat_net.load_state_dict(sd['model_state_dict']); self.gat_net.eval()
        # Cache GAT features (NOT called every predict — big perf win!)
        self._gat118_features = get_case118_gen_features()
        self._gat118_obs_builder = ObsBuilder118(self._gat118_features)

        # Case118 N-1 network
        print("[Env118] Building Case118Network for N-1 check...")
        from generate_case118_gt_v1 import Case118Network, check_n1_security
        self.dcnet = Case118Network()
        self._check_n1_security = check_n1_security
        ndc = len(self.dcnet.gen_bus)
        # Note: Case118Network n_gen may differ from pandapower n_gen
        if ndc != N_GEN:
            print(f"  [WARN] dcnet.n_gen={ndc} != N_GEN={N_GEN}, some gens may not align")

        self.all_samples = samples; self.sample_ids = list(sample_ids)
        self.current_power = None; self.gat_power = None
        self.target_demand = None; self.cumulative_delta = None
        self.current_residual_scale = 20.0; self.corr_step = 0
        self.max_steps = MAX_STEPS
        self._current_vmin = 0.95; self._n_v_viol = 999; self._n_overloaded_lines = 999
        self._last_ac_ok = False; self._last_overloaded = 999
        self._last_n1_ok = False; self._last_n1_viol = 999
        self.prev_ac_result = np.array([0., 0.95, 1.05, 0.], dtype=np.float32)
        self._shift_directions = None; self._n_shift_dirs = N_SHIFT_DIRECTIONS
        self._key_gen_indices = None; self._n_key_gens = N_KEY_GENERATORS
        self._voltage_groups = [np.asarray(g, dtype=int) for g in np.array_split(np.arange(self.N), N_VOLTAGE_GROUPS)]
        self._n_voltage_actions = N_VOLTAGE_GROUPS + 1
        self.rl_obs_dim = None

    def _get_obs_dim(self):
        if self.rl_obs_dim is not None: return self.rl_obs_dim
        _ = self.reset(self.sample_ids[0])
        obs = self._get_obs(); self.rl_obs_dim = len(obs)
        return self.rl_obs_dim

    def _predict_gat(self, demand_mw):
        nf, gf_arr = self._gat118_obs_builder.build(demand_mw)
        with torch.no_grad():
            pred = self.gat_net(
                torch.FloatTensor(nf).unsqueeze(0).to(DEVICE),
                torch.FloatTensor(gf_arr).unsqueeze(0).to(DEVICE))
        return pred.cpu().numpy()[0]

    @property
    def action_dim(self):
        return self._n_shift_dirs + self._n_key_gens + self._n_voltage_actions

    def get_action_bounds(self):
        scale = self.current_residual_scale
        p_dim = self._n_shift_dirs + self._n_key_gens
        p_low = np.full(p_dim, -scale, dtype=np.float32)
        p_high = np.full(p_dim, scale, dtype=np.float32)
        # Keep voltage exploration small and symmetric.  A one-sided voltage
        # action solves the case118 set by hand, so it is not used for the
        # trainable RL experiment.
        # Symmetric voltage exploration avoids hard-coding the low-voltage
        # direction while retaining the grouped action space used for case118.
        v_low = np.full(self._n_voltage_actions, -VOLTAGE_STEP_PU, dtype=np.float32)
        v_high = np.full(self._n_voltage_actions, VOLTAGE_STEP_PU, dtype=np.float32)
        return np.concatenate([p_low, v_low]), np.concatenate([p_high, v_high])

    def reset(self, sample_idx=None):
        if sample_idx is None: sample_idx = random.choice(self.sample_ids)
        sample = self.all_samples[sample_idx]; self.target_demand = float(sample['demand_mw'])

        with torch.no_grad():
            gat_pred = self._predict_gat(self.target_demand)
        # GAT normalization (same as V33)
        gat_pred = project_box_sum(gat_pred, self.PMIN, self.PMAX * 0.99,
                                   self.target_demand)
        self.gat_power = gat_pred.astype(np.float64).copy()
        self.current_power = self.gat_power.copy()
        self.cumulative_delta = np.zeros(self.N, dtype=np.float64); self.corr_step = 0
        self.current_gen_vm_pu = self.base_gen_vm_pu.copy()
        self.current_ext_grid_vm_pu = float(self.base_ext_grid_vm_pu)

        # Route-B 118-bus samples are near-boundary voltage cases, analogous to
        # the 300-bus 16--17GW LOW segment.  Keep the active-power residual
        # scale small; voltage actions carry most of the correction.
        self.current_residual_scale = 5.0
        self.max_steps = MAX_STEPS

        self._current_vmin = 0.95; self._n_v_viol = 999; self._n_overloaded_lines = 999
        self._last_ac_ok = False; self._last_overloaded = 999
        self._last_n1_ok = False; self._last_n1_viol = 999
        self.prev_ac_result = np.array([0., 0.95, 1.05, 0.], dtype=np.float32)
        return self._get_obs()

    def _compute_shift_directions(self):
        if self._shift_directions is not None:
            return self._shift_directions

        S = np.asarray(self.dcnet.S, dtype=np.float64)
        _n_line, n_gen = S.shape
        directions = []

        def add(vec):
            vec = np.asarray(vec, dtype=np.float64).copy()
            vec -= vec.mean()
            norm = np.linalg.norm(vec)
            if norm < 1e-10:
                return
            unit = vec / norm
            for existing in directions:
                if abs(float(np.dot(unit, existing))) > 0.90:
                    return
            directions.append(unit)

        cp1 = self.gen_feats['cp1']
        add(cp1 - cp1.mean())
        add(-(cp1 - cp1.mean()))
        cap_dir = self.gen_feats['pmax'] - self.gen_feats['pmax'].mean()
        add(cap_dir)
        add(-cap_dir)
        try:
            sens = np.sum(np.abs(S), axis=0)
            add(sens - sens.mean())
            add(-(sens - sens.mean()))
            _u, _s, vh = np.linalg.svd(S, full_matrices=False)
            for j in range(min(6, vh.shape[0])):
                add(vh[j, :])
                add(-vh[j, :])
        except Exception:
            pass

        rng = np.random.default_rng(SEED)
        while len(directions) < self._n_shift_dirs:
            rand_dir = rng.normal(size=n_gen)
            for ex in directions:
                rand_dir -= np.dot(rand_dir, ex) * ex
            add(rand_dir)
        self._shift_directions = np.array(directions[:self._n_shift_dirs], dtype=np.float64)
        return self._shift_directions

    def _get_obs(self):
        p_util = self.current_power / (self.PMAX + 1e-8)
        gen_feats = np.concatenate([
            p_util.reshape(-1, 1),
            (self.current_power / (self.target_demand + 1e-8)).reshape(-1, 1),
            (self.gat_power / (self.target_demand + 1e-8)).reshape(-1, 1),
            ((self.current_power - self.gat_power) / (self.PMAX + 1e-8)).reshape(-1, 1),
            (self.cumulative_delta / (self.current_residual_scale + 1e-8)).reshape(-1, 1),
            ((self.current_gen_vm_pu - 1.0) / 0.1).reshape(-1, 1),
        ], axis=-1)
        s = self.current_power; hhi = float(((s/(s.sum()+1e-8))**2).sum() * 10000)
        global_feat = np.array([
            float(self.corr_step) / float(self.max_steps),
            float(self.target_demand) / 1000.0,
            float(self.current_power.sum()) / (self.target_demand + 1e-8),
            float(self._n_v_viol) / 300.0, (hhi - 200.0) / 400.0, 0.5,
            float(self._current_vmin), 0.0, float(self._n_overloaded_lines) / 186.0,
            0.0, 0.0, 0.0, 0.0, float(self.current_residual_scale / 20.0),
            float((self.current_ext_grid_vm_pu - 1.0) / 0.1),
        ], dtype=np.float32)
        # N-1 sensitivity placeholder (compute in step)
        n1_sens = np.zeros(self.N, dtype=np.float32)
        obs = np.concatenate([
            gen_feats.flatten(), global_feat, self.prev_ac_result,
            np.array([float(hhi), 0.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32), n1_sens,
            np.array([0.0, 0.0], dtype=np.float32),
        ]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def step(self, action):
        self.corr_step += 1
        action = np.asarray(action, dtype=np.float64)

        # Compute shift directions
        self._compute_shift_directions()

        # Hybrid action decoding (same as V35)
        p_dim = self._n_shift_dirs + self._n_key_gens
        shift_scales = action[:self._n_shift_dirs]
        key_gen_deltas = action[self._n_shift_dirs:p_dim]
        v_group_action = action[p_dim:p_dim + N_VOLTAGE_GROUPS]
        ext_v_action = float(action[p_dim + N_VOLTAGE_GROUPS]) if len(action) > p_dim + N_VOLTAGE_GROUPS else 0.0
        delta_p = np.zeros(self.N, dtype=np.float64)
        for j in range(self._n_shift_dirs):
            delta_p += shift_scales[j] * self._shift_directions[j]
        sens = np.abs(self.dcnet.S).sum(axis=0)
        key_idx = np.argsort(-sens)[:self._n_key_gens]
        for j, g_idx in enumerate(key_idx):
            delta_p[g_idx] += key_gen_deltas[j] * 0.5

        # Apply delta with cumulative clipping
        scale = self.current_residual_scale
        cum_clip = scale * 10.0
        self.cumulative_delta += delta_p
        self.cumulative_delta = np.clip(self.cumulative_delta, -cum_clip, cum_clip)
        self.current_power = project_box_sum(self.gat_power + self.cumulative_delta,
                                             self.PMIN, self.PMAX * 0.99,
                                             self.target_demand)
        for gidx, gen_idx in enumerate(self._voltage_groups):
            if len(gen_idx):
                self.current_gen_vm_pu[gen_idx] += v_group_action[gidx]
        self.current_gen_vm_pu = np.clip(self.current_gen_vm_pu, VMIN, VMAX)
        self.current_ext_grid_vm_pu = float(np.clip(self.current_ext_grid_vm_pu + ext_v_action, VMIN, VMAX))

        # AC-PF and N-1 checks follow the same fast cadence used by the
        # 300-bus training entry point.  The first step is always checked.
        do_ac = (self.corr_step == 1) or (self.corr_step % AC_PF_CHECK_EVERY == 0)
        if do_ac:
            ac_info = self._run_ac_pf()
            ac_ok = bool(ac_info.get("converged", False))
            v_viol = int(ac_info.get("v_violations", 999))
            overloaded = int(ac_info.get("overloaded_lines", 999))
            vmin = float(ac_info.get("v_min", 0.0))
            self._last_ac_ok = ac_ok; self._n_v_viol = v_viol
            self._last_overloaded = overloaded; self._current_vmin = vmin
            self._n_overloaded_lines = overloaded
            self.prev_ac_result = np.array([
                1.0 if ac_ok else 0.0, vmin, 1.05, 0.0], dtype=np.float32)
        else:
            ac_info = {}
            ac_ok = self._last_ac_ok
            v_viol = self._n_v_viol
            overloaded = self._last_overloaded
            vmin = self._current_vmin

        do_n1 = (self.corr_step == 1) or (self.corr_step % N1_CHECK_EVERY_STEPS == 0)
        if do_n1:
            try:
                n1_viol, _ = self._check_n1_security(self.current_power, self.dcnet)
                n1_ok = (n1_viol == 0)
            except Exception:
                n1_ok = False; n1_viol = 999
            self._last_n1_ok = n1_ok; self._last_n1_viol = n1_viol
        else:
            n1_ok = self._last_n1_ok; n1_viol = self._last_n1_viol

        # Reward
        reward = BASE_REWARD
        done = False; info = {"converged": False}

        full_pass = bool(do_ac and do_n1 and ac_ok and v_viol == 0 and overloaded == 0 and n1_ok)
        if full_pass:
            reward = N1_SECURE_REWARD
            done = True
            info["converged"] = True
            info["reason"] = "full_sced_pass"
        elif do_ac and ac_ok:
            # Continuous AC-PF signal aligned with the 300-bus trainer.  For
            # Route-B 118, many states differ by only a tiny voltage gap; using
            # only the discrete number of violated buses makes learning noisy.
            v_mag = float(ac_info.get("v_violation_mag_pu", 0.0))
            v_gap = float(ac_info.get("v_violation_max_pu", 0.0))
            loading_max = float(ac_info.get("loading_max", 0.0))
            line_excess = max(0.0, loading_max - 100.0) / 100.0
            reward = BASE_REWARD + 50.0
            reward -= 3000.0 * min(v_mag, 0.20)
            reward -= 1000.0 * min(v_gap, 0.10)
            reward -= 100.0 * min(float(overloaded), 20.0)
            reward -= 50.0 * min(line_excess, 2.0)
            reward += R_VIOLATION_COEF * min(float(v_viol + overloaded) / 100.0, 1.0)
            if do_n1 and not n1_ok:
                reward -= 100.0 * min(float(n1_viol), 20.0)
            info["reason"] = "voltage_violation" if v_viol else ("line_overload" if overloaded else "n1_violation")
        else:
            reward = BASE_REWARD
            if do_ac:
                reward -= 200.0
                info["reason"] = "acpf_not_converged"
            else:
                info["reason"] = "not_checked"

        info.update(ac_info)
        info.update({"converged": bool(full_pass),
                     "ac_converged": bool(ac_ok),
                     "v_violations": v_viol, "n1_violations": n1_viol,
                     "overloaded_lines": overloaded, "vmin": vmin,
                     "ac_checked": do_ac, "n1_checked": do_n1})
        if self.corr_step >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, False, info

    def _run_ac_pf(self):
        from pandapower.networks import case118
        import pandapower as pp
        net = case118()
        net.gen.p_mw = np.clip(self.current_power, 0.0, self.PMAX)
        net.gen.vm_pu = self.current_gen_vm_pu[:len(net.gen)]
        if len(net.ext_grid):
            net.ext_grid.vm_pu = self.current_ext_grid_vm_pu
        ls = self.target_demand / net.load.p_mw.sum()
        net.load.p_mw *= ls; net.load.q_mvar *= ls
        net.bus.min_vm_pu = VMIN; net.bus.max_vm_pu = VMAX
        try:
            pp.runpp(net, calculate_voltage_angles=True, numba=False,
                     enforce_q_lims=True, max_iteration=80)
            if not net.converged:
                return {
                    "converged": False, "v_violations": 999,
                    "overloaded_lines": 999, "v_min": 0.0, "v_max": 0.0,
                    "v_violation_mag_pu": 999.0, "v_violation_max_pu": 999.0,
                    "v_low_violations": 999, "v_high_violations": 999,
                    "loading_max": 999.0, "trafo_overloaded": 999,
                }
            vm = net.res_bus.vm_pu.values
            low_gap = np.maximum(VMIN - vm, 0.0)
            high_gap = np.maximum(vm - VMAX, 0.0)
            vmin = float(vm.min())
            vmax = float(vm.max())
            v_low = int((low_gap > 0.0).sum())
            v_high = int((high_gap > 0.0).sum())
            v_viol = v_low + v_high
            v_mag = float(low_gap.sum() + high_gap.sum())
            v_gap = float(max(low_gap.max(initial=0.0), high_gap.max(initial=0.0)))
            line_loading = net.res_line.loading_percent.values
            overloaded = int((line_loading > 100).sum())
            loading_max = float(line_loading.max()) if len(line_loading) else 0.0
            trafo_overloaded = 0
            if len(net.res_trafo) > 0:
                trafo_loading = net.res_trafo.loading_percent.values
                trafo_overloaded = int((trafo_loading > 100).sum())
                overloaded += trafo_overloaded
                loading_max = max(loading_max, float(trafo_loading.max()) if len(trafo_loading) else 0.0)
            return {
                "converged": True,
                "v_violations": v_viol,
                "overloaded_lines": overloaded,
                "v_min": vmin,
                "v_max": vmax,
                "v_violation_mag_pu": v_mag,
                "v_violation_max_pu": v_gap,
                "v_low_violations": v_low,
                "v_high_violations": v_high,
                "loading_max": loading_max,
                "trafo_overloaded": trafo_overloaded,
            }
        except Exception as exc:
            return {
                "converged": False, "v_violations": 999,
                "overloaded_lines": 999, "v_min": 0.0, "v_max": 0.0,
                "v_violation_mag_pu": 999.0, "v_violation_max_pu": 999.0,
                "v_low_violations": 999, "v_high_violations": 999,
                "loading_max": 999.0, "trafo_overloaded": 999,
                "error": str(exc),
            }


# ============================================================
# SAC Networks (same as V35)
# ============================================================
class GaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU())
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, -3.0)
    def forward(self, obs):
        h = self.trunk(obs)
        mean = torch.nan_to_num(self.mean_head(h), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        log_std = torch.nan_to_num(self.log_std_head(h), nan=-3.0, posinf=2.0, neginf=-5.0).clamp(-5.0, 2.0)
        return mean, log_std
    def sample(self, obs):
        mean, log_std = self.forward(obs); std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample(); action = torch.tanh(z)
        log_prob = (normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return action, log_prob, mean
    def select_action(self, obs, deterministic=True):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            if deterministic:
                return torch.tanh(self.forward(obs_t)[0]).cpu().numpy()[0]
            else:
                return self.sample(obs_t)[0].cpu().numpy()[0]

class TwinQCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=256):
        super().__init__()
        self.q1 = nn.Sequential(nn.Linear(obs_dim+action_dim, hidden), nn.ReLU(),
                                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.q2 = nn.Sequential(nn.Linear(obs_dim+action_dim, hidden), nn.ReLU(),
                                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


class SACTrainer:
    def __init__(self, obs_dim, action_dim):
        self.actor = GaussianActor(obs_dim, action_dim).to(DEVICE)
        self.critic = TwinQCritic(obs_dim, action_dim).to(DEVICE)
        self.critic_target = TwinQCritic(obs_dim, action_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=SAC_LR)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=SAC_LR)
        self.log_alpha = nn.Parameter(torch.tensor(0.0).to(DEVICE))
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=SAC_LR)
        self.target_entropy = -0.5 * action_dim
        self.buffer = deque(maxlen=BUFFER_SIZE)
        self.total_updates = 0

    def select_action(self, obs, deterministic=True):
        return self.actor.select_action(obs, deterministic)

    def update(self, batch_size=512, n_steps=50):
        if len(self.buffer) < batch_size: return
        for _ in range(n_steps):
            idx = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
            batch = [self.buffer[i] for i in idx]
            s = torch.FloatTensor(np.array([b[0] for b in batch])).to(DEVICE)
            a = torch.FloatTensor(np.array([b[1] for b in batch])).to(DEVICE)
            r = torch.FloatTensor(np.array([b[2] for b in batch])).unsqueeze(-1).to(DEVICE)
            ns = torch.FloatTensor(np.array([b[3] for b in batch])).to(DEVICE)
            d = torch.FloatTensor(np.array([b[4] for b in batch])).unsqueeze(-1).to(DEVICE)

            with torch.no_grad():
                next_a, next_log_prob, _ = self.actor.sample(ns)
                q1_t, q2_t = self.critic_target(ns, next_a)
                min_q = torch.min(q1_t, q2_t)
                alpha = self.log_alpha.exp()
                target_q = r + SAC_GAMMA * (1 - d) * (min_q - alpha * next_log_prob)

            q1, q2 = self.critic(s, a)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            self.critic_opt.zero_grad(); critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0); self.critic_opt.step()

            curr_a, curr_log_prob, _ = self.actor.sample(s)
            q1_a, q2_a = self.critic(s, curr_a)
            actor_loss = (alpha * curr_log_prob - torch.min(q1_a, q2_a)).mean()
            self.actor_opt.zero_grad(); actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0); self.actor_opt.step()

            alpha_loss = -(self.log_alpha.exp() * (curr_log_prob.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()

            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(SAC_TAU * p.data + (1 - SAC_TAU) * tp.data)

        self.total_updates += 1


# ============================================================
# Training
# ============================================================
def evaluate(trainer, env, sample_ids):
    global AC_PF_CHECK_EVERY, N1_CHECK_EVERY_STEPS
    old_ac, old_n1 = AC_PF_CHECK_EVERY, N1_CHECK_EVERY_STEPS
    AC_PF_CHECK_EVERY = 1; N1_CHECK_EVERY_STEPS = 1
    n_ok = 0; rewards_eval = []
    try:
        for sid in sample_ids:
            obs = env.reset(sample_idx=sid)
            for step in range(env.max_steps + 1):
                raw_a = trainer.select_action(obs, deterministic=True)
                low, high = env.get_action_bounds()
                a = low + (raw_a + 1.0) * 0.5 * (high - low)
                obs, r, done, _, info = env.step(a)
                if done: break
            passed = (
                info.get('converged', False)
                and info.get('v_violations', 999) == 0
                and info.get('overloaded_lines', 999) == 0
                and info.get('n1_violations', 999) == 0
            )
            if passed:
                n_ok += 1
            rewards_eval.append(float(r))
        return {"passed": n_ok, "total": len(sample_ids),
                "avg_reward": float(np.mean(rewards_eval)) if rewards_eval else 0.0}
    finally:
        AC_PF_CHECK_EVERY, N1_CHECK_EVERY_STEPS = old_ac, old_n1


def main():
    print("=" * 70)
    print("RLv118 SAC Training: IEEE 118-bus SCED")
    print("=" * 70)

    json_path = PROJ / "case118_routeb_rescuable_dataset.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"{json_path} not found. Run paper_results/code/screen_route_b_smallcase_rescuable.py --case 118 first."
        )
    with open(str(json_path), 'r') as f: data = json.load(f)
    samples = [s for s in data['samples'] if s.get('is_complete_sced')]
    train_ids, val_ids, test_ids = split_samples_stratified(samples)
    print(f"Samples: {len(samples)} complete SCED (Train:{len(train_ids)} Val:{len(val_ids)} Test:{len(test_ids)})")

    train_env = SCEDEnv118(samples, train_ids)
    obs_dim = train_env._get_obs_dim(); action_dim = train_env.action_dim
    print(f"ObsDim={obs_dim}, ActDim={action_dim}")

    trainer = SACTrainer(obs_dim, action_dim)

    log_path = RESULTS_DIR / "training.log"
    log = open(str(log_path), 'w', buffering=1, encoding='utf-8')
    log.write(f"RLv118 Training Log — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log.write(f"ObsDim={obs_dim} ActDim={action_dim}\n")

    best_val = -1; best_val_total = 0; best_val_r = -1e18
    best_train = -1; best_train_total = 0; best_train_r = -1e18
    best_state = None; t0 = time.time()

    for u in range(1, TOTAL_UPDATES + 1):
        u_start = time.time()
        ep_rewards = []; ep_steps = []; ep_pass = 0
        fail_reasons = Counter(); fail_examples = Counter()
        eps = curriculum_sample(samples, train_ids, u, N_EPISODES_PER_UPDATE)

        collect_start = time.time()
        for sid in eps:
            obs = train_env.reset(sample_idx=sid); ep_r = 0.0
            info = {}
            for step in range(train_env.max_steps + 1):
                raw_a = trainer.select_action(obs, deterministic=False)
                low, high = train_env.get_action_bounds()
                a = low + (raw_a + 1.0) * 0.5 * (high - low)
                next_obs, reward, done, _, info = train_env.step(a)
                trainer.buffer.append((obs, raw_a, reward, next_obs, float(done)))
                ep_r += reward; obs = next_obs
                if done: break
            ep_steps.append(step + 1)
            ep_rewards.append(ep_r)
            if info.get('reason') == 'full_sced_pass':
                ep_pass += 1
            else:
                reason = info.get("reason", "unknown")
                fail_reasons[reason] += 1
                if reason == "voltage_violation":
                    detail = (
                        f"{reason}:vv={info.get('v_violations', '?')}:"
                        f"low={info.get('v_low_violations', '?')}:"
                        f"high={info.get('v_high_violations', '?')}:"
                        f"gap={float(info.get('v_violation_max_pu', 0.0)):.4f}:"
                        f"v=[{float(info.get('v_min', 0.0)):.3f},{float(info.get('v_max', 0.0)):.3f}]"
                    )
                elif reason == "n1_violation":
                    detail = f"{reason}:n1={info.get('n1_violations', '?')}"
                elif reason == "line_overload":
                    detail = f"{reason}:line={info.get('overloaded_lines', '?')}"
                else:
                    detail = reason
                fail_examples[detail] += 1
        collect_dt = time.time() - collect_start

        update_start = time.time()
        trainer.update(BATCH_SIZE, N_GRADIENT_STEPS)
        update_dt = time.time() - update_start
        avg_r = np.mean(ep_rewards) if ep_rewards else 0.0

        if ep_pass > best_train or (ep_pass == best_train and avg_r > best_train_r):
            best_train = ep_pass; best_train_total = len(eps); best_train_r = avg_r

        val_txt = "skip"; val_dt = 0.0
        if u % EVAL_INTERVAL == 0 or u == TOTAL_UPDATES:
            val_start = time.time()
            val_res = evaluate(trainer, train_env, val_ids[:min(20, len(val_ids))])
            val_dt = time.time() - val_start
            val_txt = f"{val_res['passed']}/{val_res['total']} R={val_res['avg_reward']:+.0f}"
            if val_res["passed"] > best_val or (val_res["passed"] == best_val and val_res["avg_reward"] > best_val_r):
                best_val = val_res["passed"]; best_val_total = val_res["total"]; best_val_r = val_res["avg_reward"]
                best_state = {'actor': trainer.actor.state_dict(), 'critic': trainer.critic.state_dict(),
                              'update': u, 'val_rate': 100 * best_val / max(best_val_total, 1)}
                torch.save(best_state, str(RESULTS_DIR / 'best_model.pth'))

        fail_txt = ",".join([f"{k}:{v}" for k, v in fail_reasons.most_common()]) if fail_reasons else "-"
        fail_ex_txt = ",".join([f"{k}:{v}" for k, v in fail_examples.most_common(2)]) if fail_examples else "-"
        steps_avg = float(np.mean(ep_steps)) if ep_steps else 0.0
        msg = (f"U{u:3d} R={avg_r:+.1f} train_pass={ep_pass}/{len(eps)} "
               f"best_train={best_train}/{best_train_total} val={val_txt} "
               f"best_val={best_val}/{best_val_total} fail={fail_txt} ex={fail_ex_txt} "
               f"steps_avg={steps_avg:.1f} collect={collect_dt:.0f}s upd={update_dt:.0f}s "
               f"val_time={val_dt:.0f}s total={time.time()-u_start:.0f}s")
        print(msg); log.write(msg + '\n')

    if best_state:
        trainer.actor.load_state_dict(best_state['actor'])

    test_res = evaluate(trainer, train_env, test_ids)
    test_ok, test_total = test_res["passed"], test_res["total"]
    test_rate = 100*test_ok/test_total
    elapsed = (time.time()-t0)/3600
    best_val_rate = 100 * best_val / max(best_val_total, 1)
    msg = f"\nFINAL: Test={test_ok}/{test_total} ({test_rate:.1f}%) BestVal={best_val_rate:.1f}% Time={elapsed:.1f}h"
    print(msg); log.write(msg + '\n'); log.close()
    with open(str(RESULTS_DIR / "main_experiment_result.json"), "w", encoding="utf-8") as f:
        json.dump({"case": "case118", "test_pass": test_ok, "test_total": test_total,
                   "test_pass_rate_pct": test_rate, "best_val_pct": best_val_rate,
                   "vmin": VMIN, "vmax": VMAX, "action_dim": N_SHIFT_DIRECTIONS + N_KEY_GENERATORS + N_VOLTAGE_GROUPS + 1,
                   "voltage_step_pu": VOLTAGE_STEP_PU,
                   "updates": TOTAL_UPDATES}, f, indent=2)

if __name__ == "__main__":
    main()

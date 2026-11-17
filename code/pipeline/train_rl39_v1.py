"""RL corrector for IEEE 39-bus SCED (SAC, 9 generators)."""
import argparse, sys, os, time, json, random, warnings, copy, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from collections import defaultdict, deque, Counter; from pathlib import Path
warnings.filterwarnings("ignore")
PROJ = Path(
    os.environ.get("ROUTE_B_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).resolve(); sys.path.insert(0, str(PROJ))

N_GEN = 9; DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOTAL_UPDATES = 100; N_EP = 10; N_GRAD = 30; BATCH = 256; BUFFER = 100000
MAX_STEPS = 40; MAX_STEPS_H = 60; EVAL_INT = 10; N_PER_SEG = 5
BASE_REWARD = -10.0
N1_SECURE_REWARD = 20000.0
R_VIOLATION_COEF = -20.0
from case39_route_b_utils import CASE39_VMIN, CASE39_VMAX, apply_case39_route_b_limits
VMIN = CASE39_VMIN; VMAX = CASE39_VMAX; SEED = 42
AC_PF_CHECK_EVERY = 5; N1_CHECK_EVERY_STEPS = 5
VOLTAGE_STEP_PU = 0.002
RESULTS = PROJ / "rl39_routeb_main"; RESULTS.mkdir(exist_ok=True)
N_COMPRESSED_SHIFTS = 4
N_COMPRESSED_KEYS = 2

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

def split_samples(samples, tr=0.70, vr=0.15):
    rng = random.Random(SEED); by = defaultdict(list)
    for i, s in enumerate(samples): by[round(s['demand_mw']/1000, 1)].append(i)
    tid, vid, xid = [], [], []
    for gw in sorted(by.keys()):
        pool = list(by[gw]); rng.shuffle(pool); n = len(pool)
        nt = max(1, int(n*tr)); nv = max(1, int(n*vr))
        tid.extend(pool[:nt]); vid.extend(pool[nt:nt+nv]); xid.extend(pool[nt+nv:])
    return tid, vid, xid

def curriculum(samples, pool, update, n_total=10):
    by = defaultdict(list)
    for sid in pool: by[round(samples[sid]['demand_mw']/1000, 1)].append(sid)
    if update <= 30:
        allowed = sorted(by.keys()); n_per = max(1, n_total//len(allowed))
        sel = []
        for gw in allowed:
            p = by[gw]; n = min(n_per, len(p))
            if p: sel.extend(random.sample(p, n))
    else:
        high = [sid for gw in by for sid in by[gw] if gw >= 6.6]
        low = [sid for gw in by for sid in by[gw] if gw < 6.6]
        nh = int(n_total*0.8); nl = n_total-nh
        sel = []
        if high: sel.extend(random.sample(high, min(nh, len(high))))
        if low: sel.extend(random.sample(low, min(nl, len(low))))
    while len(sel) < n_total:
        all_p = [sid for gw in by for sid in by[gw]]
        if all_p: sel.append(random.choice(all_p))
        else: break
    random.shuffle(sel); return sel[:n_total]

class Env39:
    def __init__(self, samples, sample_ids, action_encoding="full9", basis_seed=20260721):
        from pandapower.networks import case39
        net = case39(); self.N = N_GEN
        apply_case39_route_b_limits(net)
        self.PMIN = net.gen.min_p_mw.values.astype(np.float64); self.PMAX = net.gen.max_p_mw.values.astype(np.float64)
        self.base_gen_vm_pu = np.asarray(net.gen.vm_pu.values, dtype=np.float64).copy()
        if len(self.base_gen_vm_pu) != N_GEN:
            self.base_gen_vm_pu = np.ones(N_GEN, dtype=np.float64)
        self.base_gen_vm_pu = np.clip(self.base_gen_vm_pu, VMIN, VMAX)
        if len(net.ext_grid):
            self.base_ext_vm_pu = float(net.ext_grid.vm_pu.values[0])
        else:
            self.base_ext_vm_pu = 1.0
        self.base_ext_vm_pu = float(np.clip(self.base_ext_vm_pu, VMIN, VMAX))
        from generate_case39_gt_v1 import Case39Network, check_n1
        self.dcnet = Case39Network(); self._check_n1 = check_n1

        from train_gat39_v1 import GATPredictor, ObsBuilder, get_features
        gf = get_features(); self.ob = ObsBuilder(gf)
        src=[]; dst=[]
        for i in range(N_GEN):
            for j in range(N_GEN):
                if i!=j: src.append(i); dst.append(j)
        ei = torch.LongTensor([src,dst])
        self.gat = GATPredictor(n_units=N_GEN, edge_index=ei).to(DEVICE)
        ckpt = torch.load(str(PROJ/"gat39_results"/"gat39_best.pth"), map_location=DEVICE, weights_only=False)
        self.gat.load_state_dict(ckpt["model_state_dict"]); self.gat.eval()

        if action_encoding not in {"full9", "physics6", "matched6"}:
            raise ValueError(f"Unsupported case39 action encoding: {action_encoding}")
        self.action_encoding = action_encoding
        self.basis_seed = int(basis_seed)
        if action_encoding == "full9":
            self._n_shift_dirs = 0
            self._n_key_gens = N_GEN
        else:
            self._n_shift_dirs = N_COMPRESSED_SHIFTS
            self._n_key_gens = N_COMPRESSED_KEYS
        self._shift_directions = None
        self._key_gen_indices = None

        self.all_samples = samples; self.sample_ids = list(sample_ids)
        self.gat_power = None; self.current_power = None; self.target_demand = None
        self.cum_delta = None; self.scale = 50.0; self.step_count = 0; self.max_steps = MAX_STEPS
        self._last_ac_ok = False; self._last_vv = 999; self._last_ov = 999
        self._last_vmin = 0.0; self._last_vmax = 0.0
        self._last_n1_ok = False; self._last_n1_viol = 999
        self.obs_dim = None

    def _get_obs_dim(self):
        if self.obs_dim is not None: return self.obs_dim
        _ = self.reset(self.sample_ids[0]); self.obs_dim = len(self._get_obs()); return self.obs_dim

    @property
    def action_dim(self):
        return self._n_shift_dirs + self._n_key_gens + N_GEN + 1

    def get_action_bounds(self):
        s = self.scale
        p_dim = self._n_shift_dirs + self._n_key_gens
        p_low = np.full(p_dim, -s, dtype=np.float32)
        p_high = np.full(p_dim, s, dtype=np.float32)
        v_low = np.full(N_GEN + 1, -VOLTAGE_STEP_PU, dtype=np.float32)
        v_high = np.full(N_GEN + 1, VOLTAGE_STEP_PU, dtype=np.float32)
        return np.concatenate([p_low, v_low]), np.concatenate([p_high, v_high])

    def _predict_gat(self, demand):
        nf, gf_arr = self.ob.build(demand)
        with torch.no_grad():
            pred = self.gat(torch.FloatTensor(nf).unsqueeze(0).to(DEVICE),
                           torch.FloatTensor(gf_arr).unsqueeze(0).to(DEVICE))
        return pred.cpu().numpy()[0]

    @staticmethod
    def _orthonormalize(candidates, n_wanted):
        directions = []
        for candidate in candidates:
            vec = np.asarray(candidate, dtype=np.float64).copy()
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
            if len(directions) == n_wanted:
                break
        return directions

    def _build_action_basis(self):
        if self._shift_directions is not None and self._key_gen_indices is not None:
            return
        if self.action_encoding == "full9":
            self._shift_directions = np.zeros((0, N_GEN), dtype=np.float64)
            self._key_gen_indices = np.arange(N_GEN, dtype=int)
            return

        if self.action_encoding == "matched6":
            rng = np.random.default_rng(self.basis_seed)
            candidates = [rng.standard_normal(N_GEN) for _ in range(10 * N_GEN)]
            directions = self._orthonormalize(candidates, self._n_shift_dirs)
            if len(directions) != self._n_shift_dirs:
                raise RuntimeError("Could not construct the matched case39 action basis")
            self._shift_directions = np.asarray(directions, dtype=np.float64)
            self._key_gen_indices = np.sort(
                rng.choice(N_GEN, size=self._n_key_gens, replace=False)
            ).astype(int)
            return

        S = np.asarray(self.dcnet.S, dtype=np.float64)
        lodf = np.asarray(self.dcnet.lodf, dtype=np.float64)
        limits = np.maximum(
            np.asarray(self.dcnet.effective_limit, dtype=np.float64), 1.0
        )
        gradients = []
        for outage in range(lodf.shape[1]):
            for monitored in range(lodf.shape[0]):
                if monitored == outage or abs(lodf[monitored, outage]) < 0.05:
                    continue
                grad = S[monitored] + lodf[monitored, outage] * S[outage]
                gradients.append(grad / limits[monitored])
        gradient_matrix = np.asarray(gradients, dtype=np.float64)
        if gradient_matrix.ndim != 2 or gradient_matrix.shape[0] == 0:
            gradient_matrix = S.copy()
        _, _, vh = np.linalg.svd(gradient_matrix, full_matrices=False)
        directions = self._orthonormalize(
            list(vh) + list(S), self._n_shift_dirs
        )
        if len(directions) != self._n_shift_dirs:
            raise RuntimeError("Could not construct the PTDF/LODF case39 action basis")
        self._shift_directions = np.asarray(directions, dtype=np.float64)
        sensitivity = np.sum(np.abs(gradient_matrix), axis=0)
        self._key_gen_indices = np.argsort(-sensitivity)[:self._n_key_gens]

    def reset(self, sample_idx=None):
        if sample_idx is None: sample_idx = random.choice(self.sample_ids)
        s = self.all_samples[sample_idx]; self.target_demand = float(s['demand_mw'])
        with torch.no_grad():
            gp = self._predict_gat(self.target_demand)
        gp = project_box_sum(gp, self.PMIN, self.PMAX*0.99, self.target_demand)
        self.gat_power = gp.astype(np.float64).copy()
        self.current_power = self.gat_power.copy(); self.cum_delta = np.zeros(N_GEN, dtype=np.float64)
        self.current_gen_vm_pu = self.base_gen_vm_pu.copy()
        self.current_ext_vm_pu = float(self.base_ext_vm_pu)
        self.step_count = 0
        self._last_ac_ok = False; self._last_vv = 999; self._last_ov = 999
        self._last_vmin = 0.0; self._last_vmax = 0.0
        self._last_n1_ok = False; self._last_n1_viol = 999
        self.scale = 20.0
        self.max_steps = MAX_STEPS
        return self._get_obs()

    def _get_obs(self):
        pu = self.current_power/(self.PMAX+1e-8)
        gf = np.concatenate([pu.reshape(-1,1),
            (self.current_power/(self.target_demand+1e-8)).reshape(-1,1),
            (self.gat_power/(self.target_demand+1e-8)).reshape(-1,1),
            ((self.current_power-self.gat_power)/(self.PMAX+1e-8)).reshape(-1,1),
            (self.cum_delta/(self.scale+1e-8)).reshape(-1,1),
            ((self.current_gen_vm_pu - 1.0) / 0.1).reshape(-1,1)], axis=-1)
        hhi = float(((self.current_power/(self.current_power.sum()+1e-8))**2).sum()*10000)
        glob = np.array([self.step_count/max(self.max_steps,1), self.target_demand/1000.0,
            self.current_power.sum()/(self.target_demand+1e-8), 0.0, (hhi-200)/400, 0.5,
            0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.scale/50.0], dtype=np.float32)
        glob = np.concatenate([glob, np.array([(self.current_ext_vm_pu - 1.0) / 0.1], dtype=np.float32)])
        obs = np.concatenate([gf.flatten(), glob, np.array([float(hhi),0.0,0.0],dtype=np.float32),
            np.zeros(N_GEN,dtype=np.float32), np.array([0.0,0.0],dtype=np.float32)]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def step(self, action):
        self.step_count += 1; action = np.asarray(action, dtype=np.float64)
        self._build_action_basis()
        p_dim = self._n_shift_dirs + self._n_key_gens
        shift_scales = action[:self._n_shift_dirs]
        key_deltas = action[self._n_shift_dirs:p_dim]
        v_action = action[p_dim:p_dim + N_GEN]
        ext_v_action = float(action[p_dim + N_GEN]) if len(action) > p_dim + N_GEN else 0.0
        p_action = self._shift_directions.T @ shift_scales
        p_action[self._key_gen_indices] += key_deltas
        cl = self.scale*8.0
        self.cum_delta = np.clip(self.cum_delta+p_action, -cl, cl)
        self.current_power = project_box_sum(self.gat_power+self.cum_delta,
                                             self.PMIN, self.PMAX*0.99,
                                             self.target_demand)
        self.current_gen_vm_pu = np.clip(self.current_gen_vm_pu + v_action, VMIN, VMAX)
        self.current_ext_vm_pu = float(np.clip(self.current_ext_vm_pu + ext_v_action, VMIN, VMAX))

        do_ac = (self.step_count == 1) or (self.step_count % AC_PF_CHECK_EVERY == 0)
        if do_ac:
            import pandapower as pp; from pandapower.networks import case39
            net = case39(); apply_case39_route_b_limits(net)
            net.gen.p_mw = np.clip(self.current_power, 0, self.PMAX)
            net.gen.vm_pu = self.current_gen_vm_pu[:len(net.gen)]
            if len(net.ext_grid):
                net.ext_grid.vm_pu = self.current_ext_vm_pu
            ls = self.target_demand/net.load.p_mw.sum(); net.load.p_mw*=ls; net.load.q_mvar*=ls
            net.bus.min_vm_pu=VMIN; net.bus.max_vm_pu=VMAX
            ac_info = {}
            ac_ok=False; vv=999; ov=999; vmin=0.0; vmax=0.0
            try:
                pp.runpp(net, calculate_voltage_angles=True, numba=False,
                         max_iteration=100, enforce_q_lims=True)
                if net.converged:
                    vm = net.res_bus.vm_pu.values
                    low_gap = np.maximum(VMIN - vm, 0.0)
                    high_gap = np.maximum(vm - VMAX, 0.0)
                    vmin=float(vm.min()); vmax=float(vm.max())
                    vv=int((low_gap>0.0).sum() + (high_gap>0.0).sum())
                    v_mag = float(low_gap.sum() + high_gap.sum())
                    v_gap = float(max(low_gap.max(initial=0.0), high_gap.max(initial=0.0)))
                    ol = net.res_line.loading_percent.values
                    ov=int((ol>100).sum())
                    loading_max = float(ol.max()) if len(ol) else 0.0
                    trafo_ov = 0
                    if len(net.res_trafo)>0:
                        tl = net.res_trafo.loading_percent.values
                        trafo_ov = int((tl>100).sum())
                        ov += trafo_ov
                        loading_max = max(loading_max, float(tl.max()) if len(tl) else 0.0)
                    ac_ok=True
                    ac_info = {
                        "v_violation_mag_pu": v_mag,
                        "v_violation_max_pu": v_gap,
                        "loading_max": loading_max,
                        "trafo_overloaded": trafo_ov,
                    }
            except Exception:
                pass
            self._last_ac_ok = ac_ok; self._last_vv = vv; self._last_ov = ov
            self._last_vmin = vmin; self._last_vmax = vmax
        else:
            ac_info = {}
            ac_ok = self._last_ac_ok; vv = self._last_vv; ov = self._last_ov
            vmin = self._last_vmin; vmax = self._last_vmax

        do_n1 = (self.step_count == 1) or (self.step_count % N1_CHECK_EVERY_STEPS == 0)
        if do_n1:
            try:
                nv, _ = self._check_n1(self.current_power, self.dcnet); n1_ok = (nv==0)
            except Exception:
                nv = 999; n1_ok = False
            self._last_n1_ok = n1_ok; self._last_n1_viol = nv
        else:
            nv = self._last_n1_viol; n1_ok = self._last_n1_ok

        reward = BASE_REWARD; done = False
        info = {"converged": False, "v_violations": vv, "overloaded_lines": ov}
        full_pass = bool(do_ac and do_n1 and ac_ok and vv == 0 and ov == 0 and n1_ok)
        if full_pass:
            reward = N1_SECURE_REWARD
            done = True
            info["converged"] = True
            info["reason"] = "full_sced_pass"
        elif do_ac and ac_ok:
            v_mag = float(ac_info.get("v_violation_mag_pu", 0.0))
            v_gap = float(ac_info.get("v_violation_max_pu", 0.0))
            loading_max = float(ac_info.get("loading_max", 0.0))
            line_excess = max(0.0, loading_max - 100.0) / 100.0
            reward = BASE_REWARD + 50.0
            reward -= 3000.0 * min(v_mag, 0.20)
            reward -= 1000.0 * min(v_gap, 0.10)
            reward -= 100.0 * min(float(ov), 20.0)
            reward -= 50.0 * min(line_excess, 2.0)
            reward += R_VIOLATION_COEF * min(float(vv + ov) / 100.0, 1.0)
            if do_n1 and not n1_ok:
                reward -= 100.0 * min(float(nv), 20.0)
            info["reason"] = "voltage_violation" if vv else ("line_overload" if ov else "n1_violation")
        else:
            reward = BASE_REWARD - 200.0 if do_ac else BASE_REWARD
            info["reason"] = "acpf_not_converged" if do_ac else "not_checked"
        if self.step_count >= self.max_steps: done = True
        info.update(ac_info)
        info.update({"ac_converged": bool(ac_ok), "n1_violations": nv, "vmin": vmin, "vmax": vmax,
                     "ac_checked": do_ac, "n1_checked": do_n1})
        return self._get_obs(), reward, done, False, info

class GaussianActor(nn.Module):
    def __init__(self, od, ad, h=256):
        super().__init__(); self.t = nn.Sequential(nn.Linear(od,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU())
        self.mh = nn.Linear(h,ad); self.ls = nn.Linear(h,ad)
        nn.init.zeros_(self.mh.weight); nn.init.zeros_(self.mh.bias)
        nn.init.zeros_(self.ls.weight); nn.init.constant_(self.ls.bias, -3.0)
    def forward(self, x):
        h = self.t(x)
        m = torch.nan_to_num(self.mh(h), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        ls = torch.nan_to_num(self.ls(h), nan=-3.0, posinf=2.0, neginf=-5.0).clamp(-5.0,2.0)
        return m, ls
    def sample(self, x):
        m, ls = self.forward(x); s = ls.exp()
        n = torch.distributions.Normal(m,s); z = n.rsample(); a = torch.tanh(z)
        lp = (n.log_prob(z)-torch.log(1-a.pow(2)+1e-6)).sum(-1,keepdim=True)
        return a, lp, m
    def select_action(self, x, det=True):
        with torch.no_grad():
            xt = torch.FloatTensor(x).unsqueeze(0).to(DEVICE)
            return torch.tanh(self.forward(xt)[0]).cpu().numpy()[0] if det else self.sample(xt)[0].cpu().numpy()[0]

class TwinQ(nn.Module):
    def __init__(self,od,ad,h=256):
        super().__init__()
        self.q1 = nn.Sequential(nn.Linear(od+ad,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))
        self.q2 = nn.Sequential(nn.Linear(od+ad,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))
    def forward(self,x,a): xa=torch.cat([x,a],-1); return self.q1(xa), self.q2(xa)

class SACTrainer:
    def __init__(self,od,ad):
        self.actor = GaussianActor(od,ad).to(DEVICE)
        self.critic = TwinQ(od,ad).to(DEVICE); self.ct = TwinQ(od,ad).to(DEVICE)
        self.ct.load_state_dict(self.critic.state_dict())
        self.ao = torch.optim.Adam(self.actor.parameters(),lr=3e-4)
        self.co = torch.optim.Adam(self.critic.parameters(),lr=3e-4)
        self.la = nn.Parameter(torch.tensor(0.0).to(DEVICE)); self.alo = torch.optim.Adam([self.la],lr=3e-4)
        self.te = -0.5*ad; self.buf = deque(maxlen=BUFFER); self.gamma=0.99; self.tau=0.005
    def select_action(self, x, det=True): return self.actor.select_action(x, det)
    def update(self):
        if len(self.buf) < BATCH: return
        for _ in range(N_GRAD):
            idx = np.random.choice(len(self.buf), min(BATCH,len(self.buf)), replace=False)
            batch = [self.buf[i] for i in idx]
            s = torch.FloatTensor(np.array([b[0] for b in batch])).to(DEVICE)
            a = torch.FloatTensor(np.array([b[1] for b in batch])).to(DEVICE)
            r = torch.FloatTensor(np.array([b[2] for b in batch])).unsqueeze(-1).to(DEVICE)
            ns = torch.FloatTensor(np.array([b[3] for b in batch])).to(DEVICE)
            d = torch.FloatTensor(np.array([b[4] for b in batch])).unsqueeze(-1).to(DEVICE)
            with torch.no_grad():
                na, nlp, _ = self.actor.sample(ns)
                q1t, q2t = self.ct(ns, na); mq = torch.min(q1t,q2t)
                alpha = self.la.exp(); tq = r + self.gamma*(1-d)*(mq-alpha*nlp)
            q1, q2 = self.critic(s, a)
            cl = F.mse_loss(q1,tq)+F.mse_loss(q2,tq)
            self.co.zero_grad(); cl.backward(); nn.utils.clip_grad_norm_(self.critic.parameters(),1.0); self.co.step()
            ca, clp, _ = self.actor.sample(s); q1a, q2a = self.critic(s, ca)
            al = (alpha*clp-torch.min(q1a,q2a)).mean()
            self.ao.zero_grad(); al.backward(); nn.utils.clip_grad_norm_(self.actor.parameters(),1.0); self.ao.step()
            alo = -(self.la.exp()*(clp.detach()+self.te)).mean()
            self.alo.zero_grad(); alo.backward(); self.alo.step()
            for p, tp in zip(self.critic.parameters(), self.ct.parameters()):
                tp.data.copy_(self.tau*p.data+(1-self.tau)*tp.data)

def evaluate(trainer, env, ids):
    global AC_PF_CHECK_EVERY, N1_CHECK_EVERY_STEPS
    old_ac, old_n1 = AC_PF_CHECK_EVERY, N1_CHECK_EVERY_STEPS
    AC_PF_CHECK_EVERY = 1; N1_CHECK_EVERY_STEPS = 1
    ok=0; rewards_eval=[]; steps_eval=[]; runtime_eval=[]
    try:
        for sid in ids:
            eval_start = time.perf_counter()
            obs=env.reset(sample_idx=sid)
            for st in range(env.max_steps):
                ra=trainer.select_action(obs,det=True); lo,hi=env.get_action_bounds()
                a=lo+(ra+1.0)*0.5*(hi-lo); obs,r,done,_,info=env.step(a)
                if done: break
            passed = (
                info.get('converged',False)
                and info.get('v_violations',999)==0
                and info.get('overloaded_lines',999)==0
                and info.get('n1_violations',999)==0
            )
            if passed:
                ok+=1
            rewards_eval.append(float(r))
            steps_eval.append(int(st+1))
            runtime_eval.append(float(time.perf_counter()-eval_start))
        return {"passed": ok, "total": len(ids),
                "avg_reward": float(np.mean(rewards_eval)) if rewards_eval else 0.0,
                "avg_steps": float(np.mean(steps_eval)) if steps_eval else 0.0,
                "avg_runtime_sec": float(np.mean(runtime_eval)) if runtime_eval else 0.0}
    finally:
        AC_PF_CHECK_EVERY, N1_CHECK_EVERY_STEPS = old_ac, old_n1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=TOTAL_UPDATES)
    parser.add_argument(
        "--action-encoding",
        choices=["full9", "physics6", "matched6"],
        default="full9",
    )
    parser.add_argument("--basis-seed", type=int, default=20260721)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    results_dir = Path(args.output_dir).resolve() if args.output_dir else (
        RESULTS if args.action_encoding == "full9"
        else (PROJ / f"rl39_action_ablation_{args.action_encoding}")
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    print("="*60); print("RL39 SAC Training"); print("="*60)
    data_path = PROJ / "case39_routeb_rescuable_dataset.json"
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run paper_results/code/screen_route_b_smallcase_rescuable.py --case 39 first."
        )
    print(f"Dataset: {data_path}")
    with open(str(data_path)) as f: data=json.load(f)
    samples = data['samples']; tid, vid, xid = split_samples(samples)
    print(f"Samples: {len(samples)} (T:{len(tid)} V:{len(vid)} X:{len(xid)})")
    env = Env39(samples, tid, args.action_encoding, args.basis_seed)
    od = env._get_obs_dim(); ad = env.action_dim
    p_dim = env._n_shift_dirs + env._n_key_gens
    print(
        f"ObsDim={od}, ActDim={ad}, encoding={args.action_encoding}, "
        f"active_dim={p_dim}"
    )
    trainer = SACTrainer(od, ad)
    log = open(str(results_dir/"training.log"), 'w', buffering=1)
    log.write(
        f"RL39 Log\nObsDim={od} ActDim={ad} "
        f"encoding={args.action_encoding} active_dim={p_dim}\n"
    )
    best_val=-1; best_val_total=0; best_val_r=-1e18
    best_train=-1; best_train_total=0; best_train_r=-1e18
    best_st=None; t0=time.time(); history=[]; val_history=[]
    for u in range(1, args.updates+1):
        us=time.time(); eps=curriculum(samples, tid, u, N_EP)
        rewards=[]; ep_steps=[]; ep_pass=0; fail_reasons=Counter(); fail_examples=Counter()
        collect_start=time.time()
        for sid in eps:
            obs=env.reset(sample_idx=sid); er=0.0; info={}
            for st in range(env.max_steps+1):
                ra=trainer.select_action(obs,det=False); lo,hi=env.get_action_bounds()
                a=lo+(ra+1.0)*0.5*(hi-lo); no,r,done,_,info=env.step(a)
                trainer.buf.append((obs,ra,r,no,float(done))); er+=r; obs=no
                if done: break
            ep_steps.append(st+1)
            rewards.append(er)
            if info.get('reason') == 'full_sced_pass':
                ep_pass+=1
            else:
                reason=info.get('reason','unknown')
                fail_reasons[reason]+=1
                if reason == 'voltage_violation':
                    detail=(
                        f"{reason}:vv={info.get('v_violations','?')}:"
                        f"vgap={float(info.get('v_violation_max_pu',0.0)):.4f}"
                    )
                elif reason == 'n1_violation':
                    detail=f"{reason}:n1={info.get('n1_violations','?')}"
                elif reason == 'line_overload':
                    detail=f"{reason}:line={info.get('overloaded_lines','?')}"
                else:
                    detail=reason
                fail_examples[detail]+=1
        collect_dt=time.time()-collect_start
        update_start=time.time(); trainer.update(); update_dt=time.time()-update_start
        ar=np.mean(rewards) if rewards else 0
        if ep_pass > best_train or (ep_pass == best_train and ar > best_train_r):
            best_train=ep_pass; best_train_total=len(eps); best_train_r=ar
        val_txt='skip'; val_dt=0.0
        if u%EVAL_INT==0 or u==args.updates:
            val_start=time.time()
            val_res = evaluate(trainer, env, vid[:min(15,len(vid))])
            val_dt=time.time()-val_start
            val_txt=f"{val_res['passed']}/{val_res['total']} R={val_res['avg_reward']:+.0f}"
            if val_res['passed'] > best_val or (val_res['passed'] == best_val and val_res['avg_reward'] > best_val_r):
                best_val=val_res['passed']; best_val_total=val_res['total']; best_val_r=val_res['avg_reward']
                best_st={
                    'actor': copy.deepcopy(trainer.actor.state_dict()),
                    'update': u,
                    'val': 100*best_val/max(best_val_total,1),
                }
            val_history.append({
                "update": int(u),
                "passed": int(val_res["passed"]),
                "total": int(val_res["total"]),
                "avg_reward": float(val_res["avg_reward"]),
            })
        fail_txt=','.join([f"{k}:{v}" for k,v in fail_reasons.most_common()]) if fail_reasons else '-'
        fail_ex_txt=','.join([f"{k}:{v}" for k,v in fail_examples.most_common(2)]) if fail_examples else '-'
        steps_avg=float(np.mean(ep_steps)) if ep_steps else 0.0
        msg=(f"U{u:3d} R={ar:+.1f} train_pass={ep_pass}/{len(eps)} "
             f"best_train={best_train}/{best_train_total} val={val_txt} "
             f"best_val={best_val}/{best_val_total} fail={fail_txt} ex={fail_ex_txt} "
             f"steps_avg={steps_avg:.1f} collect={collect_dt:.0f}s upd={update_dt:.0f}s "
             f"val_time={val_dt:.0f}s total={time.time()-us:.0f}s")
        print(msg); log.write(msg+'\n')
        history.append({
            "update": int(u),
            "avg_reward": float(ar),
            "train_passed": int(ep_pass),
            "train_total": int(len(eps)),
            "avg_steps": float(steps_avg),
        })
    if best_st: trainer.actor.load_state_dict(best_st['actor'])
    test_res = evaluate(trainer, env, xid)
    xok, xto = test_res['passed'], test_res['total']
    xr = 100*xok/xto; elapsed=(time.time()-t0)/3600
    best_val_rate=100*best_val/max(best_val_total,1)
    msg=f"\nFINAL: Test={xok}/{xto} ({xr:.1f}%) BestVal={best_val_rate:.1f}% Time={elapsed:.1f}h"
    print(msg); log.write(msg+'\n'); log.close()
    if best_st: torch.save(best_st, str(results_dir/"best_model.pth"))
    result = {"case": "case39", "test_pass": xok, "test_total": xto,
              "test_pass_rate_pct": xr, "best_val_pct": best_val_rate,
              "vmin": VMIN, "vmax": VMAX, "action_encoding": args.action_encoding,
              "basis_seed": int(args.basis_seed), "active_action_dim": int(p_dim),
              "action_dim": int(ad), "actor_parameter_count": int(sum(p.numel() for p in trainer.actor.parameters())),
              "avg_test_reward": float(test_res["avg_reward"]),
              "avg_test_steps": float(test_res["avg_steps"]),
              "avg_test_runtime_sec": float(test_res["avg_runtime_sec"]),
              "training_runtime_sec": float(time.time()-t0), "updates": int(args.updates)}
    with open(str(results_dir/"main_experiment_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(str(results_dir/"training_history.json"), "w", encoding="utf-8") as f:
        json.dump({"config": result, "history": history, "val_history": val_history}, f, indent=2)

if __name__=="__main__": main()

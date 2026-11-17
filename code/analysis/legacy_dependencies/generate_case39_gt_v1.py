#!/usr/bin/env python3
"""GT data generator for IEEE 39-bus. Small but tightly constrained system."""
import json, time, copy, warnings, numpy as np
import pandapower as pp; from pandapower.networks import case39
import gurobipy as gp; from gurobipy import GRB
warnings.filterwarnings("ignore")
try:
    from paper_results.code.case39_route_b_utils import apply_case39_route_b_limits
except Exception:
    apply_case39_route_b_limits = None

OUTPUT = "case39_gt_v1_dataset.json"
FL_SCALE = 1.0; RESERVE_FRAC = 0.0; N_PER = 30; VMIN = 0.90; VMAX = 1.10
DEMAND_GW = [6.2, 6.4, 6.6, 6.7]
MAX_ITERS = 50; QP_TIME = 60; DELTA = 30.0

class Case39Network:
    def __init__(self):
        self.net_base = case39(); self.n_gen = len(self.net_base.gen)
        self.n_bus = len(self.net_base.bus); self.n_line = len(self.net_base.line)
        self.n_trafo = len(self.net_base.trafo)
        self.gen_bus = self.net_base.gen.bus.values.astype(int)
        self.gen_pmin = np.clip(self.net_base.gen.min_p_mw.values.astype(float), 0, 1)
        self.gen_pmax = self.net_base.gen.max_p_mw.values.astype(float)
        pc = self.net_base.poly_cost; self.cp1 = np.zeros(self.n_gen); self.cp2 = np.zeros(self.n_gen)
        for _, row in pc.iterrows():
            g = int(row['element'])
            if 0 <= g < self.n_gen:
                self.cp2[g] = float(row.get('cp2_eur_per_mw2', 0)); self.cp1[g] = float(row.get('cp1_eur_per_mw', 30))
        if self.cp2.sum() == 0:
            self.cp2[:] = 0.002; self.cp1[:] = np.linspace(35, 20, self.n_gen)
        self.base_load = self.net_base.load.p_mw.sum()
        print(f"case39: {self.n_gen} gens, {self.n_bus} buses, {self.n_line} lines, cap={self.gen_pmax.sum():.0f}MW, load={self.base_load:.0f}MW")
        self.S = self._compute_S(); self.flow_limit = self._calibrate_limits()
        self.effective_limit = self.flow_limit / FL_SCALE
        self.lodf = self._compute_lodf()
        self.n1_list = []
        for k in range(self.n_line):
            for l in range(self.n_line):
                if l != k and abs(self.lodf[l,k]) >= 0.05 and self.effective_limit[l] >= 1:
                    self.n1_list.append((k, l, +1)); self.n1_list.append((k, l, -1))
        print(f"N1 constraints: {len(self.n1_list)}")

    def _acpf(self, P, ls=None):
        n = copy.deepcopy(self.net_base); n.gen.p_mw = np.clip(P, 0, self.gen_pmax)
        if ls is None: ls = P.sum() / self.base_load
        n.load.p_mw *= ls; n.load.q_mvar *= ls
        if apply_case39_route_b_limits is not None:
            apply_case39_route_b_limits(n)
        n.bus.min_vm_pu = VMIN; n.bus.max_vm_pu = VMAX
        try:
            pp.runpp(n, numba=False, max_iteration=100)
            return (n.res_line.p_from_mw.values.copy() if n.converged and len(n.res_line) > 0 else np.zeros(self.n_line)), n.converged
        except: return np.zeros(self.n_line), False

    def _compute_S(self):
        Pb = np.ones(self.n_gen) * (6300.0 / self.n_gen); Pb = np.clip(Pb, self.gen_pmin, self.gen_pmax)
        fb, ok = self._acpf(Pb)
        if not ok: return np.zeros((self.n_line, self.n_gen))
        S = np.zeros((self.n_line, self.n_gen))
        for i in range(self.n_gen):
            Pp = Pb.copy(); Pp[i] += DELTA; Pp = np.clip(Pp, self.gen_pmin, self.gen_pmax)
            fp, okp = self._acpf(Pp)
            if okp: S[:, i] = (fp - fb) / DELTA
        return np.clip(S, -100, 100)

    def _calibrate_limits(self):
        L = np.ones(self.n_line) * 1000.0
        for i, row in self.net_base.line.iterrows():
            vn = self.net_base.bus.vn_kv.values[int(row['from_bus'])]
            L[i] = 2000 if vn >= 300 else 1200
        try:
            nt = copy.deepcopy(self.net_base); ls = 7000/self.base_load
            nt.load.p_mw *= ls; nt.load.q_mvar *= ls; pp.runopp(nt, verbose=False, numba=False)
            if nt.converged:
                for i in range(min(len(L), len(nt.res_line))):
                    f = abs(nt.res_line.p_from_mw.values[i])
                    if f > 0.1: L[i] = max(L[i], f * 6)
        except: pass
        return L

    def _compute_lodf(self):
        L = self.n_line; lodf = np.zeros((L, L))
        Pr = np.ones(self.n_gen) * (6500.0/self.n_gen); Pr = np.clip(Pr, self.gen_pmin, self.gen_pmax)
        fb, ok = self._acpf(Pr)
        if not ok: return lodf
        for k in range(L):
            if abs(fb[k]) < 0.5: continue
            no = copy.deepcopy(self.net_base); no.line.at[k, 'in_service'] = False
            no.gen.p_mw = np.clip(Pr, 0, self.gen_pmax); ls = Pr.sum()/self.base_load
            no.load.p_mw *= ls; no.load.q_mvar *= ls
            if apply_case39_route_b_limits is not None:
                apply_case39_route_b_limits(no)
            no.bus.min_vm_pu = VMIN; no.bus.max_vm_pu = VMAX
            try:
                pp.runpp(no, numba=False, max_iteration=100)
                if no.converged and len(no.res_line) == L:
                    df = no.res_line.p_from_mw.values - fb
                    for l in range(L):
                        if l != k and abs(fb[k]) > 0.01: lodf[l, k] = df[l] / fb[k]
            except: pass
        return np.clip(lodf, -5, 5)

def solve_qp(dc, demand, noise=0.02):
    cp1 = dc.cp1*(1+np.random.randn(dc.n_gen)*noise); cp2 = np.maximum(dc.cp2, 1e-6)
    m = gp.Model("ED"); m.setParam("OutputFlag", 0); m.setParam("TimeLimit", QP_TIME)
    p = m.addMVar(dc.n_gen, lb=dc.gen_pmin, ub=dc.gen_pmax)
    m.setObjective(cp1@p + p@np.diag(cp2)@p, GRB.MINIMIZE)
    m.addConstr(p.sum() == demand); m.addConstr(p.sum() <= dc.gen_pmax.sum() - RESERVE_FRAC*demand)
    m.optimize()
    return (p.X.copy(), m.ObjVal) if m.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT] and m.SolCount > 0 else (None, None)

def check_n1(P, dc):
    f = dc.S @ P; viol = 0; bad = set()
    for k in range(dc.n_line):
        if abs(f[k]) < 0.1: continue
        fp = f + dc.lodf[:, k] * f[k]
        for l in range(dc.n_line):
            if l != k and abs(fp[l]) > dc.effective_limit[l]: viol += 1; bad.add(l)
    return viol, bad

def correct_n1(dc, demand, Pi):
    Pc = Pi.copy()
    for it in range(MAX_ITERS):
        nv, bad = check_n1(Pc, dc)
        if nv == 0: return Pc, it, True
        m = gp.Model("N1"); m.setParam("OutputFlag", 0); m.setParam("TimeLimit", QP_TIME)
        p = m.addMVar(dc.n_gen, lb=dc.gen_pmin, ub=dc.gen_pmax)
        m.setObjective(dc.cp1@p + p@np.diag(np.maximum(dc.cp2,1e-6))@p, GRB.MINIMIZE)
        m.addConstr(p.sum() == demand); m.addConstr(p.sum() <= dc.gen_pmax.sum() - RESERVE_FRAC*demand)
        fv = dc.S @ p; n_added = 0
        for k, l, sgn in dc.n1_list:
            if l not in bad or n_added >= 100: continue
            coef = dc.S[l,:] + dc.lodf[l,k]*dc.S[k,:]
            rhs = dc.effective_limit[l] - dc.lodf[l,k]*dc.S[k,:]@Pc
            m.addConstr(coef@p <= rhs if sgn>0 else -coef@p <= rhs); n_added += 1
        m.optimize()
        if m.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT] and m.SolCount > 0: Pc = p.X.copy()
        else: return Pc, it, False
    return Pc, MAX_ITERS, False

def verify_acpf(dc, P, demand):
    n = copy.deepcopy(dc.net_base); n.gen.p_mw = np.clip(P, 0, dc.gen_pmax)
    ls = demand/dc.base_load; n.load.p_mw *= ls; n.load.q_mvar *= ls
    if apply_case39_route_b_limits is not None:
        apply_case39_route_b_limits(n)
    n.bus.min_vm_pu = VMIN; n.bus.max_vm_pu = VMAX
    try:
        pp.runpp(n, numba=False, max_iteration=100)
        if not n.converged: return {"ac_pf_ok": False, "v_violations": 999, "overloaded_lines": 999}
        vmin = float(n.res_bus.vm_pu.min())
        vv = int((n.res_bus.vm_pu < VMIN).sum())
        ov = int((n.res_line.loading_percent > 100).sum())
        to = int((n.res_trafo.loading_percent > 100).sum()) if len(n.res_trafo) > 0 else 0
        return {"ac_pf_ok": True, "v_violations": vv, "overloaded_lines": ov, "trafo_overloaded": to,
                "vmin": vmin, "vmax": float(n.res_bus.vm_pu.max()), "max_line_load": float(n.res_line.loading_percent.max())}
    except: return {"ac_pf_ok": False, "v_violations": 999, "overloaded_lines": 999}

def main():
    print("="*60); print("IEEE 39-bus GT Generator"); print("="*60)
    t0 = time.time(); dc = Case39Network(); samples = []
    for gw in DEMAND_GW:
        demand = gw*1000; print(f"\n[{gw}GW] Generating {N_PER} samples...")
        for i in range(N_PER):
            sid = len(samples)
            Ped, obj = solve_qp(dc, demand)
            if Ped is None: continue
            nv_init, _ = check_n1(Ped, dc)
            Pcorr, n_iter, n1_ok = correct_n1(dc, demand, Ped) if nv_init > 0 else (Ped, 0, True)
            nv_final, _ = check_n1(Pcorr, dc) if n1_ok else (nv_init, set())
            ac = verify_acpf(dc, Pcorr, demand)
            is_complete = (ac["ac_pf_ok"] and ac["v_violations"]==0 and ac["overloaded_lines"]==0
                          and ac.get("trafo_overloaded",0)==0 and nv_final==0)
            samples.append({"id": sid, "demand_gw": gw, "demand_mw": demand, "dispatch": Pcorr.tolist(),
                           "n_gen": dc.n_gen, "is_complete_sced": is_complete,
                           "n1_initial_violations": int(nv_init), "n1_final_violations": int(nv_final),
                           "n1_correction_iters": n_iter, **ac})
            if (i+1)%10==0:
                st = "COMPLETE" if is_complete else "PARTIAL"
                print(f"  [{i+1:3d}] {st} V={ac.get('vmin',0):.3f} N1i={nv_init} N1f={nv_final}")
    out = {"system": "IEEE 39-bus", "n_gen": dc.n_gen, "n_bus": dc.n_bus, "n_line": dc.n_line,
           "base_load_mw": float(dc.base_load), "demand_levels_gw": DEMAND_GW,
           "n_total": len(samples), "n_complete": sum(1 for s in samples if s["is_complete_sced"]),
           "samples": samples}
    with open(OUTPUT, 'w') as f: json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nDone: {len(samples)} samples, {out['n_complete']} complete, {time.time()-t0:.0f}s")

if __name__ == "__main__": main()

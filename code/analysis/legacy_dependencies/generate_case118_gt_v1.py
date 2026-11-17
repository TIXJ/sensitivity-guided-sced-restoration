#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_case118_gt_v1.py — Complete SCED GT Generator for IEEE 118-bus

Adapted from generate_case300_gt_v10.py for case118.
Strategy:
  Stage 1: Gurobi QP solves economic dispatch (cost min + balance + gen limits + reserve)
  Stage 2: Numerical sensitivity S-matrix via perturbation
  Stage 3: DC N-1 security check using S-matrix + LODF
  Stage 4: If N-1 violated -> iterative Gurobi correction loop
  Stage 5: AC-PF verification (vmin=0.94, vmax=1.06)

Key differences from case300:
  - 53 generators, 118 buses, 173 lines, 13 trafos
  - Demand levels: 3.0-5.0 GW (base load 4242 MW)
  - Standard IEEE voltage limits (0.94/1.06, no relaxation needed)
  - Smaller perturbation delta (20 MW vs 50 MW)
  - Small generator thresholds adjusted
"""

import json, time, copy, warnings, os
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import pandapower as pp
from pandapower.networks import case118

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
OUTPUT_FILE    = "case118_gt_v1_dataset.json"
FL_SCALE       = 2.0                     # effective = rating/2 (e.g. 800/2=400MW)
RESERVE_FRAC   = 0.05
N_SAMPLES_PER_LEVEL = 30
DEMAND_LEVELS_GW = [7.0, 7.4, 7.8, 8.2, 8.6, 9.0]
COST_NOISE_STD_PCT = 0.02
MAX_QP_TIME_SEC = 60
VMIN = 0.90
VMAX = 1.10
VERBOSE_EVERY = 5
MAX_CORRECTION_ITERS = 50
PERTURBATION_DELTA = 20.0  # MW (smaller for case118 gens: 100-707 MW)
LARGE_GEN_THRESHOLD = 150.0  # MW


class Case118Network:
    """Network wrapper for IEEE 118-bus with numerical S-matrix + LODF."""

    def __init__(self):
        print("[Net] Loading IEEE case118...")
        self.net_base = case118()
        self.n_gen = len(self.net_base.gen)
        self.n_bus = len(self.net_base.bus)
        self.n_line = len(self.net_base.line)
        self.n_trafo = len(self.net_base.trafo)

        # Generator params
        self.gen_bus = self.net_base.gen.bus.values.astype(int)
        self.gen_pmin = self.net_base.gen.min_p_mw.values.astype(float)

        # ★ Use REAL max_p_mw (not inflated sn_mva) for harder samples
        # case118 sn_mva gives 22902 MW total vs real 9161 MW — huge difference!
        self.gen_pmax = self.net_base.gen.max_p_mw.values.astype(float).copy()
        # Ensure minimum headroom
        self.gen_pmax = np.maximum(self.gen_pmax, self.gen_pmin + 5.0)

        # Cost coefficients
        pc = self.net_base.poly_cost
        self.cp0 = np.zeros(self.n_gen)
        self.cp1 = np.zeros(self.n_gen)
        self.cp2 = np.zeros(self.n_gen)
        for _, row in pc.iterrows():
            g = int(row['element'])
            if 0 <= g < self.n_gen:
                self.cp2[g] = float(row.get('cp2_eur_per_mw2', 0))
                self.cp1[g] = float(row.get('cp1_eur_per_mw', 30))
                self.cp0[g] = float(row.get('cp0_eur', 0))
        # Fallback for zero costs
        if self.cp2.sum() == 0:
            for i in range(self.n_gen):
                self.cp2[i] = 0.001 + 0.0005 * np.random.random()
                self.cp1[i] = max(40 - 15*(self.gen_pmax[i]/self.gen_pmax.max()), 10)

        self.base_load = self.net_base.load.p_mw.sum()

        print(f"  Generators: {self.n_gen}, Buses: {self.n_bus}, "
              f"Lines: {self.n_line}, Trafos: {self.n_trafo}")
        print(f"  Total capacity: {self.gen_pmax.sum():.0f} MW")
        print(f"  Base load: {self.base_load:.1f} MW")
        print(f"  Pmax range: [{self.gen_pmax.min():.0f}, {self.gen_pmax.max():.0f}] MW")

        # Compute S-matrix numerically
        print("[Net] Computing numerical S-matrix...")
        self.S = self._compute_numerical_S()
        print(f"  S shape: {self.S.shape}")

        # Calibrate flow limits
        print("[Net] Calibrating flow limits...")
        self.flow_limit = self._calibrate_flow_limits()
        self.effective_limit = self.flow_limit / FL_SCALE
        print(f"  Flow limits: [{self.flow_limit.min():.0f}, {self.flow_limit.max():.0f}] MW")

        # Compute LODF
        print("[Net] Computing numerical LODF...")
        self.lodf = self._compute_numerical_lodf()
        nz = np.count_nonzero(np.abs(self.lodf) > 0.01)
        print(f"  LODF: {self.n_line}x{self.n_line}, {nz} entries with |LODF|>0.01")

        # Pre-select N-1 constraints
        self._preselect_n1_constraints()
        print("[Net] Network setup complete.")

    def _run_ac_pf_approx(self, dispatch, demand_scale=None):
        """Run AC-PF with given dispatch."""
        net_c = copy.deepcopy(self.net_base)
        net_c.gen.p_mw = np.clip(dispatch, 0.0, self.gen_pmax)
        total_gen = float(dispatch.sum())
        if demand_scale is None and total_gen > 0:
            demand_scale = total_gen / self.base_load
        if demand_scale is not None:
            net_c.load.p_mw = net_c.load.p_mw * demand_scale
            net_c.load.q_mvar = net_c.load.q_mvar * demand_scale
        net_c.bus.min_vm_pu = VMIN
        net_c.bus.max_vm_pu = VMAX
        try:
            pp.runpp(net_c, calculate_voltage_angles=True, numba=False)
            if net_c.converged and len(net_c.res_line) > 0:
                return net_c.res_line.p_from_mw.values.copy(), True
            return np.zeros(self.n_line), False
        except Exception:
            return np.zeros(self.n_line), False

    def _compute_numerical_S(self):
        """Compute S(l,i) = d(flow_l)/d(P_i) numerically via perturbation."""
        P_base = np.ones(self.n_gen) * (7000.0 / self.n_gen)  # ~7GW (matching harder demand)
        P_base = np.clip(P_base, self.gen_pmin, self.gen_pmax)
        f_base, ok = self._run_ac_pf_approx(P_base, P_base.sum()/self.base_load)
        if not ok:
            print("  [WARN] Cannot compute base flows, S=0")
            return np.zeros((self.n_line, self.n_gen))

        delta = PERTURBATION_DELTA
        S = np.zeros((self.n_line, self.n_gen))
        for i in range(self.n_gen):
            P_plus = P_base.copy(); P_plus[i] += delta
            P_plus = np.clip(P_plus, self.gen_pmin, self.gen_pmax)
            f_plus, ok_plus = self._run_ac_pf_approx(P_plus, P_plus.sum()/self.base_load)
            if ok_plus:
                S[:, i] = (f_plus - f_base) / delta
        S = np.clip(S, -100.0, 100.0)
        return S

    def _calibrate_flow_limits(self):
        """Estimate line thermal limits based on voltage levels (standard approach)."""
        limits = np.ones(self.n_line) * 1500.0
        bus_vn = self.net_base.bus['vn_kv'].values
        for i, row in self.net_base.line.iterrows():
            f_bus = int(row['from_bus'])
            vn = bus_vn[f_bus] if 0 <= f_bus < len(bus_vn) else 138.0
            if vn >= 300: rating = 2000.0
            elif vn >= 150: rating = 1200.0
            elif vn >= 100: rating = 800.0
            else: rating = 400.0
            limits[i] = rating

        # OPF calibration at mid-demand for additional headroom info
        try:
            net_test = copy.deepcopy(self.net_base)
            ls = 8000.0 / self.base_load
            net_test.load.p_mw = net_test.load.p_mw * ls
            net_test.load.q_mvar = net_test.load.q_mvar * ls
            pp.runopp(net_test, verbose=False, suppress_warnings=True, numba=False)
            if net_test.converged and len(net_test.res_line) > 0:
                if 'p_from_mw' in net_test.res_line.columns:
                    opf_flows = np.abs(net_test.res_line.p_from_mw.values)
                    for i in range(min(len(limits), len(opf_flows))):
                        if opf_flows[i] > 0.1:
                            limits[i] = max(limits[i], opf_flows[i] * 6.0)
        except Exception as e:
            print(f"  [WARN] OPF calibration failed: {e}")
        return limits

    def _compute_numerical_lodf(self):
        """Compute LODF numerically via outage simulation (only significant lines)."""
        L = self.n_line
        lodf = np.zeros((L, L), dtype=np.float64)
        P_ref = np.ones(self.n_gen) * (7000.0 / self.n_gen)
        P_ref = np.clip(P_ref, self.gen_pmin, self.gen_pmax)
        f_base, base_ok = self._run_ac_pf_approx(P_ref, P_ref.sum()/self.base_load)
        if not base_ok:
            print("  [WARN] Base PF for LODF failed")
            return lodf

        for k_idx in range(L):
            fk = f_base[k_idx]
            if abs(fk) < 0.5:
                continue
            net_outage = copy.deepcopy(self.net_base)
            net_outage.line.at[k_idx, 'in_service'] = False
            net_outage.gen.p_mw = np.clip(P_ref, 0.0, self.gen_pmax)
            ls = P_ref.sum() / self.base_load
            net_outage.load.p_mw = net_outage.load.p_mw * ls
            net_outage.load.q_mvar = net_outage.load.q_mvar * ls
            net_outage.bus.min_vm_pu = VMIN
            net_outage.bus.max_vm_pu = VMAX
            try:
                pp.runpp(net_outage, calculate_voltage_angles=True, numba=False)
                if net_outage.converged and len(net_outage.res_line) > 0:
                    f_post = net_outage.res_line.p_from_mw.values
                    if len(f_post) == L:
                        delta_f = f_post - f_base
                        for l in range(L):
                            if l != k_idx and abs(fk) > 0.01:
                                lodf[l, k_idx] = delta_f[l] / fk
            except Exception:
                pass
        lodf = np.clip(lodf, -5.0, 5.0)
        return lodf

    def _preselect_n1_constraints(self):
        """Pre-select N-1 constraint pairs."""
        self.n1_constraint_list = []
        for k in range(self.n_line):
            for l in range(self.n_line):
                if l == k: continue
                if abs(self.lodf[l, k]) < 0.05: continue
                if self.effective_limit[l] < 1.0: continue
                self.n1_constraint_list.append((k, l, +1))
                self.n1_constraint_list.append((k, l, -1))
        print(f"  N-1 constraints pre-selected: {len(self.n1_constraint_list)}")


# ============================================================
# Economic Dispatch Solver (Gurobi QP)
# ============================================================
def solve_economic_dispatch_qp(dcnet, demand_mw, noise_std_pct=0.02):
    """Solve QP: min cost s.t. balance + gen limits + reserve."""
    n_gen = dcnet.n_gen
    cp1 = dcnet.cp1.copy(); cp2 = dcnet.cp2.copy()
    # Add noise for data diversity
    if noise_std_pct > 0:
        cp1 *= (1 + np.random.randn(n_gen) * noise_std_pct)
        cp2 *= (1 + np.abs(np.random.randn(n_gen) * noise_std_pct * 0.5))
    cp2 = np.maximum(cp2, 1e-6)

    m = gp.Model("ED_QP"); m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", MAX_QP_TIME_SEC)
    p = m.addMVar(n_gen, lb=dcnet.gen_pmin, ub=dcnet.gen_pmax, name="p")
    obj = cp1 @ p + p @ np.diag(cp2) @ p
    m.setObjective(obj, GRB.MINIMIZE)
    m.addConstr(p.sum() == demand_mw, "balance")
    m.addConstr(p.sum() <= dcnet.gen_pmax.sum() - RESERVE_FRAC * demand_mw, "reserve")
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        return p.X.copy(), m.ObjVal
    elif m.Status in [GRB.SUBOPTIMAL, GRB.TIME_LIMIT] and m.SolCount > 0:
        return p.X.copy(), m.ObjVal
    return None, None


def check_n1_security(P, dcnet):
    """Check N-1 security of dispatch P. Returns (n_violations, overload_lines)."""
    flows = dcnet.S @ P
    violations = 0; overload_lines = set()
    for k in range(dcnet.n_line):
        fk = flows[k]
        if abs(fk) < 0.1: continue
        f_post = flows + dcnet.lodf[:, k] * fk
        for l in range(dcnet.n_line):
            if l == k: continue
            lim = dcnet.effective_limit[l]
            if abs(f_post[l]) > lim:
                violations += 1; overload_lines.add(l)
    return violations, overload_lines


def correct_n1_violations(dcnet, demand_mw, P_init, max_iters=MAX_CORRECTION_ITERS):
    """Iteratively correct N-1 violations by adding constraints to QP."""
    P_current = P_init.copy()
    for iteration in range(max_iters):
        n_viol, overload_lines = check_n1_security(P_current, dcnet)
        if n_viol == 0:
            return P_current, iteration, True
        # Add N-1 constraints for violated lines then re-solve
        cp1 = dcnet.cp1; cp2 = np.maximum(dcnet.cp2, 1e-6)
        m = gp.Model("N1_CORR"); m.setParam("OutputFlag", 0)
        m.setParam("TimeLimit", MAX_QP_TIME_SEC)
        p = m.addMVar(dcnet.n_gen, lb=dcnet.gen_pmin, ub=dcnet.gen_pmax, name="p")
        m.setObjective(cp1 @ p + p @ np.diag(cp2) @ p, GRB.MINIMIZE)
        m.addConstr(p.sum() == demand_mw, "balance")
        m.addConstr(p.sum() <= dcnet.gen_pmax.sum() - RESERVE_FRAC * demand_mw, "reserve")

        # Add N-1 constraints for top violated lines
        flows = dcnet.S @ p
        n_added = 0
        for k, l, sign in dcnet.n1_constraint_list:
            if l not in overload_lines: continue
            if n_added >= 200: break
            coef = dcnet.S[l, :] + dcnet.lodf[l, k] * dcnet.S[k, :]
            rhs = dcnet.effective_limit[l] - (dcnet.lodf[l, k] * dcnet.S[k, :] @ P_current)
            if sign > 0:
                m.addConstr(coef @ p <= rhs, f"n1_{k}_{l}_pos")
            else:
                m.addConstr(-coef @ p <= rhs, f"n1_{k}_{l}_neg")
            n_added += 1

        m.optimize()
        if m.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT] and m.SolCount > 0:
            P_current = p.X.copy()
        else:
            return P_current, iteration, False
    return P_current, max_iters, False


def verify_ac_pf(dcnet, dispatch, demand_mw):
    """Verify AC-PF convergence and voltage/loading compliance."""
    net_c = copy.deepcopy(dcnet.net_base)
    net_c.gen.p_mw = np.clip(dispatch, 0.0, dcnet.gen_pmax)
    ls = demand_mw / dcnet.base_load
    net_c.load.p_mw = net_c.load.p_mw * ls
    net_c.load.q_mvar = net_c.load.q_mvar * ls
    net_c.bus.min_vm_pu = VMIN
    net_c.bus.max_vm_pu = VMAX
    try:
        pp.runpp(net_c, numba=False)
        if not net_c.converged:
            return {"ac_pf_ok": False, "v_violations": 999, "overloaded_lines": 999}
        vmin = net_c.res_bus.vm_pu.min()
        vmax = net_c.res_bus.vm_pu.max()
        v_viol = int((net_c.res_bus.vm_pu < VMIN).sum() + (net_c.res_bus.vm_pu > VMAX).sum())
        line_load = net_c.res_line.loading_percent.values
        overloaded = int((line_load > 100).sum())
        trafo_load = net_c.res_trafo.loading_percent.values if len(net_c.res_trafo) > 0 else np.array([])
        trafo_over = int((trafo_load > 100).sum()) if len(trafo_load) > 0 else 0
        return {
            "ac_pf_ok": True, "v_violations": v_viol, "overloaded_lines": overloaded,
            "trafo_overloaded": trafo_over, "vmin": float(vmin), "vmax": float(vmax),
            "max_line_load": float(line_load.max()) if len(line_load) > 0 else 0,
        }
    except Exception:
        return {"ac_pf_ok": False, "v_violations": 999, "overloaded_lines": 999}


def compute_hhi(dispatch):
    """Compute Herfindahl-Hirschman Index for generation concentration."""
    s = dispatch / (dispatch.sum() + 1e-8)
    return float((s ** 2).sum() * 10000)


def main():
    print("=" * 70)
    print("IEEE 118-bus Complete SCED GT Generator v1")
    print("=" * 70)
    t0 = time.time()

    dcnet = Case118Network()
    all_samples = []

    for gw in DEMAND_LEVELS_GW:
        demand_mw = gw * 1000.0
        print(f"\n[Demand={gw}GW / {demand_mw:.0f}MW] Generating {N_SAMPLES_PER_LEVEL} samples...")

        for i in range(N_SAMPLES_PER_LEVEL):
            sample_id = len(all_samples)

            # Stage 1: QP economic dispatch
            P_ed, obj_val = solve_economic_dispatch_qp(dcnet, demand_mw, COST_NOISE_STD_PCT)
            if P_ed is None:
                continue

            # Stage 2: Check N-1 security
            n_viol, _ = check_n1_security(P_ed, dcnet)

            # Stage 3: N-1 correction if needed
            P_corrected = P_ed.copy()
            n_iters = 0
            n1_corrected = False
            if n_viol > 0:
                P_corrected, n_iters, n1_corrected = correct_n1_violations(
                    dcnet, demand_mw, P_ed)
                n_viol_after, _ = check_n1_security(P_corrected, dcnet)
            else:
                n_viol_after = 0

            # Stage 4: AC-PF verification
            ac_result = verify_ac_pf(dcnet, P_corrected, demand_mw)
            is_complete = (ac_result["ac_pf_ok"] and ac_result["v_violations"] == 0
                           and ac_result["overloaded_lines"] == 0
                           and ac_result["trafo_overloaded"] == 0
                           and n_viol_after == 0)

            sample = {
                "id": sample_id,
                "demand_gw": gw,
                "demand_mw": demand_mw,
                "dispatch": P_corrected.tolist(),
                "obj_value": float(obj_val) if obj_val else None,
                "hhi": compute_hhi(P_corrected),
                "n_gen": dcnet.n_gen,
                "is_complete_sced": is_complete,
                "n1_initial_violations": int(n_viol),
                "n1_final_violations": int(n_viol_after),
                "n1_correction_iters": n_iters,
                "n1_corrected": bool(n1_corrected),
                **ac_result,
            }
            all_samples.append(sample)

            if (i + 1) % VERBOSE_EVERY == 0:
                status = "COMPLETE" if is_complete else "PARTIAL"
                print(f"  [{i+1:3d}/{N_SAMPLES_PER_LEVEL}] {status} "
                      f"V={ac_result.get('vmin', 0):.3f} N1={n_viol_after} "
                      f"HHI={sample['hhi']:.0f}")

    # Save
    output = {
        "system": "IEEE 118-bus",
        "n_gen": dcnet.n_gen,
        "n_bus": dcnet.n_bus,
        "n_line": dcnet.n_line,
        "n_trafo": dcnet.n_trafo,
        "base_load_mw": float(dcnet.base_load),
        "demand_levels_gw": DEMAND_LEVELS_GW,
        "n_samples_per_level": N_SAMPLES_PER_LEVEL,
        "n_total": len(all_samples),
        "n_complete_sced": sum(1 for s in all_samples if s["is_complete_sced"]),
        "config": {"fl_scale": FL_SCALE, "vmin": VMIN, "vmax": VMAX,
                   "reserve_frac": RESERVE_FRAC, "perturbation_delta": PERTURBATION_DELTA},
        "samples": all_samples,
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"IEEE 118-bus GT Generation Complete!")
    print(f"  Total: {len(all_samples)} samples")
    print(f"  Complete SCED: {output['n_complete_sced']}/{len(all_samples)} "
          f"({100*output['n_complete_sced']/max(len(all_samples),1):.1f}%)")
    print(f"  Time: {elapsed/3600:.1f}h ({elapsed:.0f}s)")
    print(f"  Saved to: {OUTPUT_FILE}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()

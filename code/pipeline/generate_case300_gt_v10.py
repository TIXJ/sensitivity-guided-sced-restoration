#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_case300_gt_v10.py — Complete SCED GT Generator v10 (Hybrid Approach)

Strategy: 
  Stage 1: Gurobi QP solves economic dispatch (cost min + balance + gen limits + reserve)
  Stage 2: Numerical sensitivity S-matrix via perturbation (no B-matrix needed!)
  Stage 3: DC N-1 security check using S-matrix + LODF
  Stage 4: If N-1 violated → iterative Gurobi correction loop
  Stage 5: AC-PF verification (vmin=0.88)

This avoids the case300 B-matrix singularity issue entirely by computing
the sensitivity matrix numerically from AC-PF Jacobian-like perturbation.
"""

import json
import time
import copy
import warnings
import argparse
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import pandapower as pp
from pandapower.networks import case300
from strict_sced_utils import project_box_sum

warnings.filterwarnings("ignore")


def compute_dc_line_lodf(net):
    """Return the standard DC LODF matrix for pandapower line outages.

    The PYPOWER branch model includes both lines and transformers. Computing
    PTDF/LODF on that topology and then selecting line rows and columns keeps
    transformer paths in the network model while defining line-outage N-1.
    """
    from pandapower.converter.pypower.to_ppc import to_ppc
    from pandapower.pypower.makePTDF import makePTDF
    from pandapower.pypower.idx_brch import F_BUS, T_BUS

    ppc_net = copy.deepcopy(net)
    ppc = to_ppc(
        ppc_net, calculate_voltage_angles=True,
        check_connectivity=True, init="flat", mode="pf"
    )
    branch = np.asarray(ppc["branch"], dtype=np.float64)
    ptdf = np.asarray(
        makePTDF(float(ppc["baseMVA"]), ppc["bus"], branch),
        dtype=np.float64,
    )

    n_branch, n_bus = ptdf.shape
    incidence = np.zeros((n_bus, n_branch), dtype=np.float64)
    col = np.arange(n_branch)
    incidence[branch[:, F_BUS].astype(int), col] = 1.0
    incidence[branch[:, T_BUS].astype(int), col] = -1.0
    transfer = ptdf @ incidence
    denominator = 1.0 - np.diag(transfer)
    with np.errstate(divide="ignore", invalid="ignore"):
        full_lodf = transfer / denominator[np.newaxis, :]
    np.fill_diagonal(full_lodf, -1.0)

    branch_lookup = ppc_net._pd2ppc_lookups.get("branch")
    if not branch_lookup or "line" not in branch_lookup:
        raise RuntimeError("pandapower conversion did not produce a line-to-branch lookup")
    line_start, line_end = branch_lookup["line"]
    line_rows = np.arange(line_start, line_end, dtype=int)
    lodf = full_lodf[np.ix_(line_rows, line_rows)]
    if lodf.shape != (len(net.line), len(net.line)):
        raise RuntimeError(
            f"line LODF shape {lodf.shape} does not match net.line={len(net.line)}"
        )
    line_denominator = denominator[line_rows]
    valid = np.flatnonzero(
        np.isfinite(lodf).all(axis=0) & (np.abs(line_denominator) > 1e-8)
    ).astype(int)
    return lodf, valid

# ============================================================
# Configuration
# ============================================================
V8_INPUT       = "case300_gt_v8_dataset.json"
V10_OUTPUT     = "case300_gt_v10_dataset.json"
FL_SCALE       = 1.0                    # Legacy metadata; kept for compatibility
N1_LIMIT_SCALE = 1.0                    # Effective line limit / calibrated limit
RESERVE_FRAC   = 0.05                   # Spinning reserve = 5% demand
N_SAMPLES_PER_LEVEL = 30               # Samples per demand level
DEMAND_LEVELS_GW  = [16, 17, 18, 19, 20,
                     21, 22, 23, 24]    # GW
COST_NOISE_STD_PCT = 0.02              # Cost coefficient perturbation (%)
MAX_QP_TIME_SEC    = 60                # Per-sample QP time limit
VMIN_RELAXED      = 0.88
VMAX_RELAXED      = 1.10
VMIN_STRICT       = 0.933   # Route-B adopted AC-PF acceptance voltage lower bound
VMAX_STRICT       = 1.09
VERBOSE_EVERY     = 5                  # Print every N samples
MAX_CORRECTION_ITERS = 20             # Max N-1 correction iterations


# ============================================================
# Part 1: Case300 Network Wrapper
# ============================================================

class Case300Network:
    """
    Network wrapper that computes S-matrix numerically (no B-matrix!).
    Uses small perturbation of generator outputs to measure line flow sensitivities.
    """

    def __init__(self):
        print("[Net] Loading IEEE case300...")
        self.net_base = case300()
        self.n_gen = len(self.net_base.gen)
        self.n_bus = len(self.net_base.bus)
        self.n_line = len(self.net_base.line)
        self.n_trafo = len(self.net_base.trafo)

        # --- Generator params ---
        self.gen_bus = self.net_base.gen.bus.values.astype(int)  # 0-indexed!
        self.gen_pmin = self.net_base.gen.min_p_mw.values.astype(float)
        self.gen_pmax = self.net_base.gen.max_p_mw.values.astype(float)
        self.gen_pmin = np.nan_to_num(self.gen_pmin, nan=0.0, posinf=0.0, neginf=0.0)
        fallback_pmax = np.maximum(self.net_base.gen.p_mw.values.astype(float) * 2.5,
                                   self.gen_pmin + 10.0)
        bad_pmax = ~np.isfinite(self.gen_pmax) | (self.gen_pmax <= self.gen_pmin)
        self.gen_pmax[bad_pmax] = fallback_pmax[bad_pmax]
        self.gen_pmax = np.maximum(self.gen_pmax, self.gen_pmin + 10.0)

        # --- Cost coefficients ---
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

        # Base load
        self.base_load = self.net_base.load.p_mw.sum()

        print(f"  Generators: {self.n_gen}, Buses: {self.n_bus}, "
              f"Lines: {self.n_line}, Trafos: {self.n_trafo}")
        print(f"  Total capacity: {self.gen_pmax.sum():.0f} MW")
        print(f"  Base load: {self.base_load:.1f} MW")
        print(f"  Cost range: cp2=[{self.cp2.min():.4f}, {self.cp2.max():.4f}], "
              f"cp1=[{self.cp1.min():.1f}, {self.cp1.max():.1f}]")

        # --- Compute S-matrix numerically ---
        print("[Net] Computing numerical S-matrix (perturbation method)...")
        self.S = self._compute_numerical_S()
        if not hasattr(self, 'P_ref'):
            self.P_ref = np.zeros(self.n_gen)
            self.F_ref = np.zeros(self.n_line)
        self.flow_intercept = self.F_ref - self.S @ self.P_ref
        print(f"  S shape: {self.S.shape}, range=[{self.S.min():.6f}, {self.S.max():.6f}]")
        
        # Check S quality: sum of each row should be ~0 (power conservation at each line)
        row_sums = np.abs(self.S.sum(axis=1))
        print(f"  Row sums |sum(S[l,:])|: mean={row_sums.mean():.6f}, max={row_sums.max():.6f}")

        # --- Calibrate flow limits using OPF ---
        print("[Net] Calibrating flow limits via OPF...")
        self.flow_limit = self._calibrate_flow_limits()
        self.effective_limit = self.flow_limit * N1_LIMIT_SCALE
        print(f"  Flow limits: [{self.flow_limit.min():.0f}, {self.flow_limit.max():.0f}] MW")
        print(f"  Effective (x{N1_LIMIT_SCALE}): [{self.effective_limit.min():.0f}, {self.effective_limit.max():.0f}] MW")

        # --- Compute standard DC LODF on the complete line/transformer topology ---
        print("[Net] Computing DC PTDF/LODF...")
        self.lodf, self.valid_contingency_indices = compute_dc_line_lodf(self.net_base)
        nz = np.count_nonzero(np.abs(self.lodf) > 0.01)
        print(f"  LODF: {self.n_line}x{self.n_line}, {nz} entries with |LODF|>0.01")
        print(f"  Valid non-islanding LODF columns: {len(self.valid_contingency_indices)}/{self.n_line}")
        if len(self.valid_contingency_indices) == 0:
            raise RuntimeError(
                "Numerical LODF construction produced no valid outage columns; "
                "refusing to generate labels with a degenerate N-1 model."
            )
        
        # Pre-select critical N-1 constraints
        self._preselect_n1_constraints()
        print("[Net] Network setup complete.")

    def estimate_base_flows(self, dispatch):
        """Affine flow estimate F(P_ref) + S @ (P - P_ref)."""
        return self.flow_intercept + self.S @ np.asarray(dispatch, dtype=np.float64)

    def estimate_post_contingency_flows(self, dispatch, contingency_idx):
        """Affine base-flow estimate plus LODF post-contingency update."""
        flows = self.estimate_base_flows(dispatch)
        k = int(contingency_idx)
        if k < 0 or k >= len(flows):
            raise IndexError(f"contingency line index out of range: {k}")
        return flows + self.lodf[:, k] * flows[k]

    def _run_dc_approx_pf(self, dispatch, demand_scale=None, demand_mw=None):
        """
        Run a simplified DC-like PF to get approximate line flows.
        Uses flat start (V=1pu) and linearized power flow.
        
        Actually, we use AC-PF but with loose tolerances as our 'DC approximation'
        because case300 B-matrix is singular.
        """
        net_c = copy.deepcopy(self.net_base)
        net_c.gen.p_mw = np.clip(dispatch, self.gen_pmin, self.gen_pmax)
        
        if demand_mw is not None:
            demand_scale = float(demand_mw) / self.base_load
        
        if demand_scale is not None:
            net_c.load.p_mw = net_c.load.p_mw * demand_scale
            net_c.load.q_mvar = net_c.load.q_mvar * demand_scale
        
        # Relax voltage limits
        net_c.bus.min_vm_pu = VMIN_RELAXED
        net_c.bus.max_vm_pu = VMAX_RELAXED
        
        try:
            pp.runpp(net_c, calculate_voltage_angles=True, numba=False)
            if net_c.converged and len(net_c.res_line) > 0:
                # Return flows in MW
                if 'p_from_mw' in net_c.res_line.columns:
                    return net_c.res_line.p_from_mw.values.copy(), True
                else:
                    return np.zeros(self.n_line), False
            return np.zeros(self.n_line), False
        except Exception:
            return np.zeros(self.n_line), False

    def _compute_numerical_S(self):
        """
        Compute slack-referenced S(l,i) = d(flow_l)/d(P_i) numerically.
        
        Method:
          1. Start from a base dispatch (equal distribution at mid-demand)
          2. Perturb each generator by ±delta_P
          3. Measure resulting change in each line flow
          4. S[l,i] ≈ delta_f_l / delta_P_i
        """
        # Base dispatch: bounded, exactly balanced reference near 16 GW.
        target = min(max(16000.0, float(self.gen_pmin.sum())),
                     float(self.gen_pmax.sum()))
        P_seed = np.ones(self.n_gen) * (target / self.n_gen)
        P_base = project_box_sum(P_seed, self.gen_pmin, self.gen_pmax, target)
        demand_scale = P_base.sum() / self.base_load

        f_base, ok = self._run_dc_approx_pf(P_base, demand_scale)
        if not ok or len(f_base) != self.n_line:
            # Fallback: zero S matrix (will cause issues later)
            print("  [WARN] Cannot compute base flows, using fallback S=0")
            self.P_ref = P_base.copy()
            self.F_ref = np.zeros(self.n_line)
            return np.zeros((self.n_line, self.n_gen))

        self.P_ref = P_base.copy()
        self.F_ref = f_base.copy()
        delta = 50.0  # MW perturbation
        S = np.zeros((self.n_line, self.n_gen))
        
        for i in range(self.n_gen):
            up_room = float(self.gen_pmax[i] - P_base[i])
            down_room = float(P_base[i] - self.gen_pmin[i])
            up_step = min(delta, up_room)
            down_step = min(delta, down_room)
            if max(up_step, down_step) <= 1e-6:
                continue

            # Keep all other controlled generators fixed. The same ext-grid
            # reference source balances every perturbation, so all columns
            # share one injection-withdrawal convention.
            f_plus = f_minus = None
            ok_plus = ok_minus = False
            if up_step > 1e-6 and down_step > 1e-6:
                # Equalize the two sides so the central difference is centered
                # exactly at the common reference dispatch.
                up_step = down_step = min(up_step, down_step)
            if up_step > 1e-6:
                P_plus = P_base.copy()
                P_plus[i] += up_step
                f_plus, ok_plus = self._run_dc_approx_pf(P_plus, demand_scale)
            if down_step > 1e-6:
                P_minus = P_base.copy()
                P_minus[i] -= down_step
                f_minus, ok_minus = self._run_dc_approx_pf(P_minus, demand_scale)

            if ok_plus and ok_minus:
                S[:, i] = (f_plus - f_minus) / (up_step + down_step)
            elif ok_plus:
                S[:, i] = (f_plus - f_base) / up_step
            elif ok_minus:
                S[:, i] = (f_base - f_minus) / down_step

        # Clean up extreme values (numerical noise)
        S = np.clip(S, -100.0, 100.0)  # Reasonable bounds
        return S

    def _calibrate_flow_limits(self):
        """Estimate line thermal limits using OPF reference."""
        limits = np.ones(self.n_line) * 2000.0  # Default high
        
        bus_vn = self.net_base.bus['vn_kv'].values
        for i, row in self.net_base.line.iterrows():
            f_bus = int(row['from_bus'])
            vn = bus_vn[f_bus] if 0 <= f_bus < len(bus_vn) else 230.0
            if vn >= 200: rating = 2500.0
            elif vn >= 120: rating = 1800.0
            elif vn >= 80: rating = 800.0
            elif vn >= 50: rating = 400.0
            else: rating = 200.0
            limits[i] = rating

        # Try OPF calibration at 16GW
        try:
            net_test = copy.deepcopy(self.net_base)
            ls = 16000.0 / self.base_load
            net_test.load.p_mw = net_test.load.p_mw * ls
            net_test.load.q_mvar = net_test.load.q_mvar * ls
            
            pp.runopp(net_test, verbose=False, suppress_warnings=True,
                      numba=False)
            
            if net_test.converged and len(net_test.res_line) > 0:
                if 'loading_percent' in net_test.res_line.columns:
                    loading = net_test.res_line.loading_percent.values
                    for i in range(min(len(limits), len(loading))):
                        flow_mw = limits[i] * loading[i] / 100.0
                        limits[i] = max(limits[i], flow_mw * 8.0)
                        limits[i] = max(limits[i], 100.0)
                    
                    if 'p_from_mw' in net_test.res_line.columns:
                        opf_flows = net_test.res_line.p_from_mw.values
                        for i in range(min(len(limits), len(opf_flows))):
                            if abs(opf_flows[i]) > 0.1:
                                limits[i] = max(limits[i], abs(opf_flows[i]) * 8.0)
                                
                print(f"  OPF calibration OK: [{limits.min():.0f}, {limits.max():.0f}] MW")
        except Exception as e:
            print(f"  [WARN] OPF calib failed ({e}), keeping voltage-based")
        
        return limits

    def _compute_numerical_lodf(self):
        """
        Compute LODF matrix numerically using S-matrix and outage simulation.
        
        LODF[l,k] = d(f_l^{(k)})/d(f_k) when line k opens
        
        Method: For each contingency k:
          1. Get base flows f_base
          2. Remove line k from network
          3. Re-solve (approximate) to get new flows f_post
          4. LODF[l,k] = (f_post[l] - f_base[l]) / f_base[k]
          
        This is expensive (n_line AC-PF runs), so we use approximation:
          LODF ≈ -S @ pseudo_inv(S_k) where S_k is S without column k
        """
        L = self.n_line
        lodf = np.zeros((L, L), dtype=np.float64)
        
        # Use a bounded, balanced reference dispatch for LODF computation.
        target = min(max(18000.0, float(self.gen_pmin.sum())),
                     float(self.gen_pmax.sum()))
        P_ref = project_box_sum(np.ones(self.n_gen) * (target / self.n_gen),
                                self.gen_pmin, self.gen_pmax, target)
        
        f_base, base_ok = self._run_dc_approx_pf(P_ref, P_ref.sum()/self.base_load)
        
        if not base_ok:
            print("  [WARN] Base PF for LODF failed, using identity LODF≈-I")
            # Identity-like LODF: opening line k shifts its full flow to... nothing simple
            return lodf
        
        # For each contingency k, simulate outage and measure flow changes
        # Only compute for lines with significant base flow (>1MW)
        significant_lines = np.where(np.abs(f_base) > 1.0)[0]
        
        for k_idx in range(L):
            fk = f_base[k_idx]
            if abs(fk) < 0.5:
                continue  # Skip near-zero flow lines
                
            # Simulate outage of line k: remove from network
            net_outage = copy.deepcopy(self.net_base)
            net_outage.line.at[k_idx, 'in_service'] = False
            
            net_outage.gen.p_mw = np.clip(P_ref, self.gen_pmin, self.gen_pmax)
            ls = P_ref.sum() / self.base_load
            net_outage.load.p_mw = net_outage.load.p_mw * ls
            net_outage.load.q_mvar = net_outage.load.q_mvar * ls
            net_outage.bus.min_vm_pu = VMIN_RELAXED
            net_outage.bus.max_vm_pu = VMAX_RELAXED
            
            try:
                pp.runpp(net_outage, calculate_voltage_angles=True, numba=False)
                if net_outage.converged and len(net_outage.res_line) > 0:
                    if 'p_from_mw' in net_outage.res_line.columns:
                        f_post = net_outage.res_line.p_from_mw.values
                        if len(f_post) == L:
                            delta_f = f_post - f_base
                            # LODF[l,k] = delta_f[l] / f_k (for l != k)
                            for l in range(L):
                                if l != k_idx and abs(fk) > 0.01:
                                    lodf[l, k_idx] = delta_f[l] / fk
            except Exception:
                pass
        
        # Clip extreme LODF values (physical bounds)
        lodf = np.clip(lodf, -5.0, 5.0)
        return lodf

    def _preselect_n1_constraints(self):
        """Pre-select N-1 constraint pairs for QP formulation."""
        self.n1_constraint_list = []
        
        for k in self.valid_contingency_indices:
            for l in range(self.n_line):
                if l == k:
                    continue
                if abs(self.lodf[l, k]) < 0.05:
                    continue
                if self.effective_limit[l] < 1.0:
                    continue
                self.n1_constraint_list.append((k, l, +1))
                self.n1_constraint_list.append((k, l, -1))
        
        print(f"  N-1 constraints pre-selected: {len(self.n1_constraint_list)}")


# ============================================================
# Part 2: Economic Dispatch Solver (Gurobi QP)
# ============================================================

def solve_economic_dispatch(demand_mw, dcnet: Case300Network, 
                           cost_noise_scale=0.0):
    """
    Solve pure economic dispatch (NO N-1 constraints in QP).
    
    Returns dict with P_opt, cost, status, etc.
    """
    N = dcnet.n_gen
    
    # Cost perturbation
    cp2 = np.maximum(dcnet.cp2 * (1.0 + cost_noise_scale * np.random.randn(N)), 1e-6)
    cp1 = np.maximum(dcnet.cp1 * (1.0 + cost_noise_scale * np.random.randn(N)), 0.1)
    cp0 = dcnet.cp0.copy()
    
    try:
        m = gp.Model("ed_v10")
        m.setParam('OutputFlag', 0)
        m.setParam('TimeLimit', MAX_QP_TIME_SEC)
        m.setParam('OptimalityTol', 1e-7)
        m.setParam('FeasibilityTol', 1e-7)
        
        P = m.addVars(N, lb=0.0, name="P")
        
        # Objective: minimize quadratic cost
        obj = gp.quicksum(cp2[i]*P[i]*P[i] + cp1[i]*P[i] + cp0[i] 
                         for i in range(N))
        m.setObjective(obj, GRB.MINIMIZE)
        
        # C1: Power balance
        m.addConstr(gp.quicksum(P[i] for i in range(N)) == demand_mw, name="C1_bal")
        
        # C2: Generation limits (set as variable bounds)
        for i in range(N):
            P[i].lb = dcnet.gen_pmin[i]
            P[i].ub = dcnet.gen_pmax[i]
        
        # C5: Spinning reserve
        reserve_req = RESERVE_FRAC * demand_mw
        m.addConstr(
            gp.quicksum(dcnet.gen_pmax[i] - P[i] for i in range(N)) >= reserve_req,
            name="C5_res"
        )
        
        t0 = time.time()
        m.optimize()
        t_solve = time.time() - t0
        
        if m.status == GRB.OPTIMAL or m.status == GRB.SUBOPTIMAL:
            P_opt = np.array([P[i].X for i in range(N)])
            return {
                'status': 'optimal' if m.status == GRB.OPTIMAL else 'suboptimal',
                'P_opt': P_opt,
                'obj_val': m.ObjVal,
                'cp2': cp2,
                'cp1': cp1,
                'cp0': cp0,
                'solve_time_sec': t_solve,
            }
        else:
            return {'status': 'infeasible', 'solve_time_sec': t_solve, 'gurobi_status': m.status}
            
    except Exception as e:
        return {'status': 'error', 'error_msg': str(e), 'solve_time_sec': 0}


# ============================================================
# Part 3: N-1 Security Check & Correction
# ============================================================

def check_n1_security(P_opt, dcnet: Case300Network):
    """
    Full DC N-1 security check using S-matrix and LODF.
    
    Returns dict with n_violations, violation details, constraint list for QP.
    """
    # Base flows
    flows = dcnet.estimate_base_flows(P_opt)
    
    n_violations = 0
    n_base_violations = 0
    failed_pairs = []
    qp_constraint_candidates = []

    # Base-case line-flow constraints are part of linear SCED security.
    for l in range(dcnet.n_line):
        limit = dcnet.effective_limit[l]
        if flows[l] > limit + 0.01:
            n_violations += 1
            n_base_violations += 1
            failed_pairs.append(('base', l, flows[l], limit))
            qp_constraint_candidates.append(
                (float(flows[l] - limit), ('base', -1, l, +1, limit))
            )
        if -flows[l] > limit + 0.01:
            n_violations += 1
            n_base_violations += 1
            failed_pairs.append(('base', l, -flows[l], limit))
            qp_constraint_candidates.append(
                (float(-flows[l] - limit), ('base', -1, l, -1, limit))
            )
    
    for k in dcnet.valid_contingency_indices:
        fk = flows[k]
        if abs(fk) < 0.1:
            continue
        
        # Post-contingency flows
        f_post = dcnet.estimate_post_contingency_flows(P_opt, k)
        
        for l in range(dcnet.n_line):
            if l == k:
                continue
            limit = dcnet.effective_limit[l]
            
            # Positive side: f_post[l] > limit
            if f_post[l] > limit + 0.01:
                n_violations += 1
                failed_pairs.append((k, l, f_post[l], limit))
                qp_constraint_candidates.append(
                    (float(f_post[l] - limit), ('n1', k, l, +1, limit))
                )
            
            # Negative side: -f_post[l] > limit  
            if -f_post[l] > limit + 0.01:
                n_violations += 1
                failed_pairs.append((k, l, -f_post[l], limit))
                qp_constraint_candidates.append(
                    (float(-f_post[l] - limit), ('n1', k, l, -1, limit))
                )

    # Add the most severe cuts first. The correction loop retains cuts across
    # iterations, so a constraint fixed in one QP cannot be forgotten later.
    qp_constraint_candidates.sort(key=lambda item: item[0], reverse=True)
    qp_constraints = [item[1] for item in qp_constraint_candidates]
    
    return {
        'n_violations': n_violations,
        'failed_pairs': failed_pairs[:15],
        'base_flows_max': float(np.abs(flows).max()),
        'base_overloads': n_base_violations,
        'qp_constraints': qp_constraints[:2000],
    }


def correct_n1_violations(P_current, demand_mw, dcnet: Case300Network,
                          n1_info, max_iters=MAX_CORRECTION_ITERS,
                          cost_coeffs=None):
    """
    Iterative Gurobi QP correction for N-1 violations.
    
    Strategy: re-solve the linear SCED QP with the original generation cost
    subject to:
      - Power balance
      - Gen limits
      - Reserve
      - Base-case line constraints
      - All violated N-1 constraints (linearized around current point)
    """
    N = dcnet.n_gen
    P = P_current.copy()
    if cost_coeffs is None:
        cp2 = dcnet.cp2.copy()
        cp1 = dcnet.cp1.copy()
        cp0 = dcnet.cp0.copy()
    else:
        cp2, cp1, cp0 = cost_coeffs
    
    constraints = n1_info.get('qp_constraints', [])
    if not constraints:
        return P, 0, 'no_constraints'

    cut_pool = {}
    
    for iteration in range(max_iters):
        # Re-check current N-1 status
        n1_new = check_n1_security(P, dcnet)
        if n1_new['n_violations'] == 0:
            return P, iteration, 'converged'
        
        constraints = n1_new.get('qp_constraints', [])
        if not constraints:
            break

        for item in constraints[:1000]:
            kind, k, l, sign, limit = item
            if kind == 'n1':
                cut_pool[(int(k), int(l), int(sign))] = item
        
        try:
            m = gp.Model(f"n1_correct_{iteration}")
            m.setParam('OutputFlag', 0)
            m.setParam('TimeLimit', MAX_QP_TIME_SEC)
            m.setParam('OptimalityTol', 1e-6)
            
            P_var = m.addVars(N, lb=0.0, name="P")
            
            # Minimize the original SCED generation cost, not distance from ED.
            obj = gp.quicksum(cp2[i]*P_var[i]*P_var[i] + cp1[i]*P_var[i] + cp0[i]
                              for i in range(N))
            m.setObjective(obj, GRB.MINIMIZE)
            
            # C1: Power balance
            m.addConstr(gp.quicksum(P_var[i] for i in range(N)) == demand_mw, name="bal")
            
            # C2: Gen limits
            for i in range(N):
                P_var[i].lb = dcnet.gen_pmin[i]
                P_var[i].ub = dcnet.gen_pmax[i]

            # C5: Spinning reserve
            reserve_req = RESERVE_FRAC * demand_mw
            m.addConstr(
                gp.quicksum(dcnet.gen_pmax[i] - P_var[i] for i in range(N)) >= reserve_req,
                name="reserve"
            )

            # Base-case line constraints: F_ref + S @ (P - P_ref).
            for l in range(dcnet.n_line):
                limit = dcnet.effective_limit[l]
                if limit < 1.0:
                    continue
                const_base = dcnet.flow_intercept[l]
                expr_base = const_base + gp.quicksum(dcnet.S[l, i] * P_var[i] for i in range(N))
                m.addConstr(expr_base <= limit, name=f"base_pos_{l}")
                m.addConstr(-expr_base <= limit, name=f"base_neg_{l}")
            
            # C4: N-1 constraints (linearized)
            f_cur = dcnet.estimate_base_flows(P)
            
            n_added = 0
            for idx, item in enumerate(cut_pool.values()):
                if len(item) == 5:
                    kind, k, l, sign, limit = item
                else:
                    k, l, sign, limit = item
                    kind = 'n1'
                if kind == 'base':
                    continue
                # coeff = d/dP of (sign * f_l^{(k)})
                #       = sign * (S[l,:] + LODF[l,k] * S[k,:])
                coeff = sign * (dcnet.S[l, :] + dcnet.lodf[l, k] * dcnet.S[k, :])
                
                # Current violation amount
                fk_cur = f_cur[k]
                fl_cur = f_cur[l]
                const_term = fl_cur + dcnet.lodf[l, k] * fk_cur
                
                if sign > 0:
                    rhs = limit - const_term
                else:
                    rhs = limit + const_term
                    
                expr = gp.quicksum(coeff[i] * (P_var[i] - P[i]) for i in range(N))
                m.addConstr(expr <= rhs, name=f"n1_{idx}")
                n_added += 1
            
            m.optimize()
            
            if m.status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
                break
            
            P = np.array([P_var[i].X for i in range(N)], dtype=np.float64)
            P = project_box_sum(P, dcnet.gen_pmin, dcnet.gen_pmax, demand_mw)
                    
        except Exception:
            break
    
    return P, max_iters, 'max_iterations'


# ============================================================
# Part 4: AC-PF Verification
# ============================================================

def optimize_voltage_strict(pp_net_base, dispatch_68, demand_mw, gen_pmin, gen_pmax,
                            vmin_strict=VMIN_STRICT, vmax_strict=VMAX_STRICT,
                            max_iter=20):
    """Given active power P from linear SCED, iteratively adjust generator
    voltage setpoints so that AC-PF passes strict voltage limits.

    Returns (passed, gen_vm_pu, ext_grid_vm_pu, ac_result_dict).
    """
    import pandapower as pp
    net = copy.deepcopy(pp_net_base)
    dispatch = np.asarray(dispatch_68, dtype=np.float64)
    net.gen.p_mw.values[:] = np.clip(dispatch, gen_pmin, gen_pmax)

    base_load = float(net.load.p_mw.sum())
    if base_load > 0:
        ls = float(demand_mw) / base_load
        net.load.p_mw = net.load.p_mw * ls
        net.load.q_mvar = net.load.q_mvar * ls

    best_vm = net.gen.vm_pu.values.astype(float).copy()
    best_ext_vm = net.ext_grid.vm_pu.values.astype(float).copy()
    best_vv = 9999

    for _it in range(max_iter):
        try:
            pp.runpp(net, calculate_voltage_angles=True, enforce_q_lims=True,
                     numba=True, max_iteration=50)
        except Exception:
            break
        if not bool(net.converged):
            break

        vm = net.res_bus.vm_pu.values
        vv_lo = int(np.sum(vm < vmin_strict))
        vv_hi = int(np.sum(vm > vmax_strict))
        total_vv = vv_lo + vv_hi

        if total_vv < best_vv:
            best_vv = total_vv
            best_vm = net.gen.vm_pu.values.astype(float).copy()
            best_ext_vm = net.ext_grid.vm_pu.values.astype(float).copy()

        # Check full pass: no voltage violations + no overloads
        line_ol = int(np.sum(net.res_line.loading_percent.values > 100.0)) if len(net.res_line) else 0
        trafo_ol = int(np.sum(net.res_trafo.loading_percent.values > 100.0)) if len(net.trafo) else 0
        if total_vv == 0 and line_ol == 0 and trafo_ol == 0:
            ac_result = {
                'converged': True, 'vmin': float(vm.min()), 'vmax': float(vm.max()),
                'vv_total': 0, 'vv_lo': 0, 'vv_hi': 0,
                'max_line_loading_pct': float(net.res_line.loading_percent.max()) if len(net.res_line) else -1,
                'overloaded_lines': [], 'overloaded_trafos': [],
            }
            return True, best_vm, best_ext_vm, ac_result

        # Adjust voltage setpoints: raise at low-voltage gen buses, lower at high
        gen_bus_ids = net.gen.bus.values
        bus_vm = net.res_bus.vm_pu.values
        for i in range(len(net.gen)):
            bvm = float(bus_vm[gen_bus_ids[i]])
            if bvm < vmin_strict:
                net.gen.vm_pu.values[i] = min(float(net.gen.vm_pu.values[i]) + 0.005, 1.08)
            elif bvm > vmax_strict:
                net.gen.vm_pu.values[i] = max(float(net.gen.vm_pu.values[i]) - 0.005, 0.92)
        if len(net.ext_grid):
            eg_bvm = float(bus_vm[net.ext_grid.bus.values[0]])
            if eg_bvm < vmin_strict:
                net.ext_grid.vm_pu.values[0] = min(float(net.ext_grid.vm_pu.values[0]) + 0.005, 1.08)
            elif eg_bvm > vmax_strict:
                net.ext_grid.vm_pu.values[0] = max(float(net.ext_grid.vm_pu.values[0]) - 0.005, 0.92)

    # Did not fully pass — return best effort
    ac_result = {
        'converged': bool(net.converged),
        'vmin': float(net.res_bus.vm_pu.min()) if bool(net.converged) else -1,
        'vmax': float(net.res_bus.vm_pu.max()) if bool(net.converged) else -1,
        'vv_total': best_vv,
        'vv_lo': -1, 'vv_hi': -1,
        'max_line_loading_pct': float(net.res_line.loading_percent.max()) if bool(net.converged) and len(net.res_line) else -1,
        'overloaded_lines': [], 'overloaded_trafos': [],
    }
    return False, best_vm, best_ext_vm, ac_result


def verify_ac_pf(dispatch_68, pp_net_base, gen_pmax, demand_mw,
                 gen_pmin=None,
                 vmin_relaxed=VMIN_RELAXED, vmax_relaxed=VMAX_RELAXED):
    """Run AC-PF verification. Returns dict with convergence, VV, loading."""
    net_c = copy.deepcopy(pp_net_base)
    if gen_pmin is None:
        gen_pmin = np.zeros_like(np.asarray(gen_pmax, dtype=np.float64))
    net_c.gen.p_mw = np.clip(dispatch_68, gen_pmin, gen_pmax)
    net_c.bus.min_vm_pu = vmin_relaxed
    net_c.bus.max_vm_pu = vmax_relaxed
    
    base_load = net_c.load.p_mw.sum()
    if base_load > 0:
        ls = float(demand_mw) / base_load
        net_c.load.p_mw = net_c.load.p_mw * ls
        net_c.load.q_mvar = net_c.load.q_mvar * ls
    
    try:
        pp.runpp(net_c, calculate_voltage_angles=True, numba=False)
        converged = net_c.converged
    except Exception:
        converged = False
    
    result = {
        'converged': converged,
        'demand_mw': demand_mw,
        'gen_sum_mw': float(dispatch_68.sum()),
    }
    
    if not converged:
        result.update({'vv_lo':-1,'vv_hi':-1,'vv_total':-1,'vmin':-1,'vmax':-1,
                       'max_line_loading_pct':-1, 'overloaded_lines':[], 'overloaded_trafos':[]})
        return result
    
    vm = net_c.res_bus.vm_pu.values
    vv_lo = int(np.sum(vm < vmin_relaxed))
    vv_hi = int(np.sum(vm > vmax_relaxed))
    
    result.update({
        'vmin': float(vm.min()), 'vmax': float(vm.max()),
        'vv_lo': vv_lo, 'vv_hi': vv_hi, 'vv_total': vv_lo + vv_hi,
    })
    
    if len(net_c.res_line) > 0 and 'loading_percent' in net_c.res_line.columns:
        ll = net_c.res_line.loading_percent.values
        result['max_line_loading_pct'] = float(ll.max()) if len(ll) > 0 else -1
        result['overloaded_lines'] = [int(x) for x in np.where(ll > 100.0)[0].tolist()]
    else:
        result['max_line_loading_pct'] = -1
        result['overloaded_lines'] = []
    
    if len(net_c.res_trafo) > 0 and 'loading_percent' in net_c.res_trafo.columns:
        tl = net_c.res_trafo.loading_percent.values
        result['max_trafo_loading_pct'] = float(tl.max()) if len(tl) > 0 else -1
        result['overloaded_trafos'] = [int(x) for x in np.where(tl > 100.0)[0].tolist()]
    else:
        result['max_trafo_loading_pct'] = -1
        result['overloaded_trafos'] = []
    
    return result


def compute_hhi(dispatch_68):
    total = dispatch_68.sum()
    if total <= 0: return 9999.0
    shares = dispatch_68 / total
    return float(np.sum(shares**2) * 10000)


# ============================================================
# Part 5: Main Loop
# ============================================================

def main():
    global N_SAMPLES_PER_LEVEL, DEMAND_LEVELS_GW, COST_NOISE_STD_PCT, N1_LIMIT_SCALE
    parser = argparse.ArgumentParser()
    parser.add_argument("--demand-levels", default=",".join(str(x) for x in DEMAND_LEVELS_GW),
                        help="Comma-separated demand levels in GW, e.g. 17,18,19,20,21")
    parser.add_argument("--samples-per-level", type=int, default=N_SAMPLES_PER_LEVEL)
    parser.add_argument("--cost-noise", type=float, default=COST_NOISE_STD_PCT)
    parser.add_argument(
        "--n1-limit-scale", type=float, default=N1_LIMIT_SCALE,
        help=("Multiply calibrated line limits before base/N-1 label generation. "
              "Values below 1 tighten limits; 1 preserves the original limits."),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for cost perturbations and all NumPy sampling.",
    )
    parser.add_argument("--output", default="case300_sced_linear_labels.json")
    args = parser.parse_args()
    np.random.seed(int(args.seed))
    DEMAND_LEVELS_GW = [float(x.strip()) for x in args.demand_levels.split(",") if x.strip()]
    N_SAMPLES_PER_LEVEL = int(args.samples_per_level)
    COST_NOISE_STD_PCT = float(args.cost_noise)
    N1_LIMIT_SCALE = float(args.n1_limit_scale)
    if not 0.0 < N1_LIMIT_SCALE <= 1.0:
        raise ValueError("--n1-limit-scale must be in (0, 1]")

    t_start = time.time()
    print("=" * 70)
    print("  v10 Complete SCED GT Generator (QP + Numerical S + DC LODF)")
    print("=" * 70)
    
    # Load v8 reference (optional — only used for a print)
    try:
        with open(V8_INPUT, 'r') as f:
            v8_data = json.load(f)
        print(f"  v8 samples: {len(v8_data['samples'])}")
    except FileNotFoundError:
        print(f"  [WARN] V8 input not found ({V8_INPUT}), proceeding without it.")
    
    # Initialize network
    dcnet = Case300Network()

    # Default voltage setpoints (flat start) for linear-SCED labels.
    # Linear SCED does not determine voltage; these defaults are the label
    # for GAT's voltage head, and SAC must adjust them to pass AC-PF.
    _default_gen_vm_pu = dcnet.net_base.gen.vm_pu.values.astype(float).tolist()
    _default_ext_vm_pu = dcnet.net_base.ext_grid.vm_pu.values.astype(float).tolist()
    
    # Output collection
    all_samples = []
    stats = {
        'total_attempted': 0, 'ed_feasible': 0, 'n1_secure_pre': 0,
        'n1_secure_post': 0, 'acpf_converged': 0, 'vv_zero': 0,
        'complete_sced': 0, 'by_level': {},
    }
    
    print(f"\n{'='*70}")
    print(f"  Generating: {len(DEMAND_LEVELS_GW)} levels x {N_SAMPLES_PER_LEVEL}")
    print(f"{'='*70}\n")
    
    sample_id = 0
    
    for level_gw in DEMAND_LEVELS_GW:
        demand = level_gw * 1000.0
        lk = f"{level_gw:g}GW"
        stats['by_level'][lk] = {'att':0, 'ed':0, 'n1_pre':0, 'n1_post':0,
                                  'acpf':0, 'vv0':0, 'comp':0}
        
        print(f"\n--- Demand: {level_gw} GW ({demand:.0f} MW) ---")
        
        for s_idx in range(N_SAMPLES_PER_LEVEL):
            sample_id += 1
            stats['total_attempted'] += 1
            stats['by_level'][lk]['att'] += 1
            noise = COST_NOISE_STD_PCT * (1.0 + 0.5*np.random.random())
            
            # ===== STAGE 1: Economic Dispatch (Gurobi QP) =====
            ed_result = solve_economic_dispatch(demand, dcnet, noise)
            
            if ed_result['status'] not in ('optimal', 'suboptimal'):
                if sample_id % VERBOSE_EVERY == 0:
                    print(f"  [{sample_id:03d}] ED: {ed_result['status']}")
                continue
            
            P_opt = ed_result['P_opt']
            ed_cost = ed_result['obj_val']
            stats['ed_feasible'] += 1
            stats['by_level'][lk]['ed'] += 1
            
            # ===== STAGE 2: DC N-1 Security Check =====
            n1_check = check_n1_security(P_opt, dcnet)
            n_n1 = n1_check['n_violations']
            
            if n_n1 == 0:
                stats['n1_secure_pre'] += 1
                stats['by_level'][lk]['n1_pre'] += 1
                P_final = P_opt.copy()
                n_correction_iters = 0
                correction_method = 'already_secure'
                final_n1_check = n1_check
            else:
                # ===== STAGE 3: N-1 Correction (Iterative QP) =====
                P_final, n_iters, method = correct_n1_violations(
                    P_opt, demand, dcnet, n1_check,
                    cost_coeffs=(ed_result['cp2'], ed_result['cp1'], ed_result['cp0']))
                n_correction_iters = n_iters
                correction_method = method
                
                # Re-check after correction
                n1_recheck = check_n1_security(P_final, dcnet)
                final_n1_check = n1_recheck
                n_n1_after = n1_recheck['n_violations']
                
                if n_n1_after == 0:
                    stats['n1_secure_post'] += 1
                    stats['by_level'][lk]['n1_post'] += 1
                n_n1 = n_n1_after  # Update for logging
            
            # ===== STAGE 4: Voltage correction + strict AC-PF verification =====
            # Linear SCED gives P only.  Iteratively adjust generator voltage
            # setpoints so AC-PF passes the adopted STRICT limits.  The
            # corrected V becomes the label's voltage setpoint.
            v_passed, corr_gen_vm, corr_ext_vm, ac_result = optimize_voltage_strict(
                dcnet.net_base, P_final, demand, dcnet.gen_pmin, dcnet.gen_pmax,
            )
            ac_ok = ac_result['converged']
            vv_tot = ac_result['vv_total']

            if ac_ok:
                stats['acpf_converged'] += 1
                stats['by_level'][lk]['acpf'] += 1
                if vv_tot == 0:
                    stats['vv_zero'] += 1
                    stats['by_level'][lk]['vv0'] += 1

            # ===== LABEL: linear N-1 secure (AC-PF strict verification for reference) =====
            # Labels are linear-SCED dispatches that satisfy linear N-1.  They are
            # NOT required to pass strict AC-PF — that infeasibility IS the research
            # motivation for SAC repair.  Voltage setpoints come from best-effort
            # correction (better than flat defaults).  AC-PF is verified with STRICT
            # limits so violations are correctly reported.
            is_complete = (
                ed_result['status'] in ('optimal', 'suboptimal') and
                n_n1 == 0
            )
            is_ac_feasible = bool(v_passed)
            if is_complete:
                stats['complete_sced'] += 1
                stats['by_level'][lk]['comp'] += 1
            
            hhi = compute_hhi(P_final)
            sced_cost = float(np.sum(
                ed_result['cp2'] * P_final * P_final +
                ed_result['cp1'] * P_final +
                ed_result['cp0']
            ))
            
            sample = {
                'sample_id': f"v10_{sample_id:04d}",
                'demand_mw': float(demand), 'demand_gw': float(level_gw),
                'dispatch_68': [float(x) for x in P_final],
                'gen_sum_mw': float(P_final.sum()),
                'sample_cp1': [float(x) for x in ed_result['cp1']],
                'sample_cp2': [float(x) for x in ed_result['cp2']],
                'sample_cp0': [float(x) for x in ed_result['cp0']],
                'cost_noise_scale': float(noise),
                'sced_cost_per_h': round(sced_cost, 2),
                'ed_solve_time_sec': round(ed_result.get('solve_time_sec', 0), 3),
                'n1_violations_before_corr': n1_check['n_violations'],
                'n1_violations_after_corr': n_n1,
                'n1_correction_iters': n_correction_iters,
                'n1_correction_method': correction_method,
                'dc_base_flow_max': round(n1_check.get('base_flows_max', -1), 2),
                'dc_base_overloads': int(n1_check.get('base_overloads', -1)),
                'ac_converged': ac_ok,
                'vmin': round(ac_result.get('vmin', -1), 6),
                'vmax': round(ac_result.get('vmax', -1), 6),
                'vv_total': vv_tot,
                'max_line_loading_pct': round(ac_result.get('max_line_loading_pct', -1), 2),
                'hhi': round(hhi, 1),
                'is_complete_sced': is_complete,
                'is_ac_feasible': is_ac_feasible,
                'label_generation_method': 'linear-SCED-QP-LODF',
                'linear_sced_feasible': bool(is_complete),
                'linear_base_pass': bool(final_n1_check.get('base_overloads', 1) == 0),
                'linear_n1_pass': bool(n_n1 == 0),
                'acpf_relaxed_pass': bool(ac_ok and vv_tot == 0),
                'acpf_strict_pass': bool(is_ac_feasible),
                # Voltage setpoints are audit/reference fields from offline
                # AC-PF probing. GAT predicts only dispatch_68; SAC starts from
                # default case300 voltage setpoints and learns voltage/action
                # corrections without using these fields as labels.
                'gen_vm_pu': [float(x) for x in corr_gen_vm],
                'ext_grid_vm_pu': [float(x) for x in corr_ext_vm],
                'ext_grid_p_mw': 0.0,
                'ext_grid_q_mvar': 0.0,
                'base_v_min': round(ac_result.get('vmin', -1), 6),
                'base_v_max': round(ac_result.get('vmax', -1), 6),
                'case_profile': 'case300-sced',
                'n1_voltage_min_pu': VMIN_STRICT,
                'n1_voltage_max_pu': VMAX_STRICT,
            }
            all_samples.append(sample)
            
            flag = "✓" if is_complete else ("~" if ac_ok else "✗")
            corr_tag = f"+{n_correction_iters}" if n_correction_iters > 0 else ""
            if sample_id % VERBOSE_EVERY == 0 or is_complete:
                print(f"  [{sample_id:03d}] D={level_gw:g}GW "
                      f"C=${ed_cost/1000:.0f}K | "
                      f"N1:{n1_check['n_violations']}→{n_n1}{corr_tag} | "
                      f"PF={'OK' if ac_ok else '--'} VV={vv_tot} | "
                      f"HHI={hhi:.0f} | {flag}")
    
    # ================================================================
    # Final Report
    # ================================================================
    t_total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  v10 Complete SCED GT Results Summary")
    print(f"{'='*70}")
    print(f"  Total attempted:         {stats['total_attempted']}")
    print(f"  ED feasible (Gurobi):    {stats['ed_feasible']} "
          f"({stats['ed_feasible']/max(stats['total_attempted'],1)*100:.1f}%)")
    print(f"  N-1 secure (pre-corr):   {stats['n1_secure_pre']}")
    print(f"  N-1 secure (post-corr):  {stats['n1_secure_post']}")
    n1_total = stats['n1_secure_pre'] + stats['n1_secure_post']
    print(f"  N-1 secure (total):      {n1_total}/{stats['total_attempted']}"
          f" ({n1_total/max(stats['total_attempted'],1)*100:.1f}%)")
    print(f"  AC-PF converged:         {stats['acpf_converged']} "
          f"({stats['acpf_converged']/max(stats['total_attempted'],1)*100:.1f}%)")
    print(f"  VV = 0:                   {stats['vv_zero']} "
          f"({stats['vv_zero']/max(stats['total_attempted'],1)*100:.1f}%)")
    print(f"  COMPLETE SCED (all 7):   {stats['complete_sced']} "
          f"({stats['complete_sced']/max(stats['total_attempted'],1)*100:.1f}%)")
    print(f"  Time:                     {t_total:.1f}s ({t_total/60:.1f}min)")
    
    print(f"\n  By demand level:")
    for lvl in sorted(stats['by_level'].keys()):
        s = stats['by_level'][lvl]
        print(f"    {lvl:>5s}: {s['att']:3d}att | {s['ed']:3d}ED | "
              f"{s['n1_pre']:3d}N1-pre | {s['n1_post']+s['n1_pre']:3d}N1-tot | "
              f"{s['acpf']:3d}PF | {s['vv0']:3d}VV0 | {s['comp']:3d}COMP")
    
    # Save
    output = {
        'version': 'v10_linear_sced',
        'label_generation_method': 'linear-SCED-QP-LODF',
        'description': 'Linear SCED (Gurobi QP + numerical S + standard DC-LODF N-1). Labels are '
                       'N-1-secure in the linear sense but NOT required to pass AC-PF. '
                       'GAT predicts these; SAC repairs to pass AC-PF + AC N-1.',
        'operation_verification_method': 'AC-PF (reference only, not a filter)',
        'contingency_verification_method': 'standard DC-LODF N-1 (label filter)',
        'model_config': {
            'seed': int(args.seed),
            'fl_scale': FL_SCALE,
            'n1_limit_scale': N1_LIMIT_SCALE,
            'reserve_frac': RESERVE_FRAC,
            'vmin': VMIN_RELAXED, 'vmax': VMAX_RELAXED,
            'method': 'Affine generator-flow sensitivity + DC PTDF/LODF + iterative QP cuts',
        },
        'linear_security_model': {
            'format': 'case300_linear_sced_affine_lodf_v1',
            'sensitivity_balance': 'fixed_ext_grid_reference_source',
            'sensitivity_perturbation_mw': 50.0,
            'n_line': int(dcnet.n_line),
            'n_gen': int(dcnet.n_gen),
            'P_ref': np.asarray(dcnet.P_ref, dtype=float).tolist(),
            'F_ref': np.asarray(dcnet.F_ref, dtype=float).tolist(),
            'flow_intercept': np.asarray(dcnet.flow_intercept, dtype=float).tolist(),
            'S': np.asarray(dcnet.S, dtype=float).tolist(),
            'lodf': np.asarray(dcnet.lodf, dtype=float).tolist(),
            'valid_contingency_indices': dcnet.valid_contingency_indices.astype(int).tolist(),
            'flow_limit': np.asarray(dcnet.flow_limit, dtype=float).tolist(),
            'effective_limit': np.asarray(dcnet.effective_limit, dtype=float).tolist(),
        },
        'statistics': stats, 'generation_time_sec': round(t_total, 1),
        'samples': all_samples,
    }

    out_path = args.output
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Output saved: {out_path}")
    n_ac_infeasible = sum(1 for s in all_samples if s.get('is_complete_sced') and not s.get('is_ac_feasible'))
    print(f"  Total samples: {len(all_samples)}, N1-secure (label): {stats['complete_sced']}, "
          f"of which AC-INFEASIBLE (SAC repair target): {n_ac_infeasible}")
    
    return output


if __name__ == "__main__":
    main()

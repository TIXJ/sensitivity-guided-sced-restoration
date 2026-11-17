#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single final verifier for strict SCED experiments.

The verifier is intentionally independent from SCEDEnvV21 caches. Evaluation,
baselines, retry logic, and label screening should call this module instead of
reading env._n1_violations or re-implementing partial checks.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


@dataclass
class SCEDVerificationReport:
    passed: bool
    failures: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "failures": list(self.failures),
            "metrics": dict(self.metrics),
        }


class FullSCEDVerifier:
    """Strict AC-PF and AC contingency verifier for one dispatch."""

    def __init__(
        self,
        net_template=None,
        balance_tol_mw: float = 1e-3,
        enforce_q_lims: bool = True,
        max_loading_percent: float = 100.0,
        voltage_min_pu: Optional[float] = None,
        voltage_max_pu: Optional[float] = None,
        n1_voltage_min_pu: Optional[float] = None,
        n1_voltage_max_pu: Optional[float] = None,
        check_ext_grid_bounds: bool = True,
        check_dispatch_sum_balance: bool = False,
        ext_grid_tol: float = 1e-3,
        skip_islanding_contingencies: bool = False,
        runpp_kwargs: Optional[Dict[str, Any]] = None,
    ):
        import pandapower as pp
        from pandapower.networks import case300

        self.pp = pp
        self.net_template = copy.deepcopy(net_template) if net_template is not None else case300()
        self.balance_tol_mw = float(balance_tol_mw)
        self.enforce_q_lims = bool(enforce_q_lims)
        self.max_loading_percent = float(max_loading_percent)
        self.voltage_min_pu = voltage_min_pu
        self.voltage_max_pu = voltage_max_pu
        self.n1_voltage_min_pu = n1_voltage_min_pu
        self.n1_voltage_max_pu = n1_voltage_max_pu
        self.check_ext_grid_bounds = bool(check_ext_grid_bounds)
        self.check_dispatch_sum_balance = bool(check_dispatch_sum_balance)
        self.ext_grid_tol = float(ext_grid_tol)
        self.skip_islanding_contingencies = bool(skip_islanding_contingencies)
        self.runpp_kwargs = {
            "calculate_voltage_angles": True,
            "enforce_q_lims": self.enforce_q_lims,
            "numba": True,
            "max_iteration": 50,
            "tolerance_mva": 1e-4,
        }
        if runpp_kwargs:
            self.runpp_kwargs.update(runpp_kwargs)

        self.gen_pmin = self._series_or_default(self.net_template.gen, "min_p_mw", 0.0)
        self.gen_pmax = self._series_or_default(self.net_template.gen, "max_p_mw", np.inf)
        fallback_pmax = np.maximum(self.net_template.gen.p_mw.values.astype(float) * 2.5,
                                   self.gen_pmin + 10.0)
        bad_pmax = ~np.isfinite(self.gen_pmax) | (self.gen_pmax <= self.gen_pmin)
        self.gen_pmax[bad_pmax] = fallback_pmax[bad_pmax]

    @staticmethod
    def _series_or_default(table, col: str, default: float) -> np.ndarray:
        if col in table.columns:
            return np.nan_to_num(table[col].values.astype(float), nan=default,
                                 posinf=default, neginf=default)
        return np.full(len(table), float(default), dtype=np.float64)

    def _fresh_net(
        self,
        dispatch: np.ndarray,
        demand_mw: float,
        gen_vm_pu: Optional[Iterable[float]] = None,
        ext_grid_vm_pu: Optional[Iterable[float]] = None,
    ):
        net = copy.deepcopy(self.net_template)
        dispatch = np.asarray(dispatch, dtype=np.float64)
        if len(dispatch) != len(net.gen):
            raise ValueError(f"dispatch length {len(dispatch)} != n_gen {len(net.gen)}")
        net.gen.p_mw.values[:] = dispatch
        if gen_vm_pu is not None:
            gen_vm = np.asarray(gen_vm_pu, dtype=np.float64)
            if len(gen_vm) != len(net.gen):
                raise ValueError(f"gen_vm_pu length {len(gen_vm)} != n_gen {len(net.gen)}")
            net.gen.vm_pu.values[:] = gen_vm
        if ext_grid_vm_pu is not None:
            ext_vm = np.asarray(ext_grid_vm_pu, dtype=np.float64)
            if len(ext_vm) != len(net.ext_grid):
                raise ValueError(f"ext_grid_vm_pu length {len(ext_vm)} != n_ext_grid {len(net.ext_grid)}")
            net.ext_grid.vm_pu.values[:] = ext_vm

        base_load = float(net.load.p_mw.sum())
        if base_load > 0:
            scale = float(demand_mw) / base_load
            net.load.p_mw = net.load.p_mw * scale
            net.load.q_mvar = net.load.q_mvar * scale
        if self.voltage_min_pu is not None:
            net.bus.min_vm_pu = float(self.voltage_min_pu)
        if self.voltage_max_pu is not None:
            net.bus.max_vm_pu = float(self.voltage_max_pu)
        return net

    def _apply_n1_limits(self, net):
        if self.n1_voltage_min_pu is not None:
            net.bus.min_vm_pu = float(self.n1_voltage_min_pu)
        if self.n1_voltage_max_pu is not None:
            net.bus.max_vm_pu = float(self.n1_voltage_max_pu)

    def _runpp(self, net) -> Optional[str]:
        try:
            self.pp.runpp(net, **self.runpp_kwargs)
            if not bool(getattr(net, "converged", False)):
                return "ac_pf_not_converged"
            return None
        except Exception as exc:
            return f"ac_pf_error:{type(exc).__name__}:{exc}"

    def _is_islanding_outage(self, element_type: str, idx: int) -> bool:
        if not self.skip_islanding_contingencies:
            return False
        try:
            import networkx as nx
            import pandapower.topology as top

            net = copy.deepcopy(self.net_template)
            if element_type == "line":
                net.line.at[idx, "in_service"] = False
            elif element_type == "trafo":
                net.trafo.at[idx, "in_service"] = False
            graph = top.create_nxgraph(net, respect_switches=True, include_trafos=True)
            return nx.number_connected_components(graph) > 1
        except Exception:
            return False

    def _check_results(self, net, prefix: str, failures: List[str], metrics: Dict[str, Any]):
        if len(net.res_bus):
            vmin = float(net.res_bus.vm_pu.min())
            vmax = float(net.res_bus.vm_pu.max())
            bus_min = self._series_or_default(net.bus, "min_vm_pu", 0.9)
            bus_max = self._series_or_default(net.bus, "max_vm_pu", 1.1)
            v_low = int((net.res_bus.vm_pu.values < bus_min - 1e-9).sum())
            v_high = int((net.res_bus.vm_pu.values > bus_max + 1e-9).sum())
            metrics[f"{prefix}_v_min"] = vmin
            metrics[f"{prefix}_v_max"] = vmax
            metrics[f"{prefix}_v_violations"] = v_low + v_high
            if v_low or v_high:
                failures.append(f"{prefix}:bus_voltage_violations={v_low + v_high}")

        if len(net.res_line):
            max_line = float(net.res_line.loading_percent.max())
            n_line_over = int((net.res_line.loading_percent.values > self.max_loading_percent + 1e-9).sum())
            metrics[f"{prefix}_line_loading_max"] = max_line
            metrics[f"{prefix}_line_overloads"] = n_line_over
            if n_line_over:
                failures.append(f"{prefix}:line_overloads={n_line_over}")

        if len(net.trafo) and len(net.res_trafo):
            max_trafo = float(net.res_trafo.loading_percent.max())
            n_trafo_over = int((net.res_trafo.loading_percent.values > self.max_loading_percent + 1e-9).sum())
            metrics[f"{prefix}_trafo_loading_max"] = max_trafo
            metrics[f"{prefix}_trafo_overloads"] = n_trafo_over
            if n_trafo_over:
                failures.append(f"{prefix}:trafo_overloads={n_trafo_over}")

        if len(net.res_gen):
            if "q_mvar" in net.res_gen.columns:
                q = net.res_gen.q_mvar.values.astype(float)
                qmin = self._series_or_default(net.gen, "min_q_mvar", -np.inf)
                qmax = self._series_or_default(net.gen, "max_q_mvar", np.inf)
                q_viol = int(((q < qmin - 1e-6) | (q > qmax + 1e-6)).sum())
                metrics[f"{prefix}_gen_q_violations"] = q_viol
                if q_viol:
                    failures.append(f"{prefix}:gen_q_violations={q_viol}")

        if len(net.ext_grid) and len(net.res_ext_grid):
            ext_p = net.res_ext_grid.p_mw.values.astype(float)
            ext_q = net.res_ext_grid.q_mvar.values.astype(float)
            metrics[f"{prefix}_ext_grid_p_abs_max"] = float(np.max(np.abs(ext_p))) if len(ext_p) else 0.0
            metrics[f"{prefix}_ext_grid_q_abs_max"] = float(np.max(np.abs(ext_q))) if len(ext_q) else 0.0
            if not self.check_ext_grid_bounds:
                return
            for col, values, label in (("min_p_mw", ext_p, "p_min"), ("max_p_mw", ext_p, "p_max"),
                                       ("min_q_mvar", ext_q, "q_min"), ("max_q_mvar", ext_q, "q_max")):
                if col in net.ext_grid.columns:
                    bounds = net.ext_grid[col].values.astype(float)
                    if col.startswith("min"):
                        viol = int((values < bounds - self.ext_grid_tol).sum())
                    else:
                        viol = int((values > bounds + self.ext_grid_tol).sum())
                    if viol:
                        failures.append(f"{prefix}:ext_grid_{label}_violations={viol}")

    def verify_dispatch(
        self,
        dispatch: Iterable[float],
        demand_mw: float,
        check_base_ac: bool = True,
        check_all_ac_contingencies: bool = True,
        contingency_types: Iterable[str] = ("line", "trafo"),
        max_contingencies: Optional[int] = None,
        contingency_list: Optional[Iterable[Any]] = None,
        gen_vm_pu: Optional[Iterable[float]] = None,
        ext_grid_vm_pu: Optional[Iterable[float]] = None,
    ) -> SCEDVerificationReport:
        failures: List[str] = []
        metrics: Dict[str, Any] = {}
        dispatch = np.asarray(dispatch, dtype=np.float64)
        demand_mw = float(demand_mw)

        balance_err = float(dispatch.sum() - demand_mw)
        metrics["dispatch_sum_mw"] = float(dispatch.sum())
        metrics["demand_mw"] = demand_mw
        metrics["dispatch_minus_demand_mw"] = balance_err
        if self.check_dispatch_sum_balance:
            metrics["balance_error_mw"] = balance_err
            if abs(balance_err) > self.balance_tol_mw:
                failures.append(f"balance_error_mw={balance_err:.6f}")

        below = int((dispatch < self.gen_pmin - 1e-6).sum())
        above = int((dispatch > self.gen_pmax + 1e-6).sum())
        metrics["gen_pmin_violations"] = below
        metrics["gen_pmax_violations"] = above
        if below:
            failures.append(f"gen_pmin_violations={below}")
        if above:
            failures.append(f"gen_pmax_violations={above}")

        if check_base_ac:
            net = self._fresh_net(dispatch, demand_mw, gen_vm_pu=gen_vm_pu, ext_grid_vm_pu=ext_grid_vm_pu)
            err = self._runpp(net)
            if err:
                failures.append(f"base:{err}")
                return SCEDVerificationReport(False, failures, metrics)
            self._check_results(net, "base", failures, metrics)

        if check_all_ac_contingencies:
            cont_failures = 0
            checked = 0
            failed_contingencies = []
            types = set(contingency_types)
            contingencies = []
            if contingency_list is not None:
                for item in contingency_list:
                    if isinstance(item, dict):
                        element_type = str(item.get("type"))
                        idx = int(item.get("idx"))
                    else:
                        element_type, idx = item
                        element_type = str(element_type)
                        idx = int(idx)
                    if element_type in types:
                        contingencies.append((element_type, idx))
            else:
                if "line" in types:
                    contingencies.extend(("line", int(i)) for i in self.net_template.line.index)
                if "trafo" in types:
                    contingencies.extend(("trafo", int(i)) for i in self.net_template.trafo.index)
            if max_contingencies is not None:
                contingencies = contingencies[:int(max_contingencies)]

            for element_type, idx in contingencies:
                checked += 1
                if self._is_islanding_outage(element_type, idx):
                    metrics["n1_islanding_skipped"] = int(metrics.get("n1_islanding_skipped", 0)) + 1
                    continue
                net_c = self._fresh_net(dispatch, demand_mw, gen_vm_pu=gen_vm_pu, ext_grid_vm_pu=ext_grid_vm_pu)
                self._apply_n1_limits(net_c)
                try:
                    if element_type == "line":
                        net_c.line.at[idx, "in_service"] = False
                    elif element_type == "trafo":
                        net_c.trafo.at[idx, "in_service"] = False
                except Exception as exc:
                    failures.append(f"n1:{element_type}:{idx}:toggle_error:{exc}")
                    cont_failures += 1
                    continue

                err = self._runpp(net_c)
                if err:
                    failures.append(f"n1:{element_type}:{idx}:{err}")
                    failed_contingencies.append({
                        "type": element_type,
                        "idx": idx,
                        "reason": err,
                    })
                    cont_failures += 1
                    continue

                local_failures: List[str] = []
                local_metrics: Dict[str, Any] = {}
                self._check_results(net_c, f"n1_{element_type}_{idx}", local_failures, local_metrics)
                if local_failures:
                    failures.extend(local_failures)
                    failed_contingencies.append({
                        "type": element_type,
                        "idx": idx,
                        "reason": list(local_failures),
                    })
                    cont_failures += 1
                    if "first_n1_failure_metrics" not in metrics:
                        metrics["first_n1_failure_metrics"] = local_metrics

            metrics["n1_checked"] = checked
            metrics["n1_failures"] = cont_failures
            metrics["n1_failed_contingencies"] = failed_contingencies

        return SCEDVerificationReport(len(failures) == 0, failures, metrics)

    def verify_env(self, env, **kwargs) -> SCEDVerificationReport:
        if "gen_vm_pu" not in kwargs and hasattr(env, "current_gen_vm_pu"):
            kwargs["gen_vm_pu"] = env.current_gen_vm_pu
        if "ext_grid_vm_pu" not in kwargs and hasattr(env, "current_ext_grid_vm_pu"):
            kwargs["ext_grid_vm_pu"] = env.current_ext_grid_vm_pu
        return self.verify_dispatch(env.current_power, env.target_demand, **kwargs)

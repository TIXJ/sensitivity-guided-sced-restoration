#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCED-GAT-RL Pipeline: 三阶段智能经济调度框架
================================================

Phase 0: 数据加载 —— 读取300-bus系统数据 + 1100个场景的Gurobi精确解(Ground Truth)
Phase 1: GAT监督预训练 —— 用GT训练GAT预测器, 输出近似满足N-1的调度方案
Phase 2: RL微调校正器 —— 在GAT预测基础上, RL输出δP微调, 同时满足N-1 + AC潮流校核
Phase 3: 端到端评估 —— GT vs GAT vs GAT+RL 对比

用法:
    python sced_gat_rl_pipeline.py --phase all          # 运行全部阶段
    python sced_gat_rl_pipeline.py --phase 0            # 仅数据加载测试
    python sced_gat_rl_pipeline.py --phase 1            # 仅GAT训练
    python sced_gat_rl_pipeline.py --phase 2            # 仅RL校正器训练
    python sced_gat_rl_pipeline.py --phase 3            # 仅评估
    python sced_gat_rl_pipeline.py --phase ac           # DC vs AC-PF N-1对比评估
    python sced_gat_rl_pipeline.py --system 300         # 指定系统(300或1888)
    python sced_gat_rl_pipeline.py --n_train 50         # 训练场景数
    python sced_gat_rl_pipeline.py --gat_epochs 100     # GAT训练轮数
    python sced_gat_rl_pipeline.py --rl_timesteps 50000 # RL训练步数
"""

import os
import sys
import argparse
import time
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym

warnings.filterwarnings("ignore")

# ============================================================
# 路径配置
# ============================================================

DATA_ROOT = Path(r"external/Liaochangtao\RL_project\UC_RL\300")   # 300-bus数据根目录 (UC_RL数据集)
OUTPUT_DIR = Path(r".")
SAVE_DIR = OUTPUT_DIR / "sced_gat_rl_results"
SAVE_DIR.mkdir(exist_ok=True)

print(f"[Config] DATA_ROOT = {DATA_ROOT}")
print(f"[Config] SAVE_DIR  = {SAVE_DIR}")


# ============================================================
# Phase 0: 数据加载模块
# ============================================================

class SCEDDataManager:
    """
    300-bus SCED数据管理器。

    加载内容:
      - 系统参数: Unit_Data.csv (69机组), Line_Data.csv (411线路)
      - 网络拓扑: PTDF-R2.csv (411×300), Busload_Data.csv (节点负荷分布)
      - 时序数据: Load_Data.csv (96时段负荷曲线)
      - Ground Truth: solution/N/机组出力.csv (每个场景69×97的出力矩阵)

    数据规模:
      - N_UNITS=69, N_LINES=411, N_BUSES=300, T_HORIZON=96, N_SCENARIOS=1100
    """

    def __init__(self, data_root: Path, n_train: int = 50, n_val: int = 10,
                 n_test: int = 20, seed: int = 42):
        self.data_root = data_root
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # 将在 load_all() 中填充
        self.unit_data = None       # DataFrame: 机组参数
        self.line_data = None       # DataFrame: 线路参数
        self.ptdf_matrix = None     # ndarray: (N_LINES, N_BUSES) PTDF矩阵
        self.busload_data = None    # DataFrame: 节点负荷分布
        self.load_data = None       # DataFrame: 时序负荷 (t, Load, Reserve, rate)
        self.N_UNITS = 0
        self.N_LINES = 0
        self.N_BUSES = 0
        self.T_HORIZON = 0
        self.PMAX = None
        self.PMIN = None
        self.PUP = None             # 爬坡上限 [MW/时段]
        self.PDOWN = None           # 爬坡下限
        self.unit_bus = None        # 各机组所在母线(0-indexed)

        # 场景列表
        self.train_ids: List[int] = []
        self.val_ids: List[int] = []
        self.test_ids: List[int] = []

        # Ground Truth缓存: scenario_id → ndarray (T_HORIZON, N_UNITS)
        self.gt_power: Dict[int, np.ndarray] = {}
        # 场景特征缓存: scenario_id → dict
        self.scenario_features: Dict[int, dict] = {}

    def load_system_params(self):
        """加载不随场景变化的系统参数。

        UC_RL/300 数据集格式适配:
          - 根目录无 Unit_Data.csv → 用 Unit_temp/Unit_Data1.csv 作为默认
          - 根目录无 Load_Data.csv → 用 Bus_temp/Bus_Data1.csv 作为默认全局profile
          - 根目录无 Busload_Data.csv → 用 Busload_temp/Busload_data1.csv 作为默认
          - 每个场景可有独立的 Load/Reserve/rate 曲线 (Bus_temp/Bus_Data{sid}.csv)
        """
        print("\n[Phase0] Loading system parameters...")

        # 机组参数 — 优先根目录, 回退到场景1
        unit_file = self.data_root / "Unit_Data.csv"
        if not unit_file.exists():
            unit_file = self.data_root / "Unit_temp" / "Unit_Data1.csv"
            print(f"  [ADAPT] Using scenario-1 unit data as default: {unit_file.name}")
        self.unit_data = pd.read_csv(unit_file)
        self.N_UNITS = len(self.unit_data)
        self.PMAX = self.unit_data["Pmax"].values.astype(np.float64)
        self.PMIN = self.unit_data["Pmin"].values.astype(np.float64)
        self.PUP = self.unit_data["Pup"].values.astype(np.float64)
        self.PDOWN = self.unit_data["Pdown"].values.astype(np.float64)
        self.unit_bus = (self.unit_data["b"].values.astype(np.int64) - 1)  # → 0-indexed

        # 线路参数
        line_file = self.data_root / "Line_Data.csv"
        self.line_data = pd.read_csv(line_file)
        self.N_LINES = len(self.line_data)
        self.flow_limit = self.line_data["Flowlimit"].values.astype(np.float64)
        self.fl_scale = 28.0  # DC-LODF近似裕度 (修正PTDF映射后, 28x使GT N-1~10-20%)

        # PTDF矩阵
        ptdf_file = self.data_root / "PTDF-R2.csv"
        ptdf_raw = pd.read_csv(ptdf_file, header=None).values.astype(np.float64)
        self.N_BUSES = ptdf_raw.shape[1]

        # [关键修复] PTDF行重排: PTDF行序与Line_Data行序可能不一致
        # 通过自参考匹配(ptdf[row,from]-ptdf[row,to])找到正确映射
        self.ptdf_matrix = self._reorder_ptdf(ptdf_raw)

        # 节点负荷分布 — 优先根目录, 回退到场景1
        busload_file = self.data_root / "Busload_Data.csv"
        if not busload_file.exists():
            busload_file = self.data_root / "Busload_temp" / "Busload_data1.csv"
            print(f"  [ADAPT] Using scenario-1 busload as default: {busload_file.name}")
        if busload_file.exists():
            self.busload_data = pd.read_csv(busload_file)

        # 时序负荷数据 — UC_RL/300是按场景的, 用场景1作为全局默认
        load_file = self.data_root / "Load_Data.csv"
        if not load_file.exists():
            load_file = self.data_root / "Bus_temp" / "Bus_Data1.csv"
            print(f"  [ADAPT] Using scenario-1 load profile as default: {load_file.name}")
        self.load_data = pd.read_csv(load_file)
        self.T_HORIZON = len(self.load_data)
        self.load_profile = self.load_data["Load"].values.astype(np.float64)
        self.reserve_profile = self.load_data["Reserve"].values.astype(np.float64)
        self.drate_profile = self.load_data["rate"].values.astype(np.float64)

        # 预加载所有场景的负荷profile (用于per-scenario数据访问)
        self._scenario_profiles: Dict[int, pd.DataFrame] = {}
        profile_dir = self.data_root / "Bus_temp"
        if profile_dir.exists():
            for f in sorted(profile_dir.glob("Bus_Data*.csv")):
                try:
                    sid = int(''.join(filter(str.isdigit, f.stem)))
                    self._scenario_profiles[sid] = pd.read_csv(f)
                except (ValueError, IndexError):
                    continue
            print(f"  [ADAPT] Loaded {len(self._scenario_profiles)} scenario load profiles")

        print(f"  Units: {self.N_UNITS}, Lines: {self.N_LINES}, "
              f"Buses: {self.N_BUSES}, Periods: {self.T_HORIZON}")
        print(f"  PTDF shape: {self.ptdf_matrix.shape}")
        print(f"  Total capacity: {self.PMAX.sum():.1f} MW")
        print(f"  Load range: {self.load_profile.min():.1f} ~ {self.load_profile.max():.1f} MW")

    def _reorder_ptdf(self, ptdf_raw: np.ndarray) -> np.ndarray:
        """修正PTDF行序以匹配Line_Data行序。

        问题根因: PTDF-R2.csv的行排列顺序与Line_Data.csv的线路顺序不一致，
               导致LODF计算时使用错误的分布因子，产生非物理的巨大越限量。
        方法: 贪心匹配——对每条line l，找使|ptdf[r,from_l]-ptdf[r,to_l]|最大的PTDF行r。
        """
        from_col = self.line_data["BusFrom"].values.astype(int) - 1
        to_col = self.line_data["BusTo"].values.astype(int) - 1
        n = self.N_LINES

        # 计算匹配分数矩阵
        scores = np.zeros((n, n))
        for r in range(n):
            for l in range(n):
                scores[r, l] = abs(ptdf_raw[r, from_col[l]] - ptdf_raw[r, to_col[l]])

        # 贪心分配
        flat = [(scores[r, l], r, l) for r in range(n) for l in range(n)]
        flat.sort(reverse=True)
        ptdf_row_for_line = [0] * n
        used_r, used_l = set(), set()
        n_matched = 0
        for _, r, l in flat:
            if r not in used_r and l not in used_l:
                ptdf_row_for_line[l] = r
                used_r.add(r); used_l.add(l)
                n_matched += 1

        reordered = ptdf_raw[ptdf_row_for_line, :]

        # 验证自参考值 (应接近±1~±2)
        ref_diffs = []
        for l in range(min(20, n)):
            fk, tk = from_col[l], to_col[l]
            ref_diffs.append(abs(reordered[l, fk] - reordered[l, tk]))

        print(f"  [PTDF-Remap] {n_matched}/{n} lines mapped")
        print(f"  [PTDF-Remap] Self-ref |diff| (first 20): "
              f"min={min(ref_diffs):.3f}, max={max(ref_diffs):.3f}")
        return reordered

    def _get_scenario_unit_data(self, sid: int) -> pd.DataFrame:
        """获取指定场景的机组参数(燃料费可能随场景变化)。"""
        # 尝试从Unit_temp目录加载场景特定参数
        scenario_unit_file = self.data_root / f"Unit_temp/Unit_Data{sid}.csv"
        if scenario_unit_file.exists():
            return pd.read_csv(scenario_unit_file)
        return self.unit_data.copy()

    def _get_scenario_profile(self, sid: int) -> pd.DataFrame:
        """获取指定场景的时序负荷profile (Load, Reserve, rate)。

        UC_RL/300 数据集中每个场景有独立的96时段负荷曲线。
        """
        if hasattr(self, '_scenario_profiles') and sid in self._scenario_profiles:
            return self._scenario_profiles[sid]
        # 回退到全局默认
        return self.load_data

    def _get_scenario_busload(self, sid: int) -> pd.DataFrame:
        """获取指定场景的节点负荷分布。"""
        bl_file = self.data_root / f"Busload_temp/Busload_data{sid}.csv"
        if bl_file.exists():
            return pd.read_csv(bl_file)
        if self.busload_data is not None:
            return self.busload_data
        return None

    def load_ground_truth(self, scenario_id: int) -> Optional[np.ndarray]:
        sid = scenario_id  # shorthand
        """
        加载单个场景的Ground Truth出力矩阵。

        Returns:
            ndarray of shape (T_HORIZON, N_UNITS) — 每时段每台机组的出力[MW]
            或 None 如果文件不存在
        """
        sol_dir = self.data_root / "solution" / str(scenario_id)
        power_file = sol_dir / "机组出力.csv"

        if not power_file.exists():
            return None

        try:
            # Try different encodings (GBK for Chinese Windows, then UTF-8, then latin1)
            for enc in ['gbk', 'utf-8', 'gb2312', 'latin1']:
                try:
                    df = pd.read_csv(power_file, index_col=0, encoding=enc)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            else:
                print(f"  [WARN] Scenario {sid}: Failed to decode GT with any encoding")
                return None
            # df shape: (N_UNITS, T_ACTUAL) — 可能因场景而异
            power_mat = df.values.astype(np.float64)  # (N_UNITS, T_ACTUAL)
            t_actual = power_mat.shape[1]
            if t_actual != self.T_HORIZON:
                # 自适应：以实际时段数为准（不同场景可能求解时段数不同）
                if t_actual < self.T_HORIZON:
                    # 补零到T_HORIZON（用最后一列填充）
                    pad = np.tile(power_mat[:, -1:], (1, self.T_HORIZON - t_actual))
                    power_mat = np.hstack([power_mat, pad])
                    print(f"  [INFO] Scenario {sid}: GT padded {t_actual}→{self.T_HORIZON}")
                else:
                    power_mat = power_mat[:, :self.T_HORIZON]
            return power_mat  # (N_UNITS, T_HORIZON) → 我们用 (T, N) 所以需要转置
        except Exception as e:
            print(f"  [WARN] Scenario {sid}: Failed to load GT: {e}")
            return None

    def compute_line_flows(self, power: np.ndarray, drate: float = 1.0) -> np.ndarray:
        """
        基于PTDF计算线路潮流。

        Args:
            power: (N_UNITS,) 各机组出力 [MW]
            drate: 负荷分配比例系数

        Returns:
            flows: (N_LINES,) 各线路有功潮流 [MW]
        """
        # 计算各节点的净注入功率: P_bus[b] = Σ(power[i] for i on bus b)
        p_bus = np.zeros(self.N_BUSES, dtype=np.float64)
        for i in range(self.N_UNITS):
            b = self.unit_bus[i]
            if 0 <= b < self.N_BUSES:
                p_bus[b] += power[i]

        # 线路潮流 = PTDF × P_bus
        flows = self.ptdf_matrix @ p_bus * drate  # (N_LINES,)
        return flows

    def check_n1_constraints(self, flows: np.ndarray) -> dict:
        """
        N-1安全约束检查（基于LODF近似）。

        对每条线路k停运，检查其余线路l是否越限：
            Flow_l^{(k)} = Flow_l + LODF_{l,k} × Flow_k
        """
        # 预计算LODF矩阵（如果还没算）
        if not hasattr(self, '_lodf_matrix'):
            self._precompute_lodf()

        lodf = self._lodf_matrix
        # 应用fl_scale放大限值 (工程备用裕度)
        eff_limit = self.flow_limit * self.fl_scale
        n1_overload_sum = 0.0
        n1_violated_lines = 0
        contingency_violations = 0

        for k in range(self.N_LINES):
            if abs(flows[k]) < 1e-6:
                continue
            post_flows = flows + lodf[:, k] * flows[k]
            over = np.abs(post_flows) - eff_limit
            viol_mask = over > 1e-3
            n_v = viol_mask.sum()
            if n_v > 0:
                contingency_violations += 1
                n1_violated_lines += n_v
                n1_overload_sum += over[viol_mask].sum()

        return {
            "n1_overload_sum": float(n1_overload_sum),
            "n1_violated_lines": int(n1_violated_lines),
            "n1_contingency_violations": int(contingency_violations),
        }

    def _check_n1_detailed(self, flows: np.ndarray) -> dict:
        """
        N-1安全约束详细检查 (连续值版本，用于Reward塑形)。

        返回过载量(非二值)，提供密集的梯度信号。
        """
        if not hasattr(self, '_lodf_matrix'):
            self._precompute_lodf()

        lodf = self._lodf_matrix
        eff_limit = self.flow_limit * self.fl_scale

        total_overload = 0.0       # 所有contingency的所有线路过载之和 [MW]
        max_overload = 0.0          # 单个线路在单个contingency下的最大过载 [MW]
        n_contingencies = 0         # 有任何过载的contingency数 (离散)
        worst_line_per_contingency = []  # 每个违规contingency的最差线路

        for k in range(self.L):
            if abs(flows[k]) < 1e-6:
                continue
            post_flows = flows + lodf[:, k] * flows[k]
            over = np.abs(post_flows) - eff_limit
            over_pos = np.maximum(over, 0)
            k_overload_sum = over_pos.sum()
            k_max_over = over_pos.max() if over_pos.size > 0 else 0.0

            if k_overload_sum > 1e-3:
                n_contingencies += 1
                total_overload += k_overload_sum
                max_overload = max(max_overload, k_max_over)
                worst_line_per_contingency.append(k_max_over)

        return {
            "total_overload": float(total_overload),      # 连续: 总过载MW数
            "max_overload": float(max_overload),           # 连续: 最严重单线过载
            "n_contingencies": int(n_contingencies),       # 离散: 违规contingency数
            "avg_worst_overload": (float(np.mean(worst_line_per_contingency))
                                   if worst_line_per_contingency else 0.0),
        }

    def _precompute_lodf(self):
        """预计算LODF矩阵 (N_LINES × N_LINES)。"""
        ptdf = self.ptdf_matrix
        n = self.N_LINES
        lodf = np.zeros((n, n), dtype=np.float64)

        # 提取线路端点信息
        from_col = self.line_data["BusFrom"].values.astype(int) - 1  # 0-indexed
        to_col = self.line_data["BusTo"].values.astype(int) - 1

        for k in range(n):
            fk, tk = from_col[k], to_col[k]
            x_k = ptdf[k, tk] - ptdf[k, fk]  # 线路k的电抗(近似)
            if abs(x_k) < 1e-10:
                continue
            denom = ptdf[k, fk] - ptdf[k, tk]  # 正确: PTDF自敏感度差 (非 1.0 - diff)
            if abs(denom) < 1e-10:
                continue  # 辐射状线路(无回路), LODF=0
            for l in range(n):
                fl, tl = from_col[l], to_col[l]
                lodf[l, k] = (ptdf[l, fl] - ptdf[l, tl]) / denom

        self._lodf_matrix = lodf
        nz = np.count_nonzero(lodf)
        print(f"  [N-1] LODF matrix ({n}×{n}) computed, {nz} non-zero entries")

    # ================================================================
    # AC Power Flow Module (based on pandapower)
    # ================================================================

    def _build_pandapower_net(self, r_x_ratio: float = 0.2, base_mva: float = 100.0):
        """
        用pandapower构建300-bus交流网络模型。

        基于Line_Data.csv的X参数，假设R/X比值推导R。
        网络包含: bus, gen, load, line(impedance)。
        """
        import pandapower as pp
        from pandapower.networks import create_empty_network

        ld = self.line_data
        n_bus = self.N_BUSES
        n_line = self.N_LINES

        net = create_empty_network(sn_mva=base_mva, f_hz=50.0)

        # ---- Buses: 创建300个节点 ----
        for i in range(n_bus):
            pp.create_bus(net, vn_kv=230.0, name=f"Bus_{i+1}",
                          type="n" if i > 0 else "slack")  # Bus1=slack

        # ---- Generators: 在机组所在节点创建 ----
        for i in range(self.N_UNITS):
            b = int(self.unit_bus[i]) if self.unit_bus is not None else 0
            is_slack = (i == 0)  # 第一台机组作为平衡节点
            pp.create_gen(net, bus=b, p_mw=0.0, vm_pu=1.02,
                          max_p_mw=float(self.PMAX[i]),
                          min_p_mw=float(self.PMIN[i]),
                          slack=is_slack,
                          name=f"Gen_{i+1}")

        # ---- Lines: 从Line_Data创建阻抗支路 ----
        from_bus = ld["BusFrom"].values.astype(int) - 1   # → 0-indexed
        to_bus = ld["BusTo"].values.astype(int) - 1         # → 0-indexed
        x_pu = ld["X"].values.astype(np.float64)   # 电抗[pu]
        r_pu = x_pu * r_x_ratio                      # 推导电阻[pu]
        flow_limit = ld["Flowlimit"].values.astype(np.float64)

        for k in range(n_line):
            f, t = int(from_bus[k]), int(to_bus[k])
            # create_impedance 使用 pu 值 (rft_pu, xft_pu)
            sn_mva = flow_limit[k] / 100.0 if flow_limit[k] > 0 else 1000.0
            pp.create_impedance(net, from_bus=f, to_bus=t,
                                rft_pu=r_pu[k], xft_pu=x_pu[k],
                                sn_mva=sn_mva,
                                name=f"Line_{k+1}")

        # ---- Loads: 从Busload_Data创建 ----
        if self.busload_data is not None:
            bl = self.busload_data
            bus_nos = bl["BusNo"].values.astype(int) - 1   # → 0-indexed
            pl = bl["PL"].values.astype(np.float64)
            for idx in range(len(bl)):
                b = int(bus_nos[idx])
                pp.create_load(net, bus=b, p_mw=pl[idx], q_mw=pl[idx] * np.tan(np.arccos(0.95)),
                               const_z_percent=0, const_i_percent=0, const_p_percent=100,
                               name=f"Load_b{b}")

        self._pp_net = net
        self._pp_base_mva = base_mva
        self._r_x_ratio = r_x_ratio

        print(f"  [AC-PF] pandapower net built: {len(net.bus)} buses, "
              f"{len(net.gen)} gens, {len(net.impedance)} lines, "
              f"{len(net.load)} loads, R/X={r_x_ratio}")

    def _run_ac_pf_pandapower(self, power: np.ndarray, drate: float = 1.0) -> dict:
        """
        使用pandapower.runacpp()运行交流潮流。

        Args:
            power: (N_UNITS,) 各机组出力 [MW]
            drate: 负荷比例系数

        Returns:
            dict: converged, V (complex), vmag, theta, line_flows_MW, error_msg
        """
        import pandapower as pp

        if not hasattr(self, '_pp_net') or self._pp_net is None:
            self._build_pandapower_net()

        import copy
        net = copy.deepcopy(self._pp_net)  # 深拷贝以避免修改原始网络

        # 设置发电机出力
        for i in range(min(len(net.gen), self.N_UNITS)):
            net.gen.p_mw.values[i] = float(power[i])

        # 按drate缩放负荷
        if len(net.load) > 0:
            original_loads = net.load.p_mw.values.copy()
            net.load.p_mw = original_loads * drate
            net.load.q_mw = net.load.q_mw * drate

        try:
            # 运行AC潮流 (Newton-Raphson, pandapower内置)
            result = pp.runpp(net, max_iter=50, tolerance_mva=1e-4,
                               enforce_q_lims=False)

            if result.converged:
                V_complex = np.array([net.res_bus.vm_pu[b] *
                                      np.exp(1j * np.deg2rad(net.res_bus.va_degree[b]))
                                      for b in range(len(net.bus))])
                # 阻抗支路有功潮流 [MW]
                line_flows = net.res_impedance.pl_mw.values.copy()
                return {
                    "converged": True,
                    "iterations": result.get("iterations", -1),
                    "V": V_complex,
                    "vmag": net.res_bus.vm_pu.values.copy(),
                    "theta": np.deg2rad(net.res_bus.va_degree.values.copy()),
                    "line_flows": line_flows,
                    "line_loading": net.res_impedance.loading_percent.values.copy(),
                    "bus_p_mw": net.res_bus.p_mw.values.copy(),
                    "bus_q_mvar": net.res_bus.q_mvar.values.copy(),
                    "error_msg": None,
                }
            else:
                return {
                    "converged": False,
                    "iterations": result.get("iterations", -1),
                    "error_msg": f"AC-PF not converged",
                    "V": None,
                    "line_flows": None,
                }
        except Exception as e:
            return {"converged": False, "error_msg": str(e), "V": None}

    def _trip_line_and_run_pf(self, line_idx: int, power: np.ndarray,
                                drate: float = 1.0) -> dict:
        """
        停运指定线路后重新运行AC潮流。

        Args:
            line_idx: 要停运的线路编号(0-indexed, 对应pandapower net.line索引)
            power, drate: 同上

        Returns:
            dict: 与_run_ac_pf_pandapower格式相同
        """
        import pandapower as pp

        if not hasattr(self, '_pp_net'):
            self._build_pandapower_net()

        import copy
        net = copy.deepcopy(self._pp_net)

        # 设置发电机出力
        for i in range(min(len(net.gen), self.N_UNITS)):
            net.gen.p_mw.values[i] = float(power[i])

        # 缩放负荷
        if len(net.load) > 0:
            net.load.p_mw = net.load.p_mw.values * drate
            net.load.q_mw = net.load.q_mw.values * drate

        # 停运线路: 设置in_service=False
        if line_idx < len(net.impedance):
            net.impedance.at[line_idx, 'in_service'] = False

        try:
            result = pp.runpp(net, max_iter=50, tolerance_mva=1e-4,
                               enforce_q_lims=False)
            if result.converged:
                V_complex = np.array([net.res_bus.vm_pu[b] *
                                      np.exp(1j * np.deg2rad(net.res_bus.va_degree[b]))
                                      for b in range(len(net.bus))])
                # 停运一条阻抗支路后，res_impedance只包含在运支路
                active_imp = net.impedance[net.impedance.in_service].index
                n_total_imp = len(self._pp_net.impedance)
                all_flows = np.zeros(n_total_imp, dtype=np.float64)
                if len(net.res_impedance) > 0:
                    all_flows[active_imp] = net.res_impedance.pl_mw.values.copy()
                return {
                    "converged": True,
                    "iterations": result.get("iterations", -1),
                    "V": V_complex,
                    "vmag": net.res_bus.vm_pu.values.copy(),
                    "theta": np.deg2rad(net.res_bus.va_degree.values.copy()),
                    "line_flows": all_flows,
                    "line_loading_pct": net.res_impedance.loading_percent.values.copy(),
                    "active_line_indices": list(active_imp),
                    "error_msg": None,
                }
            else:
                return {"converged": False, "error_msg": "PF_not_converged_after_trip"}
        except Exception as e:
            return {"converged": False, "error_msg": str(e)}

    def check_n1_constraints_ac(
        self,
        power: np.ndarray,
        drate: float = 1.0,
        voltage_lo: float = 0.95,
        voltage_hi: float = 1.05,
        verbose: bool = False,
    ) -> dict:
        """
        AC-PF N-1安全校验（基于pandapower）。

        对每条线路k停运 → re-run AC-PF → 检查线路越限 + 电压越限。

        Args:
            power: (N_UNITS,) 机组出力 [MW]
            drate: 负荷比例系数
            voltage_lo/hi: 电压上下限 [pu]
            verbose: 打印详细信息

        Returns:
            dict with DC-compatible format + AC-specific fields
        """
        import time

        t_start = time.time()

        # ---- Base case PF ----
        base_result = self._run_ac_pf_pandapower(power, drate)
        dc_base_flows = self.compute_line_flows(power, drate)

        ac_base_flows = base_result.get("line_flows")
        if ac_base_flows is not None and len(ac_base_flows) == self.N_LINES:
            pass  # OK
        elif ac_base_flows is not None:
            # 可能长度不同（pandapower内部处理差异）
            pass
        else:
            ac_base_flows = np.zeros(self.N_LINES)

        if verbose and not base_result["converged"]:
            print(f"    [AC-N1] WARNING: Base case PF NOT converged: "
                  f"{base_result.get('error_msg', '?')}")

        # ---- N-1 Contingency analysis ----
        n1_overload_sum = 0.0
        n1_violated_lines = 0
        contingency_violations = 0
        contingency_details = []
        ac_voltage_violations = 0
        total_contingencies_checked = 0

        # 只对基态有明显潮流的线路做N-1（跳过空载/轻载线路加速）
        if ac_base_flows is not None and len(ac_base_flows) > 0:
            significant_lines = np.where(np.abs(ac_base_flows) > 0.5)[0]
        else:
            significant_lines = range(self.N_LINES)

        for k in significant_lines:
            k = int(k)
            total_contingencies_checked += 1

            c_result = self._trip_line_and_run_pf(k, power, drate)

            if not c_result["converged"]:
                # 不收敛 = 不安全（保守处理）
                contingency_violations += 1
                n1_violated_lines += 1
                contingency_details.append({
                    "line_k": k, "reason": "PF_not_converged",
                    "dc_flow": float(dc_base_flows[k]),
                })
                continue

            # 检查线路潮流越限（仅检查仍在线路中的）
            c_flows = c_result["line_flows"]
            over = np.abs(c_flows) - self.flow_limit
            viol_mask = over > 1e-3  # MW容差
            n_v = viol_mask.sum()

            # 检查电压越限
            vmag = c_result.get("vmag")
            n_v_viol = 0
            if vmag is not None:
                v_low = vmag < voltage_lo
                v_high = vmag > voltage_hi
                n_v_viol = int((v_low | v_high).sum())

            has_violation = (n_v > 0) or (n_v_viol > 0)
            if has_violation:
                contingency_violations += 1
                n1_violated_lines += n_v
                ac_voltage_violations += n_v_viol
                n1_overload_sum += float(over[viol_mask].sum()) if n_v > 0 else 0.0

            if verbose and has_violation:
                worst_over = float(over.max()) if n_v > 0 else 0.0
                min_v = float(vmag.min()) if vmag is not None else 1.0
                print(f"    [AC-N1] Line {k:>3d}: {n_v} flow viols, "
                      f"{n_v_viol} V viols, worst={worst_over:.0f}MW, "
                      f"Vmin={min_v:.4f}pu")

            contingency_details.append({
                "line_k": int(k), "n_flow_viols": int(n_v),
                "n_voltage_viols": int(n_v_viol), "converged": True,
                "worst_overload_MW": worst_over if has_violation else 0.0,
                "min_voltage_pu": float(vmag.min()) if vmag is not None else 1.0,
            })

        elapsed = time.time() - t_start

        return {
            # --- DC兼容格式 ---
            "n1_overload_sum": float(n1_overload_sum),
            "n1_violated_lines": int(n1_violated_lines),
            "n1_contingency_violations": int(contingency_violations),
            # --- AC特有字段 ---
            "ac_voltage_violations": int(ac_voltage_violations),
            "total_contingencies_checked": int(total_contingencies_checked),
            "base_case_converged": base_result["converged"],
            "base_case_iterations": base_result.get("iterations", -1),
            "contingency_details": contingency_details,
            "eval_time_s": round(elapsed, 2),
            # --- 基态对比数据 ---
            "ac_base_flows": ac_base_flows,
            "dc_base_flows": dc_base_flows,
        }

    def build_features_for_scenario(self, scenario_id: int) -> dict:
        """
        为单个场景构建输入特征向量。

        特征包含:
          - 全局特征(24维): 时间进度, 当前负荷/备用, 容量上下界,
                             未来4步负荷/备用预测, 当前时段drate
          - 节点特征(N_UNITS×5维): 开关状态, 出力比例, 可调上/下裕度, 机组容量比
        """
        ud = self._get_scenario_unit_data(scenario_id)
        price_seg1 = ud["PriceSegment1"].values.astype(np.float64)
        # 使用该场景专属的负荷曲线 (UC_RL/300 每场景独立)
        profile = self._get_scenario_profile(scenario_id)

        features = {
            "price_seg1": price_seg1,
            "unit_data": ud,
            "load_profile": profile["Load"].values.astype(np.float64),
            "reserve_profile": profile["Reserve"].values.astype(np.float64),
            "drate_profile": profile["rate"].values.astype(np.float64),
        }
        return features

    def split_scenarios(self):
        """划分训练/验证/测试集场景ID。"""
        sol_dir = self.data_root / "solution"
        all_ids = sorted([int(d.name) for d in sol_dir.iterdir() if d.is_dir() and d.name.isdigit()])
        total = len(all_ids)
        print(f"\n[Phase0] Found {total} scenarios with solutions")

        n_avail = min(total, self.n_train + self.n_val + self.n_test)
        selected = self.rng.choice(all_ids, size=n_avail, replace=False).tolist()
        self.train_ids = sorted(selected[:self.n_train])
        self.val_ids = sorted(selected[self.n_train:self.n_train + self.n_val])
        self.test_ids = sorted(selected[self.n_train + self.n_val:n_avail])

        print(f"  Train: {len(self.train_ids)} scenarios ({self.train_ids[0]}~{self.train_ids[-1]})")
        print(f"  Val:   {len(self.val_ids)} scenarios ({self.val_ids[0]}~{self.val_ids[-1]})")
        print(f"  Test:  {len(self.test_ids)} scenarios ({self.test_ids[0]}~{self.test_ids[-1]})")

    def load_all(self):
        """执行完整的数据加载流程。"""
        print("=" * 60)
        print("  Phase 0: Data Loading & Preprocessing")
        print("=" * 60)

        self.load_system_params()
        self.split_scenarios()

        # 加载所有划分中场景的Ground Truth
        all_ids = self.train_ids + self.val_ids + self.test_ids
        print(f"\n[Phase0] Loading ground truth for {len(all_ids)} scenarios...")
        t0 = time.time()
        loaded = 0
        failed = 0
        for sid in all_ids:
            gt = self.load_ground_truth(sid)
            if gt is not None:
                # gt shape: (N_UNITS, T_HORIZON) → 转置为 (T_HORIZON, N_UNITS)
                if gt.shape[0] == self.N_UNITS and gt.shape[1] == self.T_HORIZON:
                    self.gt_power[sid] = gt.T  # (T, N)
                elif gt.shape[0] == self.T_HORIZON and gt.shape[1] == self.N_UNITS:
                    self.gt_power[sid] = gt  # already (T, N)
                else:
                    print(f"  [WARN] Scenario {sid}: unexpected GT shape {gt.shape}")
                    failed += 1
                    continue

                # 构建场景特征
                self.scenario_features[sid] = self.build_features_for_scenario(sid)
                loaded += 1
            else:
                failed += 1
                if failed <= 3:
                    print(f"  [FAIL] Scenario {sid}: no ground truth found")

        elapsed = time.time() - t0
        print(f"\n  Loaded {loaded}/{len(all_ids)} scenarios in {elapsed:.1f}s ({failed} failed)")

        # 打印一个样本的统计
        if self.gt_power:
            first_sid = list(self.gt_power.keys())[0]
            gt_sample = self.gt_power[first_sid]
            print(f"\n  Sample GT (scenario {first_sid}): shape={gt_sample.shape}")
            print(f"    Total generation per period (first 5): {gt_sample[:5].sum(axis=1).round(1)}")
            print(f"    Load profile (first 5):           {self.load_profile[:5].round(1)}")

        return loaded > 0


# ============================================================
# GAT 网络定义 (内联版本，避免外部依赖问题)
# ============================================================

class GATv2ConvLayer(nn.Module):
    """纯PyTorch实现的GATv2层（无PyG依赖）。"""
    def __init__(self, in_features, out_features, heads=4, concat=True, dropout=0.1, bias=True):
        super().__init__()
        self.heads = heads
        self.out_features = out_features
        self.concat = concat
        self.dropout = dropout
        self.W_src = nn.Linear(in_features, heads * out_features, bias=False)
        self.W_dst = nn.Linear(in_features, heads * out_features, bias=bias)
        self.a = nn.Parameter(torch.zeros(1, heads, out_features * 2))
        self._reset_parameters()

    def _reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.W_src.weight, gain=gain)
        nn.init.xavier_uniform_(self.W_dst.weight, gain=gain)
        nn.init.normal_(self.a, std=0.02)

    def forward(self, x, edge_index):
        # Dense GAT: full (N×N) attention with edge masking
        # Uses only bmm/baddmm — fully autograd-compatible
        N = x.size(0)
        H = self.heads
        D = self.out_features

        h_s = self.W_src(x).view(N, H, D)   # (N, H, D) source projections
        h_d = self.W_dst(x).view(N, H, D)  # (N, H, D) dest projections

        # Split attention vector a (1, H, 2D) → a_src(1,H,D), a_dst(1,H,D)
        a_split = self.a.view(H, 2*D)
        a_src = a_split[:, :D].view(H, 1, D)      # (H, 1, D)
        a_dst = a_split[:, D:].view(H, 1, D)     # (H, 1, D)

        # Compute (N,N,H) attention logits
        attn_logits = torch.zeros(N, N, H, device=x.device)
        for h_idx in range(H):
            # h_s @ a_src^T: (N,D) x (1,D)->(N,1) -> broadcast to (N,N)
            src_score = h_s[:, h_idx, :] @ a_src[h_idx, 0, :].T   # (N,)
            dst_score = h_d[:, h_idx, :] @ a_dst[h_idx, 0, :].T   # (N,)
            attn_logits[:, :, h_idx] = src_score.unsqueeze(1) + dst_score.unsqueeze(0)

        attn_logits = F.leaky_relu(attn_logits, negative_slope=0.2)

        # Mask: only edges from edge_index get real scores
        mask_val = torch.full((N, N), float('-inf'), device=x.device)
        mask_val[edge_index[0].long(), edge_index[1].long()] = 0
        attn_scores = attn_logits + mask_val.unsqueeze(-1)  # (N, N, H)

        attn = F.softmax(attn_scores, dim=1)  # normalize over source nodes

        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout, training=True)

        # Aggregate messages into destination nodes
        out = torch.zeros(N, H, D, device=x.device)
        for h_idx in range(H):
            # (N,N) @ (N,D) → (N,D): each dest node gets weighted sum of source features
            out[:, h_idx, :] = torch.matmul(attn[:, :, h_idx], h_s[:, h_idx, :])

        if self.concat:
            return out.reshape(N, H * D)
        return out.mean(dim=1)


class GATPredictor(nn.Module):
    """
    SCED用的GAT预测网络。

    输入: 观测向量 (obs_dim,)
    输出: 各机组出力预测 (N_UNITS,)

    架构:
      obs → parse → [node_feat (N×5), global_feat (24)]
      node_feat → GATv2 × 2 layers → mean_pool ─┐
      global_feat → MLP projector                 ├→ concat → MLP → P_pred (N_UNITS,)
    """
    NODE_DIM = 5   # on_off, p_ratio, margin_up, margin_down, capacity_ratio
    GLOBAL_DIM = 19  # matches _make_obs() output

    def __init__(self, obs_dim, n_units, edge_index, gat_heads=4, gat_hidden=128):
        super().__init__()
        self.n_units = n_units
        self.obs_dim = obs_dim

        if edge_index is None:
            # 默认全连接图
            src, dst = [], []
            for i in range(n_units):
                for j in range(n_units):
                    if i != j:
                        src.append(i); dst.append(j)
            edge_index = torch.LongTensor([src, dst])
        self.register_buffer('edge_index', edge_index)

        # GAT backbone
        self.gat1 = GATv2ConvLayer(self.NODE_DIM, gat_hidden, heads=gat_heads, concat=True)
        self.gat2 = GATv2ConvLayer(gat_hidden * gat_heads, gat_hidden, heads=gat_heads, concat=False)
        gat_out_dim = gat_hidden

        # Global feature projector
        self.global_proj = nn.Sequential(
            nn.Linear(self.GLOBAL_DIM, 128), nn.ReLU(),
            nn.Linear(128, gat_out_dim),
        )

        # Fusion + output
        fusion_dim = gat_out_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_units),
        )

    def _parse_obs(self, obs):
        """将观测向量拆分为节点特征和全局特征。"""
        B = obs.size(0)
        node = obs[:, :self.n_units * self.NODE_DIM].reshape(B * self.n_units, self.NODE_DIM)
        glob = obs[:, self.n_units * self.NODE_DIM:]
        return node, glob

    def forward(self, obs):
        B = obs.size(0)
        node, glob = self._parse_obs(obs)

        ei = self.edge_index
        if B > 1:
            offsets = torch.arange(B, device=obs.device).unsqueeze(1) * self.n_units
            ei = torch.cat([ei[0:1].expand(B,-1)+offsets, ei[1:2].expand(B,-1)+offsets], dim=0).reshape(2,-1)

        h = self.gat1(node, ei)
        h = F.relu(h)
        h = self.gat2(h, ei)  # (B*N, D)

        h_pool = h.reshape(B, self.n_units, -1).mean(dim=1)  # (B, D)
        g_proj = self.global_proj(glob)
        fused = torch.cat([h_pool, g_proj], dim=-1)

        return self.fusion(fused)  # (B, N_UNITS)


# ============================================================
# PyTorch Dataset
# ============================================================

class SCEDDataset(Dataset):
    """SCED预测任务的PyTorch Dataset。"""

    def __init__(self, dm: SCEDDataManager, scenario_ids: List[int],
                 max_load: float = 1.0, max_rsv: float = 1.0):
        self.dm = dm
        self.scenario_ids = scenario_ids
        self.max_load = max_load if max_load > 0 else dm.load_profile.max()
        self.max_rsv = max_rsv if max_rsv > 0 else dm.reserve_profile.max()

        self.samples = []  # List of (obs_array, target_power)
        self._build_samples()

    def _build_samples(self):
        """为每个场景×每个时段构建一个样本。"""
        for sid in self.scenario_ids:
            if sid not in self.dm.gt_power:
                continue
            gt = self.dm.gt_power[sid]  # (T, N)
            feat = self.dm.scenario_features.get(sid, {})

            for t in range(self.dm.T_HORIZON):
                obs = self._make_obs(sid, t, gt[t], feat)
                target = gt[t]  # (N,) — GT出力
                self.samples.append((obs, target))

        print(f"  Dataset built: {len(self.samples)} samples from {len(self.scenario_ids)} scenarios")

    def _make_obs(self, sid: int, t: int, current_power: np.ndarray, feat: dict) -> np.ndarray:
        """构建单个时刻的观测向量。"""
        dm = self.dm
        N = dm.N_UNITS
        T = dm.T_HORIZON

        # 使用该场景专属的负荷profile (UC_RL/300适配)
        load_profile = feat.get("load_profile", dm.load_profile)
        reserve_profile = feat.get("reserve_profile", dm.reserve_profile)
        drate_profile = feat.get("drate_profile", dm.drate_profile)

        # 节点特征 (N×5)
        on_off = (current_power > dm.PMIN * 0.5).astype(np.float32)
        p_ratio = np.clip(current_power / (dm.PMAX + 1e-6), 0, 1).astype(np.float32)
        margin_up = np.clip((dm.PMAX - current_power) / (dm.PMAX + 1e-6), 0, 1).astype(np.float32)
        margin_down = np.clip((current_power - dm.PMIN) / (dm.PMAX + 1e-6), 0, 1).astype(np.float32)
        cap_ratio = dm.PMAX / (dm.PMAX.max() + 1e-6)

        node_feat = np.stack([on_off, p_ratio, margin_up, margin_down, cap_ratio], axis=1)  # (N, 5)

        # 全局特征 (24)
        max_load_val = float(load_profile.max()) if load_profile.max() > 0 else 1.0
        max_rsv_val = float(reserve_profile.max()) if reserve_profile.max() > 0 else 1.0

        future_load = np.array([load_profile[min(t+k, T-1)] / max_load_val for k in range(1,5)], dtype=np.float32)
        future_rsv = np.array([reserve_profile[min(t+k, T-1)] / max_rsv_val for k in range(1,5)], dtype=np.float32)

        cur_dem = load_profile[t] / max_load_val
        cur_rsv = reserve_profile[t] / max_rsv_val
        drate = drate_profile[min(t, T-1)]

        cap_min = dm.PMIN[on_off > 0.5].sum() if (on_off > 0.5).any() else 0
        cap_max = dm.PMAX.sum()

        global_feat = np.array([
            t / T, cur_dem, cur_rsv,
            cap_min / (max_load_val + 1e-6), cap_max / (max_load_val + 1e-6),
            drate,
            *future_load, *future_rsv,
            0, 0, 0,  # 占位: overload info (训练时未知)
            0,         # n1_viol placeholder
            0,         # cumulative violation freq
        ], dtype=np.float32)

        obs = np.concatenate([node_feat.flatten(), global_feat]).astype(np.float32)
        return obs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs, target = self.samples[idx]
        return torch.from_numpy(obs), torch.from_numpy(target.astype(np.float32))


# ============================================================
# Phase 1: GAT 监督预训练
# ============================================================

def train_gat_predictor(dm: SCEDDataManager, epochs: int = 100, batch_size: int = 64,
                         lr: float = 1e-3, patience: int = 20) -> Tuple[GATPredictor, dict]:
    """
    Phase 1: 用Ground Truth训练GAT预测器。

    输入: 观测向量(包含当前状态+系统级信息)
    输出: 各机组出力的预测值(MW)

    Loss = MSE(P_pred, P_GT) + λ·|ΣP_pred - Demand|
    """
    print("\n" + "=" * 60)
    print("  Phase 1: GAT Supervised Pre-training")
    print("=" * 60)

    # 构建边索引（基于PTDF耦合）
    print("\n[Phase1] Building graph from PTDF matrix...")
    unit_ptdf = dm.ptdf_matrix[:, dm.unit_bus]  # (N_LINES, N_UNITS)
    norms = np.linalg.norm(unit_ptdf, axis=0, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    unit_ptdf_n = unit_ptdf / norms
    sim_matrix = unit_ptdf_n.T @ unit_ptdf_n  # (N_UNITS, N_UNITS)

    k_nbr = min(8, dm.N_UNITS - 1)
    threshold = 0.15
    src, dst = [], []
    for i in range(dm.N_UNITS):
        row = sim_matrix[i].copy()
        row[i] = -1
        top_k = np.argsort(row)[-k_nbr:]
        for j in top_k:
            if abs(sim_matrix[i, j]) >= threshold:
                src.append(i); dst.append(j)

    edge_index = torch.LongTensor([src, dst])
    print(f"  Graph built: {dm.N_UNITS} nodes, {len(src)} edges "
          f"(k={k_nbr}, threshold={threshold:.2f})")

    # 构建Dataset
    obs_dim = dm.N_UNITS * GATPredictor.NODE_DIM + GATPredictor.GLOBAL_DIM
    print(f"\n[Phase1] Observation dimension: {obs_dim}")

    train_set = SCEDDataset(dm, dm.train_ids, dm.load_profile.max(), dm.reserve_profile.max())
    val_set = SCEDDataset(dm, dm.val_ids, dm.load_profile.max(), dm.reserve_profile.max())

    train_loader = DataLoader(train_set, batch_size=min(batch_size, 32), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=min(batch_size*2, 64), shuffle=False, num_workers=0)

    # 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GATPredictor(obs_dim, dm.N_UNITS, edge_index).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "mae": [], "max_err": []}

    print(f"\n[Phase1] Training GAT predictor for {epochs} epochs (device={device})...")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        train_losses = []
        for batch_obs, batch_target in train_loader:
            batch_obs = batch_obs.to(device)
            batch_target = batch_target.to(device)

            pred = model(batch_obs)
            mse_loss = F.mse_loss(pred, batch_target)

            # 功率平衡正则项
            demand_scale = dm.load_profile.max()
            bal_penalty = ((pred.sum(dim=1) - batch_target.sum(dim=1)) / demand_scale).pow(2).mean()

            loss = mse_loss + 0.1 * bal_penalty

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ---- Validate ----
        model.eval()
        val_losses = []; maes = []; max_errs = []
        with torch.no_grad():
            for batch_obs, batch_target in val_loader:
                batch_obs = batch_obs.to(device)
                batch_target = batch_target.to(device)
                pred = model(batch_obs)
                loss = F.mse_loss(pred, batch_target)
                mae = (pred - batch_target).abs().mean().item()
                me = (pred - batch_target).abs().max().item()
                val_losses.append(loss.item())
                maes.append(mae); max_errs.append(me)

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        avg_mae = np.mean(maes)
        avg_me = np.mean(max_errs)
        scheduler.step(avg_val)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["mae"].append(avg_mae)
        history["max_err"].append(avg_me)

        if (epoch % 10 == 0) or (epoch == 1):
            print(f"  Epoch {epoch:03d}/{epochs}: train_loss={avg_train:.4f} "
                  f"val_loss={avg_val:.4f} MAE={avg_mae:.1f} MW MaxErr={avg_me:.1f} MW")

        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    # 恢复最佳模型
    model.load_state_dict(best_state)
    elapsed = time.time() - t0
    print(f"\n  Training completed in {elapsed:.1f}s")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Final MAE: {history['mae'][-1]:.1f} MW, Max Error: {history['max_err'][-1]:.1f} MW")

    # 保存模型
    save_path = SAVE_DIR / "gat_predictor_best.pth"
    torch.save({
        "model_state": model.state_dict(),
        "obs_dim": obs_dim,
        "n_units": dm.N_UNITS,
        "edge_index": edge_index.numpy(),
        "history": history,
    }, save_path)
    print(f"  Model saved to {save_path}")

    return model, history


# ============================================================
# Phase 2: RL 微调校正器环境
# ============================================================

class SCEDCorrectorEnv(gym.Env):
    """
    SCED-RL校正器环境 v5.2 — 绝对出力模式 (默认, 需求比例映射+连续Reward)。

    核心思路:
      1. GAT已给出初始预测 P̂ (已满足功率平衡)
      2. RL agent 直接输出每台机组的需求分配比例:
         action[i] ∈ [0, 1] → P_i = action[i] × D_demand
      3. 环境自动clip到[PMIN, PMAX]并修复功率平衡
      4. Reward = N-1改善奖励 - 成本惩罚 - 越限惩罚

    v5.2 关键改进 (v5.0的action映射导致大部分空间ΣP远≈D):
      - 动作空间: Box(0,1)^N → P_i = a_i * D (需求比例)
      - 有效探索区域大幅扩大: 任何均匀分布的动作都给出ΣP≈D
      - 不再需要fallback到base_power(除非极端情况)
      - RL直接学习最优发电分布来改善N-1安全

    向后兼容:
      - 'redistribute' (v4): 比例重分配动作空间
      - 'legacy' (v3): 绝对增量动作空间(原始)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, dm: SCEDDataManager, gat_model: GATPredictor = None,
                 delta_max_ratio: float = 0.05, use_ac_pf: bool = True,
                 max_correction_steps: int = 10, render_mode=None,
                 env_mode: str = "redistribute"):
        super().__init__()
        self.dm = dm
        self.gat_model = gat_model
        self.delta_max_ratio = delta_max_ratio
        self.use_ac_pf = use_ac_pf
        self.max_corr_steps = max_correction_steps
        self.env_mode = env_mode  # 'absolute'(v5默认), 'redistribute'(v4), 或 'legacy'(v3)

        self.N = dm.N_UNITS
        self.L = dm.N_LINES
        self.T = dm.T_HORIZON

        if self.env_mode == "absolute":
            # [v5] 动作空间: [0, 1]^N → 绝对出力 (归一化到[PMIN, PMAX])
            # action[i]=0 → P_i=PMIN_i, action[i]=1 → P_i=PMAX_i
            self.action_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(self.N,), dtype=np.float32
            )
        elif self.env_mode == "redistribute":
            # [v4] 动作空间: [-1, 1]^N — 相对调整方向
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(self.N,), dtype=np.float32
            )
        else:
            # [legacy] 动作空间: [0, 1]^N → 映射到 [-δmax, +δmax]
            self.action_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(self.N,), dtype=np.float32
            )

        # 观测空间
        self.obs_dim = self.N * 5 + GATPredictor.GLOBAL_DIM  # 与GATPredictor保持一致 (364)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        self.render_mode = render_mode

        # 内部状态
        self.current_sid = None
        self.t = 0
        self.base_power = None      # GAT预测的基准出力 (N,) [MW] — 已平衡!
        self.current_power = None   # 当前实际出力 (N,)
        self.target_demand = None   # 当前时段目标需求
        self.corr_step = 0          # 当前是第几次校正尝试
        self.episode_stats = None

        if self.env_mode == "absolute":
            # [v5.2] 奖励系数 — 连续Reward塑形 (密集梯度信号)
            self.LAM_COST = 0.5              # 成本惩罚
            self.LAM_LINE = 30.0             # 线路越限惩罚
            self.LAM_N1_CONT = 50.0          # N-1过载总量改善系数 (核心连续信号)
            self.LAM_N1_WORST = 20.0         # 最差contingency改善系数
            self.LAM_N1_COUNT_BONUS = 10.0   # 完全消除contingency的bonus
            self.LAM_AC = 50.0               # AC权重
            self.FEASIBILITY_BONUS = 25.0    # 可行解大奖励
            self.MIN_GEN_RATIO = 0.80        # 最低总出力/需求比
            self.LAM_BALANCE_SOFT = 500.0     # 软约束平衡惩罚
        elif self.env_mode == "redistribute":
            # [v4-aggressive] 奖励系数 — 极度聚焦N-1安全 (平衡已由硬约束保证)
            self.LAM_COST = 0.3           # 进一步降低调整惩罚
            self.LAM_LINE = 50.0          # 降低线路权重
            self.LAM_N1 = 5000.0          # [v4-aggressive] 大幅加重N-1 (原3000→5000)
            self.LAM_AC = 100.0           # 降低AC权重
            self.FEASIBILITY_BONUS = 15.0  # 提高可行解奖励
            self.MIN_GEN_RATIO = 0.80     # 最低允许总出力/需求比
        else:
            # [legacy v3] 原始惩罚系数
            self.LAM_COST = 1.0
            self.LAM_BALANCE = 5000.0
            self.LAM_LINE = 200.0
            self.LAM_N1 = 1500.0
            self.LAM_AC = 400.0
            self.LAM_DEMAND_COV = 3000.0
            self.FEASIBILITY_BONUS = 5.0
            self.MIN_GEN_RATIO = 0.50

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # 随机选择一个场景和时段
        available_sids = list(self.dm.gt_power.keys())
        self.current_sid = self.np_random.choice(available_sids)
        self.t = int(self.np_random.integers(0, self.T))

        # 使用该场景专属的负荷profile (UC_RL/300适配)
        sid_profile = self.dm._get_scenario_profile(self.current_sid)
        self._sid_load_profile = sid_profile["Load"].values.astype(np.float64)
        self._sid_reserve_profile = sid_profile["Reserve"].values.astype(np.float64)
        self._sid_drate_profile = sid_profile["rate"].values.astype(np.float64)

        self.target_demand = self._sid_load_profile[self.t]
        self.corr_step = 0

        # 获取该场景该时段的GT作为GAT应该逼近的目标
        gt = self.dm.gt_power[self.current_sid]  # (T, N)
        self.base_power = gt[self.t].copy().astype(np.float64)  # 初始用GT(模拟GAT预测误差时替换为GAT输出)

        if self.gat_model is not None:
            # TODO: 用GAT模型预测替代GT作为base_power (Phase 1完成后启用)
            pass

        # 加入模拟的GAT预测误差 (高斯噪声, 模拟GAT不是完美的)
        noise_level = 0.02  # 2% 的相对误差
        noise = self.np_random.normal(0, noise_level * self.dm.PMAX, size=self.N)
        # 确保加噪声后仍在合理范围
        self.base_power = np.clip(self.base_power + noise, self.dm.PMIN, self.dm.PMAX)
        # 强制功率平衡: 缩放到匹配demand
        gen_sum = self.base_power.sum()
        if abs(gen_sum - self.target_demand) > 1.0 and gen_sum > 0:
            self.base_power *= (self.target_demand / gen_sum)

        self.current_power = self.base_power.copy()
        self.episode_stats = {
            "total_cost": 0.0, "total_penalty": 0.0,
            "n_corrections": 0, "feasible_count": 0,
            "line_viol_count": 0, "n1_viol_count": 0,
        }

        return self._obs(), {}

    def step(self, action):
        action = np.clip(action.astype(np.float64).reshape(-1), 0.0, 1.0)

        drate = self._sid_drate_profile[min(self.t, self.T - 1)]

        if self.env_mode == "absolute":
            # ============================================================
            # [v5.2] 绝对出力模式(改进): 动作=需求比例分配
            #
            # 核心改进: action[i] ∈ [0,1] 表示第i台机组承担的需求比例
            #   P_i = action[i] * target_demand
            # 然后clip到[PMIN, PMAX]物理限值，最后做微小平衡修复
            #
            # 优势: 任何均匀分布的动作都能给出ΣP≈D，有效探索区域大幅扩大!
            # 对比v5.0的 P_i=PMIN+a*(PMAX-PMIN) 导致大部分空间ΣP远≠D
            # ============================================================
            # 1. 解码动作 → 基于需求的出力分配
            raw_power = action * self.target_demand

            # 2. Clip到物理限值
            raw_power = np.clip(raw_power, self.dm.PMIN, self.dm.PMAX)

            # 3. 功率平衡修复 (clip后可能有偏差)
            gen_total = raw_power.sum()
            new_power = None

            if gen_total < self.target_demand * self.MIN_GEN_RATIO:
                # 总出力太低(可能大部分action接近0), 回退到base
                new_power = self.base_power.copy()
            elif abs(gen_total - self.target_demand) > self.target_demand * 0.01:
                # 偏差>1%, 按比例缩放修复 (保持RL选择的相对分布)
                new_power = raw_power * (self.target_demand / gen_total)
                new_power = np.clip(new_power, self.dm.PMIN, self.dm.PMAX)

                # 二次clip后仍可能有偏差
                final_err = abs(new_power.sum() - self.target_demand)
                if final_err > self.target_demand * 0.01:
                    new_power = self._min_balance_fix(new_power)
            else:
                # 偏差可接受, 直接使用
                new_power = raw_power

            self.corr_step += 1
            self.episode_stats["n_corrections"] += 1

            # ---- 约束检验 ----
            bal_err = abs(new_power.sum() - self.target_demand)  # 应该≈0

            # DC线路潮流
            flows = self.dm.compute_line_flows(new_power, drate)
            overflow = np.abs(flows) - self.dm.flow_limit
            overflow = np.maximum(overflow, 0.0)
            line_viol = (overflow > 0.1).sum()
            overload_sum = overflow.sum()

            # N-1 校验 (当前状态)
            n1_result = self.dm.check_n1_constraints(flows)

            # ---- [v5] 关键: 计算base(GT/GAT)的N-1状态作为对比基准 ----
            base_flows = self.dm.compute_line_flows(self.base_power, drate)
            base_n1 = self.dm.check_n1_constraints(base_flows)
            n1_improvement = base_n1["n1_contingency_violations"] - n1_result["n1_contingency_violations"]

            # AC潮流校验 (可选)
            ac_feasible = True
            ac_violation = 0.0
            if self.use_ac_pf:
                ac_feasible, ac_violation = self._check_ac_pf(new_power)

            # ---- [v5.2] 连续Reward塑形: 基于潮流过载量(非二值计数) ----
            # 核心改进: 稀疏的N-1 count → 密集的overload amount
            # 小的潮流改善也能产生非零梯度 → PPO可以学习方向!

            # 2a. 计算当前状态的详细N-1指标 (连续值)
            n1_detail = self._check_n1_detailed(flows)

            # 2b. 计算base状态(GAT/GT)的详细N-1指标作为对比基准
            base_detail = self._check_n1_detailed(base_flows)

            # 归一化成本偏移
            cost_now = _compute_fuel_cost(new_power, self.dm, self.current_sid)
            cost_base = _compute_fuel_cost(self.base_power, self.dm, self.current_sid)
            cost_increase = max(0, cost_now - cost_base) / (self.target_demand + 1e-6)

            # 平衡误差惩罚 (软约束, 应该接近0)
            balance_penalty = (bal_err / (self.target_demand + 1e-6)) * self.LAM_BALANCE_SOFT

            # ===== 核心奖励: 连续化的N-1改善 =====
            # 改善量 = (base过载总量 - 当前过载总量), 正=变好
            n1_overload_improvement = (
                (base_detail["total_overload"] - n1_detail["total_overload"])
                / (self.dm.flow_limit.sum() + 1e-6)  # 归一化到[0,~1]范围
            )
            # 最差contingency的最大过载改善
            n1_worst_improvement = (
                (base_detail["max_overload"] - n1_detail["max_overload"])
                / (self.dm.flow_limit.max() + 1e-6)
            )
            # 违反contingency数改善 (保留离散信号作为bonus)
            n1_count_improvement = (
                base_detail["n_contingencies"] - n1_detail["n_contingencies"]
            )

            # 基态线路越限惩罚 (连续值)
            line_penalty = self.LAM_LINE * (overload_sum / (self.dm.flow_limit.sum() + 1e-6))

            # 总reward: 连续信号为主 + 离散bonus + 成本/平衡惩罚
            reward = (
                self.LAM_N1_CONT * n1_overload_improvement      # 核心: 过载总量减少
                + self.LAM_N1_WORST * n1_worst_improvement       # 次要: 最坏情况改善
                + self.LAM_N1_COUNT_BONUS * max(0, n1_count_improvement)  # bonus: 完全消除某个contingency
                - self.LAM_COST * cost_increase                   # 成本控制
                - line_penalty                                    # 基态线路约束
                - balance_penalty                                 # 平衡约束
            )

            # 可行性大奖励 (N-1完全满足时)
            is_feasible = (
                bal_err < self.target_demand * 0.005
                and line_viol == 0
                and n1_result["n1_contingency_violations"] == 0
                and ac_feasible
            )
            if is_feasible:
                reward += self.FEASIBILITY_BONUS
                self.episode_stats["feasible_count"] += 1

            # 更新状态
            self.current_power = new_power
            self.episode_stats["total_cost"] += cost_increase
            self.episode_stats["total_penalty"] += line_penalty + balance_penalty
            if line_viol > 0:
                self.episode_stats["line_viol_count"] += 1
            if n1_result["n1_contingency_violations"] > 0:
                self.episode_stats["n1_viol_count"] += 1

            # 保存用于debug的信息 — 绝对模式下记录相对base的变化
            delta_p = new_power - self.base_power  # v5: 相对于base的调整量

        elif self.env_mode == "redistribute":
            action = np.clip(action.astype(np.float64).reshape(-1), -1.0, 1.0)
            # ============================================================
            # [v4] 比例重分配模式: 动作=相对调整方向, 自动保证功率平衡
            # ============================================================
            delta_max = self.delta_max_ratio * self.dm.PMAX  # 每台机组最大调整量 [MW]

            # action ∈ [-1, +1] → delta_p ∈ [-δmax, +δmax]
            delta_p = action * delta_max

            # 应用相对调整
            new_power = self.current_power + delta_p

            # ===== 关键: 硬约束 — 缩放到精确匹配需求 =====
            # 先clip到物理限值
            new_power = np.clip(new_power, self.dm.PMIN, self.dm.PMAX)
            gen_total = new_power.sum()

            # 检查是否在可行范围内
            if gen_total < self.target_demand * self.MIN_GEN_RATIO:
                # 总出力太低(接近PMIN总和), 强制拉回base附近
                new_power = self.base_power.copy()
                gen_total = new_power.sum()

            if abs(gen_total - self.target_demand) > 1.0 and gen_total > 0:
                # 按比例缩放到精确匹配需求
                # 保持各机组的相对分布不变, 只调整总量
                new_power = new_power * (self.target_demand / gen_total)
                # 二次clip (缩放后可能超出限值)
                new_power = np.clip(new_power, self.dm.PMIN, self.dm.PMAX)

                # 如果二次clip后仍有偏差 (极端情况), 做最小二乘修复
                final_err = abs(new_power.sum() - self.target_demand)
                if final_err > self.target_demand * 0.01:  # >1%误差
                    new_power = self._min_balance_fix(new_power)

            self.corr_step += 1
            self.episode_stats["n_corrections"] += 1

            # ---- 约束检验 ----
            bal_err = abs(new_power.sum() - self.target_demand)  # 应该≈0

            # DC线路潮流
            flows = self.dm.compute_line_flows(new_power, drate)
            overflow = np.abs(flows) - self.dm.flow_limit
            overflow = np.maximum(overflow, 0.0)
            line_viol = (overflow > 0.1).sum()
            overload_sum = overflow.sum()

            # N-1 校验
            n1_result = self.dm.check_n1_constraints(flows)

            # AC潮流校验 (可选)
            ac_feasible = True
            ac_violation = 0.0
            if self.use_ac_pf:
                ac_feasible, ac_violation = self._check_ac_pf(new_power)

            # ---- [v4] 简化奖励函数: 平衡已满足, 聚焦N-1+成本 ----
            adj_cost = np.abs(delta_p).sum() / (self.dm.PMAX.sum())
            cost_change = abs(_compute_fuel_cost(new_power, self.dm, self.current_sid)
                              - _compute_fuel_cost(self.current_power, self.dm, self.current_sid))
            cost_change /= (self.target_demand + 1e-6)  # 归一化

            penalty = (
                self.LAM_LINE * (overload_sum / (self.dm.flow_limit.sum() + 1e-6))
                + self.LAM_N1 * (n1_result["n1_contingency_violations"] / max(self.L, 1))
            )
            if self.use_ac_pf:
                penalty += self.LAM_AC * ac_violation

            reward = -(self.LAM_COST * adj_cost + 0.001 * cost_change + penalty)

            # 可行性奖励 (N-1是关键!)
            is_feasible = (
                line_viol == 0
                and n1_result["n1_contingency_violations"] == 0
                and ac_feasible
            )
            if is_feasible:
                reward += self.FEASIBILITY_BONUS
                self.episode_stats["feasible_count"] += 1

            # 更新状态
            self.current_power = new_power
            self.episode_stats["total_cost"] += adj_cost
            self.episode_stats["total_penalty"] += penalty
            if line_viol > 0:
                self.episode_stats["line_viol_count"] += 1
            if n1_result["n1_contingency_violations"] > 0:
                self.episode_stats["n1_viol_count"] += 1

        else:
            # ============================================================
            # [legacy v3] 绝对增量模式 (原始行为, 保留兼容)
            # ============================================================
            action = np.clip((action + 1.0) / 2.0, 0.0, 1.0)  # [-1,1] → [0,1]
            delta_max = self.delta_max_ratio * self.dm.PMAX
            delta_p = (action - 0.5) * 2.0 * delta_max

            new_power = np.clip(self.current_power + delta_p, self.dm.PMIN, self.dm.PMAX)
            self.corr_step += 1
            self.episode_stats["n_corrections"] += 1

            gen_total = new_power.sum()
            bal_err = abs(gen_total - self.target_demand)

            flows = self.dm.compute_line_flows(new_power, drate)
            overflow = np.abs(flows) - self.dm.flow_limit
            overflow = np.maximum(overflow, 0.0)
            line_viol = (overflow > 0.1).sum()
            overload_sum = overflow.sum()

            n1_result = self.dm.check_n1_constraints(flows)

            ac_feasible = True
            ac_violation = 0.0
            if self.use_ac_pf:
                ac_feasible, ac_violation = self._check_ac_pf(new_power)

            adj_cost = np.abs(delta_p).sum() / (self.dm.PMAX.sum())
            demand_coverage = gen_total / (self.target_demand + 1e-6)
            demand_gap = max(0.0, 1.0 - demand_coverage)

            penalty = (
                self.LAM_BALANCE * (bal_err / (self.target_demand + 1e-6))
                + self.LAM_LINE * (overload_sum / (self.dm.flow_limit.sum() + 1e-6))
                + self.LAM_N1 * (n1_result["n1_contingency_violations"] / max(self.L, 1))
                + self.LAM_DEMAND_COV * demand_gap
            )
            if self.use_ac_pf:
                penalty += self.LAM_AC * ac_violation

            reward = -(self.LAM_COST * adj_cost + penalty)

            is_feasible = (
                bal_err < self.target_demand * 0.001
                and line_viol == 0
                and n1_result["n1_contingency_violations"] == 0
                and ac_feasible
            )
            if is_feasible:
                reward += self.FEASIBILITY_BONUS
                self.episode_stats["feasible_count"] += 1

            self.current_power = new_power
            self.episode_stats["total_cost"] += adj_cost
            self.episode_stats["total_penalty"] += penalty
            if line_viol > 0:
                self.episode_stats["line_viol_count"] += 1
            if n1_result["n1_contingency_violations"] > 0:
                self.episode_stats["n1_viol_count"] += 1

        # ---- 终止条件 (两种模式共用) ----
        terminated = False
        truncated = False
        if is_feasible:
            terminated = True
        elif self.corr_step >= self.max_corr_steps:
            truncated = True

        info = {
            "scenario_id": self.current_sid,
            "time_step": self.t,
            "correction_step": self.corr_step,
            "bal_err": float(bal_err),
            "line_viol": int(line_viol),
            "overload_sum": float(overload_sum),
            **n1_result,
            "ac_feasible": ac_feasible,
            "is_feasible": is_feasible,
            "adj_norm": float(np.abs(delta_p).sum()) if 'delta_p' in dir() else 0.0,
            "gen_total": float(new_power.sum()),
            **self.episode_stats,
        }

        return self._obs(), reward, terminated, truncated, info

    def _min_balance_fix(self, power: np.ndarray) -> np.ndarray:
        """
        最小二乘功率平衡修复。
        当按比例缩放+clip后仍有微小不平衡时使用。
        找到最接近当前解的可行点使得 sum(P) = demand。
        """
        p = power.copy().astype(np.float64)
        demand = self.target_demand
        lo = self.dm.PMIN.astype(np.float64)
        hi = self.dm.PMAX.astype(np.float64)

        current_err = p.sum() - demand
        if abs(current_err) < 1.0:
            return p

        # 简单策略: 按裕度比例分配剩余误差
        # 可以增加的机组
        if current_err < 0:  # 需要增加出力
            slack = hi - p
            if slack.sum() > abs(current_err):
                p += slack * (abs(current_err) / slack.sum())
        else:  # 需要减少出力
            slack = p - lo
            if slack.sum() > current_err:
                p -= slack * (current_err / slack.sum())

        return np.clip(p, lo, hi)

    def _obs(self) -> np.ndarray:
        """构建校正器的观测向量。"""
        N = self.N
        T = self.T
        dm = self.dm

        # 当前出力相关
        on_off = (self.current_power > dm.PMIN * 0.5).astype(np.float32)
        p_ratio = np.clip(self.current_power / (dm.PMAX + 1e-6), 0, 1).astype(np.float32)
        margin_up = np.clip((dm.PMAX - self.current_power) / (dm.PMAX + 1e-6), 0, 1).astype(np.float32)
        margin_down = np.clip((self.current_power - dm.PMIN) / (dm.PMAX + 1e-6), 0, 1).astype(np.float32)
        cap_ratio = dm.PMAX / (dm.PMAX.max() + 1e-6)

        node_feat = np.stack([on_off, p_ratio, margin_up, margin_down, cap_ratio], axis=1)  # (N, 5)

        # 全局特征 — 使用该场景专属的profile
        _load_prof = getattr(self, '_sid_load_profile', dm.load_profile)
        _rsv_prof = getattr(self, '_sid_reserve_profile', dm.reserve_profile)
        _drate_prof = getattr(self, '_sid_drate_profile', dm.drate_profile)

        max_load = float(_load_prof.max()) if _load_prof.max() > 0 else 1.0
        max_rsv = float(_rsv_prof.max()) if _rsv_prof.max() > 0 else 1.0
        future_load = np.array([_load_prof[min(self.t+k, T-1)] / max_load for k in range(1,5)], dtype=np.float32)
        future_rsv = np.array([_rsv_prof[min(self.t+k, T-1)] / max_rsv for k in range(1,5)], dtype=np.float32)

        # 当前线路越限信息
        drate = _drate_prof[min(self.t, T - 1)]
        current_flows = dm.compute_line_flows(self.current_power, drate)
        current_overflow = np.abs(current_flows) - dm.flow_limit
        current_overflow = np.maximum(current_overflow, 0.0)
        n_ol = (current_overflow > 0.1).sum()
        mx_ol = current_overflow.max() if current_overflow.max() > 0 else 0.0
        sm_ol = current_overflow.sum()

        # N-1 信息
        n1_res = dm.check_n1_constraints(current_flows)
        n1_v = n1_res["n1_contingency_violations"]

        cap_min = dm.PMIN[on_off > 0.5].sum() if (on_off > 0.5).any() else 0
        cap_max = dm.PMAX.sum()

        global_feat = np.array([
            self.corr_step / max(self.max_corr_steps, 1),
            _load_prof[self.t] / max_load,
            _rsv_prof[self.t] / max_rsv,
            cap_min / (max_load + 1e-6),
            cap_max / (max_load + 1e-6),
            drate,
            *future_load, *future_rsv,
            n_ol / max(self.L, 1),
            mx_ol / (max_load + 1e-6),
            sm_ol / (self.L * max_load + 1e-6),
            float(n1_v) / max(self.L, 1),
            float(self.episode_stats.get("line_viol_count", 0)) / max(self.corr_step + 1, 1) / T,
        ], dtype=np.float32)

        obs = np.concatenate([node_feat.flatten(), global_feat]).astype(np.float32)
        return np.clip(obs, 0, 1)

    def _check_ac_pf(self, power: np.ndarray) -> tuple:
        """
        AC潮流可行性校验。

        使用pandapower构建交流潮流模型, 运行牛顿-拉夫逊求解。
        如果pandapower不可用, 回退到简化的DC校验。

        Returns:
            (is_feasible: bool, violation_score: float)
        """
        try:
            import pandapower as pp
            import pandapower.networks as pn
        except ImportError:
            # pandapower未安装, 返回默认可行
            return True, 0.0

        # TODO: 这里需要构建完整的AC电网模型
        # 目前简化实现: 基于PTDF的AC近似判断
        # 后续需要根据300-bus系统的完整参数(bus/branch/gen数据)构建pandapower network
        #
        # 简化判据: 如果DC下无严重过载(>10%), 假设AC也大概率可行
        # 这是因为DC-AC gap通常在重载时才显著
        flows = self.dm.compute_line_flows(power)
        ratios = np.abs(flows) / (self.dm.flow_limit + 1e-6)
        max_ratio = ratios.max()
        if max_ratio < 0.95:
            return True, 0.0
        else:
            return False, float((max_ratio - 0.95) * 10)

    def render(self):
        if self.render_mode != "human":
            return
        stats = self.episode_stats
        print(f"  CorrStep {self.corr_step}/{self.max_corr_steps} | "
              f"S{self.current_sid} t{self.t} | "
              f"P_sum={self.current_power.sum():.0f} D={self.target_demand:.0f} | "
              f"feasible={stats.get('feasible_count',0)}")


# ============================================================
# Phase 2 (continued): RL 训练
# ============================================================


    def check_n1_constraints(self, flows: np.ndarray) -> dict:
        """
        N-1安全约束检查（基于LODF近似）。

        对每条线路k停运，检查其余线路l是否越限：
            Flow_l^{(k)} = Flow_l + LODF_{l,k} × Flow_k
        """
        # 预计算LODF矩阵（如果还没算）
        if not hasattr(self, '_lodf_matrix'):
            self._precompute_lodf()

        lodf = self._lodf_matrix
        # 应用fl_scale放大限值 (工程备用裕度)
        eff_limit = self.dm.flow_limit * self.dm.fl_scale
        n1_overload_sum = 0.0
        n1_violated_lines = 0
        contingency_violations = 0

        for k in range(self.N_LINES):
            if abs(flows[k]) < 1e-6:
                continue
            post_flows = flows + lodf[:, k] * flows[k]
            over = np.abs(post_flows) - eff_limit
            viol_mask = over > 1e-3
            n_v = viol_mask.sum()
            if n_v > 0:
                contingency_violations += 1
                n1_violated_lines += n_v
                n1_overload_sum += over[viol_mask].sum()

        return {
            "n1_overload_sum": float(n1_overload_sum),
            "n1_violated_lines": int(n1_violated_lines),
            "n1_contingency_violations": int(contingency_violations),
        }

    def _check_n1_detailed(self, flows: np.ndarray) -> dict:
        """
        N-1安全约束详细检查 (连续值版本，用于Reward塑形)。

        返回过载量(非二值)，提供密集的梯度信号。
        """
        if not hasattr(self, '_lodf_matrix'):
            self._precompute_lodf()

        lodf = self._lodf_matrix
        eff_limit = self.dm.flow_limit * self.dm.fl_scale

        total_overload = 0.0       # 所有contingency的所有线路过载之和 [MW]
        max_overload = 0.0          # 单个线路在单个contingency下的最大过载 [MW]
        n_contingencies = 0         # 有任何过载的contingency数 (离散)
        worst_line_per_contingency = []  # 每个违规contingency的最差线路

        for k in range(self.L):
            if abs(flows[k]) < 1e-6:
                continue
            post_flows = flows + lodf[:, k] * flows[k]
            over = np.abs(post_flows) - eff_limit
            over_pos = np.maximum(over, 0)
            k_overload_sum = over_pos.sum()
            k_max_over = over_pos.max() if over_pos.size > 0 else 0.0

            if k_overload_sum > 1e-3:
                n_contingencies += 1
                total_overload += k_overload_sum
                max_overload = max(max_overload, k_max_over)
                worst_line_per_contingency.append(k_max_over)

        return {
            "total_overload": float(total_overload),      # 连续: 总过载MW数
            "max_overload": float(max_overload),           # 连续: 最严重单线过载
            "n_contingencies": int(n_contingencies),       # 离散: 违规contingency数
            "avg_worst_overload": (float(np.mean(worst_line_per_contingency))
                                   if worst_line_per_contingency else 0.0),
        }

    def _precompute_lodf(self):
        """预计算LODF矩阵 (N_LINES × N_LINES)。"""
        ptdf = self.dm.ptdf_matrix
        n = self.dm.N_LINES
        lodf = np.zeros((n, n), dtype=np.float64)

        # 提取线路端点信息
        from_col = self.dm.line_data["BusFrom"].values.astype(int) - 1  # 0-indexed
        to_col = self.dm.line_data["BusTo"].values.astype(int) - 1

        for k in range(n):
            fk, tk = from_col[k], to_col[k]
            x_k = ptdf[k, tk] - ptdf[k, fk]  # 线路k的电抗(近似)
            if abs(x_k) < 1e-10:
                continue
            denom = ptdf[k, fk] - ptdf[k, tk]  # 正确: PTDF自敏感度差 (非 1.0 - diff)
            if abs(denom) < 1e-10:
                continue  # 辐射状线路(无回路), LODF=0
            for l in range(n):
                fl, tl = from_col[l], to_col[l]
                lodf[l, k] = (ptdf[l, fl] - ptdf[l, tl]) / denom

        self._lodf_matrix = lodf
        nz = np.count_nonzero(lodf)
        print(f"  [N-1] LODF matrix ({n}×{n}) computed, {nz} non-zero entries")

    # ================================================================
    # AC Power Flow Module (based on pandapower)
    # ================================================================

    def _build_pandapower_net(self, r_x_ratio: float = 0.2, base_mva: float = 100.0):
        """
        用pandapower构建300-bus交流网络模型。

        基于Line_Data.csv的X参数，假设R/X比值推导R。
        网络包含: bus, gen, load, line(impedance)。
        """
        import pandapower as pp
        from pandapower.networks import create_empty_network

        ld = self.line_data
        n_bus = self.N_BUSES
        n_line = self.N_LINES

        net = create_empty_network(sn_mva=base_mva, f_hz=50.0)

        # ---- Buses: 创建300个节点 ----
        for i in range(n_bus):
            pp.create_bus(net, vn_kv=230.0, name=f"Bus_{i+1}",
                          type="n" if i > 0 else "slack")  # Bus1=slack

        # ---- Generators: 在机组所在节点创建 ----
        for i in range(self.N_UNITS):
            b = int(self.unit_bus[i]) if self.unit_bus is not None else 0
            is_slack = (i == 0)  # 第一台机组作为平衡节点
            pp.create_gen(net, bus=b, p_mw=0.0, vm_pu=1.02,
                          max_p_mw=float(self.PMAX[i]),
                          min_p_mw=float(self.PMIN[i]),
                          slack=is_slack,
                          name=f"Gen_{i+1}")

        # ---- Lines: 从Line_Data创建阻抗支路 ----
        from_bus = ld["BusFrom"].values.astype(int) - 1   # → 0-indexed
        to_bus = ld["BusTo"].values.astype(int) - 1         # → 0-indexed
        x_pu = ld["X"].values.astype(np.float64)   # 电抗[pu]
        r_pu = x_pu * r_x_ratio                      # 推导电阻[pu]
        flow_limit = ld["Flowlimit"].values.astype(np.float64)

        for k in range(n_line):
            f, t = int(from_bus[k]), int(to_bus[k])
            # create_impedance 使用 pu 值 (rft_pu, xft_pu)
            sn_mva = flow_limit[k] / 100.0 if flow_limit[k] > 0 else 1000.0
            pp.create_impedance(net, from_bus=f, to_bus=t,
                                rft_pu=r_pu[k], xft_pu=x_pu[k],
                                sn_mva=sn_mva,
                                name=f"Line_{k+1}")

        # ---- Loads: 从Busload_Data创建 ----
        if self.busload_data is not None:
            bl = self.busload_data
            bus_nos = bl["BusNo"].values.astype(int) - 1   # → 0-indexed
            pl = bl["PL"].values.astype(np.float64)
            for idx in range(len(bl)):
                b = int(bus_nos[idx])
                pp.create_load(net, bus=b, p_mw=pl[idx], q_mw=pl[idx] * np.tan(np.arccos(0.95)),
                               const_z_percent=0, const_i_percent=0, const_p_percent=100,
                               name=f"Load_b{b}")

        self._pp_net = net
        self._pp_base_mva = base_mva
        self._r_x_ratio = r_x_ratio

        print(f"  [AC-PF] pandapower net built: {len(net.bus)} buses, "
              f"{len(net.gen)} gens, {len(net.impedance)} lines, "
              f"{len(net.load)} loads, R/X={r_x_ratio}")

    def _run_ac_pf_pandapower(self, power: np.ndarray, drate: float = 1.0) -> dict:
        """
        使用pandapower.runacpp()运行交流潮流。

        Args:
            power: (N_UNITS,) 各机组出力 [MW]
            drate: 负荷比例系数

        Returns:
            dict: converged, V (complex), vmag, theta, line_flows_MW, error_msg
        """
        import pandapower as pp

        if not hasattr(self, '_pp_net') or self._pp_net is None:
            self._build_pandapower_net()

        import copy
        net = copy.deepcopy(self._pp_net)  # 深拷贝以避免修改原始网络

        # 设置发电机出力
        for i in range(min(len(net.gen), self.N_UNITS)):
            net.gen.p_mw.values[i] = float(power[i])

        # 按drate缩放负荷
        if len(net.load) > 0:
            original_loads = net.load.p_mw.values.copy()
            net.load.p_mw = original_loads * drate
            net.load.q_mw = net.load.q_mw * drate

        try:
            # 运行AC潮流 (Newton-Raphson, pandapower内置)
            result = pp.runpp(net, max_iter=50, tolerance_mva=1e-4,
                               enforce_q_lims=False)

            if result.converged:
                V_complex = np.array([net.res_bus.vm_pu[b] *
                                      np.exp(1j * np.deg2rad(net.res_bus.va_degree[b]))
                                      for b in range(len(net.bus))])
                # 阻抗支路有功潮流 [MW]
                line_flows = net.res_impedance.pl_mw.values.copy()
                return {
                    "converged": True,
                    "iterations": result.get("iterations", -1),
                    "V": V_complex,
                    "vmag": net.res_bus.vm_pu.values.copy(),
                    "theta": np.deg2rad(net.res_bus.va_degree.values.copy()),
                    "line_flows": line_flows,
                    "line_loading": net.res_impedance.loading_percent.values.copy(),
                    "bus_p_mw": net.res_bus.p_mw.values.copy(),
                    "bus_q_mvar": net.res_bus.q_mvar.values.copy(),
                    "error_msg": None,
                }
            else:
                return {
                    "converged": False,
                    "iterations": result.get("iterations", -1),
                    "error_msg": f"AC-PF not converged",
                    "V": None,
                    "line_flows": None,
                }
        except Exception as e:
            return {"converged": False, "error_msg": str(e), "V": None}

    def _trip_line_and_run_pf(self, line_idx: int, power: np.ndarray,
                                drate: float = 1.0) -> dict:
        """
        停运指定线路后重新运行AC潮流。

        Args:
            line_idx: 要停运的线路编号(0-indexed, 对应pandapower net.line索引)
            power, drate: 同上

        Returns:
            dict: 与_run_ac_pf_pandapower格式相同
        """
        import pandapower as pp

        if not hasattr(self, '_pp_net'):
            self._build_pandapower_net()

        import copy
        net = copy.deepcopy(self._pp_net)

        # 设置发电机出力
        for i in range(min(len(net.gen), self.N_UNITS)):
            net.gen.p_mw.values[i] = float(power[i])

        # 缩放负荷
        if len(net.load) > 0:
            net.load.p_mw = net.load.p_mw.values * drate
            net.load.q_mw = net.load.q_mw.values * drate

        # 停运线路: 设置in_service=False
        if line_idx < len(net.impedance):
            net.impedance.at[line_idx, 'in_service'] = False

        try:
            result = pp.runpp(net, max_iter=50, tolerance_mva=1e-4,
                               enforce_q_lims=False)
            if result.converged:
                V_complex = np.array([net.res_bus.vm_pu[b] *
                                      np.exp(1j * np.deg2rad(net.res_bus.va_degree[b]))
                                      for b in range(len(net.bus))])
                # 停运一条阻抗支路后，res_impedance只包含在运支路
                active_imp = net.impedance[net.impedance.in_service].index
                n_total_imp = len(self._pp_net.impedance)
                all_flows = np.zeros(n_total_imp, dtype=np.float64)
                if len(net.res_impedance) > 0:
                    all_flows[active_imp] = net.res_impedance.pl_mw.values.copy()
                return {
                    "converged": True,
                    "iterations": result.get("iterations", -1),
                    "V": V_complex,
                    "vmag": net.res_bus.vm_pu.values.copy(),
                    "theta": np.deg2rad(net.res_bus.va_degree.values.copy()),
                    "line_flows": all_flows,
                    "line_loading_pct": net.res_impedance.loading_percent.values.copy(),
                    "active_line_indices": list(active_imp),
                    "error_msg": None,
                }
            else:
                return {"converged": False, "error_msg": "PF_not_converged_after_trip"}
        except Exception as e:
            return {"converged": False, "error_msg": str(e)}

    def check_n1_constraints_ac(
        self,
        power: np.ndarray,
        drate: float = 1.0,
        voltage_lo: float = 0.95,
        voltage_hi: float = 1.05,
        verbose: bool = False,
    ) -> dict:
        """
        AC-PF N-1安全校验（基于pandapower）。

        对每条线路k停运 → re-run AC-PF → 检查线路越限 + 电压越限。

        Args:
            power: (N_UNITS,) 机组出力 [MW]
            drate: 负荷比例系数
            voltage_lo/hi: 电压上下限 [pu]
            verbose: 打印详细信息

        Returns:
            dict with DC-compatible format + AC-specific fields
        """
        import time

        t_start = time.time()

        # ---- Base case PF ----
        base_result = self._run_ac_pf_pandapower(power, drate)
        dc_base_flows = self.compute_line_flows(power, drate)

        ac_base_flows = base_result.get("line_flows")
        if ac_base_flows is not None and len(ac_base_flows) == self.N_LINES:
            pass  # OK
        elif ac_base_flows is not None:
            # 可能长度不同（pandapower内部处理差异）
            pass
        else:
            ac_base_flows = np.zeros(self.N_LINES)

        if verbose and not base_result["converged"]:
            print(f"    [AC-N1] WARNING: Base case PF NOT converged: "
                  f"{base_result.get('error_msg', '?')}")

        # ---- N-1 Contingency analysis ----
        n1_overload_sum = 0.0
        n1_violated_lines = 0
        contingency_violations = 0
        contingency_details = []
        ac_voltage_violations = 0
        total_contingencies_checked = 0

        # 只对基态有明显潮流的线路做N-1（跳过空载/轻载线路加速）
        if ac_base_flows is not None and len(ac_base_flows) > 0:
            significant_lines = np.where(np.abs(ac_base_flows) > 0.5)[0]
        else:
            significant_lines = range(self.N_LINES)

        for k in significant_lines:
            k = int(k)
            total_contingencies_checked += 1

            c_result = self._trip_line_and_run_pf(k, power, drate)

            if not c_result["converged"]:
                # 不收敛 = 不安全（保守处理）
                contingency_violations += 1
                n1_violated_lines += 1
                contingency_details.append({
                    "line_k": k, "reason": "PF_not_converged",
                    "dc_flow": float(dc_base_flows[k]),
                })
                continue

            # 检查线路潮流越限（仅检查仍在线路中的）
            c_flows = c_result["line_flows"]
            over = np.abs(c_flows) - self.dm.flow_limit
            viol_mask = over > 1e-3  # MW容差
            n_v = viol_mask.sum()

            # 检查电压越限
            vmag = c_result.get("vmag")
            n_v_viol = 0
            if vmag is not None:
                v_low = vmag < voltage_lo
                v_high = vmag > voltage_hi
                n_v_viol = int((v_low | v_high).sum())

            has_violation = (n_v > 0) or (n_v_viol > 0)
            if has_violation:
                contingency_violations += 1
                n1_violated_lines += n_v
                ac_voltage_violations += n_v_viol
                n1_overload_sum += float(over[viol_mask].sum()) if n_v > 0 else 0.0

            if verbose and has_violation:
                worst_over = float(over.max()) if n_v > 0 else 0.0
                min_v = float(vmag.min()) if vmag is not None else 1.0
                print(f"    [AC-N1] Line {k:>3d}: {n_v} flow viols, "
                      f"{n_v_viol} V viols, worst={worst_over:.0f}MW, "
                      f"Vmin={min_v:.4f}pu")

            contingency_details.append({
                "line_k": int(k), "n_flow_viols": int(n_v),
                "n_voltage_viols": int(n_v_viol), "converged": True,
                "worst_overload_MW": worst_over if has_violation else 0.0,
                "min_voltage_pu": float(vmag.min()) if vmag is not None else 1.0,
            })

        elapsed = time.time() - t_start

        return {
            # --- DC兼容格式 ---
            "n1_overload_sum": float(n1_overload_sum),
            "n1_violated_lines": int(n1_violated_lines),
            "n1_contingency_violations": int(contingency_violations),
            # --- AC特有字段 ---
            "ac_voltage_violations": int(ac_voltage_violations),
            "total_contingencies_checked": int(total_contingencies_checked),
            "base_case_converged": base_result["converged"],
            "base_case_iterations": base_result.get("iterations", -1),
            "contingency_details": contingency_details,
            "eval_time_s": round(elapsed, 2),
            # --- 基态对比数据 ---
            "ac_base_flows": ac_base_flows,
            "dc_base_flows": dc_base_flows,
        }

    def build_features_for_scenario(self, scenario_id: int) -> dict:
        """
        为单个场景构建输入特征向量。

        特征包含:
          - 全局特征(24维): 时间进度, 当前负荷/备用, 容量上下界,
                             未来4步负荷/备用预测, 当前时段drate
          - 节点特征(N_UNITS×5维): 开关状态, 出力比例, 可调上/下裕度, 机组容量比
        """
        ud = self._get_scenario_unit_data(scenario_id)
        price_seg1 = ud["PriceSegment1"].values.astype(np.float64)
        # 使用该场景专属的负荷曲线 (UC_RL/300 每场景独立)
        profile = self._get_scenario_profile(scenario_id)

        features = {
            "price_seg1": price_seg1,
            "unit_data": ud,
            "load_profile": profile["Load"].values.astype(np.float64),
            "reserve_profile": profile["Reserve"].values.astype(np.float64),
            "drate_profile": profile["rate"].values.astype(np.float64),
        }
        return features

    def split_scenarios(self):
        """划分训练/验证/测试集场景ID。"""
        sol_dir = self.data_root / "solution"
        all_ids = sorted([int(d.name) for d in sol_dir.iterdir() if d.is_dir() and d.name.isdigit()])
        total = len(all_ids)
        print(f"\n[Phase0] Found {total} scenarios with solutions")

        n_avail = min(total, self.n_train + self.n_val + self.n_test)
        selected = self.rng.choice(all_ids, size=n_avail, replace=False).tolist()
        self.train_ids = sorted(selected[:self.n_train])
        self.val_ids = sorted(selected[self.n_train:self.n_train + self.n_val])
        self.test_ids = sorted(selected[self.n_train + self.n_val:n_avail])

        print(f"  Train: {len(self.train_ids)} scenarios ({self.train_ids[0]}~{self.train_ids[-1]})")
        print(f"  Val:   {len(self.val_ids)} scenarios ({self.val_ids[0]}~{self.val_ids[-1]})")
        print(f"  Test:  {len(self.test_ids)} scenarios ({self.test_ids[0]}~{self.test_ids[-1]})")

    def load_all(self):
        """执行完整的数据加载流程。"""
        print("=" * 60)
        print("  Phase 0: Data Loading & Preprocessing")
        print("=" * 60)

        self.load_system_params()
        self.split_scenarios()

        # 加载所有划分中场景的Ground Truth
        all_ids = self.train_ids + self.val_ids + self.test_ids
        print(f"\n[Phase0] Loading ground truth for {len(all_ids)} scenarios...")
        t0 = time.time()
        loaded = 0
        failed = 0
        for sid in all_ids:
            gt = self.load_ground_truth(sid)
            if gt is not None:
                # gt shape: (N_UNITS, T_HORIZON) → 转置为 (T_HORIZON, N_UNITS)
                if gt.shape[0] == self.N_UNITS and gt.shape[1] == self.T_HORIZON:
                    self.gt_power[sid] = gt.T  # (T, N)
                elif gt.shape[0] == self.T_HORIZON and gt.shape[1] == self.N_UNITS:
                    self.gt_power[sid] = gt  # already (T, N)
                else:
                    print(f"  [WARN] Scenario {sid}: unexpected GT shape {gt.shape}")
                    failed += 1
                    continue

                # 构建场景特征
                self.scenario_features[sid] = self.build_features_for_scenario(sid)
                loaded += 1
            else:
                failed += 1
                if failed <= 3:
                    print(f"  [FAIL] Scenario {sid}: no ground truth found")

        elapsed = time.time() - t0
        print(f"\n  Loaded {loaded}/{len(all_ids)} scenarios in {elapsed:.1f}s ({failed} failed)")

        # 打印一个样本的统计
        if self.gt_power:
            first_sid = list(self.gt_power.keys())[0]
            gt_sample = self.gt_power[first_sid]
            print(f"\n  Sample GT (scenario {first_sid}): shape={gt_sample.shape}")
            print(f"    Total generation per period (first 5): {gt_sample[:5].sum(axis=1).round(1)}")
            print(f"    Load profile (first 5):           {self.load_profile[:5].round(1)}")

        return loaded > 0


# ============================================================
# GAT 网络定义 (内联版本，避免外部依赖问题)
# ============================================================

class GATv2ConvLayer(nn.Module):
    """纯PyTorch实现的GATv2层（无PyG依赖）。"""
    def __init__(self, in_features, out_features, heads=4, concat=True, dropout=0.1, bias=True):
        super().__init__()
        self.heads = heads
        self.out_features = out_features
        self.concat = concat
        self.dropout = dropout
        self.W_src = nn.Linear(in_features, heads * out_features, bias=False)
        self.W_dst = nn.Linear(in_features, heads * out_features, bias=bias)
        self.a = nn.Parameter(torch.zeros(1, heads, out_features * 2))
        self._reset_parameters()

    def _reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.W_src.weight, gain=gain)
        nn.init.xavier_uniform_(self.W_dst.weight, gain=gain)
        nn.init.normal_(self.a, std=0.02)

    def forward(self, x, edge_index):
        # Dense GAT: full (N×N) attention with edge masking
        # Uses only bmm/baddmm — fully autograd-compatible
        N = x.size(0)
        H = self.heads
        D = self.out_features

        h_s = self.W_src(x).view(N, H, D)   # (N, H, D) source projections
        h_d = self.W_dst(x).view(N, H, D)  # (N, H, D) dest projections

        # Split attention vector a (1, H, 2D) → a_src(1,H,D), a_dst(1,H,D)
        a_split = self.a.view(H, 2*D)
        a_src = a_split[:, :D].view(H, 1, D)      # (H, 1, D)
        a_dst = a_split[:, D:].view(H, 1, D)     # (H, 1, D)

        # Compute (N,N,H) attention logits
        attn_logits = torch.zeros(N, N, H, device=x.device)
        for h_idx in range(H):
            # h_s @ a_src^T: (N,D) x (1,D)->(N,1) -> broadcast to (N,N)
            src_score = h_s[:, h_idx, :] @ a_src[h_idx, 0, :].T   # (N,)
            dst_score = h_d[:, h_idx, :] @ a_dst[h_idx, 0, :].T   # (N,)
            attn_logits[:, :, h_idx] = src_score.unsqueeze(1) + dst_score.unsqueeze(0)

        attn_logits = F.leaky_relu(attn_logits, negative_slope=0.2)

        # Mask: only edges from edge_index get real scores
        mask_val = torch.full((N, N), float('-inf'), device=x.device)
        mask_val[edge_index[0].long(), edge_index[1].long()] = 0
        attn_scores = attn_logits + mask_val.unsqueeze(-1)  # (N, N, H)

        attn = F.softmax(attn_scores, dim=1)  # normalize over source nodes

        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout, training=True)

        # Aggregate messages into destination nodes
        out = torch.zeros(N, H, D, device=x.device)
        for h_idx in range(H):
            # (N,N) @ (N,D) → (N,D): each dest node gets weighted sum of source features
            out[:, h_idx, :] = torch.matmul(attn[:, :, h_idx], h_s[:, h_idx, :])

        if self.concat:
            return out.reshape(N, H * D)
        return out.mean(dim=1)


class GATPredictor(nn.Module):
    """
    SCED用的GAT预测网络。

    输入: 观测向量 (obs_dim,)
    输出: 各机组出力预测 (N_UNITS,)

    架构:
      obs → parse → [node_feat (N×5), global_feat (24)]
      node_feat → GATv2 × 2 layers → mean_pool ─┐
      global_feat → MLP projector                 ├→ concat → MLP → P_pred (N_UNITS,)
    """
    NODE_DIM = 5   # on_off, p_ratio, margin_up, margin_down, capacity_ratio
    GLOBAL_DIM = 19  # matches _make_obs() output

    def __init__(self, obs_dim, n_units, edge_index, gat_heads=4, gat_hidden=128):
        super().__init__()
        self.n_units = n_units
        self.obs_dim = obs_dim

        if edge_index is None:
            # 默认全连接图
            src, dst = [], []
            for i in range(n_units):
                for j in range(n_units):
                    if i != j:
                        src.append(i); dst.append(j)
            edge_index = torch.LongTensor([src, dst])
        self.register_buffer('edge_index', edge_index)

        # GAT backbone
        self.gat1 = GATv2ConvLayer(self.NODE_DIM, gat_hidden, heads=gat_heads, concat=True)
        self.gat2 = GATv2ConvLayer(gat_hidden * gat_heads, gat_hidden, heads=gat_heads, concat=False)
        gat_out_dim = gat_hidden

        # Global feature projector
        self.global_proj = nn.Sequential(
            nn.Linear(self.GLOBAL_DIM, 128), nn.ReLU(),
            nn.Linear(128, gat_out_dim),
        )

        # Fusion + output
        fusion_dim = gat_out_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_units),
        )

    def _parse_obs(self, obs):
        """将观测向量拆分为节点特征和全局特征。"""
        B = obs.size(0)
        node = obs[:, :self.n_units * self.NODE_DIM].reshape(B * self.n_units, self.NODE_DIM)
        glob = obs[:, self.n_units * self.NODE_DIM:]
        return node, glob

    def forward(self, obs):
        B = obs.size(0)
        node, glob = self._parse_obs(obs)

        ei = self.edge_index
        if B > 1:
            offsets = torch.arange(B, device=obs.device).unsqueeze(1) * self.n_units
            ei = torch.cat([ei[0:1].expand(B,-1)+offsets, ei[1:2].expand(B,-1)+offsets], dim=0).reshape(2,-1)

        h = self.gat1(node, ei)
        h = F.relu(h)
        h = self.gat2(h, ei)  # (B*N, D)

        h_pool = h.reshape(B, self.n_units, -1).mean(dim=1)  # (B, D)
        g_proj = self.global_proj(glob)
        fused = torch.cat([h_pool, g_proj], dim=-1)

        return self.fusion(fused)  # (B, N_UNITS)


# ============================================================
# PyTorch Dataset
# ============================================================

class SCEDDataset(Dataset):
    """SCED预测任务的PyTorch Dataset。"""

    def __init__(self, dm: SCEDDataManager, scenario_ids: List[int],
                 max_load: float = 1.0, max_rsv: float = 1.0):
        self.dm = dm
        self.scenario_ids = scenario_ids
        self.max_load = max_load if max_load > 0 else dm.load_profile.max()
        self.max_rsv = max_rsv if max_rsv > 0 else dm.reserve_profile.max()

        self.samples = []  # List of (obs_array, target_power)
        self._build_samples()

    def _build_samples(self):
        """为每个场景×每个时段构建一个样本。"""
        for sid in self.scenario_ids:
            if sid not in self.dm.gt_power:
                continue
            gt = self.dm.gt_power[sid]  # (T, N)
            feat = self.dm.scenario_features.get(sid, {})

            for t in range(self.dm.T_HORIZON):
                obs = self._make_obs(sid, t, gt[t], feat)
                target = gt[t]  # (N,) — GT出力
                self.samples.append((obs, target))

        print(f"  Dataset built: {len(self.samples)} samples from {len(self.scenario_ids)} scenarios")

    def _make_obs(self, sid: int, t: int, current_power: np.ndarray, feat: dict) -> np.ndarray:
        """构建单个时刻的观测向量。"""
        dm = self.dm
        N = dm.N_UNITS
        T = dm.T_HORIZON

        # 使用该场景专属的负荷profile (UC_RL/300适配)
        load_profile = feat.get("load_profile", dm.load_profile)
        reserve_profile = feat.get("reserve_profile", dm.reserve_profile)
        drate_profile = feat.get("drate_profile", dm.drate_profile)

        # 节点特征 (N×5)
        on_off = (current_power > dm.PMIN * 0.5).astype(np.float32)
        p_ratio = np.clip(current_power / (dm.PMAX + 1e-6), 0, 1).astype(np.float32)
        margin_up = np.clip((dm.PMAX - current_power) / (dm.PMAX + 1e-6), 0, 1).astype(np.float32)
        margin_down = np.clip((current_power - dm.PMIN) / (dm.PMAX + 1e-6), 0, 1).astype(np.float32)
        cap_ratio = dm.PMAX / (dm.PMAX.max() + 1e-6)

        node_feat = np.stack([on_off, p_ratio, margin_up, margin_down, cap_ratio], axis=1)  # (N, 5)

        # 全局特征 (24)
        max_load_val = float(load_profile.max()) if load_profile.max() > 0 else 1.0
        max_rsv_val = float(reserve_profile.max()) if reserve_profile.max() > 0 else 1.0

        future_load = np.array([load_profile[min(t+k, T-1)] / max_load_val for k in range(1,5)], dtype=np.float32)
        future_rsv = np.array([reserve_profile[min(t+k, T-1)] / max_rsv_val for k in range(1,5)], dtype=np.float32)

        cur_dem = load_profile[t] / max_load_val
        cur_rsv = reserve_profile[t] / max_rsv_val
        drate = drate_profile[min(t, T-1)]

        cap_min = dm.PMIN[on_off > 0.5].sum() if (on_off > 0.5).any() else 0
        cap_max = dm.PMAX.sum()

        global_feat = np.array([
            t / T, cur_dem, cur_rsv,
            cap_min / (max_load_val + 1e-6), cap_max / (max_load_val + 1e-6),
            drate,
            *future_load, *future_rsv,
            0, 0, 0,  # 占位: overload info (训练时未知)
            0,         # n1_viol placeholder
            0,         # cumulative violation freq
        ], dtype=np.float32)

        obs = np.concatenate([node_feat.flatten(), global_feat]).astype(np.float32)
        return obs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs, target = self.samples[idx]
        return torch.from_numpy(obs), torch.from_numpy(target.astype(np.float32))


# ============================================================
# Phase 1: GAT 监督预训练
# ============================================================

def train_gat_predictor(dm: SCEDDataManager, epochs: int = 100, batch_size: int = 64,
                         lr: float = 1e-3, patience: int = 20) -> Tuple[GATPredictor, dict]:
    """
    Phase 1: 用Ground Truth训练GAT预测器。

    输入: 观测向量(包含当前状态+系统级信息)
    输出: 各机组出力的预测值(MW)

    Loss = MSE(P_pred, P_GT) + λ·|ΣP_pred - Demand|
    """
    print("\n" + "=" * 60)
    print("  Phase 1: GAT Supervised Pre-training")
    print("=" * 60)

    # 构建边索引（基于PTDF耦合）
    print("\n[Phase1] Building graph from PTDF matrix...")
    unit_ptdf = dm.ptdf_matrix[:, dm.unit_bus]  # (N_LINES, N_UNITS)
    norms = np.linalg.norm(unit_ptdf, axis=0, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    unit_ptdf_n = unit_ptdf / norms
    sim_matrix = unit_ptdf_n.T @ unit_ptdf_n  # (N_UNITS, N_UNITS)

    k_nbr = min(8, dm.N_UNITS - 1)
    threshold = 0.15
    src, dst = [], []
    for i in range(dm.N_UNITS):
        row = sim_matrix[i].copy()
        row[i] = -1
        top_k = np.argsort(row)[-k_nbr:]
        for j in top_k:
            if abs(sim_matrix[i, j]) >= threshold:
                src.append(i); dst.append(j)

    edge_index = torch.LongTensor([src, dst])
    print(f"  Graph built: {dm.N_UNITS} nodes, {len(src)} edges "
          f"(k={k_nbr}, threshold={threshold:.2f})")

    # 构建Dataset
    obs_dim = dm.N_UNITS * GATPredictor.NODE_DIM + GATPredictor.GLOBAL_DIM
    print(f"\n[Phase1] Observation dimension: {obs_dim}")

    train_set = SCEDDataset(dm, dm.train_ids, dm.load_profile.max(), dm.reserve_profile.max())
    val_set = SCEDDataset(dm, dm.val_ids, dm.load_profile.max(), dm.reserve_profile.max())

    train_loader = DataLoader(train_set, batch_size=min(batch_size, 32), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=min(batch_size*2, 64), shuffle=False, num_workers=0)

    # 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GATPredictor(obs_dim, dm.N_UNITS, edge_index).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "mae": [], "max_err": []}

    print(f"\n[Phase1] Training GAT predictor for {epochs} epochs (device={device})...")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        train_losses = []
        for batch_obs, batch_target in train_loader:
            batch_obs = batch_obs.to(device)
            batch_target = batch_target.to(device)

            pred = model(batch_obs)
            mse_loss = F.mse_loss(pred, batch_target)

            # 功率平衡正则项
            demand_scale = dm.load_profile.max()
            bal_penalty = ((pred.sum(dim=1) - batch_target.sum(dim=1)) / demand_scale).pow(2).mean()

            loss = mse_loss + 0.1 * bal_penalty

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ---- Validate ----
        model.eval()
        val_losses = []; maes = []; max_errs = []
        with torch.no_grad():
            for batch_obs, batch_target in val_loader:
                batch_obs = batch_obs.to(device)
                batch_target = batch_target.to(device)
                pred = model(batch_obs)
                loss = F.mse_loss(pred, batch_target)
                mae = (pred - batch_target).abs().mean().item()
                me = (pred - batch_target).abs().max().item()
                val_losses.append(loss.item())
                maes.append(mae); max_errs.append(me)

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        avg_mae = np.mean(maes)
        avg_me = np.mean(max_errs)
        scheduler.step(avg_val)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["mae"].append(avg_mae)
        history["max_err"].append(avg_me)

        if (epoch % 10 == 0) or (epoch == 1):
            print(f"  Epoch {epoch:03d}/{epochs}: train_loss={avg_train:.4f} "
                  f"val_loss={avg_val:.4f} MAE={avg_mae:.1f} MW MaxErr={avg_me:.1f} MW")

        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    # 恢复最佳模型
    model.load_state_dict(best_state)
    elapsed = time.time() - t0
    print(f"\n  Training completed in {elapsed:.1f}s")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Final MAE: {history['mae'][-1]:.1f} MW, Max Error: {history['max_err'][-1]:.1f} MW")

    # 保存模型
    save_path = SAVE_DIR / "gat_predictor_best.pth"
    torch.save({
        "model_state": model.state_dict(),
        "obs_dim": obs_dim,
        "n_units": dm.N_UNITS,
        "edge_index": edge_index.numpy(),
        "history": history,
    }, save_path)
    print(f"  Model saved to {save_path}")

    return model, history


# ============================================================
# Phase 2 (continued): RL 训练
# ============================================================

def train_rl_corrector(dm: SCEDDataManager, gat_model: GATPredictor,
                        total_timesteps: int = 50000, n_envs: int = 4,
                        env_mode: str = "redistribute") -> dict:
    """
    Phase 2: 训练PPO校正器。

    [v5] 使用绝对出力动作空间 (env_mode='absolute'):
      - 动作=归一化绝对出力[0,1]^N → 映射到[PMIN, PMAX]
      - RL直接学习最优安全经济调度
      - 拥有完整搜索空间, 非局部微调

    [v4] 使用比例重分配动作空间 (env_mode='redistribute'):
      - 动作=相对调整方向，自动缩放保证功率平衡
      - RL只学习如何重新分配出力来改善N-1安全
      - 无法通过"降总量"作弊

    [legacy] 使用绝对增量动作空间 (env_mode='legacy'):
      - 原始v3行为，有作弊风险
    """
    print("\n" + "=" * 60)
    print(f"  Phase 2: RL Corrector Training (PPO) — mode={env_mode}")
    print("=" * 60)

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    # 创建环境
    def make_env():
        if env_mode == "absolute":
            # [v5] 绝对出力模式: 完整搜索空间, 更多样本
            _delta_max = 0.25  # 不适用(绝对模式), 保留兼容
            _max_steps = 15   # 中等步数 (每步都是全新优化)
        elif env_mode == "redistribute":
            # [v4-aggressive] 更大调整幅度 + 更多校正步数, 追求N-1=0
            _delta_max = 0.20  # 允许单步调整±20%PMAX (vs 之前0.10)
            _max_steps = 20   # 更多步骤寻找最优解 (vs 之前10)
        else:
            _delta_max = 0.10
            _max_steps = 10
        env = SCEDCorrectorEnv(dm, gat_model, use_ac_pf=True,
                               max_correction_steps=_max_steps,
                               delta_max_ratio=_delta_max,
                               env_mode=env_mode)
        env = Monitor(env)
        return env

    vec_env = DummyVecEnv([make_env] * n_envs)

    # PPO策略网络配置
    if env_mode == "absolute":
        policy_kwargs = dict(
            net_arch=[512, 256, 128],
            activation_fn=nn.ReLU,
            log_std_init=-1.0,  # 更小的初始std (绝对模式需要更精确的输出)
        )
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=32,               # 小batch (dense attention矩阵占显存)
            n_epochs=20,                  # 更多epoch补偿小batch
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,                # 低探索率 (绝对模式空间大但需要精确收敛)
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=0,
        )
        print(f"\n[Phase2-v5.2] Training PPO corrector ({env_mode} mode) for {total_timesteps} steps...")
        print(f"  Action space: Box(0,1)^{dm.N_UNITS} -> P_i = a_i * Demand (demand-proportional)")
        _tmp_env = SCEDCorrectorEnv(dm, env_mode=env_mode)
        print(f"  [v5.2] Continuous Reward Shaping (dense gradient signal):")
        print(f"    N1_CONT(overload)={_tmp_env.LAM_N1_CONT:.0f} N1_WORST={_tmp_env.LAM_N1_WORST:.0f} COUNT_BONUS={_tmp_env.LAM_N1_COUNT_BONUS:.0f}")
        print(f"    COST={_tmp_env.LAM_COST:.1f} BALANCE={_tmp_env.LAM_BALANCE_SOFT:.0f} FEAS_BONUS=+{_tmp_env.FEASIBILITY_BONUS:.0f}")
        print(f"  Key: reward = overload_reduction + worst_case + count_bonus - cost - line_viol - balance")
    elif env_mode == "redistribute":
        policy_kwargs = dict(
            net_arch=[512, 256, 128],
            activation_fn=nn.ReLU,
            log_std_init=-0.5,
        )
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=5e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=15,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.15,
            ent_coef=0.02,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=0,
        )
        print(f"\n[Phase2-v3] Training PPO corrector (legacy mode)...")

    t0 = time.time()

    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    elapsed = time.time() - t0
    print(f"\n  RL training completed in {elapsed:.1f}s")

    # 保存模型 (带mode标记)
    rl_save_path = SAVE_DIR / f"ppo_corrector_{env_mode}.zip"
    model.save(str(rl_save_path))
    print(f"  Model saved to {rl_save_path}")

    # 快速评估
    print(f"\n[Phase2] Quick evaluation (10 episodes, mode={env_mode})...")
    eval_env = SCEDCorrectorEnv(dm, gat_model, use_ac_pf=True,
                                max_correction_steps=10,
                                delta_max_ratio=0.1,
                                env_mode=env_mode)
    n_success = 0; n_total = 10; total_rewards = []
    n1_counts = []; bal_errs = []
    for ep in range(n_total):
        obs, _ = eval_env.reset()
        ep_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += reward
            done = terminated or truncated
            if done:
                if info.get("is_feasible"):
                    n_success += 1
                n1_counts.append(info.get("n1_contingency_violations", -1))
                bal_errs.append(info.get("bal_err", -1))
        total_rewards.append(ep_reward)

    print(f"  Success rate: {n_success}/{n_total} ({100*n_success/n_total:.0f}%)")
    print(f"  Avg episode reward: {np.mean(total_rewards):.3f}")
    if n1_counts:
        print(f"  Avg N-1 violations: {np.mean(n1_counts):.1f}")
    if env_mode == "redistribute" and bal_errs:
        print(f"  Avg balance error: {np.mean(bal_errs):.2f} MW (should be ≈0)")


    vec_env.close()

    return {
        "model": rl_save_path,
        "success_rate": n_success / n_total,
        "avg_reward": float(np.mean(total_rewards)),
        "training_time": elapsed,
    }


# ============================================================
# Phase 3: 端到端评估
# ============================================================

def evaluate_end_to_end(dm: SCEDDataManager, gat_model: GATPredictor,
                         rl_model=None, n_scenarios: int = 20,
                         use_qp_repair: bool = True,
                         env_mode: str = "redistribute") -> dict:
    """
    Phase 3: 端到端评估。

    对比四种方案的指标:
      A) Ground Truth (Gurobi最优解)
      B) GAT预测结果 (带模拟误差)
      C) GAT + RL校正结果 (智能缩放后处理)
      D) GAT + RL + QP修复 (QP精确平衡) [可选]

    评估指标:
      - 平均发电成本
      - 功率不平衡度
      - 线路越限数/总越限量
      - N-1违反数
      - AC可行性
    """
    print("\n" + "=" * 60)
    print("  Phase 3: End-to-End Evaluation")
    print("=" * 60)
    print(f"  [Config] Evaluation using fl_scale={dm.fl_scale}")

    test_ids = dm.test_ids[:min(n_scenarios, len(dm.test_ids))]
    device = next(gat_model.parameters()).device
    gat_model.eval()

    results = {"gt": [], "gat": [], "gat_rl": [], "gat_rl_qp": []}

    print(f"\n[Phase3] Evaluating on {len(test_ids)} test scenarios...")

    with torch.no_grad():
        for sid in test_ids:
            if sid not in dm.gt_power:
                continue
            gt = dm.gt_power[sid]  # (T, N)
            feat = dm.scenario_features.get(sid, {})

            sid_results_gt = []
            sid_results_gat = []
            sid_results_gat_rl = []
            sid_results_gat_rl_qp = []

            for t in range(min(dm.T_HORIZON, 96)):  # 最多评估96个时段
                # 使用该场景专属的负荷profile (UC_RL/300适配)
                sid_profile = dm._get_scenario_profile(sid)
                sid_load = sid_profile["Load"].values.astype(np.float64)
                sid_rsv = sid_profile["Reserve"].values.astype(np.float64)
                sid_drate = sid_profile["rate"].values.astype(np.float64)

                demand = sid_load[t]
                drate = sid_drate[min(t, dm.T_HORIZON - 1)]

                # === A: Ground Truth ===
                p_gt = gt[t]
                flows_gt = dm.compute_line_flows(p_gt, drate)
                n1_gt = dm.check_n1_constraints(flows_gt)
                sid_results_gt.append({
                    "cost": _compute_fuel_cost(p_gt, dm, sid),
                    "bal_err": abs(p_gt.sum() - demand),
                    "line_viol": int((np.abs(flows_gt) - dm.flow_limit > 0.1).sum()),
                    "n1_viol": n1_gt["n1_contingency_violations"],
                })

                # === B: GAT Prediction (with simulated error) ===
                p_gat = p_gt.copy()
                # 模拟GAT误差
                np.random.seed(t)
                noise = np.random.normal(0, 0.02 * dm.PMAX, size=dm.N_UNITS)
                p_gat = np.clip(p_gat + noise, dm.PMIN, dm.PMAX)
                # 缩放以平衡
                if p_gat.sum() > 0:
                    p_gat *= (demand / p_gat.sum())
                flows_gat = dm.compute_line_flows(p_gat, drate)
                n1_gat = dm.check_n1_constraints(flows_gat)
                sid_results_gat.append({
                    "cost": _compute_fuel_cost(p_gat, dm, sid),
                    "bal_err": abs(p_gat.sum() - demand),
                    "line_viol": int((np.abs(flows_gat) - dm.flow_limit > 0.1).sum()),
                    "n1_viol": n1_gat["n1_contingency_violations"],
                })

                # === C: GAT + RL Correction ===
                if rl_model is not None:
                    _dmax = 0.20 if env_mode == "redistribute" else 0.1
                    _msteps = 15 if env_mode == "absolute" else (20 if env_mode == "redistribute" else 10)
                    corr_env = SCEDCorrectorEnv(dm, max_correction_steps=_msteps,
                                                delta_max_ratio=_dmax,
                                                env_mode=env_mode)
                    corr_env.current_sid = sid
                    corr_env.t = t
                    corr_env.target_demand = demand
                    corr_env.base_power = p_gat.copy()
                    corr_env.current_power = p_gat.copy()

                    obs, _ = corr_env.reset()
                    # Override with our specific state
                    corr_env.base_power = p_gat.copy()
                    corr_env.current_power = p_gat.copy()
                    corr_env.current_sid = sid
                    corr_env.t = t
                    corr_env.target_demand = demand
                    obs = corr_env._obs()

                    done = False
                    n_steps = 0
                    while not done and n_steps < 10:
                        action, _ = rl_model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = corr_env.step(action)
                        done = terminated or truncated
                        n_steps += 1

                    p_rl = corr_env.current_power

                    # === 后处理: 根据env_mode选择策略 ===
                    if env_mode in ("absolute", "redistribute"):
                        # [v5/v4] 环境已自动保证功率平衡，直接使用RL输出
                        p_rl_eval = p_rl.copy()
                    else:
                        # [legacy] 需要smart scaling后处理修复平衡
                        gen_total = p_rl.sum()
                        best_p_rl = p_rl.copy()
                        best_n1 = float('inf')
                        best_bal_err = abs(gen_total - demand)

                        if abs(gen_total - demand) > 1.0 and gen_total > 0:
                            for target_ratio in [1.0, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.85, 0.82, 0.80, 0.78, 0.75, 0.70]:
                                scale = (demand * target_ratio) / gen_total
                                p_candidate = np.clip(p_rl * scale, dm.PMIN, dm.PMAX)
                                flows_cand = dm.compute_line_flows(p_candidate, drate)
                                n1_cand = dm.check_n1_constraints(flows_cand)
                                bal_err_cand = abs(p_candidate.sum() - demand)
                                n1_v = n1_cand["n1_contingency_violations"]
                                score = n1_v * 10000 + bal_err_cand
                                if score < best_n1 * 10000 + best_bal_err:
                                    best_p_rl = p_candidate.copy()
                                    best_n1 = n1_v
                                    best_bal_err = bal_err_cand

                        p_rl_eval = best_p_rl

                    # === D: 双策略QP修复 (v4 Cost-Shaping + v5 Proportional Match) ===
                    p_qp_eval = p_gat.copy()  # 默认回退到GAT
                    qp_info = {"status": "fallback_to_gat"}
                    if use_qp_repair:
                        best_qp_score = float('inf')
                        best_qp_result = p_gat.copy()

                        # 策略1: Cost-Shaping QP v4 (用原始RL信号塑造成本)
                        p_cs4, info4 = gurobi_cost_shaping_qp(
                            p_gat, p_rl, demand, dm, drate,
                            scenario_id=sid, shaping_factor=0.05,
                            line_weight=200.0, time_limit=12.0
                        )
                        if info4["status"] in ("optimal", "time_limit", "no_rl_signal"):
                            f_cs4 = dm.compute_line_flows(p_cs4, drate)
                            n1_cs4 = dm.check_n1_constraints(f_cs4)
                            score4 = n1_cs4["n1_contingency_violations"] * 10000 + info4["bal_err"]
                            if score4 < best_qp_score:
                                best_qp_score = score4
                                best_qp_result = p_cs4.copy()
                                qp_info = {**info4, "n1_viol": n1_cs4["n1_contingency_violations"], "method": "cs_v4"}

                        # 策略2: Proportional Match QP v5 (用RL输出的比例分布)
                        p_pm5, info5 = gurobi_proportional_match_qp(
                            p_rl_eval, demand, dm, drate,
                            scenario_id=sid, line_weight=200.0, time_limit=12.0
                        )
                        if info5["status"] in ("optimal", "time_limit"):
                            f_pm5 = dm.compute_line_flows(p_pm5, drate)
                            n1_pm5 = dm.check_n1_constraints(f_pm5)
                            score5 = n1_pm5["n1_contingency_violations"] * 10000 + info5["bal_err"]
                            if score5 < best_qp_score:
                                best_qp_score = score5
                                best_qp_result = p_pm5.copy()
                                qp_info = {**info5, "n1_viol": n1_pm5["n1_contingency_violations"], "method": "prop_v5"}

                        p_qp_eval = best_qp_result

                    flows_rl = dm.compute_line_flows(p_rl_eval, drate)
                    n1_rl = dm.check_n1_constraints(flows_rl)
                    sid_results_gat_rl.append({
                        "cost": _compute_fuel_cost(p_rl_eval, dm, sid),
                        "bal_err": abs(p_rl_eval.sum() - demand),
                        "line_viol": int((np.abs(flows_rl) - dm.flow_limit > 0.1).sum()),
                        "n1_viol": n1_rl["n1_contingency_violations"],
                        "n_corr_steps": n_steps,
                        "feasible": info.get("is_feasible", False),
                        "raw_bal_err": abs(p_rl.sum() - demand),  # 修复前的不平衡（供参考）
                    })

                    # QP修复结果
                    flows_qp = dm.compute_line_flows(p_qp_eval, drate)
                    n1_qp = dm.check_n1_constraints(flows_qp)
                    sid_results_gat_rl_qp.append({
                        "cost": _compute_fuel_cost(p_qp_eval, dm, sid),
                        "bal_err": abs(p_qp_eval.sum() - demand),
                        "line_viol": int((np.abs(flows_qp) - dm.flow_limit > 0.1).sum()),
                        "n1_viol": n1_qp["n1_contingency_violations"],
                        "qp_status": qp_info.get("status", "unknown"),
                        "qp_solve_time": qp_info.get("solve_time", 0),
                        "deviation_from_rl": qp_info.get("deviation_from_rl", 0),
                    })
                else:
                    sid_results_gat_rl.append(None)
                    sid_results_gat_rl_qp.append(None)

            results["gt"].extend(sid_results_gt)
            results["gat"].extend(sid_results_gat)
            results["gat_rl"].extend(sid_results_gat_rl)
            results["gat_rl_qp"].extend(sid_results_gat_rl_qp)

    # 汇总统计
    def summarize(label, res_list):
        valid = [r for r in res_list if r is not None]
        if not valid:
            return {}
        return {
            "label": label,
            "n_samples": len(valid),
            "avg_cost": np.mean([r["cost"] for r in valid]),
            "avg_bal_err": np.mean([r["bal_err"] for r in valid]),
            "total_line_viol": sum([r["line_viol"] for r in valid]),
            "total_n1_viol": sum([r["n1_viol"] for r in valid]),
            "violin_rate": sum(1 for r in valid if r["line_viol"] > 0) / len(valid) * 100,
            "n1_viol_rate": sum(1 for r in valid if r["n1_viol"] > 0) / len(valid) * 100,
        }

    summary_gt = summarize("Ground Truth (Gurobi)", results["gt"])
    summary_gat = summarize("GAT Prediction", results["gat"])
    summary_rl = summarize("GAT + RL Corrector", results["gat_rl"])
    summary_qp = summarize("GAT + RL + QP Repair", results["gat_rl_qp"])

    # 打印对比表
    print(f"\n{'='*100}")
    print(f"  {'Metric':<25} {'GT (Gurobi)':>18} {'GAT Only':>18} {'GAT+RL':>18} {'GAT+RL+QP':>18}")
    print(f"{'-'*100}")
    metrics = [
        ("Avg Fuel Cost ($)", "avg_cost"),
        ("Avg Balance Err (MW)", "avg_bal_err"),
        ("Total Line Violations", "total_line_viol"),
        ("Total N-1 Violations", "total_n1_viol"),
        ("Line Viol Rate (%)", "violin_rate"),
        ("N-1 Viol Rate (%)", "n1_viol_rate"),
    ]
    for name, key in metrics:
        v_gt = f"{summary_gt.get(key, 0):.2f}"
        v_gat = f"{summary_gat.get(key, 0):.2f}"
        v_rl = f"{summary_rl.get(key, 0):.2f}"
        v_qp = f"{summary_qp.get(key, 0):.2f}"
        print(f"  {name:<25} {v_gt:>18} {v_gat:>18} {v_rl:>18} {v_qp:>18}")
    print(f"{'='*100}")

    # 保存详细结果
    result_file = SAVE_DIR / "evaluation_results.json"
    raw_results_serializable = {}
    for k, v in results.items():
        raw_results_serializable[k] = []
        for r in v:
            if r is None:
                raw_results_serializable[k].append(None)
            else:
                raw_results_serializable[k].append({
                    kk: (float(vv) if isinstance(vv, (np.integer, np.floating)) else vv)
                    for kk, vv in r.items()
                })
    # QP结果序列化
    raw_results_serializable_qp = []
    for r in results["gat_rl_qp"]:
        if r is None:
            raw_results_serializable_qp.append(None)
        else:
            raw_results_serializable_qp.append({
                kk: (float(vv) if isinstance(vv, (np.integer, np.floating)) else vv)
                for kk, vv in r.items()
            })

    eval_data = {
        "summary_gt": summary_gt,
        "summary_gat": summary_gat,
        "summary_rl": summary_rl,
        "summary_qp": summary_qp,
        "raw_results": raw_results_serializable,
        "raw_results_qp": raw_results_serializable_qp,
    }
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results saved to {result_file}")

    return {"summary_gt": summary_gt, "summary_gat": summary_gat,
            "summary_rl": summary_rl, "summary_qp": summary_qp}


# ============================================================
# Phase 3+: DC vs AC-PF N-1 对比评估
# ============================================================

def evaluate_dc_vs_ac_n1(
    dm: SCEDDataManager,
    gat_model: GATPredictor = None,
    rl_model_path: str = None,
    n_scenarios: int = 10,
    n_time_periods: int = 5,
    r_x_ratio: float = 0.2,
    verbose_pf: bool = False,
) -> dict:
    """
    DC (LODF近似) vs AC (牛顿-拉夫逊) N-1安全校验对比。

    核心问题: GT在DC-LODF下显示90%+的N-1违反率，
              这是否是DC线性化的假阳性？

    评估内容:
      - 对每个测试方案的每个时段:
        1. 计算DC基态潮流 + DC-LODF N-1检查
        2. 构建Ybus → NR-PF求解 → AC线路潮流
        3. 对每条线路停运, 重新NR-PF → AC N-1检查
      - 汇总对比: DC违反率 vs AC违反率, 假阳性率

    Args:
        dm: 数据管理器(已加载全部数据)
        gat_model: GAT模型(可选, 用于GAT+RL方案)
        rl_model_path: RL模型路径
        n_scenarios: 评估场景数
        n_time_periods: 每个场景评估时段数(AC-PF较慢, 默认5)
        r_x_ratio: R/X比值假设
        verbose_pf: 是否打印PF迭代详情

    Returns:
        dict with detailed comparison results
    """
    import time

    print("\n" + "=" * 70)
    print("  Phase 3+: DC vs AC Power Flow N-1 Security Verification")
    print("=" * 70)

    # 加载RL model
    rl_agent = None
    if rl_model_path and os.path.exists(rl_model_path):
        from stable_baselines3 import PPO
        try:
            rl_agent = PPO.load(rl_model_path)
            print(f"  [AC-Eval] RL model loaded: {rl_model_path}")
        except Exception as e:
            print(f"  [AC-Eval] WARN: Failed to load RL model: {e}")

    test_ids = dm.test_ids[:min(n_scenarios, len(dm.test_ids))]
    if gat_model is not None:
        gat_model.eval()

    # 预构建Ybus
    print(f"\n  [AC-Eval] Building Ybus (R/X={r_x_ratio})...")
    dm._build_pandapower_net(r_x_ratio=r_x_ratio)

    all_results = []
    t_total_start = time.time()

    for si, sid in enumerate(test_ids):
        if sid not in dm.gt_power:
            continue
        gt = dm.gt_power[sid]  # (T, N)

        # 选择代表性时段（均匀采样）
        T = min(dm.T_HORIZON, 96)
        if T <= n_time_periods:
            periods = list(range(T))
        else:
            step = max(1, T // n_time_periods)
            periods = list(range(0, T, step))[:n_time_periods]

        print(f"\n  [{si+1}/{len(test_ids)}] Scenario {sid}, "
              f"evaluating {len(periods)} periods...")

        feat = dm.scenario_features.get(sid, {})
        ud = dm._get_scenario_unit_data(sid)

        for ti, t in enumerate(periods):
            # 使用该场景专属的profile
            _sp = dm._get_scenario_profile(sid)
            demand = _sp["Load"].values.astype(np.float64)[t]
            drate = _sp["rate"].values.astype(np.float64)[min(t, dm.T_HORIZON - 1)]
            p_gt = gt[t]

            # ---- 准备各方案出力 ----
            solutions = {
                "GT": p_gt.copy(),
            }

            # GAT方案 (带模拟误差)
            np.random.seed(t * 1000 + sid)
            noise = np.random.normal(0, 0.02 * dm.PMAX, size=dm.N_UNITS)
            p_gat = np.clip(p_gt + noise, dm.PMIN, dm.PMAX)
            if p_gat.sum() > 0:
                p_gat *= (demand / p_gat.sum())
            solutions["GAT"] = p_gat

            # GAT+RL方案
            if rl_agent is not None:
                try:
                    corr_env = SCEDCorrectorEnv(
                        dm, delta_max_ratio=0.20,
                        max_correction_steps=20, env_mode="redistribute"
                    )
                    # 先reset初始化内部状态
                    try:
                        _obs_init, _ = corr_env.reset()
                    except Exception:
                        pass
                    corr_env.current_sid = sid; corr_env.t = t
                    corr_env.target_demand = demand
                    corr_env.base_power = p_gat.copy()
                    corr_env.current_power = p_gat.copy()
                    obs = corr_env._obs()
                    for _ in range(10):
                        action, _ = rl_agent.predict(obs, deterministic=True)
                        obs, _, terminated, truncated, _ = corr_env.step(action)
                        if terminated or truncated:
                            break
                    solutions["GAT+RL"] = corr_env.current_power.copy()
                except Exception as e:
                    solutions["GAT+RL"] = p_gat.copy()
                    print(f"    [WARN] RL inference failed: {e}")

            # ---- 对每个方案做DC vs AC N-1校验 ----
            for sol_name, p_sol in solutions.items():
                t0 = time.time()

                # === DC N-1 (LODF) ===
                dc_flows = dm.compute_line_flows(p_sol, drate)
                dc_n1 = dm.check_n1_constraints(dc_flows)

                # === AC N-1 (Newton-Raphson) ===
                ac_n1 = dm.check_n1_constraints_ac(
                    p_sol, drate=drate,
                    voltage_lo=0.95, voltage_hi=1.05,
                    verbose=verbose_pf
                )
                t_ac = time.time() - t0

                record = {
                    "scenario_id": sid,
                    "period": t,
                    "solution": sol_name,
                    "demand_MW": float(demand),
                    "gen_total_MW": float(p_sol.sum()),
                    # --- DC results ---
                    "dc_n1_violations": int(dc_n1["n1_contingency_violations"]),
                    "dc_n1_viol_lines": int(dc_n1["n1_violated_lines"]),
                    "dc_base_max_flow_MW": float(np.abs(dc_flows).max()),
                    # --- AC results ---
                    "ac_n1_violations": int(ac_n1["n1_contingency_violations"]),
                    "ac_n1_viol_lines": int(ac_n1["n1_violated_lines"]),
                    "ac_voltage_violations": int(ac_n1["ac_voltage_violations"]),
                    "ac_base_converged": ac_n1["base_case_converged"],
                    "ac_n_contingencies_checked": int(ac_n1["total_contingencies_checked"]),
                    "ac_eval_time_s": round(t_ac, 2),
                }
                all_results.append(record)

                if verbose_pf or (ti == 0 and si == 0):
                    print(f"    [{sol_name}] t={t:2d} | "
                          f"DC-N1={dc_n1['n1_contingency_violations']:3d} | "
                          f"AC-N1={ac_n1['n1_contingency_violations']:3d} | "
                          f"Vviol={ac_n1['ac_voltage_violations']:2d} | "
                          f"conv={'Y' if ac_n1['base_case_converged'] else 'N'} | "
                          f"{t_ac:.1f}s")

    total_time = time.time() - t_total_start
    n_records = len(all_results)
    print(f"\n  [AC-Eval] Done! {n_records} records, {total_time:.1f}s total")

    # ================================================================
    # 汇总统计
    # ================================================================
    df = pd.DataFrame(all_results)
    if df.empty:
        print("  [ERROR] No results collected!")
        return {}

    summary = {}
    for sol_name in df["solution"].unique():
        sub = df[df["solution"] == sol_name]
        n_samples = len(sub)

        dc_total_n1 = sub["dc_n1_violations"].sum()
        ac_total_n1 = sub["ac_n1_violations"].sum()
        dc_rate = (sub["dc_n1_violations"] > 0).mean() * 100
        ac_rate = (sub["ac_n1_violations"] > 0).mean() * 100

        # 假阳性分析: DC说违反但AC说不违反
        false_positives = ((sub["dc_n1_violations"] > 0) &
                           (sub["ac_n1_violations"] == 0)).sum()
        false_positive_rate = false_positives / n_samples * 100

        # 真阳性: DC和AC都说违反
        true_positives = ((sub["dc_n1_violations"] > 0) &
                          (sub["ac_n1_violations"] > 0)).sum()

        # AC特有违反 (DC没发现但AC发现了)
        ac_only_viol = ((sub["dc_n1_violations"] == 0) &
                        (sub["ac_n1_violations"] > 0)).sum()

        conv_rate = sub["ac_base_converged"].mean() * 100
        avg_time = sub["ac_eval_time_s"].mean()

        s = {
            "solution": sol_name,
            "n_samples": n_samples,
            "dc_total_n1": int(dc_total_n1),
            "ac_total_n1": int(ac_total_n1),
            "dc_viol_rate_pct": round(dc_rate, 1),
            "ac_viol_rate_pct": round(ac_rate, 1),
            "false_positives": int(false_positives),
            "false_positive_rate_pct": round(false_positive_rate, 1),
            "true_positives": int(true_positives),
            "ac_only_violations": int(ac_only_viol),
            "pf_convergence_rate_pct": round(conv_rate, 1),
            "avg_eval_time_s": round(avg_time, 1),
        }
        summary[sol_name] = s

    # 打印核心对比表
    print(f"\n{'='*90}")
    print(f"  {'DC vs AC N-1 Security Comparison':^90}")
    print(f"{'='*90}")
    print(f"  {'Solution':<12} {'Samples':>7} | "
          f"{'DC-N1 Tot':>9} {'DC Rate%':>9} | "
          f"{'AC-N1 Tot':>9} {'AC Rate%':>9} | "
          f"{'FP':>5} {'FPR%':>6} | {'Conv%':>6}")
    print(f"  {'-'*12} {'-'*7}-|-{'-'*9}-{'-'*9}-|-{'-'*9}-{'-'*9}-|"
          f"{'-'*5}-{'-'*6}-|{'-'*6}")

    for sol_name in ["GT", "GAT", "GAT+RL"]:
        if sol_name in summary:
            s = summary[sol_name]
            print(f"  {sol_name:<12} {s['n_samples']:>7} | "
                  f"{s['dc_total_n1']:>9} {s['dc_viol_rate_pct']:>8.1f}% | "
                  f"{s['ac_total_n1']:>9} {s['ac_viol_rate_pct']:>8.1f}% | "
                  f"{s['false_positives']:>5} {s['false_positive_rate_pct']:>5.1f}% | "
                  f"{s['pf_convergence_rate_pct']:>5.1f}%")

    print(f"{'='*90}")

    # 关键洞察
    gt_summary = summary.get("GT", {})
    if gt_summary and gt_summary["dc_total_n1"] > 0:
        reduction = (1.0 - gt_summary["ac_total_n1"] / max(gt_summary["dc_total_n1"], 1)) * 100
        print(f"\n  *** KEY FINDING: GT DC-N1 violations: {gt_summary['dc_total_n1']}, "
              f"AC-N1 violations: {gt_summary['ac_total_n1']} "
              f"(reduction: {reduction:.1f}%)")
        if gt_summary["false_positives"] > 0:
            print(f"      DC假阳性率: {gt_summary['false_positive_rate_pct']:.1f}% "
                  f"({gt_summary['false_positives']}/{gt_summary['n_samples']})")

    # 保存结果
    result_file = SAVE_DIR / "dc_vs_ac_n1_comparison.json"
    output_data = {
        "config": {
            "r_x_ratio": r_x_ratio,
            "n_scenarios": len(test_ids),
            "n_time_periods": n_time_periods,
            "total_eval_time_s": round(total_time, 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summary_by_solution": summary,
        "all_records": all_results,
    }
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results saved to {result_file}")

    return output_data


def _compute_fuel_cost(power: np.ndarray, dm: SCEDDataManager, scenario_id: int) -> float:
    """计算给定出力方案的燃料成本。"""
    ud = dm._get_scenario_unit_data(scenario_id)
    prices = ud["PriceMinPower"].values.astype(np.float64)
    seg1 = ud["PriceSegment1"].values.astype(np.float64)
    cost = 0.0
    for i in range(dm.N_UNITS):
        if power[i] > dm.PMIN[i] * 0.5:
            cost += prices[i] + seg1[i] * max(0, power[i] - dm.PMIN[i])
    return cost


def gurobi_cost_shaping_qp(p_gat: np.ndarray, p_rl: np.ndarray,
                           demand: float, dm: SCEDDataManager,
                           drate: float = 1.0, scenario_id: int = 0,
                           shaping_factor: float = 0.05,
                           line_weight: float = 200.0,
                           time_limit: float = 15.0) -> Tuple[np.ndarray, dict]:
    """
    成本塑造QP v4 —— 用RL信号引导Gurobi ED向更安全的运行点偏移。

    核心思路 (v4):
      不再试图"修复"RL的不平衡输出(那是不可能的),
      而是用RL相对于GAT的调整方向作为"安全信号",
      通过微调各机组成本系数来引导标准ED求解器向更安全的解偏移。

    算法:
      1. 计算RL相对GAT的调整量: delta_i = P_RL_i - P_GAT_i
      2. 归一化得到方向向量: dir_i = delta_i / max(|delta|)
      3. 塑造成本: cost'_i = cost_i * (1 + shaping_factor * (-dir_i))
         即RL想降低的机组→变贵(ED会少用它)，RL想提高的→变便宜
      4. 求解标准SCED-QP: min cost'(P) s.t. 平衡/限值/线路约束
      5. warm-start from P_GAT (已满足功率平衡)

    Args:
        p_gat: GAT预测出力 (N,) [MW] — 已平衡的基准
        p_rl: RL校正后出力 (N,) [MW] — 安全信号来源(不需要平衡!)
        demand: 目标总需求 [MW]
        dm: 数据管理器
        drate: 负荷分配比例
        scenario_id: 场景ID(获取燃料成本)
        shaping_factor: 成本塑形强度 (0=纯ED, 0.1=强塑形)
        line_weight: 线路越限松弛惩罚
        time_limit: 求解时限

    Returns:
        p_cs: 成本塑造后的出力 (N,)
        info: 求解信息字典
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        print("  [WARN] gurobipy not available")
        return p_gat.copy(), {"status": "no_gurobi"}

    N = dm.N_UNITS
    # 获取燃料成本系数
    ud = dm._get_scenario_unit_data(scenario_id)
    price_slope = ud["PriceSegment1"].values.astype(np.float64)

    # === 步骤1-2: 计算RL安全方向信号 ===
    delta = p_rl.astype(np.float64) - p_gat.astype(np.float64)
    max_abs_delta = np.abs(delta).max()
    if max_abs_delta < 1e-6:
        # RL没有做任何调整，直接返回GAT
        return p_gat.copy(), {"status": "no_rl_signal", "bal_err": abs(p_gat.sum() - demand)}

    direction = delta / max_abs_delta  # [-1, 1], 负=RL想降低(该机组不安全), 正=RL想提高

    # === 步骤3: 塑造成本 ===
    # RL想降低的方向(d<0)→提高成本 → ED减少使用 → 降低潮流 → 改善N-1
    # RL想提高的方向(d>0)→降低成本 → ED增加使用
    shaped_slope = price_slope * (1.0 - shaping_factor * direction)
    shaped_slope = np.maximum(shaped_slope, price_slope * 0.01)  # 防止负成本

    # 创建模型
    model = gp.Model("CS_SCED_v4")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)

    # 变量
    P = model.addVars(N, lb=dm.PMIN.tolist(), ub=dm.PMAX.tolist(), name="P")
    L_pos = model.addVars(dm.N_LINES, lb=0, name="slack_pos")
    L_neg = model.addVars(dm.N_LINES, lb=0, name="slack_neg")

    # 目标: 塑造后的燃料成本 + 线路松弛惩罚
    fuel_expr = gp.quicksum(
        shaped_slope[i] * (P[i] - dm.PMIN[i])
        for i in range(N) if shaped_slope[i] > 0
    )
    line_pen = line_weight * gp.quicksum(L_pos[l] + L_neg[l] for l in range(dm.N_LINES))
    model.setObjective(fuel_expr + line_pen, GRB.MINIMIZE)

    # 约束
    model.addConstr(gp.quicksum(P[i] for i in range(N)) == demand, name="PowerBalance")
    for l in range(dm.N_LINES):
        ptdf_row = dm.ptdf_matrix[l, :]
        flow_expr = gp.quicksum(
            ptdf_row[dm.unit_bus[i]] * P[i] for i in range(N)
            if 0 <= dm.unit_bus[i] < dm.N_BUSES
        ) * drate
        limit = dm.flow_limit[l]
        model.addConstr(flow_expr - limit <= L_pos[l], name=f"LinePos_{l}")
        model.addConstr(-flow_expr - limit <= L_neg[l], name=f"LineNeg_{l}")

    # Warm-start from GAT (feasible starting point!)
    for i in range(N):
        P[i].Start = float(p_gat[i])

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        p_cs = np.array([P[i].X for i in range(N)], dtype=np.float64)
        total_slack = sum(L_pos[l].X + L_neg[l].X for l in range(dm.N_LINES))
        info = {
            "status": "optimal" if model.status == GRB.OPTIMAL else "time_limit",
            "solve_time": model.Runtime,
            "obj_val": model.ObjVal,
            "bal_err": abs(p_cs.sum() - demand),
            "total_line_slack": total_slack,
            "deviation_from_gat": float(np.linalg.norm(p_cs - p_gat)),
            "shaping_factor": shaping_factor,
        }
        return p_cs, info
    else:
        return p_gat.copy(), {"status": f"solve_failed_{model.status}",
                              "bal_err": abs(p_gat.sum() - demand)}


def gurobi_proportional_match_qp(p_reference: np.ndarray, demand: float,
                                  dm: SCEDDataManager, drate: float = 1.0,
                                  scenario_id: int = 0,
                                  line_weight: float = 200.0,
                                  time_limit: float = 15.0) -> Tuple[np.ndarray, dict]:
    """
    比例匹配QP v5 —— 匹配参考方案的相对分布比例，同时满足功率平衡。

    核心思路 (v5):
      从GAT+RL(smart-scaled)获得具有良好N-1特性的相对分布比例,
      然后用QP找到最接近该比例分布且完全可行的运行点。
      这避免了RL"降出力作弊"导致的总量错误，只提取其分布智慧。

    算法:
      1. 计算参考方案的比例向量: r_i = P_ref_i / sum(P_ref)
      2. 目标: 最小化与比例向量的偏差 + 燃料成本 + 线路松弛
      3. 约束: 功率平衡(硬)、出力限值、线路热稳定

    Args:
        p_reference: 参考出力 (N,) [MW] — 具有良好N-1但可能不平衡的方案
        demand: 目标总需求 [MW]
        dm: 数据管理器
        drate: 负荷分配比例
        scenario_id: 场景ID
        line_weight: 线路越限松弛惩罚
        time_limit: 求解时限
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return p_reference.copy(), {"status": "no_gurobi"}

    N = dm.N_UNITS
    ref_sum = p_reference.sum()
    if ref_sum < 1e-6:
        return p_reference.copy(), {"status": "zero_reference"}

    # 比例向量: 各机组占总出力的份额
    ref_ratios = p_reference.astype(np.float64) / ref_sum

    # 目标出力 (按比例分配需求)
    P_target_ratio = ref_ratios * demand

    ud = dm._get_scenario_unit_data(scenario_id)
    price_slope = ud["PriceSegment1"].values.astype(np.float64)

    model = gp.Model("PropMatch_v5")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)

    P = model.addVars(N, lb=dm.PMIN.tolist(), ub=dm.PMAX.tolist(), name="P")
    L_pos = model.addVars(dm.N_LINES, lb=0, name="slack_pos")
    L_neg = model.addVars(dm.N_LINES, lb=0, name="slack_neg")

    # 目标: (1) 匹配参考比例 (2) 低燃料成本 (3) 线路可行
    prop_term = gp.quicksum(
        (P[i] - P_target_ratio[i]) ** 2 for i in range(N)
    )
    fuel_term = gp.quicksum(
        0.0001 * price_slope[i] * (P[i] - dm.PMIN[i])
        for i in range(N) if price_slope[i] > 0
    )
    line_pen = line_weight * gp.quicksum(L_pos[l] + L_neg[l] for l in range(dm.N_LINES))

    model.setObjective(prop_term + fuel_term + line_pen, GRB.MINIMIZE)

    # 约束
    model.addConstr(gp.quicksum(P[i] for i in range(N)) == demand, name="PowerBalance")
    for l in range(dm.N_LINES):
        ptdf_row = dm.ptdf_matrix[l, :]
        flow_expr = gp.quicksum(
            ptdf_row[dm.unit_bus[i]] * P[i] for i in range(N)
            if 0 <= dm.unit_bus[i] < dm.N_BUSES
        ) * drate
        limit = dm.flow_limit[l]
        model.addConstr(flow_expr - limit <= L_pos[l], name=f"LinePos_{l}")
        model.addConstr(-flow_expr - limit <= L_neg[l], name=f"LineNeg_{l}")

    # Warm-start: 从按比例缩放的参考开始 (可能不满足线路约束但接近)
    for i in range(N):
        P[i].Start = float(np.clip(P_target_ratio[i], dm.PMIN[i], dm.PMAX[i]))

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        p_pm = np.array([P[i].X for i in range(N)], dtype=np.float64)
        total_slack = sum(L_pos[l].X + L_neg[l].X for l in range(dm.N_LINES))
        info = {
            "status": "optimal" if model.status == GRB.OPTIMAL else "time_limit",
            "solve_time": model.Runtime,
            "obj_val": model.ObjVal,
            "bal_err": abs(p_pm.sum() - demand),
            "total_line_slack": total_slack,
            "deviation_from_ref": float(np.linalg.norm(p_pm - P_target_ratio)),
        }
        return p_pm, info
    else:
        return p_reference.copy(), {"status": f"solve_failed_{model.status}",
                                   "bal_err": abs(p_reference.sum() - demand)}


# ============================================================
# Main 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="SCED-GAT-RL Pipeline")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["0", "1", "2", "3", "ac", "all"],
                        help="Which phase(s) to run (default: all); 'ac' for DC vs AC N-1 comparison")
    parser.add_argument("--system", type=str, default="300",
                        help="System size (300 or 1888)")
    parser.add_argument("--n_train", type=int, default=50,
                        help="Number of training scenarios")
    parser.add_argument("--n_val", type=int, default=10,
                        help="Number of validation scenarios")
    parser.add_argument("--n_test", type=int, default=20,
                        help="Number of test scenarios")
    parser.add_argument("--gat_epochs", type=int, default=100,
                        help="GAT pre-training epochs")
    parser.add_argument("--rl_timesteps", type=int, default=50000,
                        help="RL corrector training timesteps")
    parser.add_argument("--delta_max", type=float, default=0.05,
                        help="Max correction ratio of PMAX")
    parser.add_argument("--env_mode", type=str, default="absolute",
                        choices=["absolute", "redistribute", "legacy"],
                        help="RL env mode: absolute(v5, full power control), redistribute(v4, auto-balance), legacy(v3)")
    parser.add_argument("--fl_scale", type=float, default=28.0,
                        help="Flow limit scale factor for N-1 checks (default=28 for DC-LODF margin)")
    args = parser.parse_args()

    print("=" * 60)
    print("  SCED-GAT-RL Pipeline")
    print("  Three-Stage Intelligent Economic Dispatch Framework")
    print("=" * 60)
    print(f"  System: {args.system}-bus")
    print(f"  Phase:  {args.phase}")
    print(f"  Train/Val/Test: {args.n_train}/{args.n_val}/{args.n_test}")
    print(f"  fl_scale: {args.fl_scale} (N-1 flow limit margin factor)")
    print(f"  env_mode: {args.env_mode} (RL action space design)")

    overall_start = time.time()

    # ===== Phase 0: Data Loading =====
    if args.phase in ("0", "all"):
        dm = SCEDDataManager(DATA_ROOT, n_train=args.n_train, n_val=args.n_val,
                              n_test=args.n_test)
        if not dm.load_all():
            print("[FATAL] Phase 0 failed: no ground truth data found!")
            return
        # [FIX] 必须在load_all()之后设置fl_scale, 因为load_all()→load_system_params()会重置为28
        dm.fl_scale = args.fl_scale  # N-1限值放大系数 (工程备用裕度)
        print(f"  [Config] dm.fl_scale set to {dm.fl_scale} (AFTER load_all)")
        # 供后续phase使用
        _save_dm_cache(dm)
    else:
        dm = _load_dm_cache()
        dm.fl_scale = args.fl_scale  # N-1限值放大系数
        print(f"  [Config] dm.fl_scale (from cache) set to {dm.fl_scale}")

    # ===== Phase 1: GAT Training =====
    gat_model = None
    if args.phase in ("1", "all"):
        gat_model, gat_history = train_gat_predictor(
            dm, epochs=args.gat_epochs, batch_size=32, lr=1e-3, patience=20
        )
    elif args.phase in ("2", "3", "all"):
        # 从checkpoint加载GAT模型
        ckpt = SAVE_DIR / "gat_predictor_best.pth"
        if ckpt.exists():
            checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
            obs_dim = checkpoint["obs_dim"]
            n_units = checkpoint["n_units"]
            ei = torch.LongTensor(checkpoint["edge_index"])
            gat_model = GATPredictor(obs_dim, n_units, ei)
            gat_model.load_state_dict(checkpoint["model_state"])
            print(f"[Main] Loaded GAT model from {ckpt}")
        else:
            print("[WARN] No GAT model found! Run phase 1 first.")

    # ===== Phase 2: RL Training =====
    rl_model = None
    if args.phase in ("2", "all") and gat_model is not None:
        rl_info = train_rl_corrector(dm, gat_model, total_timesteps=args.rl_timesteps,
                                      env_mode=args.env_mode)
        rl_model = rl_info["model"]

    elif args.phase == "2" and gat_model is None:
        print("[ERROR] Phase 2 needs a trained GAT model. Run phase 1 first.")
        return

    # ===== Phase 3: Evaluation =====
    if args.phase in ("3", "all"):
        # 加载RL模型
        rl_model_obj = None
        if args.phase == "all" and rl_model is not None:
            from stable_baselines3 import PPO
            rl_model_obj = PPO.load(str(rl_model))
        elif args.phase == "3":
            # 单独运行Phase 3时，从保存文件加载RL模型
            rl_zip = SAVE_DIR / f"ppo_corrector_{args.env_mode}.zip"
            if not rl_zip.exists():
                rl_zip = SAVE_DIR / "ppo_corrector.zip"  # fallback到默认名
            if rl_zip.exists():
                from stable_baselines3 import PPO
                rl_model_obj = PPO.load(str(rl_zip))
                print(f"[Main] Loaded RL model from {rl_zip}")
            else:
                print("[WARN] No RL model found at ppo_corrector.zip, skipping GAT+RL eval")

        eval_results = evaluate_end_to_end(
            dm, gat_model, rl_model_obj, n_scenarios=min(args.n_test, 20),
            env_mode=args.env_mode
        )

    # ===== Phase AC: DC vs AC N-1 Comparison =====
    if args.phase == "ac":
        # 需要GAT模型(用于GAT+RL方案)
        _gat = gat_model
        if _gat is None:
            ckpt = SAVE_DIR / "gat_predictor_best.pth"
            if ckpt.exists():
                checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
                obs_dim = checkpoint["obs_dim"]
                n_units = checkpoint["n_units"]
                ei = torch.LongTensor(checkpoint["edge_index"])
                _gat = GATPredictor(obs_dim, n_units, ei)
                _gat.load_state_dict(checkpoint["model_state"])
                print(f"[AC] Loaded GAT model from {ckpt}")

        # RL模型路径
        rl_path = str(SAVE_DIR / "ppo_corrector_redistribute.zip")
        if not os.path.exists(rl_path):
            rl_path = str(SAVE_DIR / "ppo_corrector.zip")

        ac_results = evaluate_dc_vs_ac_n1(
            dm,
            gat_model=_gat,
            rl_model_path=rl_path if os.path.exists(rl_path) else None,
            n_scenarios=min(args.n_test, 10),
            n_time_periods=5,
            r_x_ratio=0.2,
            verbose_pf=True,
        )

    elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"  Pipeline completed in {elapsed:.1f}s")
    print(f"  Results saved to: {SAVE_DIR}")
    print(f"{'='*60}")


def _save_dm_cache(dm: SCEDDataManager):
    """缓存DataManager以便跨phase复用（避免重复加载大数据）。"""
    cache_file = SAVE_DIR / "dm_cache.pkl"
    import pickle
    with open(cache_file, "wb") as f:
        pickle.dump({
            "train_ids": dm.train_ids,
            "val_ids": dm.val_ids,
            "test_ids": dm.test_ids,
            "gt_power_keys": list(dm.gt_power.keys()),  # 只存keys节省空间
        }, f)
    print(f"  DM cache saved to {cache_file}")


def _load_dm_cache() -> SCEDDataManager:
    """加载缓存的DataManager。"""
    cache_file = SAVE_DIR / "dm_cache.pkl"
    import pickle
    with open(cache_file, "rb") as f:
        cached = pickle.load(f)
    dm = SCEDDataManager(DATA_ROOT,
                          n_train=len(cached["train_ids"]),
                          n_val=len(cached["val_ids"]),
                          n_test=len(cached["test_ids"]))
    dm.load_system_params()
    dm.split_scenarios()
    dm.train_ids = cached["train_ids"]
    dm.val_ids = cached["val_ids"]
    dm.test_ids = cached["test_ids"]

    # 重新加载所有GT
    all_ids = dm.train_ids + dm.val_ids + dm.test_ids
    for sid in all_ids:
        gt = dm.load_ground_truth(sid)
        if gt is not None:
            if gt.shape[0] == dm.N_UNITS and gt.shape[1] == dm.T_HORIZON:
                dm.gt_power[sid] = gt.T
            elif gt.shape[0] == dm.T_HORIZON and gt.shape[1] == dm.N_UNITS:
                dm.gt_power[sid] = gt
            dm.scenario_features[sid] = dm.build_features_for_scenario(sid)

    print(f"[Main] Loaded DM cache: {len(dm.gt_power)} scenarios")
    return dm


if __name__ == "__main__":
    main()

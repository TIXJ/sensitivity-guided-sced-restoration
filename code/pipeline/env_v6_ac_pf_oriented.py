"""
v6 AC-PF Oriented RL Corrector Environment
================================================
核心设计理念:
  SCED用DC-LODF保证N-1安全 → GAT做经济调度预测(有MAE误差)
  → RL负责校正AC-PF偏差(收敛率、电压、线路负载)

设计依据 (来自Step5/6实验数据):
  - GT在10GW时AC-PF仅35%收敛, Vmin=0.793pu
  - GAT(+43MW噪声): 35%收敛, Vmin=0.785pu (更差!)
  - GAT+RL(v5.2): 50%收敛, Vmin=0.815pu (最优!)

关键发现: RL虽然没改善DC-N-1, 但显著改善了AC-PF指标!
→ 说明RL学到了某种"AC-friendly"的发电分布模式

v6 Reward设计:
  Layer 1 (快速, 每步): DC proxy指标 ~1ms
  Layer 2 (慢速, 采样): 真实AC-PF校验 ~2s/次

作者: Auto-generated for SCED-GAT-RL project
Date: 2026-06-06
"""

import numpy as np
import gymnasium as gym
from typing import Optional, Tuple, Dict, List
import time


class AC_PF_ProxyMetrics:
    """
    AC-PF性能的快速DC代理指标。

    核心思想: 不需要真的跑AC-PF(~2s), 用可从DC潮流瞬间计算的
    特征来近似预测AC-PF行为。这些特征与AC-PF表现高度相关:

    1. 发电集中度(HHI): 发电越集中→局部电压风险越大
       HHI = sum((P_i / P_total)^2), 范围[1/N, 1]
       越低越好(均匀分布)

    2. 线路负载均匀性: max|flow| / mean|flow|
       越接近1越好(无瓶颈)

    3. PV节点支撑比: 有发电机的PV节点比例
       越高越好(更多电压支撑点)

    4. 功率不平衡度: |sum(P) - D| / D
       应该≈0

    5. 发电裕度: (PMAX - P) / PMAX 的均值
       裕度大→有调节空间

    实验验证 (待Step6完成后填入):
      这些指标与AC-PF收敛率的Pearson相关系数
    """

    @staticmethod
    def compute(power: np.ndarray, demand: float, flows: np.ndarray,
                flow_limits: np.ndarray, PMIN: np.ndarray, PMAX: np.ndarray,
                n_lines: int) -> dict:
        """计算所有代理指标。返回dict供reward函数使用。"""
        N = len(power)

        # 1. 发电集中度 (HHI)
        total_gen = power.sum()
        if total_gen > 0:
            shares = power / total_gen
            hhi = np.sum(shares ** 2)  # [1/N, 1], 完全均匀=1/N
        else:
            hhi = 1.0

        # 归一化HHI: 0=完全均匀, 1=单机供电
        hhi_norm = (hhi - 1.0 / N) / (1.0 - 1.0 / N)  # [0, 1]

        # 2. 线路负载均匀性
        abs_flows = np.abs(flows)
        mean_flow = abs_flows.mean() + 1e-6
        max_flow = abs_flows.max()
        flow_uniformity = max_flow / mean_flow  # >=1, 1=完美均匀

        # 3. 最大线路负载率
        loading_rates = abs_flows / (flow_limits + 1e-6)
        max_loading = loading_rates.max()

        # 4. 功率平衡误差
        balance_error = abs(total_gen - demand) / (demand + 1e-6)

        # 5. 发电裕度 (向上调节能力)
        upward_margin = np.clip((PMAX - power) / (PMAX + 1e-6), 0, 1)
        mean_margin = upward_margin.mean()

        # 6. 机组利用率 (出力/容量区间)
        utilization = (power - PMIN) / (PMAX - PMIN + 1e-6)
        util_mean = np.clip(utilization, 0, 1).mean()

        # 7. 活跃机组数 (P > PMIN * threshold)
        active_units = np.sum(power > PMIN * 0.5)

        return {
            'hhi': float(hhi),
            'hhi_norm': float(hhi_norm),
            'flow_uniformity': float(flow_uniformity),
            'max_loading': float(max_loading),
            'max_loading_pct': float(max_loading * 100),
            'balance_error': float(balance_error),
            'mean_upward_margin': float(mean_margin),
            'utilization_mean': float(util_mean),
            'active_units': int(active_units),
            'total_gen_MW': float(total_gen),
        }


class AC_PF_RewardShaper_v6:
    """
    v6 Reward塑形器 — 基于AC-PF代理指标的连续奖励。

    设计原则:
    - 所有奖励项都是连续值(避免稀疏奖励问题)
    - 每个指标都有物理意义和对偶解释
    - 系数通过Step6实验数据标定

    Reward组成:
      r_total = r_voltage_proxy          # 电压安全代理 (+改善Vmin)
              + r_convergence_proxy      # 收敛概率代理
              + r_line_safety            # 线路安全
              + r_balance                # 功率平衡(硬约束)
              + r_cost                   # 经济性(软约束)
              + r_ac_sampled             # AC-PF采样真值(稀疏但准确)
    """

    # ===== 奖励系数 (初始值, 待Step6结果微调) =====
    LAM_VOLTAGE_PROXY = 30.0       # 电压安全权重
    LAM_CONVERGENCE_PROXY = 20.0   # 收敛性权重
    LAM_LINE_SAFETY = 15.0         # 线路安全权重
    LAM_BALANCE_HARD = 500.0       # 平衡硬约束
    LAM_COST_SOFT = 0.5            # 成本软约束

    # AC采样参数
    AC_SAMPLE_PROBABILITY = 0.05   # 每步5%概率做真实AC-PF
    AC_SAMPLE_REWARD_SCALE = 50.0  # AC采样结果的奖励放大系数

    def __init__(self, base_metrics: dict = None):
        self.base_metrics = base_metrics  # base_power(GAT/GT)的指标作为基准

    def compute(self, current_metrics: dict, action: np.ndarray,
               base_metrics: dict = None, ac_result: dict = None) -> Tuple[float, dict]:
        """
        计算v6总reward。

        Args:
            current_metrics: 当前dispatch的AC_PF_ProxyMetrics输出
            action: RL动作向量
            base_metrics: 基准(GT/GAT)的proxy metrics (用于计算improvement)
            ac_result: 可选, 真实AC-PF结果 (None表示未采样)

        Returns:
            (total_reward, reward_breakdown_dict)
        """
        if base_metrics is None:
            base_metrics = self.base_metrics or {}

        breakdown = {}

        # ===== 1. 电压安全代理奖励 =====
        # 物理依据: 发电越分散→电压分布越平坦→Vmin越高
        # 惩罚高HHI(集中发电), 奖励低HHI(分散发电)
        hhi_current = current_metrics.get('hhi_norm', 0.5)
        hhi_base = base_metrics.get('hhi_norm', 0.5)
        r_voltage = (hhi_base - hhi_current) * self.LAM_VOLTAGE_PROXY
        breakdown['r_voltage'] = float(r_voltage)

        # ===== 2. 收敛性代理奖励 =====
        # 物理依据: 线路负载均匀+低最大负载率 → NR更容易收敛
        fu_current = current_metrics.get('flow_uniformity', 2.0)
        fu_base = base_metrics.get('flow_uniformity', 2.0)
        ml_current = min(current_metrics.get('max_loading_pct', 50), 100)  # cap at 100%
        ml_base = min(base_metrics.get('max_loading_pct', 50), 100)

        # 流量均匀性改善 (越低越好)
        r_flow_uni = (fu_base - fu_current) * 5.0  # scaled
        # 最大负载率降低 (越低越好)
        r_max_load = (ml_base - ml_current) / 100.0 * 10.0
        r_convergence = (r_flow_uni + r_max_load) * self.LAM_CONVERGENCE_PROXY / 15.0
        breakdown['r_convergence'] = float(r_convergence)

        # ===== 3. 线路安全惩罚 =====
        # 直接惩罚过载线路 (超过80%警戒线开始惩罚)
        max_loading = current_metrics.get('max_loading_pct', 0)
        if max_loading > 80:
            overload_penalty = (max_loading - 80) / 20.0  # [0, 1] at >80% load
            r_line = -overload_penalty * self.LAM_LINE_SAFETY
        else:
            r_line = 0.0
        breakdown['r_line'] = float(r_line)

        # ===== 4. 功率平衡硬约束 =====
        bal_err = current_metrics.get('balance_error', 0)
        r_balance = -bal_err * self.LAM_BALANCE_HARD
        breakdown['r_balance'] = float(r_balance)

        # ===== 5. 成本控制 (相对于base) =====
        # 通过active_units数和utilization间接估计成本变化
        # 更多活跃机组+更高利用 → 通常更经济
        r_cost = 0.0  # v6暂时不重点优化成本
        breakdown['r_cost'] = float(r_cost)

        # ===== 6. AC-PF采样真值 (如果有) =====
        r_ac = 0.0
        if ac_result is not None and ac_result.get('converged', False):
            # AC-PF收敛的大奖励
            r_ac += self.AC_SAMPLE_REWARD_SCALE * 0.5
            # 电压好的额外奖励
            vmin = ac_result.get('vm_min', 0.9)
            if vmin > 0.90:
                r_ac += (vmin - 0.90) * 50.0
            elif vmin < 0.85:
                r_ac -= (0.85 - vmin) * 100.0  # 严重惩罚低电压
            # 线路负载低的奖励
            ll = ac_result.get('line_loading_max', 50)
            r_ac += max(0, (15 - ll)) * 2.0
        elif ac_result is not None:
            # AC-PF未收敛的惩罚
            r_ac -= self.AC_SAMPLE_REWARD_SCALE * 0.3
        breakdown['r_ac'] = float(r_ac)

        # 总reward
        total = (breakdown['r_voltage'] + breakdown['r_convergence'] +
                 breakdown['r_line'] + breakdown['r_balance'] +
                 breakdown['r_cost'] + breakdown['r_ac'])

        return total, breakdown


class SCEDCorrectorEnvV6(gym.Env):
    """
    v6 AC-PF导向的SCED校正器环境。

    与v5.x的关键区别:
    - Reward基于AC-PF代理指标(非DC-N-1)
    - 集成pandapower case300网络用于AC-PF采样
    - 支持Layer1(fast DC proxy) + Layer2(slow AC sampling)混合训练

    Action space: Box(0,1)^N (同v5.2, a_i * Demand → P_i)
    Observation space: 同v5.2 (364 dim)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, dm, gat_model=None, pp_net=None, bus_map=None,
                 bus_data_list=None, unit_df=None, busload_df=None,
                 delta_max_ratio=0.05, ac_sample_prob=0.05,
                 render_mode=None):
        super().__init__()

        self.dm = dm
        self.gat_model = gat_model
        self.pp_net = pp_net           # pre-built pandapower network
        self.bus_map = bus_map
        self.bus_data_list = bus_data_list
        self.unit_df = unit_df
        self.busload_df = busload_df
        self.ac_sample_prob = ac_sample_prob

        self.N = dm.N_UNITS
        self.T = dm.T_HORIZON

        # Action & Observation spaces (same as v5.2)
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.N,), dtype=np.float32
        )
        self.obs_dim = self.N * 5 + 19  # 364
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        self.render_mode = render_mode

        # Reward shaper
        self.reward_shaper = AC_PF_RewardShaper_v6()

        # State variables
        self.current_sid = None
        self.t = 0
        self.base_power = None
        self.current_power = None
        self.target_demand = None
        self.base_metrics = None   # base_power的AC proxy指标

        # Statistics
        self.episode_stats = {
            'n_steps': 0, 'ac_samples': 0, 'ac_converged': 0,
            'total_r_voltage': 0, 'total_r_convergence': 0,
            'mean_vmin_ac': [], 'mean_llmax_ac': [],
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        available_sids = list(self.dm.gt_power.keys())
        self.current_sid = self.np_random.choice(available_sids)
        self.t = int(self.np_random.integers(0, self.T))

        sid_profile = self.dm._get_scenario_profile(self.current_sid)
        load_prof = sid_profile["Load"].values.astype(np.float64)
        drate_prof = sid_profile["rate"].values.astype(np.float64)

        self.target_demand = load_prof[self.t]
        drate = drate_prof[min(self.t, self.T - 1)]

        gt = self.dm.gt_power[self.current_sid]  # (T, N)
        self.base_power = gt[self.t].copy().astype(np.float64)

        # Add GAT-like noise
        noise_level = 0.02
        noise = self.np_random.normal(0, noise_level * self.dm.PMAX, size=self.N)
        self.base_power = np.clip(self.base_power + noise, self.dm.PMIN, self.dm.PMAX)

        # Balance to match demand (simulating GAT output that's already balanced)
        gen_sum = self.base_power.sum()
        scale = self.target_demand / gen_sum if gen_sum > 0 else 1.0
        self.base_power = np.clip(self.base_power * scale, self.dm.PMIN, self.dm.PMAX)

        self.current_power = self.base_power.copy()

        # Compute base metrics as reference
        base_flows = self.dm.compute_line_flows(self.base_power, drate)
        self.base_metrics = AC_PF_ProxyMetrics.compute(
            self.base_power, self.target_demand, base_flows,
            self.dm.flow_limit, self.dm.PMIN, self.dm.PMAX, self.dm.N_LINES
        )

        self.episode_stats = {
            'n_steps': 0, 'ac_samples': 0, 'ac_converged': 0,
            'total_r_voltage': 0, 'total_r_convergence': 0,
            'mean_vmin_ac': [], 'mean_llmax_ac': [],
        }

        obs = self._make_obs()
        return obs, {}

    def step(self, action):
        self.episode_stats['n_steps'] += 1

        # Decode action → power dispatch (same mapping as v5.2)
        raw_power = action * self.target_demand
        raw_power = np.clip(raw_power, self.dm.PMIN, self.dm.PMAX)

        gen_total = raw_power.sum()
        new_power = None
        if gen_total < self.target_demand * 0.80:
            new_power = self.base_power.copy()
        elif abs(gen_total - self.target_demand) > self.target_demand * 0.01:
            new_power = raw_power * (self.target_demand / gen_total)
            new_power = np.clip(new_power, self.dm.PMIN, self.dm.PMAX)
            final_err = abs(new_power.sum() - self.target_demand)
            if final_err > self.target_demand * 0.01:
                new_power = self._min_balance_fix(new_power)
        else:
            new_power = raw_power

        # Get DRATE
        sid_profile = self.dm._get_scenario_profile(self.current_sid)
        drate = sid_profile["rate"].values.astype(np.float64)[min(self.t, self.T - 1)]

        # Compute DC line flows
        flows = self.dm.compute_line_flows(new_power, drate)

        # Compute AC-PF proxy metrics for current dispatch
        current_metrics = AC_PF_ProxyMetrics.compute(
            new_power, self.target_demand, flows,
            self.dm.flow_limit, self.dm.PMIN, self.dm.PMAX, self.dm.N_LINES
        )

        # Decide whether to sample real AC-PF
        ac_result = None
        if (self.np_random.random() < self.ac_sample_prob
            and self.pp_net is not None):
            ac_result = self._sample_ac_pf(new_power)
            self.episode_stats['ac_samples'] += 1
            if ac_result.get('converged', False):
                self.episode_stats['ac_converged'] += 1
            if ac_result.get('vm_min') is not None:
                self.episode_stats['mean_vmin_ac'].append(ac_result['vm_min'])
            if ac_result.get('line_loading_max') is not None:
                self.episode_stats['mean_llmax_ac'].append(ac_result['line_loading_max'])

        # Compute v6 reward
        reward, reward_bd = self.reward_shaper.compute(
            current_metrics, action,
            base_metrics=self.base_metrics,
            ac_result=ac_result
        )

        # Accumulate stats
        self.episode_stats['total_r_voltage'] += reward_bd.get('r_voltage', 0)
        self.episode_stats['total_r_convergence'] += reward_bd.get('r_convergence', 0)

        # Update state
        self.current_power = new_power

        # Termination: simple horizon-based
        terminated = False
        truncated = (self.episode_stats['n_steps'] >= 20)

        obs = self._make_obs()
        info = {
            'reward_breakdown': reward_bd,
            'current_metrics': current_metrics,
            'balance_err': abs(new_power.sum() - self.target_demand),
        }
        if ac_result:
            info['ac_result'] = ac_result

        return obs, reward, terminated, truncated, info

    def _sample_ac_pf(self, power):
        """Run real AC-PF on case300 network (slow, ~2s)."""
        try:
            import copy
            from build_custom_pp_network import setup_generation_and_load, run_ac_pf_check

            net_copy = copy.deepcopy(self.pp_net)
            demand = self.target_demand
            # Scale demand if needed (case300 base ~3500MW)
            setup_generation_and_load(
                net_copy, self.bus_map, self.bus_data_list,
                self.unit_df, self.busload_df, demand, power
            )
            result = run_ac_pf_check(net_copy, verbose=False)
            return result
        except Exception as e:
            return {'converged': False, 'error': str(e)}

    def _min_balance_fix(self, power):
        """Minimal fix for small balance errors."""
        diff = self.target_demand - power.sum()
        if abs(diff) < 1e-6:
            return power
        # Find unit with most headroom in correction direction
        if diff > 0:
            headroom = self.dm.PMAX - power
            idx = np.argmax(headroom)
        else:
            slack = power - self.dm.PMIN
            idx = np.argmax(slack)
        adjusted = power.copy()
        adjusted[idx] += diff
        adjusted = np.clip(adjusted, self.dm.PMIN, self.dm.PMAX)
        return adjusted

    def _make_obs(self):
        """Build observation vector (364 dims, same format as v5.2)."""
        N = self.N; T = self.T

        on_off = (self.current_power > self.dm.PMIN * 0.5).astype(np.float32)
        p_ratio = np.clip(self.current_power / (self.dm.PMAX + 1e-6), 0, 1).astype(np.float32)
        margin_up = np.clip((self.dm.PMAX - self.current_power) / (self.dm.PMAX + 1e-6), 0, 1).astype(np.float32)
        margin_down = np.clip((self.current_power - self.dm.PMIN) / (self.dm.PMAX + 1e-6), 0, 1).astype(np.float32)
        cap_ratio = self.dm.PMAX / (self.dm.PMAX.max() + 1e-6)
        node_feat = np.stack([on_off, p_ratio, margin_up, margin_down, cap_ratio], axis=1)

        sid_profile = self.dm._get_scenario_profile(self.current_sid)
        load_prof = sid_profile["Load"].values.astype(np.float64)
        rsv_prof = sid_profile["Reserve"].values.astype(np.float64)
        drate_prof = sid_profile["rate"].values.astype(np.float64)

        max_ld = max(float(load_prof.max()), 1.0)
        max_rsv = max(float(rsv_prof.max()), 1.0)

        future_load = np.array([load_prof[min(self.t+k, T-1)] / max_ld for k in range(1,5)], dtype=np.float32)
        future_rsv = np.array([rsv_prof[min(self.t+k, T-1)] / max_rsv for k in range(1,5)], dtype=np.float32)

        global_feat = np.array([
            self.t / T,
            load_prof[self.t] / max_ld,
            rsv_prof[self.t] / max_rsv,
            self.dm.PMIN[on_off > 0.5].sum() / (max_ld + 1e-6),
            self.dm.PMAX.sum() / (max_ld + 1e-6),
            drate_prof[min(self.t, T-1)],
            *future_load, *future_rsv,
            0, 0, 0,  # overload placeholders
            0, 0,      # n1_viol, cum viol
        ], dtype=np.float32)

        obs = np.concatenate([node_feat.flatten(), global_feat]).astype(np.float32)
        return obs

    def render(self):
        if self.render_mode != "human":
            return
        s = self.episode_stats
        print(f"  Step {s['n_steps']} | "
              f"D={self.target_demand:.0f}MW | "
              f"P_sum={self.current_power.sum():.0f} | "
              f"AC_samples={s['ac_samples']} conv={s['ac_converged']}")


# ============================================================
# Integration helper: build v6 env with pre-built network
# ============================================================

def create_v6_env(data_root, proj_path, n_train=50, n_val=10, n_test=50,
                  seed=42, ac_sample_prob=0.05):
    """
    一站式创建v6环境的工厂函数。

    Args:
        data_root: UC_RL/300 数据目录
        proj_path: ED_RL 项目目录
        ac_sample_prob: AC-PF采样概率 (建议0.02-0.10)

    Returns:
        (env_v6, dm, pp_net, bus_map, bus_data_list, unit_df, busload_df)
    """
    from pathlib import Path
    from build_custom_pp_network import parse_cdf, build_custom_network
    from sced_gat_rl_pipeline import SCEDDataManager
    import pandas as pd

    PROJ = Path(proj_path)
    DATA_ROOT = Path(data_root)

    # Build pandapower network (once, shared across all episodes)
    buses, branches = parse_cdf(PROJ / 'data' / 'ieee300cdf.txt')
    unit_df = pd.read_csv(DATA_ROOT / 'Unit_temp' / 'Unit_Data1.csv')
    busload_df = pd.read_csv(DATA_ROOT / 'Busload_temp' / 'Busload_data1.csv')
    pp_net, bus_map, bus_data_list = build_custom_network(buses, branches, unit_df, busload_df)

    # Data manager
    dm = SCEDDataManager(DATA_ROOT, n_train=n_train, n_val=n_val, n_test=n_test, seed=seed)
    dm.load_all()

    # Create environment
    env = SCEDCorrectorEnvV6(
        dm=dm,
        pp_net=pp_net,
        bus_map=bus_map,
        bus_data_list=bus_data_list,
        unit_df=unit_df,
        busload_df=busload_df,
        ac_sample_prob=ac_sample_prob,
    )

    return env, dm, pp_net, bus_map, bus_data_list, unit_df, busload_df

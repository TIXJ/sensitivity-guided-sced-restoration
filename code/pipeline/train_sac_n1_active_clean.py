"""Clean SAC trainer for the fixed-basis active N-1 Route-B task.

The implementation intentionally contains only SAC. It keeps the validated
training structure from the submission snapshot and the corrected contracts
required by the tightened-N-1 dataset:

* projected GAT dispatch as the reset point;
* one fixed 12-shift + 20-key-generator PTDF/LODF action basis;
* orthogonal active-power targets and fixed per-device voltage controls;
* strict 50-step episode boundaries in replay;
* event-triggered AC-PF and per-step linear N-1 feedback;
* common-verifier acceptance that retains the best joint N-1/voltage state;
* a compact potential-based reward that never reads rescue labels.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--episodes-per-update", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--gradient-steps", type=int, default=30)
    parser.add_argument("--learning-starts", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-interval", type=int, default=20)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--initial-alpha", type=float, default=0.01)
    parser.add_argument(
        "--auto-alpha", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--actor-update-every", type=int, default=1)
    parser.add_argument("--lr-decay-start", type=int, default=100)
    parser.add_argument("--actor-lr-final-factor", type=float, default=1.0)
    parser.add_argument("--critic-lr-final-factor", type=float, default=1.0)
    parser.add_argument("--log-std-init", type=float, default=-3.0)
    parser.add_argument("--log-std-max", type=float, default=-2.0)
    parser.add_argument("--active-log-std-max", type=float, default=-3.5)
    parser.add_argument("--voltage-log-std-max", type=float, default=-3.0)
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=0.01,
        help=(
            "Replay-only reward scale. Reported physical rewards are unchanged; "
            "the scaled targets keep the critic in a numerically stable range."
        ),
    )
    parser.add_argument("--n1-potential-weight", type=float, default=300.0)
    parser.add_argument("--n1-progress-weight", type=float, default=600.0)
    parser.add_argument("--voltage-progress-weight", type=float, default=4000.0)
    parser.add_argument(
        "--n1-secure-bonus",
        type=float,
        default=0.0,
        help=(
            "Intermediate N-1-only bonus. It is zero by default because a "
            "positive jackpot before voltage feasibility reinforced saturated "
            "voltage actions in the joint task."
        ),
    )
    parser.add_argument("--n1-residual-weight", type=float, default=50.0)
    parser.add_argument("--voltage-sum-weight", type=float, default=5000.0)
    parser.add_argument("--voltage-max-weight", type=float, default=10000.0)
    parser.add_argument("--active-target-range-mw", type=float, default=420.0)
    parser.add_argument(
        "--voltage-target-range-pu",
        type=float,
        default=0.015,
        help=(
            "Absolute voltage-residual range. The active-required dataset "
            "needs at most 0.0141 pu in its independently verified rescues."
        ),
    )
    parser.add_argument(
        "--voltage-step-pu",
        type=float,
        default=0.002,
        help="Fixed per-step voltage range used by all 68 generators and the reference source.",
    )
    parser.add_argument(
        "--voltage-groups",
        type=int,
        default=68,
        help="Compatibility option; the clean contract always uses all 68 generators.",
    )
    parser.add_argument("--residual-scale", type=float, default=20.0)
    parser.add_argument("--cumulative-scale-mult", type=float, default=20.0)
    parser.add_argument("--ac-pf-check-every", type=int, default=5)
    parser.add_argument("--ac-event-n1-threshold-mw", type=float, default=5.0)
    parser.add_argument("--n1-check-every", type=int, default=5)
    parser.add_argument(
        "--dataset",
        default="case300_gt_n1s0575_active_required_fixed32.json",
    )
    parser.add_argument(
        "--gat-checkpoint",
        default="gatv10_results_n1s0575_candidate/gatv10_predictor_best.pth",
    )
    parser.add_argument(
        "--split-file",
        default="paper_results/case300_n1s0575_active_fixed32_split.json",
    )
    parser.add_argument(
        "--output-dir",
        default="rl_sac_n1_active_fixed101_v24_repro",
    )
    parser.add_argument("--final-test-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Build the full environment and execute one neutral step without training.",
    )
    return parser.parse_args()


ARGS = parse_args()
PROJECT = Path(__file__).resolve().parents[2]
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(1, str(PROJECT))


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (PROJECT / candidate).resolve()


DATASET_PATH = resolve(ARGS.dataset)
CHECKPOINT_PATH = resolve(ARGS.gat_checkpoint)
SPLIT_PATH = resolve(ARGS.split_file)
OUTPUT_DIR = resolve(ARGS.output_dir)
os.environ["ROUTE_B_GAT_CHECKPOINT"] = str(CHECKPOINT_PATH)

random.seed(ARGS.seed)
np.random.seed(ARGS.seed)
torch.manual_seed(ARGS.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(ARGS.seed)

import train_rlv35_complete as core

# Set every shared SAC/environment constant once, before constructing objects.
core.AC_PF_CHECK_EVERY = max(1, ARGS.ac_pf_check_every)
core.AC_EVENT_N1_OVERLOAD_MW = max(
    0.01, float(ARGS.ac_event_n1_threshold_mw)
)
core.N1_CHECK_EVERY_STEPS = max(1, ARGS.n1_check_every)
core.MAX_STEPS = ARGS.max_steps
core.MAX_STEPS_HIGH = ARGS.max_steps
core.N_GRADIENT_STEPS = ARGS.gradient_steps
core.N_EPISODES_PER_UPDATE = ARGS.episodes_per_update
core.BATCH_SIZE = ARGS.batch_size
core.SAC_LR_ACTOR = ARGS.actor_lr
core.SAC_LR_CRITIC = ARGS.critic_lr
core.SAC_LR_ALPHA = ARGS.alpha_lr
core.SAC_INIT_ALPHA = ARGS.initial_alpha
core.SAC_AUTO_ALPHA = ARGS.auto_alpha
core.SAC_LOG_STD_MAX = ARGS.log_std_max
core.SAC_LOG_STD_MAX_VECTOR = None
core.UNIFORM_RESIDUAL_MW = ARGS.residual_scale
core.UNIFORM_CUMULATIVE_MULT = ARGS.cumulative_scale_mult
core.ENFORCE_REF_SOURCE_LIMIT = True
if int(ARGS.voltage_groups) != core.N_GEN:
    raise ValueError(
        f"The fixed voltage-action contract requires --voltage-groups {core.N_GEN}"
    )
core.N_VOLTAGE_GROUPS = core.N_GEN
core.VOLTAGE_STEP_PU = float(ARGS.voltage_step_pu)


def n1_potential(overload_mw: float) -> float:
    return float(np.log1p(max(float(overload_mw), 0.0) / 10.0))


def voltage_potential_from_gaps(gaps) -> float:
    magnitudes = np.abs(np.asarray(gaps, dtype=np.float64).reshape(-1))
    if magnitudes.size == 0:
        return 0.0
    return float(np.sum(magnitudes) + np.max(magnitudes))


class BranchedGaussianActor(nn.Module):
    """SAC actor with independent feature paths and fixed action semantics.

    The 32 active-power coordinates multiply an orthonormal generator-space
    basis.  Radial squashing therefore gives the whole redispatch vector one
    physical L2 budget instead of incorrectly granting that budget to every
    coordinate. The voltage coordinates use the same construction: all
    fixed per-device voltage controls use the same independent incremental
    bounds as the convergent submission implementation.
    """

    def __init__(self, obs_dim: int, p_dim: int, v_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.p_dim = int(p_dim)
        self.v_dim = int(v_dim)
        self.action_dim = self.p_dim + self.v_dim

        def trunk():
            return nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

        self.p_trunk = trunk()
        self.v_trunk = trunk()
        self.p_mean_head = nn.Linear(hidden_dim, self.p_dim)
        self.v_mean_head = nn.Linear(hidden_dim, self.v_dim)
        self.p_log_std_head = nn.Linear(hidden_dim, self.p_dim)
        self.v_log_std_head = nn.Linear(hidden_dim, self.v_dim)
        for head in (
            self.p_mean_head, self.v_mean_head,
            self.p_log_std_head, self.v_log_std_head,
        ):
            nn.init.zeros_(head.weight)
        nn.init.zeros_(self.p_mean_head.bias)
        nn.init.zeros_(self.v_mean_head.bias)
        nn.init.constant_(self.p_log_std_head.bias, -3.0)
        nn.init.constant_(self.v_log_std_head.bias, -3.0)

    def forward(self, obs):
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        p_hidden = self.p_trunk(obs)
        v_hidden = self.v_trunk(obs)
        p_mean = self.p_mean_head(p_hidden)
        v_mean = self.v_mean_head(v_hidden)
        mean = torch.cat([p_mean, v_mean], dim=-1)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)

        p_log_std = self.p_log_std_head(p_hidden)
        v_log_std = self.v_log_std_head(v_hidden)
        log_std = torch.cat([p_log_std, v_log_std], dim=-1)
        log_std = torch.nan_to_num(
            log_std, nan=-3.0, posinf=0.0, neginf=-5.0
        ).clamp(min=-5.0)
        max_vector = torch.as_tensor(
            core.SAC_LOG_STD_MAX_VECTOR,
            dtype=log_std.dtype,
            device=log_std.device,
        )
        log_std = torch.minimum(log_std, max_vector.view(1, -1))
        return mean, log_std

    @staticmethod
    def _radial_squash(latent):
        radius = torch.linalg.vector_norm(latent, dim=-1, keepdim=True)
        radial_scale = torch.where(
            radius > 1e-6,
            torch.tanh(radius) / radius.clamp_min(1e-6),
            torch.ones_like(radius),
        )
        action = latent * radial_scale
        output_radius = torch.tanh(radius)
        radial_derivative = 1.0 - output_radius.pow(2)
        tangential_ratio = torch.where(
            radius > 1e-6,
            output_radius / radius.clamp_min(1e-6),
            torch.ones_like(radius),
        )
        log_det = torch.log(radial_derivative + 1e-6)
        dim = int(latent.shape[-1])
        if dim > 1:
            log_det = log_det + (dim - 1) * torch.log(
                tangential_ratio + 1e-6
            )
        return action, log_det

    def _squash(self, latent):
        p_action, p_log_det = self._radial_squash(
            latent[..., :self.p_dim]
        )
        v_action = torch.tanh(latent[..., self.p_dim:])
        v_log_det = torch.log(1.0 - v_action.pow(2) + 1e-6).sum(
            dim=-1, keepdim=True
        )
        return torch.cat([p_action, v_action], dim=-1), p_log_det, v_log_det

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        normal = torch.distributions.Normal(mean, log_std.exp())
        z = normal.rsample()
        action, p_log_det, v_log_det = self._squash(z)

        # Exact Jacobians for the two independent radial maps.
        log_prob = normal.log_prob(z).sum(dim=-1, keepdim=True)
        log_prob = log_prob - p_log_det - v_log_det
        return action, log_prob, mean

    def select_action(self, obs, deterministic=False):
        with torch.no_grad():
            mean, log_std = self.forward(obs)
            if deterministic:
                action, _p_log_det, _v_log_det = self._squash(mean)
                return action
            normal = torch.distributions.Normal(mean, log_std.exp())
            action, _p_log_det, _v_log_det = self._squash(normal.sample())
            return action


class CleanN1Env(core.SCEDEnvV21):
    """Current physical environment with one auditable training reward."""

    def __init__(self, samples, sample_ids, dataset_meta):
        super().__init__(
            samples,
            sample_ids,
            dataset_meta=dataset_meta,
            action_encoding="physics32orth",
            action_basis_seed=20260721,
            voltage_action_mode="incremental",
            voltage_absolute_range_pu=ARGS.voltage_target_range_pu,
            observation_mode="normalized_compact",
            generator_upper_margin=0.99,
            include_bus_voltage_feedback=True,
            initial_acpf_on_reset=True,
        )
        if self._n_voltage_groups != self.N:
            raise RuntimeError("Fixed voltage actions must map one-to-one to all generators")
        self._voltage_group_indices = [
            np.asarray([idx], dtype=int) for idx in range(self.N)
        ]

    def _compute_n1_sensitivity(self):
        step = int(getattr(self, "corr_step", 0))
        refresh = bool(
            not hasattr(self, "_clean_n1_sensitivity_cache")
            or step <= 1
            or step % max(1, int(ARGS.n1_check_every)) == 0
        )
        if refresh:
            sensitivity, overload, violations = super()._compute_n1_sensitivity()
            self._clean_n1_sensitivity_cache = (
                np.asarray(sensitivity, dtype=np.float32).copy(),
                float(overload),
                int(violations),
            )
        sensitivity, overload, violations = self._clean_n1_sensitivity_cache
        return sensitivity.copy(), float(overload), int(violations)

    def reset(self, sample_idx=None):
        if hasattr(self, "_clean_n1_sensitivity_cache"):
            del self._clean_n1_sensitivity_cache
        obs = super().reset(sample_idx=sample_idx)
        self._clean_prev_n1_potential = n1_potential(
            getattr(self, "_n1_overload_mw", 0.0)
        )
        self._clean_prev_voltage_potential = voltage_potential_from_gaps(
            getattr(self, "_bus_v_gap_pu", np.zeros(0, dtype=np.float64))
        )
        self._clean_prev_n1_secure = bool(
            int(getattr(self, "_n1_violations", 0)) == 0
            and float(getattr(self, "_n1_overload_mw", 0.0)) < 0.01
        )
        self._clean_n1_bonus_awarded = self._clean_prev_n1_secure
        return obs

    def step(self, action):
        policy_action = np.asarray(action, dtype=np.float64).copy()
        p_dim = self._n_shift_dirs + self._n_key_gens
        nominal_scale = max(float(self.current_residual_scale), 1e-8)
        target_range = max(float(ARGS.active_target_range_mw), nominal_scale)

        low, high = self.get_action_bounds()
        normalized_p_target = policy_action[:p_dim] / nominal_scale
        target_fraction = float(np.linalg.norm(normalized_p_target, ord=2))
        voltage_action_scale = max(float(high[p_dim]), 1e-8)
        voltage_target_fraction = float(
            np.max(np.abs(policy_action[p_dim:])) / voltage_action_scale
        )
        voltage_target_signed = float(np.mean(
            policy_action[p_dim:] / voltage_action_scale
        ))

        # Fixed contract: 32-D absolute P target plus one incremental voltage
        # action for each generator and one for the reference source.
        decoded_action = policy_action.copy()
        decoded_action[:p_dim] *= target_range / nominal_scale
        voltage_residual_limit = float(ARGS.voltage_target_range_pu)
        gen_residual = self.current_gen_vm_pu - self.base_gen_vm_pu
        decoded_action[p_dim:p_dim + self.N] = np.clip(
            decoded_action[p_dim:p_dim + self.N],
            -voltage_residual_limit - gen_residual,
            voltage_residual_limit - gen_residual,
        )
        ext_residual = float(
            self.current_ext_grid_vm_pu[0] - self.base_ext_grid_vm_pu[0]
        )
        decoded_action[-1] = float(np.clip(
            decoded_action[-1],
            -voltage_residual_limit - ext_residual,
            voltage_residual_limit - ext_residual,
        ))
        self.cumulative_delta.fill(0.0)
        old_scale = self.current_residual_scale
        old_cumulative_mult = core.UNIFORM_CUMULATIVE_MULT
        self.current_residual_scale = target_range
        core.UNIFORM_CUMULATIVE_MULT = 1.0
        try:
            _obs, reward, terminated, truncated, info = super().step(decoded_action)
        finally:
            self.current_residual_scale = old_scale
            core.UNIFORM_CUMULATIVE_MULT = old_cumulative_mult

        obs = np.asarray(_obs, dtype=np.float32).copy()
        executed_raw = np.divide(
            policy_action,
            np.where(np.abs(high) > 1e-12, high, 1.0),
        )
        executed_raw[p_dim:] = decoded_action[p_dim:] / voltage_action_scale
        info["clean_executed_raw_action"] = np.clip(
            executed_raw, -1.0, 1.0
        ).astype(float).tolist()
        info["clean_phase_before"] = "joint"
        info["clean_phase_after"] = "joint"
        info["clean_active_target_fraction"] = target_fraction
        info["clean_active_target_l2_mw"] = float(
            target_fraction * target_range
        )
        info["clean_voltage_target_fraction"] = voltage_target_fraction
        info["clean_voltage_target_signed"] = voltage_target_signed
        info["clean_gen_vm_min_pu"] = float(np.min(self.current_gen_vm_pu))
        info["clean_gen_vm_max_pu"] = float(np.max(self.current_gen_vm_pu))
        info["clean_gen_vm_touch_count"] = int(np.count_nonzero(
            (self.current_gen_vm_pu <= core.GEN_VM_MIN + 1e-8)
            | (self.current_gen_vm_pu >= core.GEN_VM_MAX - 1e-8)
        ))
        info["clean_ext_vm_pu"] = float(self.current_ext_grid_vm_pu[0])
        info["clean_active_target_range_mw"] = target_range
        info["clean_voltage_generator_indices"] = list(range(self.N))
        return obs, reward, terminated, truncated, info

    def _compute_reward(self):
        _legacy_reward, done, info = super()._compute_reward()
        overload = float(info.get("n1_overload_mw", 0.0))
        potential = n1_potential(overload)
        previous = float(getattr(self, "_clean_prev_n1_potential", potential))
        progress = float(np.clip(previous - potential, -2.0, 2.0))
        self._clean_prev_n1_potential = potential
        voltage_potential = float(
            getattr(self, "_clean_prev_voltage_potential", 0.0)
        )
        voltage_progress = 0.0
        n1_secure = bool(
            int(info.get("n1_violations", 0)) == 0 and overload < 0.01
        )
        entered_n1_secure = bool(
            n1_secure and not getattr(self, "_clean_prev_n1_secure", False)
        )
        award_n1_secure = bool(
            entered_n1_secure
            and not getattr(self, "_clean_n1_bonus_awarded", False)
        )
        if award_n1_secure:
            self._clean_n1_bonus_awarded = True
        self._clean_prev_n1_secure = n1_secure

        reward_phase = getattr(self, "_clean_reward_phase", "joint")
        if (
            bool(info.get("ac_checked", False))
            and bool(info.get("converged", False))
        ):
            voltage_sum = float(info.get("v_violation_mag_pu", 0.0))
            voltage_max = float(info.get("v_violation_max_pu", 0.0))
            voltage_potential = voltage_sum + voltage_max
            previous_voltage = float(
                getattr(self, "_clean_prev_voltage_potential", voltage_potential)
            )
            voltage_progress = float(
                np.clip(previous_voltage - voltage_potential, -0.10, 0.10)
            )
            self._clean_prev_voltage_potential = voltage_potential

        if done:
            reward = 20000.0
        else:
            reward = (
                -10.0
                - ARGS.n1_potential_weight * potential
                + ARGS.n1_progress_weight * progress
            )
            # The logarithmic potential is useful for large overloads but is
            # too flat in the last few MW. This residual term preserves a
            # continuous distinction between "nearly secure" and the strict
            # 0.01-MW acceptance boundary.
            reward -= ARGS.n1_residual_weight * min(overload, 10.0)
            reward += ARGS.voltage_progress_weight * voltage_progress
            if award_n1_secure:
                reward += ARGS.n1_secure_bonus
            if bool(info.get("ac_checked", False)):
                if not bool(info.get("converged", False)):
                    reward -= 1000.0
                else:
                    voltage_sum = float(info.get("v_violation_mag_pu", 0.0))
                    voltage_max = float(info.get("v_violation_max_pu", 0.0))
                    # Do not flatten the voltage signal after a severe action.
                    # The previous capped penalty made 0.2 pu and 6 pu of total
                    # violation indistinguishable to the critic.
                    reward -= ARGS.voltage_sum_weight * voltage_sum
                    reward -= ARGS.voltage_max_weight * voltage_max
                    reward -= 100.0 * min(
                        float(info.get("overloaded_lines", 0))
                        + float(info.get("trafo_overloaded", 0)),
                        10.0,
                    )
                    ref_p = float(info.get("reference_source_p_mw", 0.0))
                    ref_q = float(info.get("reference_source_q_mvar", 0.0))
                    ref_s = float(info.get("reference_source_s_mva", 0.0))
                    ref_excess = (
                        max(core.REF_SOURCE_P_MIN_MW - ref_p, 0.0)
                        + max(ref_p - core.REF_SOURCE_P_MAX_MW, 0.0)
                        + max(core.REF_SOURCE_Q_MIN_MVAR - ref_q, 0.0)
                        + max(ref_q - core.REF_SOURCE_Q_MAX_MVAR, 0.0)
                        + max(ref_s - core.REF_SOURCE_S_MAX_MVA, 0.0)
                    )
                    reward -= 2.0 * ref_excess

        info["clean_reward"] = float(reward)
        info["clean_n1_potential"] = potential
        info["clean_n1_progress"] = progress
        info["clean_entered_n1_secure"] = entered_n1_secure
        info["clean_awarded_n1_secure"] = award_n1_secure
        info["clean_voltage_potential"] = voltage_potential
        info["clean_voltage_progress"] = voltage_progress
        return float(reward), bool(done), info


def demand_key(sample: dict) -> float:
    return round(float(sample.get("demand_gw", sample["demand_mw"] / 1000.0)), 3)


def stratified_batch(samples: list[dict], ids: list[int], count: int) -> list[int]:
    groups: dict[float, list[int]] = {}
    for idx in ids:
        groups.setdefault(demand_key(samples[idx]), []).append(idx)
    selected = [random.choice(groups[key]) for key in sorted(groups)]
    pool = [idx for key in sorted(groups) for idx in groups[key] if idx not in selected]
    remaining = max(0, count - len(selected))
    if remaining:
        selected.extend(random.sample(pool, min(remaining, len(pool))))
    if len(selected) > count:
        selected = random.sample(selected, count)
    random.shuffle(selected)
    return selected


def scale_action(env: CleanN1Env, normalized_action: np.ndarray) -> np.ndarray:
    low, high = env.get_action_bounds()
    return low + (normalized_action + 1.0) * 0.5 * (high - low)


def is_pass(info: dict) -> bool:
    return bool(
        info.get("reason") == "full_sced_pass"
        or core.adopted_operation_pass(info)
    )


def evaluate(trainer, env: CleanN1Env, ids: list[int]) -> dict:
    passed = 0
    rewards = []
    steps = []
    reasons = Counter()
    n1_final = []
    voltage_final = []
    voltage_low = []
    voltage_high = []
    for sid in ids:
        obs = env.reset(sample_idx=sid)
        total_reward = 0.0
        last_info = {}
        for step in range(env.max_steps):
            raw_action = trainer.select_action(obs, deterministic=True)
            obs, reward, done, _truncated, last_info = env.step(
                scale_action(env, raw_action)
            )
            total_reward += float(reward)
            if done:
                break
        success = is_pass(last_info)
        passed += int(success)
        rewards.append(total_reward if success else min(total_reward, -1.0))
        steps.append(step + 1)
        n1_final.append(float(last_info.get("n1_overload_mw", np.inf)))
        voltage_final.append(float(last_info.get("clean_voltage_potential", np.inf)))
        voltage_low.append(int(last_info.get("v_low_violations", 0)))
        voltage_high.append(int(last_info.get("v_high_violations", 0)))
        if not success:
            reasons[last_info.get("reason", "unknown")] += 1
    return {
        "passed": passed,
        "total": len(ids),
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "avg_steps": float(np.mean(steps)) if steps else 0.0,
        "avg_final_n1_overload_mw": float(np.mean(n1_final)) if n1_final else 0.0,
        "avg_final_voltage_potential_pu": (
            float(np.mean(voltage_final)) if voltage_final else 0.0
        ),
        "avg_final_low_voltage_violations": (
            float(np.mean(voltage_low)) if voltage_low else 0.0
        ),
        "avg_final_high_voltage_violations": (
            float(np.mean(voltage_high)) if voltage_high else 0.0
        ),
        "reasons": dict(reasons),
    }


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    samples = [sample for sample in data["samples"] if sample.get("is_complete_sced")]
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train_ids = [int(idx) for idx in split["train_ids"]]
    val_ids = [int(idx) for idx in split["val_ids"]]
    test_ids = [int(idx) for idx in split["test_ids"]]
    all_ids = train_ids + val_ids + test_ids
    if not all_ids or min(all_ids) < 0 or max(all_ids) >= len(samples):
        raise ValueError("Split indices do not match the selected dataset")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_env = CleanN1Env(samples, train_ids, data)
    val_env = CleanN1Env(samples, val_ids, data)
    obs_dim = train_env._get_obs_dim()
    action_dim = train_env.action_dim
    p_dim = train_env._n_shift_dirs + train_env._n_key_gens
    log_std_max_vector = np.full(
        action_dim, ARGS.voltage_log_std_max, dtype=np.float32
    )
    log_std_max_vector[:p_dim] = ARGS.active_log_std_max
    core.SAC_LOG_STD_MAX_VECTOR = log_std_max_vector
    trainer = core.SACTrainer(obs_dim, action_dim, device=core.DEVICE)
    trainer.actor = BranchedGaussianActor(
        obs_dim,
        p_dim,
        action_dim - p_dim,
    ).to(core.DEVICE)
    trainer.actor_optimizer = torch.optim.Adam(
        trainer.actor.parameters(), lr=ARGS.actor_lr
    )
    log_std_init_vector = np.full(
        action_dim,
        min(ARGS.log_std_init, ARGS.voltage_log_std_max),
        dtype=np.float32,
    )
    log_std_init_vector[:p_dim] = min(
        ARGS.log_std_init, ARGS.active_log_std_max
    )
    with torch.no_grad():
        trainer.actor.p_log_std_head.weight.zero_()
        trainer.actor.v_log_std_head.weight.zero_()
        trainer.actor.p_log_std_head.bias.fill_(float(log_std_init_vector[0]))
        trainer.actor.v_log_std_head.bias.fill_(float(log_std_init_vector[p_dim]))

    config = {
        "dataset": str(DATASET_PATH),
        "gat_checkpoint": str(CHECKPOINT_PATH),
        "split_file": str(SPLIT_PATH),
        "train_val_test": [len(train_ids), len(val_ids), len(test_ids)],
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "action_encoding": "physics32orth",
        "active_action_dim": p_dim,
        "generator_voltage_groups": ARGS.voltage_groups,
        "voltage_action_dim": action_dim - p_dim,
        "voltage_grouping": "fixed_one_action_per_generator",
        "voltage_context_features_per_control": 0,
        "actor_architecture": "independent_active_voltage_trunks",
        "updates": ARGS.updates,
        "episodes_per_update": ARGS.episodes_per_update,
        "max_steps": ARGS.max_steps,
        "gradient_steps": ARGS.gradient_steps,
        "learning_starts": ARGS.learning_starts,
        "actor_lr": ARGS.actor_lr,
        "critic_lr": ARGS.critic_lr,
        "alpha_lr": ARGS.alpha_lr,
        "initial_alpha": ARGS.initial_alpha,
        "automatic_alpha": ARGS.auto_alpha,
        "actor_update_every_critic_steps": ARGS.actor_update_every,
        "lr_decay_start": ARGS.lr_decay_start,
        "actor_lr_final_factor": ARGS.actor_lr_final_factor,
        "critic_lr_final_factor": ARGS.critic_lr_final_factor,
        "phase_gated_action_execution": False,
        "joint_active_voltage_action_execution": True,
        "execution_phases": ["joint"],
        "replay_stores_executed_action": True,
        "log_std_initial_by_branch": {
            "active": float(log_std_init_vector[0]),
            "voltage": float(log_std_init_vector[p_dim]),
        },
        "active_log_std_max": ARGS.active_log_std_max,
        "voltage_log_std_max": ARGS.voltage_log_std_max,
        "reward_scale": ARGS.reward_scale,
        "n1_potential_weight": ARGS.n1_potential_weight,
        "n1_progress_weight": ARGS.n1_progress_weight,
        "voltage_progress_weight": ARGS.voltage_progress_weight,
        "n1_secure_bonus": ARGS.n1_secure_bonus,
        "n1_residual_weight": ARGS.n1_residual_weight,
        "voltage_sum_weight": ARGS.voltage_sum_weight,
        "voltage_max_weight": ARGS.voltage_max_weight,
        "n1_secure_bonus_mode": "first_entry_per_episode",
        "active_target_range_mw": ARGS.active_target_range_mw,
        "active_target_budget": "joint_l2_ball_in_orthonormal_basis",
        "voltage_action_mode": "fixed_per_device_incremental",
        "voltage_step_pu": ARGS.voltage_step_pu,
        "voltage_cumulative_residual_limit_pu": ARGS.voltage_target_range_pu,
        "voltage_target_budget": "independent_per_device_incremental_bounds",
        "candidate_acceptance": "none",
        "voltage_setpoint_bounds_pu": [core.GEN_VM_MIN, core.GEN_VM_MAX],
        "ac_pf_check_every": ARGS.ac_pf_check_every,
        "ac_event_n1_threshold_mw": ARGS.ac_event_n1_threshold_mw,
        "n1_check_every": ARGS.n1_check_every,
        "n1_observation_sensitivity_refresh_every": ARGS.n1_check_every,
        "timeout_is_terminal": True,
        "uses_rescue_labels": False,
    }
    save_json(OUTPUT_DIR / "config.json", config)
    save_json(
        OUTPUT_DIR / "split_ids.json",
        {"train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids},
    )

    print("=" * 70)
    print("Clean SAC: orthogonal physics-basis active N-1 correction")
    print("=" * 70)
    print(f"Dataset: {DATASET_PATH}")
    print(f"GAT checkpoint: {CHECKPOINT_PATH}")
    print(f"ObsDim={obs_dim}, ActDim={action_dim}")
    print(f"Split: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")
    print(
        f"Action dimensions: active={p_dim}, voltage="
        f"{action_dim - p_dim} ({ARGS.voltage_groups} fixed generators "
        "+ reference source)."
    )
    print(
        f"SAC: actor_lr={ARGS.actor_lr:g} critic_lr={ARGS.critic_lr:g} "
        f"alpha0={ARGS.initial_alpha:g} "
        f"auto_alpha={'on' if ARGS.auto_alpha else 'off'} "
        f"std0={np.exp(ARGS.log_std_init):.4f}"
    )
    print(
        f"Active action: shared L2 budget={ARGS.active_target_range_mw:g} MW; "
        f"actor update every {ARGS.actor_update_every} critic steps."
    )
    print(
        f"Voltage action: fixed per-device incremental bound="
        f"+/-{ARGS.voltage_step_pu:g} pu."
    )
    print(
        "P uses one L2 budget; voltage uses fixed per-device increments; no "
        "guide, no freeze, no candidate filter, "
        "and no rescue labels."
    )

    if ARGS.preflight_only:
        obs = train_env.reset(sample_idx=train_ids[0])
        initial_n1 = float(getattr(train_env, "_n1_overload_mw", 0.0))
        neutral = np.zeros(action_dim, dtype=np.float32)
        next_obs, reward, done, _truncated, info = train_env.step(
            scale_action(train_env, neutral)
        )
        checks = {
            "obs_shape_ok": tuple(obs.shape) == (obs_dim,),
            "next_obs_shape_ok": tuple(next_obs.shape) == (obs_dim,),
            "finite_observation": bool(np.isfinite(obs).all() and np.isfinite(next_obs).all()),
            "finite_reward": bool(np.isfinite(reward)),
            "gat_reset": bool(
                train_env.reset_audit.get("initial_power_source")
                == "gat_prediction_projected_to_balance"
                and not train_env.reset_audit.get("uses_dispatch_label_as_initial_power", True)
                and not train_env.reset_audit.get("uses_rescue_as_initial_state", True)
            ),
            "initial_n1_overload_mw": initial_n1,
            "neutral_step_n1_overload_mw": float(info.get("n1_overload_mw", np.nan)),
            "neutral_step_done": bool(done),
            "neutral_step_reason": info.get("reason", "unknown"),
        }
        save_json(OUTPUT_DIR / "preflight.json", checks)
        print(json.dumps(checks, indent=2))
        if not all(checks[key] for key in (
            "obs_shape_ok", "next_obs_shape_ok", "finite_observation", "finite_reward", "gat_reset"
        )):
            raise RuntimeError("Clean SAC preflight failed")
        return

    history = []
    val_history = []
    best_train_rank = (-1, -np.inf)
    best_val_rank = (-1, -np.inf)
    started = time.time()

    for update in range(1, ARGS.updates + 1):
        update_started = time.time()
        episode_ids = stratified_batch(
            samples, train_ids, ARGS.episodes_per_update
        )
        passed = 0
        episode_rewards = []
        episode_dense_rewards = []
        successful_episode_rewards = []
        failed_episode_rewards = []
        episode_steps = []
        initial_n1 = []
        best_n1 = []
        final_n1 = []
        initial_voltage = []
        best_voltage = []
        final_voltage = []
        active_target_fractions = []
        voltage_target_fractions = []
        voltage_target_signed = []
        action_execution_mismatches = []
        final_gen_vm_touch_counts = []
        final_ext_vm = []
        phase_steps = Counter()
        reasons = Counter()

        for sid in episode_ids:
            obs = train_env.reset(sample_idx=sid)
            initial_overload = float(getattr(train_env, "_n1_overload_mw", 0.0))
            episode_best_n1 = initial_overload
            initial_voltage_potential = float(
                getattr(train_env, "_clean_prev_voltage_potential", 0.0)
            )
            episode_best_voltage = initial_voltage_potential
            episode_final_voltage = initial_voltage_potential
            total_reward = 0.0
            last_info = {}

            for step in range(train_env.max_steps):
                raw_action = trainer.select_action(obs, deterministic=False)
                next_obs, reward, done, _truncated, info = train_env.step(
                    scale_action(train_env, raw_action)
                )
                boundary = bool(done or step + 1 >= train_env.max_steps)
                executed_raw_action = np.asarray(
                    info.get("clean_executed_raw_action", raw_action),
                    dtype=np.float32,
                )
                if executed_raw_action.shape != raw_action.shape:
                    raise RuntimeError(
                        "Executed action shape does not match the actor action: "
                        f"{executed_raw_action.shape} != {raw_action.shape}"
                    )
                action_execution_mismatches.append(float(
                    np.mean(np.abs(executed_raw_action - raw_action))
                ))
                trainer.store_transition(
                    obs,
                    executed_raw_action,
                    float(reward) * ARGS.reward_scale,
                    next_obs,
                    boundary,
                )
                obs = next_obs
                last_info = info
                total_reward += float(reward)
                phase_steps[info.get("clean_phase_before", "unknown")] += 1
                active_target_fractions.append(float(
                    info.get("clean_active_target_fraction", 0.0)
                ))
                voltage_target_fractions.append(float(
                    info.get("clean_voltage_target_fraction", 0.0)
                ))
                voltage_target_signed.append(float(
                    info.get("clean_voltage_target_signed", 0.0)
                ))
                episode_best_n1 = min(
                    episode_best_n1,
                    float(info.get("n1_overload_mw", episode_best_n1)),
                )
                if bool(info.get("ac_checked", False)) and bool(info.get("converged", False)):
                    episode_final_voltage = float(
                        info.get("clean_voltage_potential", episode_final_voltage)
                    )
                    episode_best_voltage = min(
                        episode_best_voltage, episode_final_voltage
                    )
                if done:
                    break

            success = is_pass(last_info)
            passed += int(success)
            episode_rewards.append(total_reward if success else min(total_reward, -1.0))
            episode_dense_rewards.append(total_reward)
            if success:
                successful_episode_rewards.append(total_reward)
            else:
                failed_episode_rewards.append(total_reward)
            episode_steps.append(step + 1)
            initial_n1.append(initial_overload)
            best_n1.append(episode_best_n1)
            final_n1.append(float(last_info.get("n1_overload_mw", np.inf)))
            initial_voltage.append(initial_voltage_potential)
            best_voltage.append(episode_best_voltage)
            final_voltage.append(episode_final_voltage)
            final_gen_vm_touch_counts.append(float(
                last_info.get("clean_gen_vm_touch_count", 0)
            ))
            final_ext_vm.append(float(last_info.get("clean_ext_vm_pu", 1.0)))
            if not success:
                reasons[last_info.get("reason", "unknown")] += 1

        update_stats = []
        if len(trainer.replay_buffer) >= ARGS.learning_starts:
            decay_denominator = max(1, ARGS.updates - ARGS.lr_decay_start)
            decay_fraction = float(np.clip(
                (update - ARGS.lr_decay_start) / decay_denominator, 0.0, 1.0
            ))
            actor_lr_now = ARGS.actor_lr * (
                1.0 + decay_fraction * (ARGS.actor_lr_final_factor - 1.0)
            )
            critic_lr_now = ARGS.critic_lr * (
                1.0 + decay_fraction * (ARGS.critic_lr_final_factor - 1.0)
            )
            for group in trainer.actor_optimizer.param_groups:
                group["lr"] = actor_lr_now
            for group in trainer.critic_optimizer.param_groups:
                group["lr"] = critic_lr_now
            actor_interval = max(1, int(ARGS.actor_update_every))
            for gradient_idx in range(ARGS.gradient_steps):
                update_stats.append(trainer.update(
                    update_actor=(gradient_idx % actor_interval == 0)
                ))

        mean_reward = float(np.mean(episode_rewards))
        row = {
            "update": update,
            "passed": passed,
            "total": len(episode_ids),
            "avg_reward": mean_reward,
            "avg_dense_reward": float(np.mean(episode_dense_rewards)),
            "avg_success_episode_reward": (
                float(np.mean(successful_episode_rewards))
                if successful_episode_rewards else None
            ),
            "avg_failed_episode_reward": (
                float(np.mean(failed_episode_rewards))
                if failed_episode_rewards else None
            ),
            "avg_steps": float(np.mean(episode_steps)),
            "avg_initial_n1_mw": float(np.mean(initial_n1)),
            "avg_best_n1_mw": float(np.mean(best_n1)),
            "avg_final_n1_mw": float(np.mean(final_n1)),
            "avg_initial_voltage_pu": float(np.mean(initial_voltage)),
            "avg_best_voltage_pu": float(np.mean(best_voltage)),
            "avg_final_voltage_pu": float(np.mean(final_voltage)),
            "avg_active_target_fraction": float(np.mean(active_target_fractions)),
            "avg_voltage_target_fraction": float(np.mean(voltage_target_fractions)),
            "avg_voltage_target_signed": float(np.mean(voltage_target_signed)),
            "avg_final_gen_vm_touch_count": float(np.mean(final_gen_vm_touch_counts)),
            "avg_final_ext_vm_pu": float(np.mean(final_ext_vm)),
            "avg_action_execution_mismatch": float(
                np.mean(action_execution_mismatches)
            ),
            "active_std_cap": float(np.exp(ARGS.active_log_std_max)),
            "voltage_std_cap": float(np.exp(ARGS.voltage_log_std_max)),
            "phase_steps": dict(phase_steps),
            "fail_reasons": dict(reasons),
        }
        history.append(row)

        train_rank = (passed, mean_reward)
        if train_rank > best_train_rank:
            best_train_rank = train_rank
            torch.save(trainer.actor.state_dict(), OUTPUT_DIR / "best_train_actor.pth")
        torch.save(trainer.actor.state_dict(), OUTPUT_DIR / "latest_actor.pth")

        val_text = "skip"
        if update % ARGS.val_interval == 0:
            val_result = evaluate(trainer, val_env, val_ids)
            val_result["update"] = update
            val_history.append(val_result)
            rank = (val_result["passed"], val_result["avg_reward"])
            if rank > best_val_rank:
                best_val_rank = rank
                torch.save(trainer.actor.state_dict(), OUTPUT_DIR / "best_actor.pth")
            val_text = f"{val_result['passed']}/{val_result['total']}"

        q_loss = 0.0
        actor_loss = 0.0
        if update_stats:
            q_loss = float(np.mean([
                item.get("q1_loss", 0.0) + item.get("q2_loss", 0.0)
                for item in update_stats
            ]))
            actor_updates = [
                item.get("actor_loss", 0.0)
                for item in update_stats
                if not item.get("actor_frozen", False)
            ]
            actor_loss = float(np.mean(actor_updates)) if actor_updates else 0.0
        row["critic_loss"] = q_loss
        row["actor_loss"] = actor_loss
        row["alpha"] = float(trainer.alpha.item())
        row["actor_lr"] = float(trainer.actor_optimizer.param_groups[0]["lr"])
        row["critic_lr"] = float(trainer.critic_optimizer.param_groups[0]["lr"])
        success_reward_text = (
            "-" if row["avg_success_episode_reward"] is None
            else f"{row['avg_success_episode_reward']:+.0f}"
        )
        failed_reward_text = (
            "-" if row["avg_failed_episode_reward"] is None
            else f"{row['avg_failed_episode_reward']:+.0f}"
        )

        print(
            f"U{update:3d} R={mean_reward:+.0f} "
            f"denseR={row['avg_dense_reward']:+.0f} "
            f"Rsucc/fail={success_reward_text}/{failed_reward_text} "
            f"pass={passed}/{len(episode_ids)} "
            f"val={val_text} n1(init/best/final)="
            f"{row['avg_initial_n1_mw']:.1f}/{row['avg_best_n1_mw']:.1f}/"
            f"{row['avg_final_n1_mw']:.1f} v(init/best/final)="
            f"{row['avg_initial_voltage_pu']:.4f}/"
            f"{row['avg_best_voltage_pu']:.4f}/"
            f"{row['avg_final_voltage_pu']:.4f} ptarget="
            f"{row['avg_active_target_fraction']:.3f} vtarget="
            f"{row['avg_voltage_target_fraction']:.3f}/"
            f"{row['avg_voltage_target_signed']:+.3f} vtouch="
            f"{row['avg_final_gen_vm_touch_count']:.1f} extvm="
            f"{row['avg_final_ext_vm_pu']:.3f} execgap="
            f"{row['avg_action_execution_mismatch']:.3f} pstd<="
            f"{row['active_std_cap']:.4f} vstd<={row['voltage_std_cap']:.4f} "
            f"qloss={q_loss:.3f} "
            f"piloss={actor_loss:.3f} alpha={trainer.alpha.item():.4f} "
            f"lr={row['actor_lr']:.1e}/{row['critic_lr']:.1e} "
            f"phase={dict(phase_steps)} buf={len(trainer.replay_buffer)} "
            f"fail={dict(reasons)} "
            f"time={time.time() - update_started:.0f}s"
        )

        save_json(
            OUTPUT_DIR / "training_results.json",
            {
                "config": config,
                "history": history,
                "validation": val_history,
                "elapsed_sec": time.time() - started,
            },
        )

    torch.save(trainer.actor.state_dict(), OUTPUT_DIR / "final_actor.pth")

    if ARGS.final_test_eval:
        selected = OUTPUT_DIR / "best_actor.pth"
        if not selected.exists():
            selected = OUTPUT_DIR / "best_train_actor.pth"
        trainer.actor.load_state_dict(
            torch.load(selected, map_location=core.DEVICE, weights_only=True)
        )
        test_env = CleanN1Env(samples, test_ids, data)
        test_result = evaluate(trainer, test_env, test_ids)
        test_result["selected_checkpoint"] = str(selected)
        save_json(OUTPUT_DIR / "test_results.json", test_result)
        print(f"TEST: {test_result['passed']}/{test_result['total']}")


if __name__ == "__main__":
    main()

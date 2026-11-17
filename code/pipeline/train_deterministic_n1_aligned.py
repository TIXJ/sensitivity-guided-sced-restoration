"""Train fair DDPG/TD3 baselines on the aligned fixed-slack environment.

Only the learning algorithm differs from V26 SAC.  Dataset, split, GAT reset,
voltage-sensitivity-augmented observation, 101-D fixed action semantics, reward,
and common verifier
are imported directly from ``train_sac_n1_active_aligned.py``.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import random
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT = Path(__file__).resolve().parents[2]
CODE_DIR = Path(__file__).resolve().parent
LEGACY_OBS_DIM = 1235
VOLTAGE_CONTEXT_DIM = 69
LEGACY_PREFIX_HEAD_DIM = LEGACY_OBS_DIM - 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("algorithm", choices=("DDPG", "TD3"))
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--episodes-per-update", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--gradient-steps", type=int, default=30)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-interval", type=int, default=20)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--noise-start", type=float, default=0.05)
    parser.add_argument("--noise-end", type=float, default=0.01)
    parser.add_argument("--noise-decay-updates", type=int, default=60)
    parser.add_argument("--ou-theta", type=float, default=0.15)
    parser.add_argument("--td3-policy-noise", type=float, default=0.05)
    parser.add_argument("--td3-noise-clip", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset", default="case300_gt_n1s0575_active_required_fixed32.json"
    )
    parser.add_argument(
        "--gat-checkpoint",
        default="gatv10_results_n1s0575_candidate/gatv10_predictor_best.pth",
    )
    parser.add_argument(
        "--split-file",
        default="paper_results/case300_n1s0575_active_fixed32_split.json",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="Build the full contract and execute one neutral action.",
    )
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def load_aligned(args: argparse.Namespace):
    old_argv = sys.argv
    try:
        sys.argv = [str(CODE_DIR / "train_sac_n1_active_aligned.py")]
        sys.path.insert(0, str(CODE_DIR))
        aligned = importlib.import_module("train_sac_n1_active_aligned")
    finally:
        sys.argv = old_argv

    aligned.DATASET_PATH = resolve(args.dataset)
    aligned.CHECKPOINT_PATH = resolve(args.gat_checkpoint)
    aligned.SPLIT_PATH = resolve(args.split_file)
    aligned.ARGS.max_steps = int(args.max_steps)
    aligned.ARGS.n1_check_every = 1
    aligned.core.MAX_STEPS = int(args.max_steps)
    aligned.core.MAX_STEPS_HIGH = int(args.max_steps)
    aligned.core.N1_CHECK_EVERY_STEPS = 1
    os.environ["ROUTE_B_GAT_CHECKPOINT"] = str(aligned.CHECKPOINT_PATH)
    return aligned


def project_action_tensor(action: torch.Tensor, p_dim: int) -> torch.Tensor:
    """Enforce the same shared active L2 ball used by the SAC actor."""
    active = action[..., :p_dim]
    radius = torch.linalg.vector_norm(active, dim=-1, keepdim=True)
    active = active / torch.maximum(radius, torch.ones_like(radius))
    voltage = action[..., p_dim:].clamp(-1.0, 1.0)
    return torch.cat([active, voltage], dim=-1)


def project_action_numpy(action: np.ndarray, p_dim: int) -> np.ndarray:
    result = np.asarray(action, dtype=np.float32).copy()
    radius = float(np.linalg.norm(result[:p_dim], ord=2))
    if radius > 1.0:
        result[:p_dim] /= radius
    result[p_dim:] = np.clip(result[p_dim:], -1.0, 1.0)
    return result


def initialize_expanded_input(linear: nn.Linear, legacy_input_dim: int) -> None:
    """Preserve the old 1235-D initialization and add learnable zero context."""
    if linear.in_features != legacy_input_dim + VOLTAGE_CONTEXT_DIM:
        return
    with torch.no_grad():
        nn.init.kaiming_uniform_(
            linear.weight[:, :legacy_input_dim], a=np.sqrt(5.0)
        )
        linear.weight[:, legacy_input_dim:].zero_()
        if linear.bias is not None:
            bound = 1.0 / np.sqrt(float(legacy_input_dim))
            nn.init.uniform_(linear.bias, -bound, bound)


def reorder_expanded_observation(obs: torch.Tensor) -> torch.Tensor:
    """Place the old three N-1 scalars before the appended voltage context."""
    if obs.shape[-1] != LEGACY_OBS_DIM + VOLTAGE_CONTEXT_DIM:
        return obs
    return torch.cat(
        [
            obs[..., :LEGACY_PREFIX_HEAD_DIM],
            obs[..., -3:],
            obs[..., LEGACY_PREFIX_HEAD_DIM:-3],
        ],
        dim=-1,
    )


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim: int, p_dim: int, v_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.p_dim = int(p_dim)

        def trunk():
            return nn.Sequential(
                nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            )

        self.p_trunk = trunk()
        self.v_trunk = trunk()
        initialize_expanded_input(self.p_trunk[0], LEGACY_OBS_DIM)
        initialize_expanded_input(self.v_trunk[0], LEGACY_OBS_DIM)
        self.p_head = nn.Linear(hidden_dim, p_dim)
        self.v_head = nn.Linear(hidden_dim, v_dim)
        nn.init.zeros_(self.p_head.weight)
        nn.init.zeros_(self.p_head.bias)
        nn.init.zeros_(self.v_head.weight)
        nn.init.zeros_(self.v_head.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        obs = reorder_expanded_observation(obs)
        active_latent = self.p_head(self.p_trunk(obs))
        radius = torch.linalg.vector_norm(active_latent, dim=-1, keepdim=True)
        scale = torch.where(
            radius > 1e-6,
            torch.tanh(radius) / radius.clamp_min(1e-6),
            torch.ones_like(radius),
        )
        active = active_latent * scale
        voltage = torch.tanh(self.v_head(self.v_trunk(obs)))
        return torch.cat([active, voltage], dim=-1)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        initialize_expanded_input(self.net[0], LEGACY_OBS_DIM + action_dim)
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        obs = reorder_expanded_observation(obs)
        if obs.shape[-1] == LEGACY_OBS_DIM + VOLTAGE_CONTEXT_DIM:
            net_input = torch.cat(
                [obs[..., :LEGACY_OBS_DIM], action, obs[..., LEGACY_OBS_DIM:]],
                dim=-1,
            )
        else:
            net_input = torch.cat([obs, action], dim=-1)
        return self.net(net_input)


class TwinCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.q1 = Critic(obs_dim, action_dim)
        self.q2 = Critic(obs_dim, action_dim)

    def forward(self, obs, action):
        return self.q1(obs, action), self.q2(obs, action)


class ReplayBuffer:
    def __init__(self, capacity: int = 200_000):
        self.data = deque(maxlen=capacity)

    def __len__(self):
        return len(self.data)

    def append(self, transition):
        self.data.append(transition)

    def sample(self, batch_size: int, device):
        indices = np.random.choice(len(self.data), batch_size, replace=False)
        batch = [self.data[index] for index in indices]
        obs = torch.as_tensor(np.asarray([row[0] for row in batch]), device=device)
        action = torch.as_tensor(np.asarray([row[1] for row in batch]), device=device)
        reward = torch.as_tensor(
            np.asarray([row[2] for row in batch], dtype=np.float32)[:, None],
            device=device,
        )
        next_obs = torch.as_tensor(np.asarray([row[3] for row in batch]), device=device)
        done = torch.as_tensor(
            np.asarray([row[4] for row in batch], dtype=np.float32)[:, None],
            device=device,
        )
        return obs, action, reward, next_obs, done


class DeterministicAgent:
    def __init__(self, args, obs_dim: int, action_dim: int, p_dim: int, device):
        self.algorithm = args.algorithm
        self.device = device
        self.p_dim = int(p_dim)
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_delay = 2 if self.algorithm == "TD3" else 1
        self.policy_noise = float(args.td3_policy_noise)
        self.noise_clip = float(args.td3_noise_clip)
        self.ou_theta = float(args.ou_theta)
        self.total_updates = 0
        self.actor_updates = 0

        v_dim = action_dim - p_dim
        self.actor = DeterministicActor(obs_dim, p_dim, v_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        critic_class = TwinCritic if self.algorithm == "TD3" else Critic
        self.critic = critic_class(obs_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=args.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=args.critic_lr
        )
        self.replay = ReplayBuffer()
        self.exploration_state = np.zeros(action_dim, dtype=np.float32)

    def reset_exploration(self, noise_std: float = 0.0) -> None:
        self.exploration_state[:] = np.random.normal(
            0.0, noise_std, size=self.exploration_state.shape
        )

    def act(self, obs: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        obs_tensor = torch.as_tensor(
            np.asarray(obs, dtype=np.float32), device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_tensor)[0].cpu().numpy()
        if noise_std > 0.0:
            innovation_std = noise_std * np.sqrt(
                max(1e-6, 2.0 * self.ou_theta - self.ou_theta ** 2)
            )
            innovation = np.random.normal(
                0.0, innovation_std, size=action.shape
            ).astype(np.float32)
            self.exploration_state += (
                -self.ou_theta * self.exploration_state + innovation
            )
            action += self.exploration_state
        return project_action_numpy(action, self.p_dim)

    @staticmethod
    def _soft_update(source, target, tau: float):
        for parameter, target_parameter in zip(source.parameters(), target.parameters()):
            target_parameter.data.mul_(1.0 - tau).add_(tau * parameter.data)

    def update(self, batch_size: int) -> dict:
        obs, action, reward, next_obs, done = self.replay.sample(
            batch_size, self.device
        )
        with torch.no_grad():
            next_action = self.actor_target(next_obs)
            if self.algorithm == "TD3":
                noise = torch.randn_like(next_action) * self.policy_noise
                noise = noise.clamp(-self.noise_clip, self.noise_clip)
                next_action = project_action_tensor(next_action + noise, self.p_dim)
                q1_target, q2_target = self.critic_target(next_obs, next_action)
                target_q = torch.minimum(q1_target, q2_target)
            else:
                target_q = self.critic_target(next_obs, next_action)
            target = reward + self.gamma * (1.0 - done) * target_q

        if self.algorithm == "TD3":
            q1, q2 = self.critic(obs, action)
            critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        else:
            critic_loss = F.mse_loss(self.critic(obs, action), target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        self.total_updates += 1
        actor_loss_value = None
        if self.total_updates % self.policy_delay == 0:
            policy_action = self.actor(obs)
            if self.algorithm == "TD3":
                actor_loss = -self.critic.q1(obs, policy_action).mean()
            else:
                actor_loss = -self.critic(obs, policy_action).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            self._soft_update(self.actor, self.actor_target, self.tau)
            self.actor_updates += 1
            actor_loss_value = float(actor_loss.item())
        self._soft_update(self.critic, self.critic_target, self.tau)
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": actor_loss_value,
        }


def current_noise(args, update: int) -> float:
    fraction = np.clip(
        (update - 1) / max(1, args.noise_decay_updates - 1), 0.0, 1.0
    )
    return float(args.noise_start + fraction * (args.noise_end - args.noise_start))


def evaluate(aligned, agent, env, ids: list[int]) -> dict:
    passed = 0
    rewards = []
    steps = []
    n1_values = []
    voltage_values = []
    reasons = Counter()
    for sid in ids:
        obs = env.reset(sample_idx=sid)
        total_reward = 0.0
        last_info = {}
        for step in range(env.max_steps):
            raw_action = agent.act(obs, noise_std=0.0)
            obs, reward, done, _truncated, last_info = env.step(
                aligned.scale_action(env, raw_action)
            )
            total_reward += float(reward)
            if done:
                break
        success = aligned.is_pass(last_info)
        passed += int(success)
        rewards.append(total_reward if success else min(total_reward, -1.0))
        steps.append(step + 1)
        n1_values.append(float(last_info.get("n1_overload_mw", np.inf)))
        voltage_values.append(
            float(last_info.get("clean_voltage_potential", np.inf))
        )
        if not success:
            reasons[str(last_info.get("reason", "unknown"))] += 1
    return {
        "passed": passed,
        "total": len(ids),
        "avg_reward": float(np.mean(rewards)),
        "avg_steps": float(np.mean(steps)),
        "avg_final_n1_overload_mw": float(np.mean(n1_values)),
        "avg_final_voltage_potential_pu": float(np.mean(voltage_values)),
        "reasons": dict(reasons),
    }


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    aligned = load_aligned(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_path = resolve(args.dataset)
    split_path = resolve(args.split_file)
    output_dir = resolve(
        args.output_dir
        or f"rl_{args.algorithm.lower()}_n1_active_aligned_v26"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    samples = [sample for sample in data["samples"] if sample.get("is_complete_sced")]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids = [int(index) for index in split["train_ids"]]
    val_ids = [int(index) for index in split["val_ids"]]
    test_ids = [int(index) for index in split["test_ids"]]

    print(f"[{args.algorithm}] Building V26 environments...", flush=True)
    train_env = aligned.CleanN1Env(samples, train_ids, data)
    val_env = aligned.CleanN1Env(samples, val_ids, data)
    obs_dim = train_env._get_obs_dim()
    action_dim = train_env.action_dim
    p_dim = train_env._n_shift_dirs + train_env._n_key_gens
    agent = DeterministicAgent(
        args, obs_dim, action_dim, p_dim, aligned.core.DEVICE
    )
    if args.algorithm == "DDPG" and agent.policy_delay != 1:
        raise RuntimeError("DDPG requires one actor update per critic update")

    config = {
        "algorithm": args.algorithm,
        "dataset": str(dataset_path),
        "gat_checkpoint": str(resolve(args.gat_checkpoint)),
        "split_file": str(split_path),
        "train_val_test": [len(train_ids), len(val_ids), len(test_ids)],
        "observation_contract": "1304D_legacy_prefix_plus_voltage_sensitivity",
        "obs_dim": obs_dim,
        "legacy_observation_prefix_dim": LEGACY_OBS_DIM,
        "voltage_context_dim": VOLTAGE_CONTEXT_DIM,
        "expanded_input_initialization": "legacy_prefix_reordered_with_zero_initialized_voltage_context_columns",
        "action_dim": action_dim,
        "active_action_contract": "32D_shared_L2_ball",
        "voltage_action_contract": "69D_fixed_per_device_incremental",
        "replay_action_contract": "raw_normalized_policy_action",
        "updates": args.updates,
        "episodes_per_update": args.episodes_per_update,
        "max_steps": args.max_steps,
        "gradient_steps": args.gradient_steps,
        "learning_starts": args.learning_starts,
        "batch_size": args.batch_size,
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "reward_scale": args.reward_scale,
        "noise_start": args.noise_start,
        "noise_end": args.noise_end,
        "noise_decay_updates": args.noise_decay_updates,
        "exploration_process": "episode-reset_ornstein_uhlenbeck",
        "ou_theta": args.ou_theta,
        "policy_delay": agent.policy_delay,
        "td3_policy_noise": args.td3_policy_noise if args.algorithm == "TD3" else None,
        "td3_noise_clip": args.td3_noise_clip if args.algorithm == "TD3" else None,
        "checkpoint_selection": "validation_pass_then_reward",
        "uses_rescue_labels": False,
    }
    save_json(output_dir / "config.json", config)
    save_json(output_dir / "split_ids.json", split)

    print(
        f"{args.algorithm}: obs={obs_dim} action={action_dim} "
        f"split={len(train_ids)}/{len(val_ids)}/{len(test_ids)} "
        f"device={aligned.core.DEVICE}"
    )
    print(
        "Deterministic state contract: 1304D = legacy 1235D + "
        "69D trainable voltage-sensitivity context; replay=raw_policy_action."
    )
    print(
        f"Update contract: critic_steps={args.gradient_steps}/outer-update, "
        f"actor_every={agent.policy_delay} critic step(s)."
    )
    if args.preflight_only:
        obs = train_env.reset(sample_idx=train_ids[0])
        raw_action = agent.act(obs)
        next_obs, reward, done, _truncated, info = train_env.step(
            aligned.scale_action(train_env, raw_action)
        )
        smoke_transition = (
            np.asarray(obs, dtype=np.float32),
            np.asarray(raw_action, dtype=np.float32),
            float(reward) * args.reward_scale,
            np.asarray(next_obs, dtype=np.float32),
            float(done),
        )
        for _ in range(args.batch_size):
            agent.replay.append(smoke_transition)
        smoke_stats = [
            agent.update(args.batch_size) for _ in range(agent.policy_delay)
        ]
        optimizer_smoke_finite = bool(all(
            np.isfinite(item["critic_loss"])
            and (
                item["actor_loss"] is None
                or np.isfinite(item["actor_loss"])
            )
            for item in smoke_stats
        ))
        checks = {
            "obs_shape_ok": tuple(obs.shape) == (obs_dim,),
            "next_obs_shape_ok": tuple(next_obs.shape) == (obs_dim,),
            "finite_observation": bool(
                np.isfinite(obs).all() and np.isfinite(next_obs).all()
            ),
            "finite_reward": bool(np.isfinite(reward)),
            "neutral_active_l2": float(np.linalg.norm(raw_action[:p_dim])),
            "optimizer_smoke_finite": optimizer_smoke_finite,
            "optimizer_smoke_actor_updates": agent.actor_updates,
            "gat_reset": bool(
                train_env.reset_audit.get("initial_power_source")
                == "gat_prediction_projected_to_balance"
            ),
            "done": bool(done),
            "reason": info.get("reason", "unknown"),
        }
        save_json(output_dir / "preflight.json", checks)
        print(json.dumps(checks, indent=2))
        if not all(
            checks[key]
            for key in (
                "obs_shape_ok", "next_obs_shape_ok", "finite_observation",
                "finite_reward", "gat_reset",
                "optimizer_smoke_finite",
            )
        ):
            raise RuntimeError("Preflight failed")
        return

    history = []
    validation_history = []
    best_actor_state = None
    best_rank = (-1, -np.inf)
    start_time = time.time()

    for update in range(1, args.updates + 1):
        episode_ids = aligned.stratified_batch(
            samples, train_ids, args.episodes_per_update
        )
        noise_std = current_noise(args, update)
        episode_rewards = []
        episode_steps = []
        train_passed = 0
        fail_reasons = Counter()
        active_action_l2 = []
        voltage_action_abs = []
        action_abs_max = []
        for sid in episode_ids:
            obs = train_env.reset(sample_idx=sid)
            agent.reset_exploration(noise_std)
            total_reward = 0.0
            last_info = {}
            for step in range(train_env.max_steps):
                raw_action = agent.act(obs, noise_std=noise_std)
                next_obs, reward, done, _truncated, last_info = train_env.step(
                    aligned.scale_action(train_env, raw_action)
                )
                boundary = bool(done or step + 1 >= train_env.max_steps)
                executed = np.asarray(
                    last_info.get("clean_executed_raw_action", raw_action),
                    dtype=np.float32,
                )
                executed = project_action_numpy(executed, p_dim)
                active_action_l2.append(float(np.linalg.norm(executed[:p_dim])))
                voltage_action_abs.append(
                    float(np.mean(np.abs(executed[p_dim:])))
                )
                action_abs_max.append(float(np.max(np.abs(executed))))
                # Recovery and clipping are deterministic parts of the
                # transition f(s, a). Store the actor's normalized action so
                # replay, critic evaluation, and deterministic policy gradients
                # all use the same coordinates.
                agent.replay.append(
                    (
                        np.asarray(obs, dtype=np.float32),
                        np.asarray(raw_action, dtype=np.float32),
                        float(reward) * args.reward_scale,
                        np.asarray(next_obs, dtype=np.float32), float(boundary),
                    )
                )
                obs = next_obs
                total_reward += float(reward)
                if done:
                    break
            success = aligned.is_pass(last_info)
            train_passed += int(success)
            episode_rewards.append(total_reward if success else min(total_reward, -1.0))
            episode_steps.append(step + 1)
            if not success:
                fail_reasons[str(last_info.get("reason", "unknown"))] += 1

        update_stats = []
        if len(agent.replay) >= max(args.learning_starts, args.batch_size):
            for _ in range(args.gradient_steps):
                update_stats.append(agent.update(args.batch_size))

        row = {
            "update": update,
            "passed": train_passed,
            "total": len(episode_ids),
            "avg_reward": float(np.mean(episode_rewards)),
            "avg_steps": float(np.mean(episode_steps)),
            "noise_std": noise_std,
            "replay_size": len(agent.replay),
            "critic_updates": agent.total_updates,
            "actor_updates": agent.actor_updates,
            "reasons": dict(fail_reasons),
            "active_action_l2": float(np.mean(active_action_l2)),
            "voltage_action_abs": float(np.mean(voltage_action_abs)),
            "action_abs_max": float(np.max(action_abs_max)),
            "critic_loss": (
                float(np.mean([item["critic_loss"] for item in update_stats]))
                if update_stats else None
            ),
        }

        val_text = "skip"
        if update % args.val_interval == 0:
            validation = evaluate(aligned, agent, val_env, val_ids)
            validation["update"] = update
            validation_history.append(validation)
            rank = (validation["passed"], validation["avg_reward"])
            if rank > best_rank:
                best_rank = rank
                best_actor_state = copy.deepcopy(agent.actor.state_dict())
                torch.save(best_actor_state, output_dir / "best_actor.pth")
            val_text = f"{validation['passed']}/{validation['total']}"

        torch.save(agent.actor.state_dict(), output_dir / "latest_actor.pth")
        history.append(row)
        print(
            f"U{update:3d} R={row['avg_reward']:+.0f} "
            f"pass={train_passed}/{len(episode_ids)} val={val_text} "
            f"noise={noise_std:.3f} replay={len(agent.replay)} "
            f"updates={agent.total_updates}/{agent.actor_updates} "
            f"pL2={row['active_action_l2']:.3f} "
            f"vabs={row['voltage_action_abs']:.3f} "
            f"amax={row['action_abs_max']:.3f} "
            f"fail={dict(fail_reasons)}"
        )

    torch.save(agent.actor.state_dict(), output_dir / "final_actor.pth")
    if best_actor_state is None:
        best_actor_state = copy.deepcopy(agent.actor.state_dict())
        torch.save(best_actor_state, output_dir / "best_actor.pth")
    agent.actor.load_state_dict(best_actor_state)
    agent.actor_target.load_state_dict(best_actor_state)
    test_env = aligned.CleanN1Env(samples, test_ids, data)
    test_result = evaluate(aligned, agent, test_env, test_ids)
    save_json(output_dir / "test_results.json", test_result)
    save_json(
        output_dir / "training_results.json",
        {
            "config": config,
            "history": history,
            "validation": validation_history,
            "test": test_result,
            "elapsed_sec": time.time() - start_time,
        },
    )
    print(
        f"TEST {args.algorithm}: {test_result['passed']}/{test_result['total']} "
        f"R={test_result['avg_reward']:+.0f} "
        f"N1={test_result['avg_final_n1_overload_mw']:.6f}"
    )


if __name__ == "__main__":
    main()

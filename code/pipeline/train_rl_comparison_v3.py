# -*- coding: utf-8 -*-
"""
train_rl_comparison_v3.py — RL Algorithm Comparison (OPTIMIZED SPEED VERSION)

Same algorithms as V2 but with accelerated training:
  - AC-PF check every 5 steps (was 2) → 2.5x faster
  - 30 updates × 10 episodes × 30 grad steps → ~1h/algorithm
  - Max 50 steps per episode → 1.6x faster
  - Total speedup: ~30x vs V2

DDPG / TD3 / PPO vs SAC (V35 reference).
"""
import sys, os, json, time, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, defaultdict
import random
from pathlib import Path

PROJ = Path(r".")
sys.path.insert(0, str(PROJ))

# ★ Monkey-patch V35 constants for speed BEFORE importing env
import train_rlv35_complete as _v35
_v35.AC_PF_CHECK_EVERY = 5        # was 2 → 2.5x faster
_v35.N1_CHECK_EVERY_STEPS = 5     # was 2
_v35.N_GRADIENT_STEPS = 30        # was 50
_v35.N_EPISODES_PER_UPDATE = 10   # was 20
_v35.MAX_STEPS = 50               # was 80
_v35.MAX_STEPS_HIGH = 70          # was 120

# Now import the rest normally
from train_rlv35_complete import (
    SCEDEnvV21, split_samples_stratified, stratified_sample_from_pool,
    curriculum_sample, DEVICE,
    N_SHIFT_DIRECTIONS, N_KEY_GENERATORS,
    BATCH_SIZE, BUFFER_SIZE, EVAL_INTERVAL, N_PER_SEGMENT_EVAL
)

# Local overrides
FAST_TOTAL_UPDATES = 30
FAST_EVAL_INTERVAL = 5
FAST_BATCH_SIZE = 256  # smaller batch → faster training start

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

RESULTS_DIR = PROJ / "rl_comparison_v3_results"
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================
# Shared Networks (same as V2)
# ============================================================
class DDPGActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh())
    def forward(self, x): return self.net(x)

class DDPGCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, x, a): return self.net(torch.cat([x, a], dim=-1))

class TD3TwinCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.q1 = DDPGCritic(obs_dim, act_dim, hidden)
        self.q2 = DDPGCritic(obs_dim, act_dim, hidden)
    def forward(self, x, a): return self.q1(x, a), self.q2(x, a)

class PPOGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.mean_head = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
    def forward(self, x):
        h = self.net(x)
        return self.mean_head(h), self.log_std.clamp(-5.0, 2.0)
    def sample(self, x):
        mean, log_std = self.forward(x); std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample(); action = torch.tanh(z)
        lp = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = lp.sum(-1, keepdim=True)
        return action, log_prob, mean
    def get_log_prob(self, x, action):
        mean, log_std = self.forward(x); std = log_std.exp().clamp(1e-6, 1e6)
        # NaN guard
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)
        std = torch.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1e-6)
        normal = torch.distributions.Normal(mean, std)
        z = torch.atanh(action.clamp(-0.999, 0.999))
        lp = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        lp = torch.nan_to_num(lp, nan=-10.0, posinf=0.0, neginf=-10.0)
        return lp.sum(-1, keepdim=True)

class PPOValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, x): return self.net(x)

# ============================================================
# DDPG Trainer (optimized)
# ============================================================
class DDPGTrainer:
    def __init__(self, obs_dim, act_dim, lr=1e-3, gamma=0.99, tau=0.005, policy_freq=2):
        self.actor = DDPGActor(obs_dim, act_dim).to(DEVICE)
        self.actor_target = DDPGActor(obs_dim, act_dim).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = DDPGCritic(obs_dim, act_dim).to(DEVICE)
        self.critic_target = DDPGCritic(obs_dim, act_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma, self.tau, self.policy_freq = gamma, tau, policy_freq
        self.buffer = deque(maxlen=BUFFER_SIZE)
        self._total_grad_steps = 0

    def select_action(self, obs, deterministic=True):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            a = self.actor(obs_t)
            if not deterministic:
                noise = (torch.randn_like(a) * 0.3).clamp(-0.5, 0.5)
                a = (a + noise).clamp(-1.0, 1.0)
            return a.cpu().numpy()[0]

    def update(self, batch_size=256, n_steps=30):
        if len(self.buffer) < batch_size: return
        for gs in range(n_steps):
            idx = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
            batch = [self.buffer[i] for i in idx]
            s = torch.FloatTensor(np.array([b[0] for b in batch])).to(DEVICE)
            a = torch.FloatTensor(np.array([b[1] for b in batch])).to(DEVICE)
            r = torch.FloatTensor(np.array([b[2] for b in batch])).unsqueeze(-1).to(DEVICE)
            ns = torch.FloatTensor(np.array([b[3] for b in batch])).to(DEVICE)
            d = torch.FloatTensor(np.array([b[4] for b in batch])).unsqueeze(-1).to(DEVICE)
            with torch.no_grad():
                target_q = r + self.gamma * (1-d) * self.critic_target(ns, self.actor_target(ns))
            q = self.critic(s, a); critic_loss = F.mse_loss(q, target_q)
            self.critic_opt.zero_grad(); critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0); self.critic_opt.step()
            if gs % self.policy_freq == 0:
                actor_loss = -self.critic(s, self.actor(s)).mean()
                self.actor_opt.zero_grad(); actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0); self.actor_opt.step()
                for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                    tp.data.copy_(self.tau * p.data + (1-self.tau) * tp.data)
                for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                    tp.data.copy_(self.tau * p.data + (1-self.tau) * tp.data)
            self._total_grad_steps += 1

# ============================================================
# TD3 Trainer (optimized)
# ============================================================
class TD3Trainer:
    def __init__(self, obs_dim, act_dim, lr=1e-3, gamma=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.actor = DDPGActor(obs_dim, act_dim).to(DEVICE)
        self.actor_target = DDPGActor(obs_dim, act_dim).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = TD3TwinCritic(obs_dim, act_dim).to(DEVICE)
        self.critic_target = TD3TwinCritic(obs_dim, act_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma, self.tau = gamma, tau
        self.policy_noise, self.noise_clip, self.policy_freq = policy_noise, noise_clip, policy_freq
        self.buffer = deque(maxlen=BUFFER_SIZE)
        self._total_grad_steps = 0

    def select_action(self, obs, deterministic=True):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            a = self.actor(obs_t)
            if not deterministic:
                noise = (torch.randn_like(a) * 0.3).clamp(-0.5, 0.5)
                a = (a + noise).clamp(-1.0, 1.0)
            return a.cpu().numpy()[0]

    def update(self, batch_size=256, n_steps=30):
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
                noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_a = (self.actor_target(ns) + noise).clamp(-1, 1)
                q1_t, q2_t = self.critic_target(ns, next_a)
                target_q = r + self.gamma * (1-d) * torch.min(q1_t, q2_t)
            q1, q2 = self.critic(s, a)
            critic_loss = F.mse_loss(q1, target_q.detach()) + F.mse_loss(q2, target_q.detach())
            self.critic_opt.zero_grad(); critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0); self.critic_opt.step()
            if self._total_grad_steps % self.policy_freq == 0:
                actor_loss = -self.critic.q1(s, self.actor(s)).mean()
                self.actor_opt.zero_grad(); actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0); self.actor_opt.step()
                for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                    tp.data.copy_(self.tau * p.data + (1-self.tau) * tp.data)
                for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                    tp.data.copy_(self.tau * p.data + (1-self.tau) * tp.data)
            self._total_grad_steps += 1

# ============================================================
# PPO Trainer (optimized)
# ============================================================
class PPOTrainer:
    def __init__(self, obs_dim, act_dim, lr=1e-4, gamma=0.99, lam=0.95, clip_eps=0.2,
                 epochs=3, entropy_coef=0.01, log_std_init=0.0,
                 learn_log_std=True, target_kl=0.05, critic_lr=None):
        self.actor = PPOGaussianActor(obs_dim, act_dim).to(DEVICE)
        with torch.no_grad():
            self.actor.mean_head.weight.zero_()
            self.actor.mean_head.bias.zero_()
            self.actor.log_std.fill_(float(log_std_init))
        self.actor.log_std.requires_grad_(bool(learn_log_std))
        self.critic = PPOValueNet(obs_dim).to(DEVICE)
        self.actor_opt = torch.optim.Adam(
            [p for p in self.actor.parameters() if p.requires_grad], lr=lr
        )
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=lr if critic_lr is None else critic_lr
        )
        self.gamma, self.lam, self.clip_eps, self.epochs = gamma, lam, clip_eps, epochs
        self.entropy_coef = float(entropy_coef)
        self.learn_log_std = bool(learn_log_std)
        self.target_kl = float(target_kl)
        self.total_updates = 0; self.buffer = []
        self.last_update_stats = {}

    def select_action(self, obs, deterministic=True):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            if deterministic:
                mean, _ = self.actor(obs_t); return torch.tanh(mean).cpu().numpy()[0]
            else:
                action, log_prob, mean = self.actor.sample(obs_t)
                value = self.critic(obs_t)
                return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0]

    def store(self, s, a, r, ns, d, log_prob, value):
        self.buffer.append((s, a, r, ns, d, log_prob, value))

    def update(self, batch_size=256, n_steps=30):
        if len(self.buffer) < 32: return
        T = len(self.buffer)
        all_s = torch.FloatTensor(np.array([b[0] for b in self.buffer])).to(DEVICE)
        all_a = torch.FloatTensor(np.array([b[1] for b in self.buffer])).to(DEVICE)
        all_r = torch.as_tensor(np.array([b[2] for b in self.buffer]), dtype=torch.float32, device=DEVICE).reshape(-1)
        all_ns = torch.FloatTensor(np.array([b[3] for b in self.buffer])).to(DEVICE)
        all_d = torch.as_tensor(np.array([b[4] for b in self.buffer]), dtype=torch.float32, device=DEVICE).reshape(-1)
        all_log_prob = torch.as_tensor(np.array([b[5] for b in self.buffer]), dtype=torch.float32, device=DEVICE).reshape(-1, 1)
        all_value = torch.as_tensor(np.array([b[6] for b in self.buffer]), dtype=torch.float32, device=DEVICE).reshape(-1)
        with torch.no_grad():
            last_value = self.critic(all_ns[-1:]).squeeze(-1)
            advantages = torch.zeros(T, device=DEVICE); returns = torch.zeros(T, device=DEVICE)
            gae = 0.0
            for t in reversed(range(T)):
                next_val = last_value[0] if t == T-1 else all_value[t+1]
                delta = all_r[t] + self.gamma*(1-all_d[t])*next_val - all_value[t]
                gae = delta + self.gamma*self.lam*(1-all_d[t])*gae
                advantages[t] = gae; returns[t] = gae + all_value[t]
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        kl_values = []; clip_values = []; actor_losses = []
        minibatches = 0
        for _ in range(self.epochs):
            perm = torch.randperm(T, device=DEVICE)
            stop_epoch = False
            for mb_start in range(0, T, batch_size):
                mb_idx = perm[mb_start:mb_start+batch_size]
                if len(mb_idx) < 4: continue
                mb_s, mb_a = all_s[mb_idx], all_a[mb_idx]
                mb_adv, mb_ret = advantages[mb_idx], returns[mb_idx]
                mb_old_lp = all_log_prob[mb_idx]
                new_lp = self.actor.get_log_prob(mb_s, mb_a)
                log_ratio = (new_lp - mb_old_lp).clamp(-20.0, 20.0)
                ratio = log_ratio.exp()
                # NaN guard: skip update if values exploded
                if torch.isnan(ratio).any() or torch.isinf(ratio).any():
                    continue
                clip_adv = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * mb_adv.unsqueeze(-1)
                actor_loss = -torch.min(ratio*mb_adv.unsqueeze(-1), clip_adv).mean()
                mean, log_std = self.actor(mb_s)
                entropy = (log_std + 0.5*np.log(2*np.pi*np.e)).sum(-1).mean()
                actor_loss = actor_loss - self.entropy_coef*entropy
                self.actor_opt.zero_grad(); actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0); self.actor_opt.step()
                critic_loss = F.mse_loss(self.critic(mb_s).squeeze(-1), mb_ret)
                if torch.isnan(critic_loss):
                    continue
                self.critic_opt.zero_grad(); critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0); self.critic_opt.step()
                approx_kl = float((ratio - 1.0 - log_ratio).mean().detach().cpu())
                clip_fraction = float(((ratio - 1.0).abs() > self.clip_eps).float().mean().detach().cpu())
                kl_values.append(approx_kl); clip_values.append(clip_fraction)
                actor_losses.append(float(actor_loss.detach().cpu())); minibatches += 1
                if self.target_kl > 0 and approx_kl > self.target_kl:
                    stop_epoch = True
                    break
            if stop_epoch:
                break
        self.last_update_stats = {
            "kl": float(np.mean(kl_values)) if kl_values else 0.0,
            "clip_fraction": float(np.mean(clip_values)) if clip_values else 0.0,
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "minibatches": int(minibatches),
        }
        self.buffer = []; self.total_updates += 1

# ============================================================
# Training & Eval (shared)
# ============================================================
def train_rl(trainer, algo_name, samples, train_ids, val_ids, test_ids,
             save_dir, log_fh, is_ppo=False):
    os.makedirs(save_dir, exist_ok=True)
    train_env = SCEDEnvV21(samples, train_ids)
    obs_dim = train_env._get_obs_dim(); action_dim = train_env.action_dim

    best_val = 0.0; best_state = None; best_update = 0
    t0 = time.time()
    log_fh.write(f"\n{'='*50}\nTraining {algo_name} — {FAST_TOTAL_UPDATES} updates (fast)\n{'='*50}\n")
    log_fh.flush()

    for u in range(1, FAST_TOTAL_UPDATES + 1):
        u_start = time.time()
        eps = curriculum_sample(
            train_env.all_samples,
            [i for i in range(len(train_env.all_samples))
             if train_env.all_samples[i].get('is_complete_sced')],
            u, 10)
        ep_rewards = []; ep_conv = 0
        for sid in eps:
            obs = train_env.reset(sample_idx=sid); ep_r = 0.0
            for step in range(train_env.max_steps + 1):
                if is_ppo:
                    raw_a, lp, val = trainer.select_action(obs, deterministic=False)
                else:
                    raw_a = trainer.select_action(obs, deterministic=False)
                    lp, val = None, None
                low, high = train_env.get_action_bounds()
                a = low + (raw_a + 1.0) * 0.5 * (high - low)
                next_obs, reward, done, _, info = train_env.step(a)
                ep_r += reward
                if is_ppo:
                    trainer.store(obs, raw_a, reward, next_obs, float(done),
                                  lp[0] if lp is not None else 0.0,
                                  val[0] if val is not None else 0.0)
                else:
                    trainer.buffer.append((obs, raw_a, reward, next_obs, float(done)))
                obs = next_obs
                if done: break
            ep_rewards.append(ep_r)
            if info.get('converged', False): ep_conv += 1

        trainer.update(batch_size=FAST_BATCH_SIZE, n_steps=30)
        dt = time.time() - u_start; avg_r = np.mean(ep_rewards) if ep_rewards else 0

        if u % FAST_EVAL_INTERVAL == 0 or u == 1:
            msg = f"  U{u:2d} R={avg_r:+.0f} Conv={ep_conv}/{len(eps)} {dt:.0f}s"
            print(msg); log_fh.write(msg+'\n'); log_fh.flush()

        if u % FAST_EVAL_INTERVAL == 0 or u == FAST_TOTAL_UPDATES:
            val_ok, val_total = evaluate(trainer, samples, val_ids, is_ppo)
            val_rate = 100*val_ok/val_total
            msg = f"  EVAL U{u}: Val={val_ok}/{val_total} ({val_rate:.1f}%)"
            print(msg); log_fh.write(msg+'\n'); log_fh.flush()
            if val_rate >= best_val:
                best_val = val_rate; best_update = u
                best_state = {'actor': trainer.actor.state_dict(), 'update': u, 'val_rate': val_rate}

    if best_state: trainer.actor.load_state_dict(best_state['actor'])
    test_ok, test_total = evaluate(trainer, samples, test_ids, is_ppo)
    test_rate = 100*test_ok/test_total; total_time = time.time()-t0
    msg = (f"\n  {algo_name} FINAL: Test={test_ok}/{test_total} ({test_rate:.1f}%) "
           f"Best@U{best_update}={best_val:.1f}% Time={total_time/3600:.1f}h")
    print(msg); log_fh.write(msg+'\n'); log_fh.flush()
    torch.save(best_state, os.path.join(save_dir, f'{algo_name.lower()}_best.pth'))
    return test_ok, test_total, test_rate, total_time, best_val

def evaluate(trainer, samples, sample_ids, is_ppo=False):
    eval_ids = stratified_sample_from_pool(samples, sample_ids, N_PER_SEGMENT_EVAL)
    if len(eval_ids) == 0: eval_ids = sample_ids
    env = SCEDEnvV21(samples, sample_ids)
    n_ok = 0
    for sid in eval_ids:
        obs = env.reset(sample_idx=sid)
        for step in range(env.max_steps + 1):
            raw_a = trainer.select_action(obs, deterministic=True)
            low, high = env.get_action_bounds()
            a = low + (raw_a + 1.0) * 0.5 * (high - low)
            obs, r, done, _, info = env.step(a)
            if done: break
        ok = (info.get('converged', False) and info.get('v_violations', 999) == 0
              and info.get('n1_violations', 999) == 0 and info.get('overloaded_lines', 999) == 0)
        if ok: n_ok += 1
    return n_ok, len(eval_ids)

# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='all',
                        choices=['DDPG','TD3','PPO','all'])
    args = parser.parse_args()

    print("="*60)
    print("RL Comparison V3 (OPTIMIZED SPEED)")
    print(f"Updates={FAST_TOTAL_UPDATES}, AC-PF every 5 steps")
    print(f"Batch={FAST_BATCH_SIZE}, GradSteps=30, Episodes=10")
    print("="*60)

    with open(str(PROJ/'case300_gt_v10_dataset.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
    samples = [s for s in data['samples'] if s.get('is_complete_sced')]
    train_ids, val_ids, test_ids = split_samples_stratified(samples)
    print(f"Samples: {len(samples)} (Train:{len(train_ids)} Val:{len(val_ids)} Test:{len(test_ids)})")

    obs_dim_test = SCEDEnvV21(samples, train_ids)._get_obs_dim()
    action_dim = N_SHIFT_DIRECTIONS + N_KEY_GENERATORS
    print(f"ObsDim={obs_dim_test}, ActDim={action_dim}")

    log_path = RESULTS_DIR / "training.log"
    log_fh = open(str(log_path), 'w', buffering=1, encoding='utf-8')
    log_fh.write(f"RL Comparison V3 (Fast) — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    results = {}
    algo_list = ['DDPG','TD3','PPO'] if args.algo == 'all' else [args.algo]

    for algo in algo_list:
        print(f"\n{'#'*50}\n# {algo}\n{'#'*50}")
        if algo == 'DDPG':
            trainer = DDPGTrainer(obs_dim_test, action_dim)
            ok, total, rate, t, bv = train_rl(trainer, 'DDPG', samples, train_ids, val_ids, test_ids,
                                              str(RESULTS_DIR/'ddpg'), log_fh, is_ppo=False)
            results['DDPG'] = {'test_ok': ok, 'test_total': total, 'test_rate': rate,
                               'time_h': t/3600, 'best_val': bv}
        elif algo == 'TD3':
            trainer = TD3Trainer(obs_dim_test, action_dim)
            ok, total, rate, t, bv = train_rl(trainer, 'TD3', samples, train_ids, val_ids, test_ids,
                                              str(RESULTS_DIR/'td3'), log_fh, is_ppo=False)
            results['TD3'] = {'test_ok': ok, 'test_total': total, 'test_rate': rate,
                              'time_h': t/3600, 'best_val': bv}
        elif algo == 'PPO':
            trainer = PPOTrainer(obs_dim_test, action_dim)
            ok, total, rate, t, bv = train_rl(trainer, 'PPO', samples, train_ids, val_ids, test_ids,
                                              str(RESULTS_DIR/'ppo'), log_fh, is_ppo=True)
            results['PPO'] = {'test_ok': ok, 'test_total': total, 'test_rate': rate,
                              'time_h': t/3600, 'best_val': bv}

    results['SAC (V35)'] = {'test_ok': 37, 'test_total': 44, 'test_rate': 84.1,
                             'time_h': 13.0, 'note': 'P2 deterministic (from V35)'}

    print("\n"+"="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Algo':<12} {'Test':>8} {'Rate':>7} {'BestVal':>8} {'Time':>8}")
    print("-"*45)
    for n, r in results.items():
        print(f"{n:<12} {r['test_ok']:>4}/{r['test_total']:<3} {r['test_rate']:>6.1f}% "
              f"{r.get('best_val',0):>7.1f}% {r['time_h']:>7.1f}h")
        log_fh.write(f"{n:<12} {r['test_ok']:>4}/{r['test_total']:<3} {r['test_rate']:>6.1f}% "
                     f"{r.get('best_val',0):>7.1f}% {r['time_h']:>7.1f}h\n")

    out_path = RESULTS_DIR / 'rl_algorithm_comparison_v3_results.json'
    with open(str(out_path), 'w') as f: json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    log_fh.close()

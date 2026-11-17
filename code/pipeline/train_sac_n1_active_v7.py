"""Train SAC on the tightened-N-1 active-redispatch Route-B dataset.

This is an independent copy of ``_train_scratch_full.py``.  Its defaults and
exploration schedule are specialized for the active-power stress experiment;
the original training entry point remains unchanged.
"""
import sys, os, json, time, argparse, copy, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from collections import deque, Counter; import random
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("algo", nargs="?", default="SAC", choices=["SAC", "DDPG", "TD3", "PPO"])
parser.add_argument("--updates", type=int, default=100)
parser.add_argument(
    "--dataset",
    default="case300_gt_n1s0575_active_required.json",
    help="Linear-SCED dataset used by the RL environment.",
)
parser.add_argument(
    "--gat-checkpoint",
    default="gatv10_results_n1s0575_candidate/gatv10_predictor_best.pth",
    help="GAT checkpoint that matches --dataset.",
)
parser.add_argument("--val-interval", type=int, default=10)
parser.add_argument("--episodes-per-update", type=int, default=10)
parser.add_argument("--learning-starts", type=int, default=256)
parser.add_argument("--gradient-steps", type=int, default=30,
                    help="Full SAC updates per outer update.")
parser.add_argument("--ac-pf-check-every", type=int, default=5)
parser.add_argument("--n1-check-every", type=int, default=1,
                    help="Training-time linear N-1 feedback interval; final evaluation remains strict.")
parser.add_argument("--actor-freeze-updates", type=int, default=0,
                    help="Number of outer updates with critic-only training; standard Route-B SAC uses 0.")
parser.add_argument("--actor-update-interval", type=int, default=1)
parser.add_argument("--actor-inner-update-interval", type=int, default=1,
                    help="Within each outer update, update the actor once every N critic steps.")
parser.add_argument("--critic-warmup-updates", type=int, default=0,
                    help="Optional critic-only warm-up; zero keeps actor learning unfrozen.")
parser.add_argument("--min-success-episodes-before-actor", type=int, default=0,
                    help="Do not update the SAC actor until replay collection has observed this many successful episodes.")
parser.add_argument("--safe-candidate-eval-samples", type=int, default=0,
                    help="After SAC actor updates, accept the candidate only if it improves on this many train samples.")
parser.add_argument("--train-rollback-probe-samples", type=int, default=0,
                    help="Deterministic train samples used to reject actor regressions.")
parser.add_argument("--train-probe-interval", type=int, default=0,
                    help="Outer-update interval for the deterministic train rollback probe.")
parser.add_argument("--min-val-save-pass", type=int, default=5,
                    help="Do not save a validation best_actor unless at least this many validation samples pass.")
parser.add_argument("--actor-saturation-guard", type=float, default=0.0,
                    help="Skip actor updates when probe mean-action saturation exceeds this threshold.")
parser.add_argument("--actor-lr", type=float, default=3e-4)
parser.add_argument("--critic-lr", type=float, default=3e-4)
parser.add_argument("--alpha-lr", type=float, default=3e-4)
parser.add_argument("--det-actor-lr", type=float, default=1e-4,
                    help="Actor LR for DDPG/TD3; separated from SAC for stable high-dimensional control.")
parser.add_argument("--det-critic-lr", type=float, default=3e-4,
                    help="Critic LR for DDPG/TD3.")
parser.add_argument("--det-gradient-steps", type=int, default=30,
                    help="Gradient steps per outer update for DDPG/TD3.")
parser.add_argument("--det-learning-starts", type=int, default=2500,
                    help="Replay transitions collected before DDPG/TD3 gradient updates begin.")
parser.add_argument("--det-critic-warmup-steps", type=int, default=120,
                    help="Critic-only gradient steps after replay warm-up before deterministic actor updates.")
parser.add_argument("--det-policy-freq", type=int, default=2,
                    help="One DDPG/TD3 actor update per N critic updates after critic warm-up.")
parser.add_argument("--det-early-stop-patience", type=int, default=0,
                    help="Optional DDPG/TD3 early stopping. Disabled for complete 100-update curves.")
parser.add_argument("--det-val-rollback", action=argparse.BooleanOptionalAction, default=True,
                    help="Restore the best validated deterministic actor when validation regresses.")
parser.add_argument("--det-lr-decay-start", type=int, default=40,
                    help="Outer update where DDPG actor/critic learning-rate decay begins.")
parser.add_argument("--det-actor-lr-final-factor", type=float, default=0.10)
parser.add_argument("--det-critic-lr-final-factor", type=float, default=0.33)
parser.add_argument("--det-noise-start", type=float, default=0.30,
                    help="Initial normalized rollout-noise std for DDPG/TD3.")
parser.add_argument("--det-noise-end", type=float, default=0.05,
                    help="Final normalized rollout-noise std for DDPG/TD3.")
parser.add_argument("--det-noise-decay-updates", type=int, default=60,
                    help="Outer updates over which DDPG/TD3 rollout noise decays.")
parser.add_argument("--det-reward-scale", type=float, default=0.001,
                    help="Replay-only reward scale for DDPG/TD3; reported rewards remain physical/unscaled.")
parser.add_argument("--td3-actor-lr", type=float, default=1e-3)
parser.add_argument("--td3-critic-lr", type=float, default=1e-3)
parser.add_argument("--td3-gradient-steps", type=int, default=30)
parser.add_argument("--td3-learning-starts", type=int, default=256)
parser.add_argument("--td3-noise-start", type=float, default=0.30)
parser.add_argument("--td3-noise-end", type=float, default=0.30)
parser.add_argument("--td3-noise-decay-updates", type=int, default=1)
parser.add_argument("--td3-reward-scale", type=float, default=1.0)
parser.add_argument("--td3-lr-decay-start", type=int, default=100)
parser.add_argument("--td3-actor-lr-final-factor", type=float, default=1.0)
parser.add_argument("--td3-critic-lr-final-factor", type=float, default=1.0)
parser.add_argument("--td3-val-rollback", action=argparse.BooleanOptionalAction, default=False,
                    help="Optional TD3 validation rollback. Disabled in the reproduced baseline profile.")
parser.add_argument("--td3-policy-noise", type=float, default=0.20,
                    help="TD3 target-policy smoothing noise std.")
parser.add_argument("--td3-noise-clip", type=float, default=0.50,
                    help="TD3 target-policy smoothing noise clip.")
parser.add_argument("--td3-critic-warmup-steps", type=int, default=0,
                    help="TD3-specific critic-only steps; twin critics need longer calibration than DDPG.")
parser.add_argument("--td3-policy-freq", type=int, default=2,
                    help="TD3-specific delayed actor update frequency.")
parser.add_argument("--ppo-lr", type=float, default=1e-4,
                    help="PPO actor learning rate.")
parser.add_argument("--ppo-critic-lr", type=float, default=3e-4)
parser.add_argument("--ppo-log-std-init", type=float, default=-2.3,
                    help="Initial PPO log std; -2.3 gives std about 0.10 near the actor mean.")
parser.add_argument("--ppo-log-std-final", type=float, default=-4.0,
                    help="Final scheduled PPO log std; -4 gives std about 0.018.")
parser.add_argument("--ppo-std-decay-updates", type=int, default=60)
parser.add_argument("--ppo-learn-log-std", action=argparse.BooleanOptionalAction, default=False,
                    help="Learn PPO variance instead of using the default deterministic annealing schedule.")
parser.add_argument("--ppo-reward-scale", type=float, default=0.001,
                    help="Training-only PPO reward scale; printed rewards remain unscaled.")
parser.add_argument("--ppo-entropy-coef", type=float, default=0.0,
                    help="PPO entropy coefficient. Zero avoids variance inflation in the 101-D action space.")
parser.add_argument("--ppo-epochs", type=int, default=5)
parser.add_argument("--ppo-target-kl", type=float, default=0.05,
                    help="Stop a PPO epoch early if joint-action KL exceeds this value.")
parser.add_argument("--sac-init-alpha", type=float, default=0.2)
parser.add_argument("--auto-alpha", action=argparse.BooleanOptionalAction, default=True,
                    help="Enable SAC temperature auto-tuning. Keep enabled when --learn-log-std is enabled.")
parser.add_argument("--sac-log-std-bias", type=float, default=-3.0)
parser.add_argument("--sac-log-std-max", type=float, default=-2.0,
                    help="Upper clamp for SAC log_std; keeps stochastic rollouts from destroying AC-PF.")
parser.add_argument("--stochastic-steps", type=int, default=-1,
                    help="Use stochastic SAC actions only for the first N correction steps in training rollouts.")
parser.add_argument("--sac-rollout-mode", default="sample",
                    choices=["sample", "mean_noise", "mean"],
                    help="SAC rollout action mode. 'sample' is the standard stochastic SAC collector.")
parser.add_argument("--rollout-noise-start", type=float, default=0.06,
                    help="Std of normalized Gaussian noise added to actor-mean rollout actions at update 1.")
parser.add_argument("--rollout-noise-end", type=float, default=0.02,
                    help="Std of normalized Gaussian noise added to actor-mean rollout actions after decay.")
parser.add_argument("--rollout-noise-decay-updates", type=int, default=50,
                    help="Number of updates used to decay mean-noise exploration.")
parser.add_argument("--learn-log-std", action=argparse.BooleanOptionalAction, default=True,
                    help="Learn exploration variance. Standard SAC enables this together with automatic alpha.")
parser.add_argument("--reward-scale", type=float, default=1.0,
                    help="Replay reward scale; printed rewards remain unscaled.")
parser.add_argument("--failure-gated-replay", action=argparse.BooleanOptionalAction, default=False,
                    help="For SAC, prevent positive dense rewards from failed episodes entering replay.")
parser.add_argument("--failure-reward-offset", type=float, default=30.0,
                    help="Subtract this from each failed-episode SAC replay reward while preserving reward ordering.")
parser.add_argument("--success-bc-steps", type=int, default=0,
                    help="Optional self-imitation updates; disabled for the pure-SAC experiment.")
parser.add_argument("--success-bc-batch", type=int, default=128)
parser.add_argument("--success-bc-weight", type=float, default=0.50)
parser.add_argument("--success-buffer-size", type=int, default=20000)
parser.add_argument("--voltage-explore-gain", type=float, default=0.0,
                    help="Training-only voltage exploration bias from the latest AC-PF voltage gap.")
parser.add_argument("--active-explore-gain", type=float, default=0.0,
                    help="Training-only active-power exploration bias from voltage-gap proxy.")
parser.add_argument("--n1-guide-gain", type=float, default=0.0,
                    help="Normalized PTDF/LODF relief bias added to early SAC exploration.")
parser.add_argument("--n1-guide-until", type=int, default=60,
                    help="Last outer update using N-1-guided exploration.")
parser.add_argument("--n1-guide-decay-start", type=int, default=30,
                    help="Outer update where the N-1 exploration bias starts decaying.")
parser.add_argument("--voltage-explore-until", type=int, default=0,
                    help="Apply P+V guided exploration through this update; 0 disables it.")
parser.add_argument("--residual-scale", type=float, default=20.0,
                    help="Uniform per-step active-power residual action scale for all Route-B samples.")
parser.add_argument("--cumulative-scale-mult", type=float, default=20.0,
                    help="Cumulative active-power residual clip as a multiple of --residual-scale.")
parser.add_argument("--max-steps", type=int, default=50,
                    help="Unified maximum SAC correction steps.")
parser.add_argument("--guided-decay-start", type=int, default=30,
                    help="Linearly decay guided exploration after this update.")
parser.add_argument("--val-rollback", action=argparse.BooleanOptionalAction, default=False,
                    help="Optionally restore the complete best SAC state when validation degrades.")
parser.add_argument("--variant", default="default",
                    choices=["default", "no-gat", "no-voltage", "no-gat-no-voltage"])
parser.add_argument(
    "--action-encoding",
    default="physics32fixed",
    choices=["physics32", "physics32fixed", "full68", "matched32"],
    help=(
        "Active-power action representation: dynamic PTDF/LODF 12+20, "
        "fixed network-wide PTDF/LODF 12+20, "
        "68 direct generator controls, or a dimension-matched nonphysical 12+20 basis."
    ),
)
parser.add_argument("--action-basis-seed", type=int, default=20260721)
parser.add_argument(
    "--n1-limit-scale",
    type=float,
    default=1.0,
    help="Multiply stored effective LODF limits for the action-compression stress task.",
)
parser.add_argument(
    "--final-test-eval",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Evaluate the selected final actor on the complete-hybrid held-out test task.",
)
parser.add_argument("--split-file", default="paper_results/case300_n1s0575_active_split.json",
                    help="Shared master split JSON with train_ids/val_ids/test_ids.")
parser.add_argument("--output-dir", default="rl_sac_n1_active_v7",
                    help="Override output directory. Defaults to rl_scratch_<algo>[_variant].")
args = parser.parse_args()
algo_name = args.algo
if algo_name != "SAC":
    parser.error("train_sac_n1_active.py is a SAC-only training entry point")
variant_name = args.variant
PROJ_DIR = Path(
    os.environ.get("ROUTE_B_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).resolve()
CODE_DIR = PROJ_DIR / "paper_results" / "code"
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(1, str(PROJ_DIR))
dataset_path = Path(args.dataset)
if not dataset_path.is_absolute():
    dataset_path = PROJ_DIR / dataset_path
gat_checkpoint = Path(args.gat_checkpoint)
if not gat_checkpoint.is_absolute():
    gat_checkpoint = PROJ_DIR / gat_checkpoint
os.environ["ROUTE_B_GAT_CHECKPOINT"] = str(gat_checkpoint)

import train_rlv35_complete as _v35
_v35.AC_PF_CHECK_EVERY = max(1, args.ac_pf_check_every)
_v35.N1_CHECK_EVERY_STEPS = max(1, args.n1_check_every)
_v35.N_GRADIENT_STEPS = args.gradient_steps; _v35.N_EPISODES_PER_UPDATE = args.episodes_per_update
_v35.BATCH_SIZE = 256
_v35.SAC_INIT_ALPHA = args.sac_init_alpha
_v35.SAC_LR_ACTOR = args.actor_lr
_v35.SAC_LR_CRITIC = args.critic_lr
_v35.SAC_LR_ALPHA = args.alpha_lr
_v35.SAC_AUTO_ALPHA = args.auto_alpha
_v35.SAC_LOG_STD_MAX = args.sac_log_std_max
_v35.ENFORCE_REF_SOURCE_LIMIT = True
_v35.UNIFORM_RESIDUAL_MW = args.residual_scale
_v35.UNIFORM_CUMULATIVE_MULT = args.cumulative_scale_mult
_v35.MAX_STEPS = args.max_steps; _v35.MAX_STEPS_HIGH = args.max_steps
SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

from train_rlv35_complete import (SCEDEnvV21, split_samples_stratified,
    DEVICE, N_SHIFT_DIRECTIONS, N_KEY_GENERATORS, BATCH_SIZE, BUFFER_SIZE,
    project_box_sum, adopted_operation_pass)
try:
    from route_b_split_utils import load_master_split
except Exception:
    load_master_split = None
TOTAL_UPDATES=args.updates; BATCH=256; DEV=DEVICE
VAL_EVAL_INTERVAL=args.val_interval
EPISODES_PER_UPDATE=args.episodes_per_update

class Actor(nn.Module):
    def __init__(self,od,ad,h=256,zero_output=True):
        super().__init__()
        hidden1 = nn.Linear(od,h)
        hidden2 = nn.Linear(h,h)
        output = nn.Linear(h,ad)
        if zero_output:
            nn.init.zeros_(output.weight); nn.init.zeros_(output.bias)
        self.net=nn.Sequential(hidden1,nn.ReLU(),hidden2,nn.ReLU(),output,nn.Tanh())
    def forward(self,x): return self.net(x)
class Critic(nn.Module):
    def __init__(self,od,ad,h=256,zero_output=True):
        super().__init__()
        hidden1 = nn.Linear(od+ad,h)
        hidden2 = nn.Linear(h,h)
        output = nn.Linear(h,1)
        if zero_output:
            nn.init.zeros_(output.weight); nn.init.zeros_(output.bias)
        self.net=nn.Sequential(hidden1,nn.ReLU(),hidden2,nn.ReLU(),output)
    def forward(self,x,a): return self.net(torch.cat([x,a],-1))
class TwinCritic(nn.Module):
    def __init__(self,od,ad,h=256,zero_output=True): super().__init__(); self.q1=Critic(od,ad,h,zero_output); self.q2=Critic(od,ad,h,zero_output)
    def forward(self,x,a): return self.q1(x,a),self.q2(x,a)

class DDPG:
    def __init__(self,od,ad,actor_lr,critic_lr,gradient_steps,critic_warmup,policy_freq):
        self.actor=Actor(od,ad).to(DEV); self.at=Actor(od,ad).to(DEV); self.at.load_state_dict(self.actor.state_dict())
        self.critic=Critic(od,ad).to(DEV); self.ct=Critic(od,ad).to(DEV); self.ct.load_state_dict(self.critic.state_dict())
        self.ao=torch.optim.Adam(self.actor.parameters(),lr=actor_lr); self.co=torch.optim.Adam(self.critic.parameters(),lr=critic_lr)
        self.buf=deque(maxlen=BUFFER_SIZE); self.gs=0; self.actor_updates=0; self.gradient_steps=gradient_steps
        self.critic_warmup=max(0,int(critic_warmup)); self.policy_freq=max(1,int(policy_freq))
    def act(self,obs,det=True,noise_std=0.0):
        with torch.no_grad(): o=torch.FloatTensor(obs).unsqueeze(0).to(DEV); a=self.actor(o)
        if not det and noise_std>0: a=(a+torch.randn_like(a)*noise_std).clamp(-1,1)
        return a.cpu().numpy()[0]
    def update(self):
        if len(self.buf)<BATCH: return
        for _ in range(self.gradient_steps):
            idx=np.random.choice(len(self.buf),min(BATCH,len(self.buf)),replace=False)
            batch=[self.buf[i] for i in idx]
            s=torch.FloatTensor(np.array([b[0] for b in batch])).to(DEV); a=torch.FloatTensor(np.array([b[1] for b in batch])).to(DEV)
            r=torch.FloatTensor(np.array([b[2] for b in batch])).unsqueeze(-1).to(DEV)
            ns=torch.FloatTensor(np.array([b[3] for b in batch])).to(DEV); d=torch.FloatTensor(np.array([b[4] for b in batch])).unsqueeze(-1).to(DEV)
            with torch.no_grad(): tq=r+0.99*(1-d)*self.ct(ns,self.at(ns))
            cl=F.mse_loss(self.critic(s,a),tq); self.co.zero_grad(); cl.backward(); nn.utils.clip_grad_norm_(self.critic.parameters(),1.0); self.co.step()
            actor_due = self.gs >= self.critic_warmup and ((self.gs-self.critic_warmup)%self.policy_freq==0)
            if actor_due:
                al=-self.critic(s,self.actor(s)).mean(); self.ao.zero_grad(); al.backward(); nn.utils.clip_grad_norm_(self.actor.parameters(),1.0); self.ao.step()
                for p,tp in zip(self.actor.parameters(),self.at.parameters()): tp.data.copy_(0.005*p+0.995*tp)
                self.actor_updates += 1
            for p,tp in zip(self.critic.parameters(),self.ct.parameters()): tp.data.copy_(0.005*p+0.995*tp)
            self.gs+=1

class TD3:
    def __init__(self,od,ad,actor_lr,critic_lr,gradient_steps,policy_noise,noise_clip,critic_warmup,policy_freq):
        # Standard TD3 initialization/profile retained because it is the
        # empirically validated comparator on the corrected 277-sample set.
        self.actor=Actor(od,ad,zero_output=False).to(DEV); self.at=Actor(od,ad,zero_output=False).to(DEV); self.at.load_state_dict(self.actor.state_dict())
        self.critic=TwinCritic(od,ad,zero_output=False).to(DEV); self.ct=TwinCritic(od,ad,zero_output=False).to(DEV); self.ct.load_state_dict(self.critic.state_dict())
        self.ao=torch.optim.Adam(self.actor.parameters(),lr=actor_lr); self.co=torch.optim.Adam(self.critic.parameters(),lr=critic_lr)
        self.buf=deque(maxlen=BUFFER_SIZE); self.gs=0; self.actor_updates=0; self.gradient_steps=gradient_steps
        self.policy_noise=policy_noise; self.noise_clip=noise_clip
        self.critic_warmup=max(0,int(critic_warmup)); self.policy_freq=max(1,int(policy_freq))
    def act(self,obs,det=True,noise_std=0.0):
        with torch.no_grad(): o=torch.FloatTensor(obs).unsqueeze(0).to(DEV); a=self.actor(o)
        if not det and noise_std>0: a=(a+torch.randn_like(a)*noise_std).clamp(-1,1)
        return a.cpu().numpy()[0]
    def update(self):
        if len(self.buf)<BATCH: return
        for _ in range(self.gradient_steps):
            idx=np.random.choice(len(self.buf),min(BATCH,len(self.buf)),replace=False)
            batch=[self.buf[i] for i in idx]
            s=torch.FloatTensor(np.array([b[0] for b in batch])).to(DEV); a=torch.FloatTensor(np.array([b[1] for b in batch])).to(DEV)
            r=torch.FloatTensor(np.array([b[2] for b in batch])).unsqueeze(-1).to(DEV)
            ns=torch.FloatTensor(np.array([b[3] for b in batch])).to(DEV); d=torch.FloatTensor(np.array([b[4] for b in batch])).unsqueeze(-1).to(DEV)
            with torch.no_grad():
                noise=(torch.randn_like(a)*self.policy_noise).clamp(-self.noise_clip,self.noise_clip); na=(self.at(ns)+noise).clamp(-1,1); q1t,q2t=self.ct(ns,na)
                tq=r+0.99*(1-d)*torch.min(q1t,q2t)
            q1,q2=self.critic(s,a); cl=F.mse_loss(q1,tq.detach())+F.mse_loss(q2,tq.detach())
            self.co.zero_grad(); cl.backward(); nn.utils.clip_grad_norm_(self.critic.parameters(),1.0); self.co.step()
            actor_due = self.gs >= self.critic_warmup and ((self.gs-self.critic_warmup)%self.policy_freq==0)
            if actor_due:
                al=-self.critic.q1(s,self.actor(s)).mean(); self.ao.zero_grad(); al.backward(); nn.utils.clip_grad_norm_(self.actor.parameters(),1.0); self.ao.step()
                for p,tp in zip(self.actor.parameters(),self.at.parameters()): tp.data.copy_(0.005*p+0.995*tp)
                self.actor_updates += 1
                for p,tp in zip(self.critic.parameters(),self.ct.parameters()): tp.data.copy_(0.005*p+0.995*tp)
            self.gs+=1

# Importing the PPO comparison module mutates shared V35 hyperparameters.
# Keep it out of SAC/DDPG/TD3 processes entirely.
PPO = None
if algo_name == "PPO":
    from train_rl_comparison_v3 import PPOTrainer as PPO

# Lock shared settings after all algorithm-specific imports. This also guards
# against future modules silently changing SAC's update schedule.
_v35.AC_PF_CHECK_EVERY = max(1, args.ac_pf_check_every)
_v35.N1_CHECK_EVERY_STEPS = max(1, args.n1_check_every)
_v35.N_GRADIENT_STEPS = args.gradient_steps
_v35.N_EPISODES_PER_UPDATE = args.episodes_per_update
_v35.BATCH_SIZE = 256
_v35.MAX_STEPS = args.max_steps
_v35.MAX_STEPS_HIGH = args.max_steps
_v35.SAC_INIT_ALPHA = args.sac_init_alpha
_v35.SAC_LR_ACTOR = args.actor_lr
_v35.SAC_LR_CRITIC = args.critic_lr
_v35.SAC_LR_ALPHA = args.alpha_lr
_v35.SAC_AUTO_ALPHA = args.auto_alpha
_v35.SAC_LOG_STD_MAX = args.sac_log_std_max
_v35.ENFORCE_REF_SOURCE_LIMIT = True
_v35.UNIFORM_RESIDUAL_MW = args.residual_scale
_v35.UNIFORM_CUMULATIVE_MULT = args.cumulative_scale_mult

# SAC scratch using V35 SACTrainer
from train_rlv35_complete import SACTrainer as SAC_Scratch

class RouteBAblationEnv(SCEDEnvV21):
    def __init__(self, *env_args, **env_kwargs):
        super().__init__(
            *env_args,
            **env_kwargs,
            action_encoding=args.action_encoding,
            action_basis_seed=args.action_basis_seed,
        )
        if not (0.0 < float(args.n1_limit_scale) <= 1.0):
            raise ValueError("--n1-limit-scale must be in (0, 1]")
        self.dcnet.effective_limit = (
            np.asarray(self.dcnet.effective_limit, dtype=np.float64)
            * float(args.n1_limit_scale)
        )

    def reset(self, sample_idx=None):
        obs = super().reset(sample_idx=sample_idx)
        if "no-gat" in variant_name:
            p0 = self.PMAX / (self.PMAX.sum() + 1e-8) * self.target_demand
            p0 = project_box_sum(p0, self.PMIN, self.PMAX * 0.99, self.target_demand)
            self.gat_power = p0.astype(np.float64).copy()
            self.current_power = self.gat_power.copy()
            self.cumulative_delta = np.zeros(self.N, dtype=np.float64)
            self.reset_audit.update({
                "initial_power_source": "capacity_proportional_no_gat",
                "uses_gat_prediction_as_initial_power": False,
            })
            obs = self._get_obs()
        return obs

    def step(self, action):
        if "no-voltage" in variant_name:
            action = np.asarray(action, dtype=np.float64).copy()
            p_dim = self._n_shift_dirs + self._n_key_gens
            if len(action) > p_dim:
                action[p_dim:] = 0.0
        return super().step(action)


EnvClass = RouteBAblationEnv

print(f"Training {algo_name} ({variant_name}) for {TOTAL_UPDATES} updates on {DEV}")
print(
    f"Optimizer config: grad_steps={_v35.N_GRADIENT_STEPS}, "
    f"learning_starts={args.learning_starts}, actor_updates_per_outer={_v35.N_GRADIENT_STEPS}, "
    f"actor_gate=none, "
    f"alpha0={_v35.SAC_INIT_ALPHA:g}, auto_alpha={_v35.SAC_AUTO_ALPHA}, log_std_bias={args.sac_log_std_bias:g}, "
    f"log_std_max={args.sac_log_std_max:g}, rollout={args.sac_rollout_mode}, "
    f"reward_scale={args.reward_scale:g}, actor_lr={args.actor_lr:g}, "
    f"critic_lr={args.critic_lr:g}, fixed_log_std={not args.learn_log_std}, "
    f"stochastic_steps={args.stochastic_steps}, "
    f"failure_gated_replay={args.failure_gated_replay}, failure_offset={args.failure_reward_offset:g}, "
    f"success_bc_steps={args.success_bc_steps}, val_rollback={args.val_rollback}, "
    f"acpf_every={_v35.AC_PF_CHECK_EVERY}, n1_every={_v35.N1_CHECK_EVERY_STEPS}, "
    f"voltage_explore_gain={args.voltage_explore_gain:g}, "
    f"active_explore_gain={args.active_explore_gain:g}, "
    f"n1_guide={args.n1_guide_gain:g}@U1-{args.n1_guide_until} "
    f"decay_from_U{args.n1_guide_decay_start}, "
    f"voltage_explore_until={args.voltage_explore_until}, "
    f"enforce_ref_source={_v35.ENFORCE_REF_SOURCE_LIMIT}, "
    f"action_encoding={args.action_encoding}, "
    f"n1_limit_scale={args.n1_limit_scale:g}"
)
if algo_name == "SAC" and _v35.N_GRADIENT_STEPS != args.gradient_steps:
    raise RuntimeError(
        f"SAC gradient-step configuration was mutated: "
        f"requested={args.gradient_steps}, actual={_v35.N_GRADIENT_STEPS}"
    )
with open(dataset_path, 'r', encoding='utf-8') as f: data=json.load(f)
samples=[s for s in data['samples'] if s.get('is_complete_sced')]
if args.split_file:
    if load_master_split is None:
        raise RuntimeError("route_b_split_utils could not be imported")
    split = load_master_split(args.split_file)
    recorded_dataset = split.get("dataset")
    if recorded_dataset and Path(recorded_dataset).resolve() != dataset_path.resolve():
        raise ValueError(
            f"Split dataset mismatch: split={recorded_dataset}, requested={dataset_path}"
        )
    train_ids, val_ids, test_ids = split["train_ids"], split["val_ids"], split["test_ids"]
    split_source = str(args.split_file)
else:
    train_ids,val_ids,test_ids = split_samples_stratified(samples)
    split_source = "generated_by_split_samples_stratified"
all_split_ids = train_ids + val_ids + test_ids
if not all_split_ids or min(all_split_ids) < 0 or max(all_split_ids) >= len(samples):
    raise ValueError(f"Split indices are invalid for dataset with {len(samples)} samples")
env=EnvClass(samples,train_ids,dataset_meta=data); od=env._get_obs_dim(); ad=env.action_dim
print(f"ObsDim={od}, ActDim={ad}")
print(
    f"Action encoding: {args.action_encoding} "
    f"(active={env._n_shift_dirs + env._n_key_gens}, "
    f"voltage={env._n_voltage_actions}, basis_seed={args.action_basis_seed})"
)
print(f"Split: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")
print(f"Split source: {split_source}")
print(f"Dataset: {dataset_path}")
print(f"GAT checkpoint: {gat_checkpoint}")

def demand_key(sample):
    return round(float(sample.get('demand_gw', float(sample['demand_mw'])/1000.0)), 3)

if algo_name=='DDPG':
    t=DDPG(od,ad,args.det_actor_lr,args.det_critic_lr,args.det_gradient_steps,
           args.det_critic_warmup_steps,args.det_policy_freq); is_ppo=False; is_sac=False
elif algo_name=='TD3':
    t=TD3(od,ad,args.td3_actor_lr,args.td3_critic_lr,args.td3_gradient_steps,
          args.td3_policy_noise,args.td3_noise_clip,args.td3_critic_warmup_steps,
          args.td3_policy_freq); is_ppo=False; is_sac=False
elif algo_name=='PPO':
    t=PPO(od,ad,lr=args.ppo_lr,epochs=args.ppo_epochs,
          entropy_coef=args.ppo_entropy_coef,log_std_init=args.ppo_log_std_init,
          learn_log_std=args.ppo_learn_log_std,target_kl=args.ppo_target_kl,
          critic_lr=args.ppo_critic_lr)
    is_ppo=True; is_sac=False
else: t=SAC_Scratch(od,ad,device=DEV); is_ppo=False; is_sac=True
if algo_name in ('DDPG','TD3'):
    shown_actor_lr = args.td3_actor_lr if algo_name == 'TD3' else args.det_actor_lr
    shown_critic_lr = args.td3_critic_lr if algo_name == 'TD3' else args.det_critic_lr
    shown_grad_steps = args.td3_gradient_steps if algo_name == 'TD3' else args.det_gradient_steps
    shown_noise_start = args.td3_noise_start if algo_name == 'TD3' else args.det_noise_start
    shown_noise_end = args.td3_noise_end if algo_name == 'TD3' else args.det_noise_end
    shown_reward_scale = args.td3_reward_scale if algo_name == 'TD3' else args.det_reward_scale
    shown_replay_warmup = args.td3_learning_starts if algo_name == 'TD3' else args.det_learning_starts
    print(
        f"{algo_name} optimizer: actor_lr={shown_actor_lr:g} critic_lr={shown_critic_lr:g} "
        f"grad_steps={shown_grad_steps} noise={shown_noise_start:g}->{shown_noise_end:g} "
        f"reward_scale={shown_reward_scale:g} replay_warmup={shown_replay_warmup} "
        f"critic_warmup={args.td3_critic_warmup_steps if algo_name == 'TD3' else args.det_critic_warmup_steps} "
        f"policy_freq={args.td3_policy_freq if algo_name == 'TD3' else args.det_policy_freq}"
    )
elif is_ppo:
    print(
        f"PPO optimizer: actor_lr={args.ppo_lr:g} critic_lr={args.ppo_critic_lr:g} epochs={args.ppo_epochs} "
        f"log_std={args.ppo_log_std_init:g}->{args.ppo_log_std_final:g} "
        f"reward_scale={args.ppo_reward_scale:g} entropy_coef={args.ppo_entropy_coef:g}"
    )
if is_sac:
    with torch.no_grad():
        t.actor.log_std_head.weight.zero_()
        t.actor.log_std_head.bias.fill_(args.sac_log_std_bias)
    if not args.learn_log_std:
        for param in t.actor.log_std_head.parameters():
            param.requires_grad_(False)


# Cheap fixed probes reveal mean-policy saturation without extra AC-PF runs.
probe_ids = []
seen_probe_gw = set()
for sid in train_ids:
    gw = demand_key(samples[sid])
    if gw not in seen_probe_gw:
        seen_probe_gw.add(gw)
        probe_ids.append(sid)
policy_probe_obs = np.asarray([env.reset(sample_idx=sid) for sid in probe_ids], dtype=np.float32)

def stratified_random_batch(samples, ids, n_total):
    by_gw={}
    for sid in ids:
        by_gw.setdefault(demand_key(samples[sid]), []).append(sid)
    keys=sorted(by_gw)
    selected=[]
    for k in keys:
        selected.append(random.choice(by_gw[k]))
    remaining=max(0, n_total-len(selected))
    pool=[sid for k in keys for sid in by_gw[k] if sid not in selected]
    if remaining and pool:
        selected.extend(random.sample(pool, min(remaining, len(pool))))
    if len(selected) > n_total:
        selected=random.sample(selected, n_total)
    random.shuffle(selected)
    return selected

def fixed_stratified_probe_ids(samples, ids, n_total):
    if n_total <= 0:
        return []
    by_gw={}
    for sid in ids:
        by_gw.setdefault(demand_key(samples[sid]), []).append(sid)
    selected=[]
    for k in sorted(by_gw):
        selected.append(by_gw[k][0])
        if len(selected) >= n_total:
            return selected
    for k in sorted(by_gw):
        for sid in by_gw[k][1:]:
            selected.append(sid)
            if len(selected) >= n_total:
                return selected
    return selected

def eval_policy(env, trainer, ids):
    ok=0; total=0; rewards_eval=[]; dense_rewards_eval=[]; reasons=Counter()
    steps_eval=[]; n1_passed=0; n1_overloads=[]
    for sid in ids:
        obs=env.reset(sample_idx=sid); last_info={}; er=0.0
        for _ in range(env.max_steps+1):
            if is_sac:
                raw=trainer.select_action(obs,deterministic=True)
            elif is_ppo:
                ret=trainer.select_action(obs,deterministic=True); raw=ret[0]
            else:
                raw=trainer.act(obs,det=True)
            lo,hi=env.get_action_bounds(); a=lo+(raw+1.0)*0.5*(hi-lo)
            obs,r,done,_trunc,last_info=env.step(a)
            er += float(np.nan_to_num(r,nan=-100.0,posinf=1000.0,neginf=-1000.0))
            if done: break
        passed=(last_info.get('reason')=='full_sced_pass' or adopted_operation_pass(last_info))
        # Keep the dense step reward for learning, but make the reported
        # episode score rank every failed trajectory below every success.
        er_report = er if passed else min(er, -1.0)
        ok += int(passed); total += 1; rewards_eval.append(er_report); dense_rewards_eval.append(er)
        steps_eval.append(int(env.corr_step))
        n1_viol = int(last_info.get('n1_violations', 9999))
        n1_passed += int(n1_viol == 0)
        n1_overloads.append(float(last_info.get('n1_overload_mw', 9999.0)))
        if not passed: reasons[last_info.get('reason','unknown')] += 1
    return {
        "passed": ok,
        "total": total,
        "avg_reward": float(np.mean(rewards_eval)) if rewards_eval else 0.0,
        "avg_dense_reward": float(np.mean(dense_rewards_eval)) if dense_rewards_eval else 0.0,
        "avg_steps": float(np.mean(steps_eval)) if steps_eval else 0.0,
        "n1_passed": int(n1_passed),
        "avg_n1_overload_mw": float(np.mean(n1_overloads)) if n1_overloads else 0.0,
        "max_n1_overload_mw": float(np.max(n1_overloads)) if n1_overloads else 0.0,
        "reasons": dict(reasons),
    }

def deterministic_probe_rank(result):
    """Lexicographic score for guarding the deployable actor mean."""
    overload = float(result.get("avg_n1_overload_mw", 1e12))
    if not np.isfinite(overload):
        overload = 1e12
    return (
        int(result.get("passed", 0)),
        int(result.get("n1_passed", 0)),
        -overload,
        float(result.get("avg_dense_reward", -1e18)),
    )

def apply_guided_exploration(raw_action, env, update_num):
    """Bias stochastic training actions toward reducing the last observed voltage gap.

    The active-power part uses only the current PTDF/LODF basis and local
    voltage-gap proxy; it does not read stored rescue actions.
    """
    if args.voltage_explore_gain <= 0.0 and args.active_explore_gain <= 0.0:
        return raw_action
    if args.voltage_explore_until > 0 and update_num > args.voltage_explore_until:
        return raw_action
    if args.guided_decay_start > 0 and update_num > args.guided_decay_start:
        end = max(args.voltage_explore_until, args.guided_decay_start + 1)
        guide_scale = max(0.0, (end - update_num + 1) / max(1, end - args.guided_decay_start + 1))
    else:
        guide_scale = 1.0
    if guide_scale <= 0.0:
        return raw_action
    p_dim = env._n_shift_dirs + env._n_key_gens
    out = np.asarray(raw_action, dtype=np.float64).copy()
    gap = np.asarray(getattr(env, "_gen_bus_v_gap_pu", np.zeros(env.N)), dtype=np.float64)
    if gap.shape[0] != env.N or not np.isfinite(gap).all():
        return raw_action
    gap_norm = np.clip(gap / 0.05, -1.0, 1.0)

    if args.active_explore_gain > 0.0 and p_dim > 0:
        try:
            if env._shift_directions is None:
                env._compute_shift_directions()
            if env._key_gen_indices is None:
                env._select_key_generators()
        except Exception:
            pass

        if getattr(env, "_shift_directions", None) is not None:
            dirs = np.asarray(env._shift_directions, dtype=np.float64)
            scores = dirs @ gap_norm
            max_abs = max(float(np.max(np.abs(scores))) if len(scores) else 0.0, 1e-6)
            out[:env._n_shift_dirs] = np.clip(
                out[:env._n_shift_dirs] + guide_scale * args.active_explore_gain * scores / max_abs,
                -1.0, 1.0,
            )

        key_idx = getattr(env, "_key_gen_indices", None)
        if key_idx is not None and len(key_idx):
            key_bias = gap_norm[np.asarray(key_idx, dtype=int)]
            out[env._n_shift_dirs:p_dim] = np.clip(
                out[env._n_shift_dirs:p_dim] + guide_scale * args.active_explore_gain * key_bias,
                -1.0, 1.0,
            )

    if len(out) <= p_dim:
        return out.astype(np.float32)

    group_bias = []
    for gen_idx in env._voltage_group_indices:
        if len(gen_idx):
            group_bias.append(float(np.mean(gap[gen_idx])))
        else:
            group_bias.append(0.0)
    ext_bias = float(np.mean(gap))
    v_bias = np.asarray(group_bias + [ext_bias], dtype=np.float64)
    raw_bias = guide_scale * args.voltage_explore_gain * v_bias / max(float(_v35.VOLTAGE_STEP_PU), 1e-6)
    # Generator-bus gaps do not reveal every remote load-bus violation. Add a
    # small common-mode correction from the latest real AC-PF extrema so low
    # remote buses can be restored without any offline rescue voltage target.
    last_ac = np.asarray(getattr(env, "prev_ac_result", np.zeros(4)), dtype=np.float64)
    if last_ac.shape == (4,) and bool(last_ac[0]) and np.isfinite(last_ac).all():
        low_gap = max(0.0, 0.933 - float(last_ac[1]))
        high_gap = max(0.0, float(last_ac[2]) - 1.09)
        common_bias = np.clip((low_gap - high_gap) / 0.02, -1.0, 1.0)
        raw_bias += guide_scale * args.voltage_explore_gain * common_bias
    out[p_dim:] = np.clip(out[p_dim:] + raw_bias, -1.0, 1.0)
    return out.astype(np.float32)

def apply_n1_guided_exploration(raw_action, env, update_num):
    """Bias exploration along instantaneous PTDF/LODF overload relief.

    The guide is calculated only from the current dispatch and the environment
    sensitivity model.  Dataset rescue evidence is deliberately inaccessible
    here and is never used as an action target.
    """
    if args.n1_guide_gain <= 0.0 or update_num > args.n1_guide_until:
        return raw_action
    if update_num <= args.n1_guide_decay_start:
        decay = 1.0
    else:
        span = max(1, args.n1_guide_until - args.n1_guide_decay_start)
        decay = max(0.0, (args.n1_guide_until - update_num + 1) / span)
    if decay <= 0.0:
        return raw_action

    try:
        if env._shift_directions is None:
            env._compute_shift_directions()
        if env._key_gen_indices is None:
            env._select_key_generators()
        sensitivity = np.asarray(env._n1_sensitivity, dtype=np.float64)
        directions = np.asarray(env._shift_directions, dtype=np.float64)
    except Exception:
        return raw_action
    if sensitivity.shape != (env.N,) or not np.isfinite(sensitivity).all():
        return raw_action

    out = np.asarray(raw_action, dtype=np.float64).copy()
    gain = float(args.n1_guide_gain) * float(decay)
    if len(directions):
        # Positive score means movement along this coordinate reduces the
        # current aggregate post-contingency overload.
        shift_scores = -(directions @ sensitivity)
        denom = max(float(np.max(np.abs(shift_scores))), 1e-9)
        out[:env._n_shift_dirs] += gain * shift_scores / denom

    key_indices = np.asarray(env._key_gen_indices, dtype=int)
    if len(key_indices):
        key_scores = -sensitivity[key_indices]
        denom = max(float(np.max(np.abs(key_scores))), 1e-9)
        start = env._n_shift_dirs
        stop = start + env._n_key_gens
        out[start:stop] += 0.5 * gain * key_scores / denom
    return np.clip(out, -1.0, 1.0).astype(np.float32)

def current_rollout_noise(update_num):
    denom = max(1, int(args.rollout_noise_decay_updates) - 1)
    frac = min(1.0, max(0.0, (int(update_num) - 1) / denom))
    return float(args.rollout_noise_start) + frac * (
        float(args.rollout_noise_end) - float(args.rollout_noise_start)
    )

def current_deterministic_noise(update_num):
    start = args.td3_noise_start if algo_name == 'TD3' else args.det_noise_start
    end = args.td3_noise_end if algo_name == 'TD3' else args.det_noise_end
    decay = args.td3_noise_decay_updates if algo_name == 'TD3' else args.det_noise_decay_updates
    denom = max(1, int(decay) - 1)
    frac = min(1.0, max(0.0, (int(update_num) - 1) / denom))
    return float(start) + frac * (float(end) - float(start))

def set_deterministic_learning_rates(trainer, update_num):
    """Anneal off-policy learning rates after the policy reaches its usual plateau."""
    if algo_name == "TD3":
        start = int(args.td3_lr_decay_start)
        actor_base = float(args.td3_actor_lr)
        critic_base = float(args.td3_critic_lr)
        actor_final = float(args.td3_actor_lr_final_factor)
        critic_final = float(args.td3_critic_lr_final_factor)
    else:
        start = int(args.det_lr_decay_start)
        actor_base = float(args.det_actor_lr)
        critic_base = float(args.det_critic_lr)
        actor_final = float(args.det_actor_lr_final_factor)
        critic_final = float(args.det_critic_lr_final_factor)
    if int(update_num) <= start:
        frac = 0.0
    else:
        frac = min(1.0, (int(update_num) - start) / max(1, TOTAL_UPDATES - start))
    actor_factor = 1.0 + frac * (actor_final - 1.0)
    critic_factor = 1.0 + frac * (critic_final - 1.0)
    actor_lr = actor_base * actor_factor
    critic_lr = critic_base * critic_factor
    for group in trainer.ao.param_groups:
        group["lr"] = actor_lr
    for group in trainer.co.param_groups:
        group["lr"] = critic_lr
    return actor_lr, critic_lr

def current_ppo_log_std(update_num):
    denom = max(1, int(args.ppo_std_decay_updates) - 1)
    frac = min(1.0, max(0.0, (int(update_num) - 1) / denom))
    return float(args.ppo_log_std_init) + frac * (
        float(args.ppo_log_std_final) - float(args.ppo_log_std_init)
    )

def sac_rollout_action(trainer, obs, update_num, step_num):
    deterministic_tail = args.stochastic_steps >= 0 and step_num >= args.stochastic_steps
    if args.sac_rollout_mode == "sample" and not deterministic_tail:
        raw_action = trainer.select_action(obs, deterministic=False)
    else:
        raw_action = trainer.select_action(obs, deterministic=True)
        if args.sac_rollout_mode == "mean_noise" and not deterministic_tail:
            sigma = current_rollout_noise(update_num)
            if sigma > 0.0:
                raw_action = np.asarray(raw_action, dtype=np.float32) + np.random.normal(
                    0.0, sigma, size=np.asarray(raw_action).shape
                ).astype(np.float32)
    return np.clip(raw_action, -1.0, 1.0).astype(np.float32)

def success_bc_update(trainer, success_buffer, n_steps):
    if n_steps <= 0 or not success_buffer:
        return 0
    n_done = 0
    bs = min(int(args.success_bc_batch), len(success_buffer))
    for _ in range(int(n_steps)):
        batch = random.sample(list(success_buffer), bs)
        states = torch.as_tensor(np.asarray([b[0] for b in batch], dtype=np.float32), device=DEV)
        target_actions = torch.as_tensor(np.asarray([b[1] for b in batch], dtype=np.float32), device=DEV)
        mean, _log_std = trainer.actor.forward(states)
        pred_actions = torch.tanh(mean)
        loss = float(args.success_bc_weight) * F.mse_loss(pred_actions, target_actions)
        if not torch.isfinite(loss):
            continue
        trainer.actor_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(trainer.actor.parameters(), 1.0)
        trainer.actor_optimizer.step()
        n_done += 1
    return n_done

def snapshot_sac_state(trainer):
    """Capture every trainable SAC component needed for a consistent rollback."""
    return {
        "actor": copy.deepcopy(trainer.actor.state_dict()),
        "critic": copy.deepcopy(trainer.critic.state_dict()),
        "critic_target": copy.deepcopy(trainer.critic_target.state_dict()),
        "actor_optimizer": copy.deepcopy(trainer.actor_optimizer.state_dict()),
        "critic_optimizer": copy.deepcopy(trainer.critic_optimizer.state_dict()),
        "alpha_optimizer": copy.deepcopy(trainer.alpha_optimizer.state_dict()),
        "log_alpha": trainer.log_alpha.detach().clone(),
        "total_updates": int(trainer.total_updates),
    }

def restore_sac_state(trainer, state):
    """Restore actor, critics, optimizers and entropy temperature together."""
    trainer.actor.load_state_dict(state["actor"])
    trainer.critic.load_state_dict(state["critic"])
    trainer.critic_target.load_state_dict(state["critic_target"])
    trainer.actor_optimizer.load_state_dict(state["actor_optimizer"])
    trainer.critic_optimizer.load_state_dict(state["critic_optimizer"])
    trainer.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
    with torch.no_grad():
        trainer.log_alpha.copy_(state["log_alpha"].to(trainer.device))
    trainer.total_updates = int(state["total_updates"])

val_env=EnvClass(samples,val_ids,dataset_meta=data)
rollback_probe_count = max(
    int(args.train_rollback_probe_samples),
    int(args.safe_candidate_eval_samples),
)
rollback_probe_ids = fixed_stratified_probe_ids(
    samples, train_ids, rollback_probe_count
) if is_sac else []
success_bc_buffer = (
    deque(maxlen=int(args.success_buffer_size))
    if is_sac and int(args.success_bc_steps) > 0 else None
)
rewards=[]; pass_history=[]; val_history=[]; best_r=-1e9; best_pass=-1; best_total=0; best_val=-1; best_val_total=0; best_val_r=-1e18; t0=time.time()
success_episode_total = 0
best_eval_actor_state = None
det_bad_val_checks = 0
suffix_parts = []
if variant_name != "default":
    suffix_parts.append(variant_name.replace('-', '_'))
if args.action_encoding != "physics32":
    suffix_parts.append(args.action_encoding)
if abs(float(args.n1_limit_scale) - 1.0) > 1e-12:
    suffix_parts.append(f"n1s{float(args.n1_limit_scale):.3f}".replace('.', 'p'))
suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
save_dir=Path(args.output_dir) if args.output_dir else PROJ_DIR/f'rl_scratch_{algo_name.lower()}{suffix}'
if not save_dir.is_absolute():
    save_dir = PROJ_DIR / save_dir
save_dir.mkdir(exist_ok=True)
best_probe_pass = -1
best_probe_reward = -1e18
best_probe_rank = None
best_probe_state = None
best_val_state = None
with open(str(save_dir/'split_ids.json'),'w') as f:
    json.dump({
        "train_ids":train_ids,
        "val_ids":val_ids,
        "test_ids":test_ids,
        "split_source":split_source,
        "dataset":str(dataset_path.resolve()),
        "n_samples":len(samples),
        "dropped_by_lodf_audit":data.get("lodf_correction_summary",{}).get("dropped_sample_ids",[]),
        "action_encoding": args.action_encoding,
        "action_basis_seed": int(args.action_basis_seed),
        "n1_limit_scale": float(args.n1_limit_scale),
    },f,indent=2)
if is_sac and rollback_probe_ids:
    probe_start = time.time()
    probe_res = eval_policy(env, t, rollback_probe_ids)
    best_probe_pass = int(probe_res["passed"])
    best_probe_reward = float(probe_res["avg_reward"])
    best_probe_rank = deterministic_probe_rank(probe_res)
    best_probe_state = snapshot_sac_state(t)
    print(
        f"Initial train probe: {best_probe_pass}/{len(rollback_probe_ids)} "
        f"R={best_probe_reward:+.0f} ({time.time()-probe_start:.0f}s)"
    )

for u in range(1,TOTAL_UPDATES+1):
    stop_training = False
    det_actor_lr_now = None
    det_critic_lr_now = None
    if not is_sac and not is_ppo:
        det_actor_lr_now, det_critic_lr_now = set_deterministic_learning_rates(t, u)
    if is_ppo and not args.ppo_learn_log_std:
        with torch.no_grad():
            t.actor.log_std.fill_(current_ppo_log_std(u))
    eps=stratified_random_batch(samples,train_ids,EPISODES_PER_UPDATE)
    us=time.time(); ep_rs=[]; ep_dense_rs=[]; ep_steps=[]; ep_pass=0; fail_reasons=Counter(); fail_examples=Counter()
    collect_start=time.time()
    for sid in eps:
        obs=env.reset(sample_idx=sid); er=0.0
        last_info={}
        sac_episode_transitions=[]
        for step in range(env.max_steps+1):
            if is_sac:
                raw_a=sac_rollout_action(t, obs, u, step)
                raw_a=apply_guided_exploration(raw_a, env, u)
                raw_a=apply_n1_guided_exploration(raw_a, env, u)
                lo,hi=env.get_action_bounds(); a=lo+(raw_a+1.0)*0.5*(hi-lo)
                no,r,done,_,info=env.step(a); r=float(np.nan_to_num(r,nan=-100.0,posinf=1000.0,neginf=-1000.0)); er+=r
                sac_episode_transitions.append((obs, raw_a, r, no, float(done)))
            elif is_ppo:
                ret=t.select_action(obs,deterministic=False)
                raw_a,lp,v=ret[0],ret[1],ret[2]
                lo,hi=env.get_action_bounds(); a=lo+(raw_a+1.0)*0.5*(hi-lo)
                no,r,done,_,info=env.step(a); r=float(np.nan_to_num(r,nan=-100.0,posinf=1000.0,neginf=-1000.0)); er+=r
                # A max-step reset is also an episode boundary. Leaving it as
                # done=False makes GAE bootstrap into the next, unrelated
                # operating sample and corrupts the policy advantages.
                ppo_boundary = bool(done or step >= env.max_steps)
                t.store(obs,raw_a,r*args.ppo_reward_scale,no,float(ppo_boundary),lp,v)
            else:
                raw_a=t.act(obs,det=False,noise_std=current_deterministic_noise(u))
                lo,hi=env.get_action_bounds(); a=lo+(raw_a+1.0)*0.5*(hi-lo)
                no,r,done,_,info=env.step(a); r=float(np.nan_to_num(r,nan=-100.0,posinf=1000.0,neginf=-1000.0)); er+=r
                det_reward_scale = args.td3_reward_scale if algo_name == 'TD3' else args.det_reward_scale
                t.buf.append((obs,raw_a,r*det_reward_scale,no,float(done)))
            last_info=info
            obs=no
            if done: break
        ep_steps.append(step + 1)
        if (last_info.get('reason') == 'full_sced_pass' or adopted_operation_pass(last_info)):
            ep_pass += 1
        else:
            reason = last_info.get('reason', 'unknown')
            fail_reasons[reason] += 1
            if reason == 'voltage_violation':
                detail = (
                    f"{reason}:vv={last_info.get('v_violations', '?')}"
                    f":low={last_info.get('v_low_violations', '?')}"
                    f":high={last_info.get('v_high_violations', '?')}"
                    f":vgap={last_info.get('v_violation_max_pu', 0):.4f}"
                )
            elif reason == 'acpf_only_n1_fail':
                detail = f"{reason}:n1={last_info.get('n1_violations', '?')}"
            elif reason == 'line_overload':
                detail = f"{reason}:line={last_info.get('overloaded_lines', '?')}"
            else:
                detail = reason
            fail_examples[detail] += 1
        episode_passed = (last_info.get('reason') == 'full_sced_pass' or adopted_operation_pass(last_info))
        if is_sac:
            for s0, a0, r0, s1, d0 in sac_episode_transitions:
                replay_r = r0 if (episode_passed or not args.failure_gated_replay) else float(r0) - float(args.failure_reward_offset)
                t.replay_buffer.push(s0, a0, replay_r * args.reward_scale, s1, d0)
                if episode_passed and success_bc_buffer is not None:
                    success_bc_buffer.append((s0, a0))
        er_report = er if episode_passed else min(er, -1.0)
        ep_dense_rs.append(er)
        ep_rs.append(er_report)
    collect_dt=time.time()-collect_start
    update_start=time.time()
    performed_updates = 0
    performed_actor_updates = 0
    performed_success_bc = 0
    sac_update_stats = []
    available_success_episodes = success_episode_total + ep_pass
    actor_gate_txt = ''
    if is_sac:
        if len(t.replay_buffer) >= args.learning_starts:
            for _grad_idx in range(_v35.N_GRADIENT_STEPS):
                update_result = t.update(update_actor=True)
                sac_update_stats.append(update_result)
                performed_updates += 1
                performed_actor_updates += 1
    elif is_ppo:
        t.update(batch_size=256,n_steps=30)
        performed_updates = 1
    else:
        # DDPG/TD3 need a genuinely populated replay buffer before either the
        # critic or actor is updated. Previously --learning-starts was silently
        # ignored here, so both networks started fitting the first, mostly
        # unsuccessful rollout batch and the deterministic policy collapsed.
        det_learning_starts = args.td3_learning_starts if algo_name == 'TD3' else args.det_learning_starts
        if len(t.buf) >= max(BATCH, int(det_learning_starts)):
            actor_updates_before = t.actor_updates
            t.update()
            performed_updates = t.gradient_steps
            performed_actor_updates = t.actor_updates - actor_updates_before
        else:
            performed_updates = 0
    update_dt=time.time()-update_start
    probe_txt = ''
    probe_due = (
        int(args.train_probe_interval) > 0
        and u % int(args.train_probe_interval) == 0
        and u % VAL_EVAL_INTERVAL != 0
    )
    if is_sac and rollback_probe_ids and (performed_actor_updates > 0 or performed_success_bc > 0) and probe_due:
        probe_start = time.time()
        probe_res = eval_policy(env, t, rollback_probe_ids)
        probe_pass = int(probe_res["passed"])
        probe_reward = float(probe_res["avg_reward"])
        candidate_rank = deterministic_probe_rank(probe_res)
        if best_probe_state is None or best_probe_rank is None or candidate_rank > best_probe_rank:
            best_probe_pass = probe_pass
            best_probe_reward = probe_reward
            best_probe_rank = candidate_rank
            best_probe_state = snapshot_sac_state(t)
            probe_txt = f" probe=keep:{probe_pass}/{len(rollback_probe_ids)}"
        else:
            restore_sac_state(t, best_probe_state)
            probe_txt = (
                f" probe=rollback:{probe_pass}/{len(rollback_probe_ids)}"
                f"<=best:{best_probe_pass}/{len(rollback_probe_ids)}"
            )
        update_dt += time.time() - probe_start
    avg_r=np.mean(ep_rs) if ep_rs else 0
    avg_dense_r=np.mean(ep_dense_rs) if ep_dense_rs else 0
    rewards.append(avg_r); pass_history.append({"update": u, "passed": ep_pass, "total": len(eps), "dense_reward": float(avg_dense_r)})

    if ep_pass > best_pass or (ep_pass == best_pass and avg_r > best_r):
        best_pass=ep_pass; best_total=len(eps); best_r=avg_r
        torch.save(t.actor.state_dict(),str(save_dir/'best_train_actor.pth'))
    candidate_txt = ""
    torch.save(t.actor.state_dict(),str(save_dir/'latest_actor.pth'))
    success_episode_total = available_success_episodes

    val_txt='skip'; val_dt=0.0
    if u % VAL_EVAL_INTERVAL == 0:
        val_start=time.time()
        val_res=eval_policy(val_env,t,val_ids)
        val_dt=time.time()-val_start
        val_history.append({"update":u,**val_res})
        val_txt=f"{val_res['passed']}/{val_res['total']} R={val_res['avg_reward']:+.0f}"
        can_save_val = int(val_res['passed']) >= int(args.min_val_save_pass)
        val_improved = False
        if can_save_val and (
            val_res['passed'] > best_val or (val_res['passed'] == best_val and val_res['avg_reward'] > best_val_r)
        ):
            best_val=val_res['passed']; best_val_total=val_res['total']; best_val_r=val_res['avg_reward']
            torch.save(t.actor.state_dict(),str(save_dir/'best_actor.pth'))
            best_eval_actor_state = copy.deepcopy(t.actor.state_dict())
            val_improved = True
            if not is_sac and not is_ppo:
                det_bad_val_checks = 0
            if is_sac and args.val_rollback:
                best_val_state = snapshot_sac_state(t)
        elif not can_save_val:
            val_txt += f" nosave<{args.min_val_save_pass}"
        elif args.val_rollback and is_sac and best_val_state is not None:
            restore_sac_state(t, best_val_state)
            val_txt += " rollback"
        if not is_sac and not is_ppo and best_eval_actor_state is not None and not val_improved:
            det_bad_val_checks += 1
            det_rollback_enabled = args.td3_val_rollback if algo_name == "TD3" else args.det_val_rollback
            if det_rollback_enabled:
                t.actor.load_state_dict(best_eval_actor_state)
                t.at.load_state_dict(best_eval_actor_state)
                # Adam moments accumulated while the actor was degrading can
                # immediately push a restored policy back in the same direction.
                t.ao.state.clear()
                val_txt += " actor_rollback"
            if args.det_early_stop_patience > 0 and det_bad_val_checks >= args.det_early_stop_patience:
                stop_training = True
                val_txt += f" early_stop:{det_bad_val_checks}"

    fail_txt = ','.join([f"{k}:{v}" for k,v in fail_reasons.most_common()]) if fail_reasons else '-'
    fail_ex_txt = ','.join([f"{k}:{v}" for k,v in fail_examples.most_common(2)]) if fail_examples else '-'
    steps_avg=float(np.mean(ep_steps)) if ep_steps else 0.0
    if is_sac:
        update_txt = f"{performed_updates}:critic+{performed_actor_updates}:actor+sbc{performed_success_bc}"
        if sac_update_stats:
            q_losses = [
                float(s.get('q1_loss', 0.0)) + float(s.get('q2_loss', 0.0))
                for s in sac_update_stats
                if np.isfinite(float(s.get('q1_loss', 0.0)))
                and np.isfinite(float(s.get('q2_loss', 0.0)))
            ]
            actor_losses = [
                float(s.get('actor_loss', 0.0)) for s in sac_update_stats
                if float(s.get('actor_loss', 0.0)) != 0.0
                and np.isfinite(float(s.get('actor_loss', 0.0)))
            ]
            q_txt = f" qloss={np.mean(q_losses):.3f}" if q_losses else ""
            pi_txt = f" piloss={np.mean(actor_losses):.3f}" if actor_losses else ""
            update_txt += q_txt + pi_txt
    elif not is_ppo:
        update_txt = f"{performed_updates}:critic+{performed_actor_updates}:actor"
    else:
        update_txt = str(performed_updates)
    policy_txt = ''
    if is_sac and policy_probe_obs is not None:
        with torch.no_grad():
            probe_t = torch.as_tensor(policy_probe_obs, dtype=torch.float32, device=DEV)
            probe_mean, probe_log_std = t.actor.forward(probe_t)
            probe_action = torch.tanh(probe_mean)
            p_dim = env._n_shift_dirs + env._n_key_gens
            probe_p = probe_action[:, :p_dim]
            probe_v = probe_action[:, p_dim:]
            probe_std = probe_log_std.exp()
            policy_txt = (
                f" amean={probe_action.abs().mean().item():.4f}"
                f" amax={probe_action.abs().max().item():.4f}"
                f" pmean={probe_p.abs().mean().item():.4f}"
                f" vmean={probe_v.abs().mean().item():.4f}"
                f" vsigned={probe_v.mean().item():+.4f}"
                f" pstd={probe_std[:, :p_dim].mean().item():.4f}"
                f" vstd={probe_std[:, p_dim:].mean().item():.4f}"
                f" alpha={t.alpha.item():.4f}"
                f" rollout={args.sac_rollout_mode}:noise={current_rollout_noise(u):.3f}"
            )
    elif policy_probe_obs is not None:
        with torch.no_grad():
            probe_t = torch.as_tensor(policy_probe_obs, dtype=torch.float32, device=DEV)
            if is_ppo:
                probe_mean, probe_log_std = t.actor.forward(probe_t)
                probe_action = torch.tanh(probe_mean)
                ppo_stats = getattr(t, "last_update_stats", {})
                extra_std = (
                    f" std={probe_log_std.exp().mean().item():.4f}"
                    f" kl={ppo_stats.get('kl', 0.0):.4f}"
                    f" clip={ppo_stats.get('clip_fraction', 0.0):.3f}"
                    f" mb={ppo_stats.get('minibatches', 0)}"
                )
            else:
                probe_action = t.actor(probe_t)
                extra_std = (
                    f" noise={current_deterministic_noise(u):.3f}"
                    f" alr={det_actor_lr_now:.1e} clr={det_critic_lr_now:.1e}"
                )
            p_dim = env._n_shift_dirs + env._n_key_gens
            policy_txt = (
                f" amean={probe_action.abs().mean().item():.4f}"
                f" amax={probe_action.abs().max().item():.4f}"
                f" pmean={probe_action[:, :p_dim].abs().mean().item():.4f}"
                f" vmean={probe_action[:, p_dim:].abs().mean().item():.4f}"
                f" vsigned={probe_action[:, p_dim:].mean().item():+.4f}"
                f"{extra_std}"
            )
    gate_txt = f" gate={actor_gate_txt}" if is_sac and actor_gate_txt else ""
    sbuf_txt = len(success_bc_buffer) if is_sac and success_bc_buffer is not None else '-'
    replay_size = len(t.replay_buffer) if is_sac else (len(t.buf) if not is_ppo else len(t.buffer))
    print(f"U{u:3d} R={avg_r:+.1f} denseR={avg_dense_r:+.1f} train_pass={ep_pass}/{len(eps)} best_train={best_pass}/{best_total} val={val_txt} best_val={best_val}/{best_val_total} fail={fail_txt} ex={fail_ex_txt} steps_avg={steps_avg:.1f} updates={update_txt}{gate_txt}{candidate_txt}{probe_txt} succ_total={success_episode_total if is_sac else '-'} buf={replay_size} sbuf={sbuf_txt}{policy_txt} collect={collect_dt:.0f}s upd={update_dt:.0f}s val_time={val_dt:.0f}s total={time.time()-us:.0f}s")
    run_config = {
        "dataset": str(dataset_path),
        "gat_checkpoint": str(gat_checkpoint),
        "split_source": split_source,
        "updates": TOTAL_UPDATES,
        "episodes_per_update": EPISODES_PER_UPDATE,
        "max_steps": args.max_steps,
        "action_encoding": args.action_encoding,
        "action_basis_seed": int(args.action_basis_seed),
        "active_action_dim": int(env._n_shift_dirs + env._n_key_gens),
        "voltage_action_dim": int(env._n_voltage_actions),
        "n1_limit_scale": float(args.n1_limit_scale),
    }
    if algo_name in ("DDPG", "TD3"):
        is_td3 = algo_name == "TD3"
        run_config.update({
            "actor_lr": args.td3_actor_lr if is_td3 else args.det_actor_lr,
            "critic_lr": args.td3_critic_lr if is_td3 else args.det_critic_lr,
            "gradient_steps": args.td3_gradient_steps if is_td3 else args.det_gradient_steps,
            "learning_starts": args.td3_learning_starts if is_td3 else args.det_learning_starts,
            "critic_warmup_steps": args.td3_critic_warmup_steps if is_td3 else args.det_critic_warmup_steps,
            "policy_freq": args.td3_policy_freq if is_td3 else args.det_policy_freq,
            "noise_start": args.td3_noise_start if is_td3 else args.det_noise_start,
            "noise_end": args.td3_noise_end if is_td3 else args.det_noise_end,
            "noise_decay_updates": args.td3_noise_decay_updates if is_td3 else args.det_noise_decay_updates,
            "reward_scale": args.td3_reward_scale if is_td3 else args.det_reward_scale,
            "early_stop_patience": args.det_early_stop_patience,
            "validation_actor_rollback": args.td3_val_rollback if is_td3 else args.det_val_rollback,
            "lr_decay_start": args.td3_lr_decay_start if is_td3 else args.det_lr_decay_start,
            "actor_lr_final_factor": args.td3_actor_lr_final_factor if is_td3 else args.det_actor_lr_final_factor,
            "critic_lr_final_factor": args.td3_critic_lr_final_factor if is_td3 else args.det_critic_lr_final_factor,
            "td3_policy_noise": args.td3_policy_noise if algo_name == "TD3" else None,
            "td3_noise_clip": args.td3_noise_clip if algo_name == "TD3" else None,
        })
    elif algo_name == "PPO":
        run_config.update({
            "lr": args.ppo_lr,
            "critic_lr": args.ppo_critic_lr,
            "epochs": args.ppo_epochs,
            "log_std_init": args.ppo_log_std_init,
            "log_std_final": args.ppo_log_std_final,
            "std_decay_updates": args.ppo_std_decay_updates,
            "learn_log_std": args.ppo_learn_log_std,
            "reward_scale": args.ppo_reward_scale,
            "entropy_coef": args.ppo_entropy_coef,
            "target_kl": args.ppo_target_kl,
        })
    else:
        run_config.update({
            "actor_lr": args.actor_lr,
            "critic_lr": args.critic_lr,
            "alpha_lr": args.alpha_lr,
            "gradient_steps": args.gradient_steps,
            "learning_starts": args.learning_starts,
            "critic_warmup_updates": args.critic_warmup_updates,
            "actor_inner_update_interval": args.actor_inner_update_interval,
            "reward_scale": args.reward_scale,
            "initial_alpha": args.sac_init_alpha,
            "automatic_alpha": args.auto_alpha,
            "learn_log_std": args.learn_log_std,
            "rollout_mode": args.sac_rollout_mode,
            "rollout_noise_start": args.rollout_noise_start,
            "rollout_noise_end": args.rollout_noise_end,
            "n1_guide_gain": args.n1_guide_gain,
            "n1_guide_until": args.n1_guide_until,
            "n1_guide_decay_start": args.n1_guide_decay_start,
            "residual_scale_mw": args.residual_scale,
            "cumulative_scale_mult": args.cumulative_scale_mult,
        })
    with open(str(save_dir/'rewards.json'),'w') as f:
        json.dump({"algo":algo_name,"variant":variant_name,"config":run_config,"rewards":rewards,"pass_history":pass_history,"val_history":val_history,"best_train_pass":best_pass,"best_train_total":best_total,"best_val_pass":best_val,"best_val_total":best_val_total,"best_val_avg_reward":best_val_r},f,indent=2)
    if stop_training:
        print(f"{algo_name} early stop at U{u}; restoring best validation actor {best_val}/{best_val_total}.")
        break

if best_eval_actor_state is not None:
    t.actor.load_state_dict(best_eval_actor_state)
torch.save(t.actor.state_dict(),str(save_dir/'final_actor.pth'))
training_runtime_sec = time.time() - t0
if args.final_test_eval:
    test_env = EnvClass(samples, test_ids, dataset_meta=data)
    test_eval_start = time.perf_counter()
    test_result = eval_policy(test_env, t, test_ids)
    test_eval_elapsed = time.perf_counter() - test_eval_start
    test_result.update({
        "action_encoding": args.action_encoding,
        "action_basis_seed": int(args.action_basis_seed),
        "active_action_dim": int(test_env._n_shift_dirs + test_env._n_key_gens),
        "total_action_dim": int(test_env.action_dim),
        "actor_parameter_count": int(
            sum(parameter.numel() for parameter in t.actor.parameters())
        ),
        "n1_limit_scale": float(args.n1_limit_scale),
        "evaluation_mode": "complete_hybrid_actor_mean",
        "dataset": str(dataset_path),
        "split_source": split_source,
        "training_runtime_sec": float(training_runtime_sec),
        "total_runtime_sec": float(test_eval_elapsed),
        "avg_runtime_sec": float(test_eval_elapsed / max(len(test_ids), 1)),
    })
    with open(str(save_dir/'test_results.json'), 'w') as f:
        json.dump(test_result, f, indent=2)
    print(
        f"Final test: {test_result['passed']}/{test_result['total']} "
        f"R={test_result['avg_reward']:+.1f} reasons={test_result['reasons']}"
    )
elapsed=(time.time()-t0)/3600
final_r = rewards[-1] if rewards else 0.0
print(f"{algo_name} done: {elapsed:.1f}h, best R={best_r:+.0f}, final R={final_r:+.0f}")

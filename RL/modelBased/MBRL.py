# Model-Based RL Example
# 1. Dynamics Model
# 2. Gradient (1a) & CEM (1b) Planners
# 3. MPC Loop (Approach 2)
# 4. Model-Based Policy Optimization (MBPO)
# 5. Logging, checkpointing, GPU support, configuration

#Have to write code ; this is overall template 

import argparse
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import gym

# -------------------------
# Utility: set seeds
# -------------------------
def set_seed(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)

# -------------------------
# 1. Dynamics Model
# -------------------------
class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),       nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

# -------------------------
# Reward Function (Pendulum)
# -------------------------
def reward_fn(state, action):
    theta = torch.atan2(state[:,1], state[:,0])
    theta_dot = state[:,2]
    # cost = theta^2 + 0.1*theta_dot^2 + 0.001*action^2
    return -(theta**2 + 0.1*theta_dot**2 + 0.001*action.pow(2).sum(dim=-1))

# -------------------------
# 2a. Gradient-Based Planner
# -------------------------
class GradientPlanner:
    def __init__(self, model, horizon=15, iters=30, lr=0.05, device='cpu'):
        self.model = model
        self.horizon = horizon
        self.iters = iters
        self.lr = lr
        self.device = device

    def plan(self, state):
        state = state.to(self.device)
        batch_size, state_dim = state.shape
        action_dim = batch_size  # assume 1D action for simplicity
        # Initialize action sequence (batch=1)
        actions = torch.zeros(1, self.horizon, action_dim, device=self.device, requires_grad=True)
        optimizer = optim.Adam([actions], lr=self.lr)

        for _ in range(self.iters):
            optimizer.zero_grad()
            s = state.clone()
            total_reward = 0.0
            for t in range(self.horizon):
                a = torch.tanh(actions[:,t])
                s = self.model(s, a)
                total_reward += reward_fn(s, a)
            # Maximize reward
            loss = -total_reward.mean()
            loss.backward()
            optimizer.step()
        # Return first action
        return torch.tanh(actions[:,0]).detach()

# -------------------------
# 2b. CEM-Based Planner
# -------------------------
class CEMPlanner:
    def __init__(self, model, horizon=20, pop_size=500, elite_frac=0.1, iters=5, device='cpu'):
        self.model = model
        self.horizon = horizon
        self.pop_size = pop_size
        self.elite_frac = elite_frac
        self.iters = iters
        self.device = device

    def plan(self, state):
        state = state.to(self.device)
        batch, state_dim = state.shape
        action_dim = batch
        mu = torch.zeros(self.horizon, action_dim, device=self.device)
        sigma = torch.ones(self.horizon, action_dim, device=self.device)
        n_elite = int(self.pop_size * self.elite_frac)

        for _ in range(self.iters):
            seqs = mu.unsqueeze(0) + sigma.unsqueeze(0) * torch.randn(self.pop_size, self.horizon, action_dim, device=self.device)
            rewards = torch.zeros(self.pop_size, device=self.device)
            for i in range(self.pop_size):
                s = state.clone()
                total = 0.0
                for t in range(self.horizon):
                    a = torch.tanh(seqs[i,t]).unsqueeze(0)
                    s = self.model(s, a)
                    total += reward_fn(s, a)
                rewards[i] = total
            elite_idx = rewards.topk(n_elite).indices
            elite_seqs = seqs[elite_idx]
            mu = elite_seqs.mean(dim=0)
            sigma = elite_seqs.std(dim=0) + 1e-6

        return torch.tanh(mu[0]).unsqueeze(0)

# -------------------------
# 3. MPC Agent Loop
# -------------------------
class MPCAgent:
    def __init__(self, env, model, planner, cfg, writer=None):
        self.env = env
        self.model = model
        self.planner = planner
        self.cfg = cfg
        self.dataset = []
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.model_lr)
        self.writer = writer

    def collect_episode(self, ep):
        state = self.env.reset()
        done = False
        ep_reward = 0
        step = 0
        while not done and step < self.cfg.max_ep_steps:
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            a_t = self.planner.plan(s_t).cpu().numpy().flatten()
            next_s, r, done, _ = self.env.step(a_t)
            self.dataset.append((state, a_t, next_s))
            ep_reward += r
            state = next_s
            step += 1

            # Retrain model every N steps
            if len(self.dataset) >= self.cfg.batch_size and step % self.cfg.retrain_every == 0:
                self._train_model(step)
        if self.writer:
            self.writer.add_scalar('MPC/episode_reward', ep_reward, ep)
        return ep_reward

    def _train_model(self, step):
        batch = random.sample(self.dataset, self.cfg.batch_size)
        s_b = torch.tensor([b[0] for b in batch], dtype=torch.float32)
        a_b = torch.tensor([b[1] for b in batch], dtype=torch.float32).unsqueeze(-1)
        s_next = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        pred = self.model(s_b, a_b)
        loss = nn.MSELoss()(pred, s_next)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.writer:
            self.writer.add_scalar('MPC/model_loss', loss.item(), step)

# -------------------------
# 4. Policy Network & MBPO
# -------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, s): return torch.tanh(self.net(s))


def model_based_policy_optimization(env, model, policy, real_data, cfg, writer=None):
    policy_opt = optim.Adam(policy.parameters(), lr=cfg.policy_lr)
    model_opt = optim.Adam(model.parameters(), lr=cfg.model_lr)
    for it in range(cfg.mbpo_iters):
        # Train model
        if len(real_data) >= cfg.batch_size:
            batch = random.sample(real_data, cfg.batch_size)
            s_b = torch.tensor([b[0] for b in batch], dtype=torch.float32)
            a_b = torch.tensor([b[1] for b in batch], dtype=torch.float32).unsqueeze(-1)
            s_next = torch.tensor([b[2] for b in batch], dtype=torch.float32)
            pred = model(s_b, a_b)
            loss_m = nn.MSELoss()(pred, s_next)
            model_opt.zero_grad(); loss_m.backward(); model_opt.step()
            if writer: writer.add_scalar('MBPO/model_loss', loss_m.item(), it)
        # Synthetic rollouts
        synthetic = []
        for s, _, _ in random.sample(real_data, min(len(real_data), cfg.batch_size)):
            st = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            for _ in range(cfg.rollout_len):
                a = policy(st)
                st = model(st, a)
                synthetic.append((st.squeeze(0).numpy(), a.squeeze(0).detach().numpy(), None))
        # Train policy by imitation
        if synthetic:
            batch = random.sample(synthetic, min(len(synthetic), cfg.batch_size))
            s_b = torch.tensor([b[0] for b in batch], dtype=torch.float32)
            a_target = torch.tensor([b[1] for b in batch], dtype=torch.float32)
            a_pred = policy(s_b)
            loss_p = nn.MSELoss()(a_pred, a_target)
            policy_opt.zero_grad(); loss_p.backward(); policy_opt.step()
            if writer: writer.add_scalar('MBPO/policy_loss', loss_p.item(), it)

# -------------------------
# 5. Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='Pendulum-v1')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    # hyperparameters
    parser.add_argument('--horizon',      type=int, default=15)
    parser.add_argument('--pop_size',     type=int, default=500)
    parser.add_argument('--elite_frac',   type=float, default=0.1)
    parser.add_argument('--retrain_every',type=int, default=5)
    parser.add_argument('--batch_size',   type=int, default=64)
    parser.add_argument('--mbpo_iters',   type=int, default=500)
    parser.add_argument('--rollout_len',  type=int, default=5)
    parser.add_argument('--model_lr',     type=float, default=1e-3)
    parser.add_argument('--policy_lr',    type=float, default=1e-3)
    args = parser.parse_args()

    # Setup
    env = gym.make(args.env)
    set_seed(args.seed, env)
    writer = SummaryWriter(log_dir='logs')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Build components
    model = DynamicsModel(state_dim, action_dim).to(args.device)
    planner = CEMPlanner(model, horizon=args.horizon, pop_size=args.pop_size,
                         elite_frac=args.elite_frac, iters=5, device=args.device)
    mpc_agent = MPCAgent(env, model, planner, args, writer)
    policy = PolicyNetwork(state_dim, action_dim).to(args.device)

    # 1) MPC data collection
    for ep in range(args.episodes):
        ep_reward = mpc_agent.collect_and_plan(ep)
        print(f"[MPC] Episode {ep} reward: {ep_reward:.2f}")

    # 2) Model-Based Policy Optimization
    model_based_policy_optimization(env, model, policy, mpc_agent.dataset,
                                   args, writer)
    print("MBPO policy training complete.")
    # Save final policy
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(policy.state_dict(), 'checkpoints/policy.pth')
    writer.close()

if __name__ == '__main__':
    main()


"""Minimal PPO implementation compatible with the GazeboEnv interface.

Drop-in replacement for stable_baselines3.PPO for environments that extend
GazeboEnv.  Requires only numpy and torch — no SB3 dependency.

Usage
-----
    from gazebo_gymnasium.ppo import PPO, train

    env = MyGazeboEnv(...)
    model = PPO(env, hidden_size=64, lr=3e-4)
    train(model, env, total_steps=500_000)
    model.save("my_model.pt")

    # Or load and evaluate
    model = PPO.load("my_model.pt", env)
    obs, _ = env.reset()
    for _ in range(1000):
        action = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

Interface parity with SB3
--------------------------
    PPO(env, ...)         — construct (no "MlpPolicy" string needed)
    model.learn(n, ...)   — train for n environment steps
    model.predict(obs)    — return deterministic action
    model.save(path)      — save weights
    PPO.load(path, env)   — restore weights
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


# ---------------------------------------------------------------------------
# Shared MLP backbone
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ActorCritic(nn.Module):
    """Shared-trunk actor-critic. Works for both discrete and continuous action spaces."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int, discrete: bool):
        super().__init__()
        self.discrete = discrete
        self.trunk = _MLP(obs_dim, hidden_size)
        self.pi_head = nn.Linear(hidden_size, act_dim)
        self.v_head = nn.Linear(hidden_size, 1)
        if not discrete:
            self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        logits_or_mean = self.pi_head(h)
        v = self.v_head(h).squeeze(-1)
        return logits_or_mean, v

    def dist(self, obs: torch.Tensor):
        h = self.trunk(obs)
        lm = self.pi_head(h)
        if self.discrete:
            return Categorical(logits=lm)
        return Normal(lm, self.log_std.exp())

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.trunk(obs)
        return self.v_head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

@dataclass
class _Buffer:
    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    def clear(self):
        self.obs.clear(); self.actions.clear(); self.rewards.clear()
        self.values.clear(); self.log_probs.clear(); self.dones.clear()

    def size(self) -> int:
        return len(self.rewards)


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------

class PPO:
    """Minimal PPO compatible with the GazeboEnv interface.

    Parameters
    ----------
    env:
        A Gymnasium-compatible environment (GazeboEnv subclass).
    hidden_size:
        Width of the two-layer MLP trunk.
    lr:
        Adam learning rate.
    gamma:
        Discount factor.
    gae_lambda:
        GAE lambda for advantage estimation.
    clip_range:
        PPO clip epsilon.
    ent_coef:
        Entropy bonus coefficient.
    vf_coef:
        Value loss coefficient.
    n_steps:
        Rollout length (steps collected before each update).
    n_epochs:
        Gradient epochs per update.
    batch_size:
        Mini-batch size within each epoch.
    device:
        "cpu", "cuda", or "auto".
    verbose:
        0 = silent, 1 = per-rollout stats.
    tensorboard_log:
        Directory for TensorBoard SummaryWriter (None = disabled).
    """

    def __init__(
        self,
        env,
        *,
        hidden_size: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_steps: int = 2048,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: str = "auto",
        verbose: int = 1,
        tensorboard_log: Optional[str] = None,
    ):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.verbose = verbose

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        import gymnasium as gym
        obs_space = env.observation_space
        act_space = env.action_space
        obs_dim = int(np.prod(obs_space.shape))
        self.discrete = isinstance(act_space, gym.spaces.Discrete)
        act_dim = act_space.n if self.discrete else int(np.prod(act_space.shape))

        self.net = _ActorCritic(obs_dim, act_dim, hidden_size, self.discrete).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        self._writer = None
        if tensorboard_log is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(tensorboard_log)
            except ImportError:
                pass

        self._total_steps = 0
        self._buf = _Buffer()

    # ------------------------------------------------------------------ public API

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Return a deterministic action (mean for continuous, argmax for discrete)."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits_or_mean, _ = self.net(obs_t)
            if self.discrete:
                action = logits_or_mean.argmax(dim=-1)
            else:
                action = logits_or_mean
        return action.cpu().numpy()

    def learn(
        self,
        total_timesteps: int,
        callback: Optional[Callable] = None,
        progress_bar: bool = False,
    ) -> "PPO":
        """Train for total_timesteps environment steps."""
        obs, _ = self.env.reset()
        ep_reward = 0.0
        ep_len = 0
        rollout_count = 0
        start_time = time.time()

        while self._total_steps < total_timesteps:
            # --- collect rollout ---
            self._buf.clear()
            for _ in range(self.n_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    dist = self.net.dist(obs_t)
                    action_t = dist.sample()
                    log_prob = dist.log_prob(action_t)
                    if not self.discrete:
                        log_prob = log_prob.sum(-1)
                    value = self.net.value(obs_t)

                action_np = action_t.cpu().numpy()
                if self.discrete:
                    action_np = int(action_np)

                next_obs, reward, terminated, truncated, info = self.env.step(action_np)
                done = terminated or truncated

                self._buf.obs.append(obs.copy())
                self._buf.actions.append(action_t.cpu().numpy())
                self._buf.rewards.append(float(reward))
                self._buf.values.append(value.item())
                self._buf.log_probs.append(log_prob.item())
                self._buf.dones.append(done)

                obs = next_obs
                ep_reward += reward
                ep_len += 1
                self._total_steps += 1

                if done:
                    if self.verbose >= 1:
                        elapsed = time.time() - start_time
                        fps = self._total_steps / max(elapsed, 1e-9)
                        print(f"  steps={self._total_steps:8d} | "
                              f"ep_len={ep_len:4d} | reward={ep_reward:.2f} | fps={fps:.0f}")
                    if self._writer:
                        self._writer.add_scalar("rollout/ep_reward", ep_reward, self._total_steps)
                        self._writer.add_scalar("rollout/ep_len", ep_len, self._total_steps)
                    ep_reward = 0.0
                    ep_len = 0
                    obs, _ = self.env.reset()

                if self._total_steps >= total_timesteps:
                    break

            # --- compute advantages (GAE) ---
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                last_value = self.net.value(obs_t).item()
            advantages = self._compute_gae(last_value)
            returns = [a + v for a, v in zip(advantages, self._buf.values)]

            # --- PPO update ---
            loss_info = self._update(advantages, returns)
            rollout_count += 1

            if self.verbose >= 1 and rollout_count % 1 == 0:
                elapsed = time.time() - start_time
                fps = self._total_steps / max(elapsed, 1e-9)
                print(f"[update {rollout_count}] steps={self._total_steps} "
                      f"fps={fps:.0f} "
                      f"loss={loss_info['loss']:.4f} "
                      f"vf={loss_info['vf_loss']:.4f} "
                      f"ent={loss_info['ent']:.4f}")
            if self._writer:
                for k, v in loss_info.items():
                    self._writer.add_scalar(f"train/{k}", v, self._total_steps)

        return self

    def save(self, path: str) -> None:
        """Save model weights and hyperparameters."""
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".pt")
        torch.save({
            "state_dict": self.net.state_dict(),
            "discrete": self.discrete,
            "hidden_size": self.net.trunk.net[0].out_features,
            "act_dim": self.net.pi_head.out_features,
            "obs_dim": self.net.trunk.net[0].in_features,
        }, p)
        print(f"[PPO] Saved → {p}")

    @classmethod
    def load(cls, path: str, env, **kwargs) -> "PPO":
        """Load a previously saved model."""
        data = torch.load(path, weights_only=True)
        obj = cls(env, hidden_size=data["hidden_size"], **kwargs)
        obj.net.load_state_dict(data["state_dict"])
        obj.net.eval()
        return obj

    # ------------------------------------------------------------------ internals

    def _compute_gae(self, last_value: float) -> List[float]:
        advantages = [0.0] * self._buf.size()
        gae = 0.0
        next_val = last_value
        for t in reversed(range(self._buf.size())):
            next_non_terminal = 0.0 if self._buf.dones[t] else 1.0
            delta = (self._buf.rewards[t]
                     + self.gamma * next_val * next_non_terminal
                     - self._buf.values[t])
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            next_val = self._buf.values[t]
        return advantages

    def _update(self, advantages: List[float], returns: List[float]) -> dict:
        obs_arr = torch.tensor(np.array(self._buf.obs), dtype=torch.float32, device=self.device)
        acts_arr = torch.tensor(np.array(self._buf.actions), device=self.device)
        if self.discrete:
            acts_arr = acts_arr.long().squeeze(-1)
        old_lp = torch.tensor(self._buf.log_probs, dtype=torch.float32, device=self.device)
        adv = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret = torch.tensor(returns, dtype=torch.float32, device=self.device)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = obs_arr.shape[0]
        total_loss = total_vf = total_ent = 0.0
        updates = 0

        for _ in range(self.n_epochs):
            idx = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                mb = idx[start:start + self.batch_size]
                dist = self.net.dist(obs_arr[mb])
                lp = dist.log_prob(acts_arr[mb])
                if not self.discrete:
                    lp = lp.sum(-1)
                ratio = (lp - old_lp[mb]).exp()
                a = adv[mb]
                loss_clip = -torch.min(
                    ratio * a,
                    ratio.clamp(1 - self.clip_range, 1 + self.clip_range) * a,
                ).mean()
                v_pred = self.net.value(obs_arr[mb])
                loss_vf = 0.5 * (v_pred - ret[mb]).pow(2).mean()
                ent = dist.entropy()
                if not self.discrete:
                    ent = ent.sum(-1)
                loss_ent = -ent.mean()
                loss = loss_clip + self.vf_coef * loss_vf + self.ent_coef * loss_ent
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()
                total_loss += loss_clip.item()
                total_vf += loss_vf.item()
                total_ent += -loss_ent.item()
                updates += 1

        return {
            "loss": total_loss / max(updates, 1),
            "vf_loss": total_vf / max(updates, 1),
            "ent": total_ent / max(updates, 1),
        }

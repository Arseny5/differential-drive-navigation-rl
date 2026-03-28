"""
Advantage Actor-Critic (A2C) with Generalized Advantage Estimation.
"""

import numpy as np
import torch

from .base_agent import BaseAgent
from .networks import PolicyNetwork, ValueNetwork


class ActorCriticAgent(BaseAgent):

    def __init__(self, obs_dim: int, act_dim: int, config: dict):
        super().__init__(obs_dim, act_dim, config)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.policy_lr = config.get("learning_rate", 3e-4)
        self.value_lr = config.get("value_lr", 1e-3)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.max_action = config.get("max_action", 2.0)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        hidden = config.get("hidden_dims", [128, 128])

        self.policy = PolicyNetwork(obs_dim, act_dim, hidden).to(self.device)
        self.value = ValueNetwork(obs_dim, hidden).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.value_lr)

    def select_action(self, obs, training=True):
        obs_t = self._to_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            if training:
                action, log_prob = self.policy.sample(obs_t)
            else:
                action = self.policy(obs_t)
                log_prob = self.policy.log_prob(obs_t, action)
        raw = action.squeeze(0).cpu().numpy()
        clipped = np.clip(raw, -self.max_action, self.max_action)
        return clipped, log_prob.item(), raw

    def get_value(self, obs):
        obs_t = self._to_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            return self.value(obs_t).item()

    def update(self, transitions):
        obs_t = self._to_tensor(np.array([t["obs"] for t in transitions]))
        actions_t = self._to_tensor(np.array([t["action_raw"] for t in transitions]))
        rewards = [t["reward"] for t in transitions]
        dones = [float(t["done"]) for t in transitions]

        with torch.no_grad():
            values = self.value(obs_t).tolist()
            if not transitions[-1]["done"]:
                next_obs = self._to_tensor(transitions[-1]["next_obs"]).unsqueeze(0)
                next_value = self.value(next_obs).item()
            else:
                next_value = 0.0

        advantages, returns = self.compute_gae(
            rewards, values, next_value, dones, self.gamma, self.gae_lambda,
        )
        advantages_t = self._to_tensor(advantages)
        returns_t = self._to_tensor(returns)

        if len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # Policy
        log_probs = self.policy.log_prob(obs_t, actions_t)
        entropy = self.policy.entropy(obs_t).mean()
        policy_loss = -(log_probs * advantages_t).mean()

        self.policy_optimizer.zero_grad()
        (policy_loss - self.entropy_coef * entropy).backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()

        # Value
        pred_values = self.value(obs_t)
        value_loss = ((pred_values - returns_t) ** 2).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
        self.value_optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }

    def save(self, path):
        torch.save({
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
            "config": self.config,
        }, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(data["policy"])
        self.value.load_state_dict(data["value"])

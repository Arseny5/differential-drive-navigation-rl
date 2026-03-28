"""
Vanilla REINFORCE (Monte-Carlo Policy Gradient).
"""

import numpy as np
import torch

from .base_agent import BaseAgent
from .networks import PolicyNetwork


class ReinforceAgent(BaseAgent):

    def __init__(self, obs_dim: int, act_dim: int, config: dict):
        super().__init__(obs_dim, act_dim, config)
        self.gamma = config.get("gamma", 0.99)
        self.lr = config.get("learning_rate", 1e-3)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.max_action = config.get("max_action", 2.0)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        hidden = config.get("hidden_dims", [128, 128])

        self.policy = PolicyNetwork(obs_dim, act_dim, hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

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

    def update(self, transitions):
        rewards = [t["reward"] for t in transitions]
        returns = self.compute_returns(rewards, self.gamma)
        returns_t = self._to_tensor(returns)

        if len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        returns_t = returns_t.clamp(-5.0, 5.0)

        obs_t = self._to_tensor(np.array([t["obs"] for t in transitions]))
        actions_t = self._to_tensor(np.array([t["action_raw"] for t in transitions]))

        log_probs = self.policy.log_prob(obs_t, actions_t)
        entropy = self.policy.entropy(obs_t).mean()

        policy_loss = -(log_probs * returns_t).mean()
        loss = policy_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {"policy_loss": policy_loss.item(), "entropy": entropy.item()}

    def save(self, path):
        torch.save({"policy": self.policy.state_dict(), "config": self.config}, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(data["policy"])

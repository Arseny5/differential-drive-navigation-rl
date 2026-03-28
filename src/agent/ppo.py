"""
Proximal Policy Optimization (PPO) with clipped surrogate objective and GAE.

Accumulates transitions over multiple episodes before performing PPO updates,
which provides more stable training than updating after every single episode.
"""

import numpy as np
import torch

from .base_agent import BaseAgent
from .networks import PolicyNetwork, ValueNetwork


class PPOAgent(BaseAgent):

    def __init__(self, obs_dim: int, act_dim: int, config: dict):
        super().__init__(obs_dim, act_dim, config)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.policy_lr = config.get("learning_rate", 3e-4)
        self.value_lr = config.get("value_lr", 1e-3)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.max_action = config.get("max_action", 2.0)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.clip_eps = config.get("clip_eps", 0.2)
        self.ppo_epochs = config.get("ppo_epochs", 4)
        self.mini_batch_size = config.get("mini_batch_size", 64)
        self.rollout_episodes = config.get("rollout_episodes", 8)
        hidden = config.get("hidden_dims", [128, 128])

        self.policy = PolicyNetwork(obs_dim, act_dim, hidden).to(self.device)
        self.value = ValueNetwork(obs_dim, hidden).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.value_lr)

        # Rollout buffer: accumulate episodes before updating
        self._buffer: list[list[dict]] = []

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
        """Buffer episodes; only perform PPO update every rollout_episodes episodes."""
        self._buffer.append(transitions)

        if len(self._buffer) < self.rollout_episodes:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # Flatten all buffered episodes, computing GAE per-episode
        all_obs, all_actions, all_old_lp, all_advantages, all_returns = [], [], [], [], []

        for ep_transitions in self._buffer:
            obs_t = self._to_tensor(np.array([t["obs"] for t in ep_transitions]))
            actions_t = self._to_tensor(np.array([t["action_raw"] for t in ep_transitions]))
            rewards = [t["reward"] for t in ep_transitions]
            dones = [float(t["done"]) for t in ep_transitions]

            with torch.no_grad():
                old_lp = self.policy.log_prob(obs_t, actions_t)
                values = self.value(obs_t).tolist()
                if not ep_transitions[-1]["done"]:
                    next_obs = self._to_tensor(ep_transitions[-1]["next_obs"]).unsqueeze(0)
                    next_value = self.value(next_obs).item()
                else:
                    next_value = 0.0

            advantages, returns = self.compute_gae(
                rewards, values, next_value, dones, self.gamma, self.gae_lambda,
            )

            all_obs.append(obs_t)
            all_actions.append(actions_t)
            all_old_lp.append(old_lp)
            all_advantages.extend(advantages)
            all_returns.extend(returns)

        self._buffer.clear()

        obs_t = torch.cat(all_obs)
        actions_t = torch.cat(all_actions)
        old_log_probs = torch.cat(all_old_lp)
        advantages_t = self._to_tensor(all_advantages)
        returns_t = self._to_tensor(all_returns)

        # Normalize advantages across the entire rollout
        if len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        n = len(obs_t)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.mini_batch_size):
                end = min(start + self.mini_batch_size, n)
                mb_idx = indices[start:end]

                mb_obs = obs_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages_t[mb_idx]
                mb_returns = returns_t[mb_idx]

                # Policy update with clipped surrogate
                new_log_probs = self.policy.log_prob(mb_obs, mb_actions)
                ratio = (new_log_probs - mb_old_log_probs).exp()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy = self.policy.entropy(mb_obs).mean()

                self.policy_optimizer.zero_grad()
                (policy_loss - self.entropy_coef * entropy).backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()

                # Value update
                pred_values = self.value(mb_obs)
                value_loss = ((pred_values - mb_returns) ** 2).mean()

                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
                self.value_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1

        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
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

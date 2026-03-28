"""
Trust Region Policy Optimization (TRPO) for continuous control.

Natural gradient via conjugate gradient + line search with KL constraint.
"""

import numpy as np
import torch

from .base_agent import BaseAgent
from .networks import PolicyNetwork, ValueNetwork


class TRPOAgent(BaseAgent):

    def __init__(self, obs_dim: int, act_dim: int, config: dict):
        super().__init__(obs_dim, act_dim, config)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.max_kl = config.get("max_kl", 0.01)
        self.cg_iters = config.get("cg_iters", 10)
        self.backtrack_iters = config.get("backtrack_iters", 10)
        self.backtrack_coef = config.get("backtrack_coef", 0.5)
        self.damping = config.get("damping", 0.1)
        self.value_lr = config.get("value_lr", 1e-3)
        self.entropy_coef = config.get("entropy_coef", 0.0)
        self.max_action = config.get("max_action", 2.0)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        hidden = config.get("hidden_dims", [128, 128])

        self.policy = PolicyNetwork(obs_dim, act_dim, hidden).to(self.device)
        self.value = ValueNetwork(obs_dim, hidden).to(self.device)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.value_lr)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def _flat_params(self) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for p in self.policy.parameters()])

    def _set_flat_params(self, flat: torch.Tensor):
        offset = 0
        for p in self.policy.parameters():
            n = p.numel()
            p.data.copy_(flat[offset : offset + n].view_as(p))
            offset += n

    def _flat_grad(self, loss, retain_graph=False, create_graph=False) -> torch.Tensor:
        grads = torch.autograd.grad(
            loss, self.policy.parameters(),
            retain_graph=retain_graph, create_graph=create_graph,
        )
        return torch.cat([g.contiguous().view(-1) for g in grads])

    # ------------------------------------------------------------------
    # Fisher-vector product  H*v  via double backprop through KL
    # ------------------------------------------------------------------

    def _fisher_vector_product(self, obs: torch.Tensor,
                               old_mean: torch.Tensor, old_std: torch.Tensor,
                               v: torch.Tensor) -> torch.Tensor:
        new_dist = self.policy.get_distribution(obs)
        kl = torch.distributions.kl_divergence(
            torch.distributions.Normal(old_mean, old_std), new_dist,
        ).sum(-1).mean()

        grads = self._flat_grad(kl, retain_graph=True, create_graph=True)
        gvp = (grads * v).sum()
        hvp = self._flat_grad(gvp)
        return hvp + self.damping * v

    # ------------------------------------------------------------------
    # Conjugate gradient  H^{-1} g
    # ------------------------------------------------------------------

    def _conjugate_gradient(self, obs, old_mean, old_std, b) -> torch.Tensor:
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rdotr = r.dot(r)

        for _ in range(self.cg_iters):
            Ap = self._fisher_vector_product(obs, old_mean, old_std, p)
            alpha = rdotr / (p.dot(Ap) + 1e-8)
            x += alpha * p
            r -= alpha * Ap
            new_rdotr = r.dot(r)
            if new_rdotr < 1e-10:
                break
            p = r + (new_rdotr / (rdotr + 1e-8)) * p
            rdotr = new_rdotr

        return x

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, transitions):
        obs_t = self._to_tensor(np.array([t["obs"] for t in transitions]))
        actions_t = self._to_tensor(np.array([t["action_raw"] for t in transitions]))
        rewards = [t["reward"] for t in transitions]
        dones = [float(t["done"]) for t in transitions]

        # GAE
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

        # ---- Save old distribution (detached) ----
        with torch.no_grad():
            old_dist = self.policy.get_distribution(obs_t)
            old_mean = old_dist.loc.clone()
            old_std = old_dist.scale.clone()

        # ---- Policy gradient ----
        log_probs = self.policy.log_prob(obs_t, actions_t)
        entropy = self.policy.entropy(obs_t).mean()
        surrogate = -(log_probs * advantages_t).mean() - self.entropy_coef * entropy

        g = self._flat_grad(surrogate)

        # ---- Natural gradient direction (CG) ----
        step_dir = self._conjugate_gradient(obs_t, old_mean, old_std, g)

        # ---- Step size from trust region ----
        shs = step_dir.dot(self._fisher_vector_product(obs_t, old_mean, old_std, step_dir))
        max_step = torch.sqrt(2 * self.max_kl / (shs + 1e-8))
        full_step = max_step * step_dir

        # ---- Line search ----
        old_params = self._flat_params()
        old_surrogate = surrogate.item()
        accepted = False
        kl_val = 0.0

        for i in range(self.backtrack_iters):
            new_params = old_params - (self.backtrack_coef ** i) * full_step
            self._set_flat_params(new_params)

            with torch.no_grad():
                new_log_probs = self.policy.log_prob(obs_t, actions_t)
                new_entropy = self.policy.entropy(obs_t).mean()
                new_surrogate = -(new_log_probs * advantages_t).mean() - self.entropy_coef * new_entropy

                new_dist = self.policy.get_distribution(obs_t)
                kl = torch.distributions.kl_divergence(
                    torch.distributions.Normal(old_mean, old_std), new_dist,
                ).sum(-1).mean()
                kl_val = kl.item()

            if kl_val < self.max_kl and new_surrogate.item() <= old_surrogate:
                accepted = True
                break

        if not accepted:
            self._set_flat_params(old_params)
            kl_val = 0.0

        # ---- Value update ----
        pred_values = self.value(obs_t)
        value_loss = ((pred_values - returns_t) ** 2).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
        self.value_optimizer.step()

        policy_loss_val = -(self.policy.log_prob(obs_t, actions_t).detach() * advantages_t).mean().item()

        return {
            "policy_loss": policy_loss_val,
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "kl": kl_val,
            "step_accepted": int(accepted),
        }

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

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

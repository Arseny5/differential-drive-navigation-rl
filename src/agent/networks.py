"""
Neural networks for continuous control: Gaussian policy and value function.
"""

import torch
import torch.nn as nn


def _build_mlp(in_dim: int, out_dim: int, hidden_dims: list[int], activation=nn.Tanh) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(activation())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class PolicyNetwork(nn.Module):
    """
    Gaussian policy: outputs mean, learnable log_std (clamped).
    π(a|s) = N(μ(s), diag(σ²)), σ = exp(clamp(log_std)).
    """

    LOG_STD_MIN = -2.0   # std >= ~0.14 — prevents policy collapse
    LOG_STD_MAX = 0.5     # std <= ~1.65 — prevents too noisy actions

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list[int] = None,
                 log_std_init: float = -0.5):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]
        self.mean_net = _build_mlp(obs_dim, act_dim, hidden_dims)
        self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean_net(obs)

    def _clamped_log_std(self) -> torch.Tensor:
        return self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)

    def get_distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.forward(obs)
        std = self._clamped_log_std().exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        dist = self.get_distribution(obs)
        return dist.log_prob(action).sum(-1)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.get_distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        dist = self.get_distribution(obs)
        return dist.entropy().sum(-1)


class ValueNetwork(nn.Module):
    """V(s) → scalar."""

    def __init__(self, obs_dim: int, hidden_dims: list[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]
        self.net = _build_mlp(obs_dim, 1, hidden_dims)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)

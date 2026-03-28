"""
Base class for all RL agents.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch


class BaseAgent(ABC):

    def __init__(self, obs_dim: int, act_dim: int, config: dict):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config = config
        self.device = torch.device("cpu")

    @abstractmethod
    def select_action(self, obs: np.ndarray, training: bool = True) -> tuple[np.ndarray, float, np.ndarray]:
        """Returns (action_clipped_for_env, log_prob, action_raw_from_policy)."""

    def get_value(self, obs: np.ndarray) -> float | None:
        return None

    @abstractmethod
    def update(self, transitions: list[dict]) -> dict:
        """Single-episode update. Returns metrics dict."""

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.float().to(self.device)
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    @staticmethod
    def compute_returns(rewards: list[float], gamma: float) -> list[float]:
        returns: list[float] = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        return returns

    @staticmethod
    def compute_gae(
        rewards: list[float],
        values: list[float],
        next_value: float,
        dones: list[float],
        gamma: float,
        gae_lambda: float,
    ) -> tuple[list[float], list[float]]:
        advantages: list[float] = []
        gae = 0.0
        vals = list(values) + [next_value]
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * vals[t + 1] * mask - vals[t]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages.insert(0, gae)
        returns = [a + v for a, v in zip(advantages, values)]
        return advantages, returns

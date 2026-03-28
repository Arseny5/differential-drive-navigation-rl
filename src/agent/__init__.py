"""
Agent factory.
"""

from .reinforce import ReinforceAgent
from .reinforce_baseline import ReinforceBaselineAgent
from .actor_critic import ActorCriticAgent
from .trpo import TRPOAgent

AGENT_MAP = {
    "reinforce": ReinforceAgent,
    "reinforce_baseline": ReinforceBaselineAgent,
    "actor_critic": ActorCriticAgent,
    "trpo": TRPOAgent,
}

AGENT_NAMES = list(AGENT_MAP.keys())


def create_agent(method: str, obs_dim: int, act_dim: int, config: dict):
    cls = AGENT_MAP.get(method)
    if cls is None:
        raise ValueError(f"Unknown agent method '{method}'. Choose from {AGENT_NAMES}")
    return cls(obs_dim, act_dim, config)

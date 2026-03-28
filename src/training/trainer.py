"""
Custom training loop (no external RL library).
Works with all agent types: REINFORCE, REINFORCE+Baseline, Actor-Critic, TRPO.
"""

import json
import os
from pathlib import Path

import numpy as np

from src.environment import DiffDriveEnv
from src.agent.base_agent import BaseAgent
from src.training.logger import log_episode, log_message


def run_episode(env: DiffDriveEnv, agent: BaseAgent, max_steps: int):
    """
    Run a single episode and collect transitions.

    Returns: (transitions, total_reward, step_count, success, info)
    """
    obs, info = env.reset()
    transitions: list[dict] = []
    total_reward = 0.0

    for _ in range(max_steps):
        action, log_prob, action_raw = agent.select_action(obs, training=True)
        next_obs, reward, terminated, truncated, info = env.step(action)

        transitions.append({
            "obs": obs,
            "action": action,
            "action_raw": action_raw,
            "reward": reward,
            "log_prob": log_prob,
            "next_obs": next_obs,
            "done": terminated or truncated,
        })

        total_reward += reward
        obs = next_obs
        if terminated or truncated:
            break

    success = bool(info.get("is_success", False))
    return transitions, total_reward, info["step_count"], success, info


def train(
    env: DiffDriveEnv,
    agent: BaseAgent,
    num_episodes: int = 3000,
    max_steps: int = 500,
    log_interval: int = 50,
    checkpoint_interval: int = 500,
    log_dir: str = "logs",
    model_save_dir: str = "models",
    history_path: str | None = None,
) -> list[dict]:
    """
    Main training loop.

    Returns: list of per-episode metrics dicts.
    """
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    csv_path = os.path.join(log_dir, "train_stats.csv")
    with open(csv_path, "w") as f:
        f.write("episode,reward,length,success\n")

    history: list[dict] = []
    best_avg_reward = -float("inf")

    for ep in range(1, num_episodes + 1):
        transitions, total_reward, step_count, success, _ = run_episode(env, agent, max_steps)
        metrics = agent.update(transitions)

        record = {
            "episode": ep,
            "reward": total_reward,
            "steps": step_count,
            "success": int(success),
            **metrics,
        }
        history.append(record)

        with open(csv_path, "a") as f:
            f.write(f"{ep},{total_reward:.4f},{step_count},{int(success)}\n")

        if ep % log_interval == 0:
            recent = history[-log_interval:]
            avg_r = np.mean([h["reward"] for h in recent])
            avg_s = np.mean([h["success"] for h in recent])
            avg_len = np.mean([h["steps"] for h in recent])
            log_episode(
                episode=ep, reward=avg_r, steps=int(avg_len), success=avg_s > 0.5,
                success_rate=f"{avg_s:.1%}",
                loss=metrics.get("policy_loss", 0),
            )
            if avg_r > best_avg_reward:
                best_avg_reward = avg_r
                best_path = os.path.join(model_save_dir, "best_model.pth")
                agent.save(best_path)

        if ep % checkpoint_interval == 0:
            ckpt_path = os.path.join(model_save_dir, f"checkpoint_{ep}.pth")
            agent.save(ckpt_path)
            log_message(f"Checkpoint saved: {ckpt_path}")

    final_path = os.path.join(model_save_dir, "final_model.pth")
    agent.save(final_path)
    log_message(f"Final model saved to {final_path}")

    if history_path is None:
        history_path = os.path.join(log_dir, "training_history.json")
    save_history(history_path, history)
    log_message(f"Training history saved to {history_path}")

    return history


def save_history(path: str, history: list[dict]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2, default=str)

"""
Train selected algorithms (default: all) and produce comparison plots + evaluation table.

Usage: python run/compare.py [--num_episodes 3000] [--seed 42]
"""

import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.environment import DiffDriveEnv
from src.agent import create_agent, AGENT_NAMES
from src.utils import config
from src.training.logger import setup_logger, log_message
from src.training.trainer import train, run_episode


ALGO_CONFIGS = {
    "reinforce": dict(
        learning_rate=config.REINFORCE_LR,
        hidden_dims=config.REINFORCE_HIDDEN_DIMS,
        entropy_coef=config.REINFORCE_ENTROPY_COEF,
        max_grad_norm=config.REINFORCE_MAX_GRAD_NORM,
    ),
    "reinforce_baseline": dict(
        learning_rate=config.BASELINE_POLICY_LR,
        value_lr=config.BASELINE_VALUE_LR,
        hidden_dims=config.BASELINE_HIDDEN_DIMS,
        entropy_coef=config.BASELINE_ENTROPY_COEF,
        max_grad_norm=config.BASELINE_MAX_GRAD_NORM,
    ),
    "actor_critic": dict(
        learning_rate=config.AC_POLICY_LR,
        value_lr=config.AC_VALUE_LR,
        hidden_dims=config.AC_HIDDEN_DIMS,
        entropy_coef=config.AC_ENTROPY_COEF,
        gae_lambda=config.AC_GAE_LAMBDA,
        rollout_episodes=config.AC_ROLLOUT_EPISODES,
        max_grad_norm=config.AC_MAX_GRAD_NORM,
    ),
    "trpo": dict(
        hidden_dims=config.TRPO_HIDDEN_DIMS,
        value_lr=config.TRPO_VALUE_LR,
        max_kl=config.TRPO_MAX_KL,
        cg_iters=config.TRPO_CG_ITERS,
        backtrack_iters=config.TRPO_BACKTRACK_ITERS,
        backtrack_coef=config.TRPO_BACKTRACK_COEF,
        damping=config.TRPO_DAMPING,
        gae_lambda=config.TRPO_GAE_LAMBDA,
        entropy_coef=config.TRPO_ENTROPY_COEF,
        rollout_episodes=config.TRPO_ROLLOUT_EPISODES,
        max_grad_norm=config.TRPO_MAX_GRAD_NORM,
    ),
    "ppo": dict(
        learning_rate=config.PPO_POLICY_LR,
        value_lr=config.PPO_VALUE_LR,
        hidden_dims=config.PPO_HIDDEN_DIMS,
        entropy_coef=config.PPO_ENTROPY_COEF,
        gae_lambda=config.PPO_GAE_LAMBDA,
        clip_eps=config.PPO_CLIP_EPS,
        ppo_epochs=config.PPO_EPOCHS,
        mini_batch_size=config.PPO_MINI_BATCH_SIZE,
        rollout_episodes=config.PPO_ROLLOUT_EPISODES,
        max_grad_norm=config.PPO_MAX_GRAD_NORM,
    ),
}


def quick_evaluate(agent, env, num_episodes=20):
    rewards, lengths, successes = [], [], []
    for _ in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        while True:
            action, _, _ = agent.select_action(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        success = bool(info.get("is_success", False))
        rewards.append(total_reward)
        lengths.append(info["step_count"])
        successes.append(int(success))
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Compare all algorithms")
    p.add_argument("--num_episodes", type=int, default=config.NUM_EPISODES)
    p.add_argument("--max_steps", type=int, default=config.MAX_STEPS)
    p.add_argument("--eval_episodes", type=int, default=config.EVAL_NUM_EPISODES)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--log_dir", type=str, default=config.LOG_DIR)
    p.add_argument("--model_dir", type=str, default=config.MODEL_SAVE_DIR)
    p.add_argument("--plot_dir", type=str, default=config.PLOT_DIR)
    p.add_argument("--methods", type=str, nargs="+", default=AGENT_NAMES, choices=AGENT_NAMES)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logger(level=config.LOG_LEVEL)

    env_kwargs = dict(
        max_steps=args.max_steps,
        max_v=config.MAX_V,
        obstacle_radius=config.OBSTACLE_RADIUS,
        goal_radius=config.GOAL_RADIUS,
        reward_goal=config.REWARD_GOAL,
        reward_collision=config.REWARD_COLLISION,
        reward_step=config.REWARD_STEP,
        progress_scale=config.PROGRESS_SCALE,
        spin_penalty=config.SPIN_PENALTY,
    )

    eval_results = {}

    for method in args.methods:
        log_message(f"\n{'='*50}")
        log_message(f"  Training: {method}")
        log_message(f"{'='*50}")

        env = DiffDriveEnv(**env_kwargs)
        env.reset(seed=args.seed)

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        agent_cfg = {
            **ALGO_CONFIGS[method],
            "gamma": config.GAMMA,
            "max_action": config.MAX_V,
            "device": "cpu",
        }
        agent = create_agent(method, obs_dim, act_dim, agent_cfg)

        log_dir = os.path.join(args.log_dir, method)
        model_dir = os.path.join(args.model_dir, method)

        train(
            env=env, agent=agent,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            log_interval=config.LOG_INTERVAL,
            checkpoint_interval=config.CHECKPOINT_INTERVAL,
            log_dir=log_dir,
            model_save_dir=model_dir,
        )

        # Load best model for evaluation
        best_path = os.path.join(model_dir, "best_model.pth")
        if os.path.exists(best_path):
            agent.load(best_path)

        eval_env = DiffDriveEnv(**env_kwargs)
        eval_env.reset(seed=args.seed + 9999)
        metrics = quick_evaluate(agent, eval_env, num_episodes=args.eval_episodes)
        eval_results[method] = metrics

        log_message(f"{method}: reward={metrics['mean_reward']:.2f}, "
                    f"success={metrics['success_rate']:.1%}, "
                    f"length={metrics['mean_length']:.1f}")

    # ── Summary table ──
    print(f"\n{'='*70}")
    print(f"{'Method':<25} {'Reward':>10} {'Success':>10} {'Ep. Length':>12}")
    print(f"{'-'*70}")
    for method, m in eval_results.items():
        print(f"{method:<25} {m['mean_reward']:>10.2f} {m['success_rate']:>10.1%} {m['mean_length']:>12.1f}")
    print(f"{'='*70}")

    # Save
    os.makedirs(args.log_dir, exist_ok=True)
    with open(os.path.join(args.log_dir, "comparison_results.json"), "w") as f:
        json.dump(eval_results, f, indent=2)

    # ── Auto-plot ──
    log_message("Generating comparison plots...")
    from run.plot import load_csv, plot_comparison

    datasets = {}
    for method in args.methods:
        csv = os.path.join(args.log_dir, method, "train_stats.csv")
        if os.path.exists(csv):
            datasets[method] = load_csv(csv)

    if datasets:
        import matplotlib
        matplotlib.use("Agg")
        plot_comparison(datasets, save_dir=args.plot_dir)

    log_message("All done!")


if __name__ == "__main__":
    main()

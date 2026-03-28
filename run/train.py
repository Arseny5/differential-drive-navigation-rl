"""
Training script for Differential Drive Navigation.
Usage: python run/train.py [--agent_method reinforce] [--num_episodes 3000] ...
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import config
from src.environment import DiffDriveEnv
from src.agent import create_agent, AGENT_NAMES
from src.training.logger import setup_logger, log_message
from src.training.trainer import train


def parse_args():
    p = argparse.ArgumentParser(description="Train Diff Drive Navigation agent")

    # Agent
    p.add_argument("--agent_method", type=str, default=config.AGENT_METHOD, choices=AGENT_NAMES)

    # Environment
    p.add_argument("--max_steps", type=int, default=config.MAX_STEPS)
    p.add_argument("--dt", type=float, default=config.DT)
    p.add_argument("--wheelbase", type=float, default=config.WHEELBASE)
    p.add_argument("--max_v", type=float, default=config.MAX_V)
    p.add_argument("--obstacle_radius", type=float, default=config.OBSTACLE_RADIUS)
    p.add_argument("--goal_radius", type=float, default=config.GOAL_RADIUS)

    # Rewards
    p.add_argument("--reward_goal", type=float, default=config.REWARD_GOAL)
    p.add_argument("--reward_collision", type=float, default=config.REWARD_COLLISION)
    p.add_argument("--reward_step", type=float, default=config.REWARD_STEP)
    p.add_argument("--progress_scale", type=float, default=config.PROGRESS_SCALE)
    p.add_argument("--spin_penalty", type=float, default=config.SPIN_PENALTY)

    # Training
    p.add_argument("--num_episodes", type=int, default=config.NUM_EPISODES)
    p.add_argument("--gamma", type=float, default=config.GAMMA)
    p.add_argument("--log_interval", type=int, default=config.LOG_INTERVAL)
    p.add_argument("--checkpoint_interval", type=int, default=config.CHECKPOINT_INTERVAL)

    # Hyperparams (will override per-algorithm defaults)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--value_lr", type=float, default=None)
    p.add_argument("--entropy_coef", type=float, default=None)
    p.add_argument("--hidden_dims", type=int, nargs="+", default=None)
    p.add_argument("--gae_lambda", type=float, default=None)

    # TRPO-specific
    p.add_argument("--max_kl", type=float, default=None)
    p.add_argument("--cg_iters", type=int, default=None)
    p.add_argument("--damping", type=float, default=None)

    # Paths
    p.add_argument("--log_dir", type=str, default=config.LOG_DIR)
    p.add_argument("--model_save_dir", type=str, default=config.MODEL_SAVE_DIR)
    p.add_argument("--load_model", type=str, default=None)

    p.add_argument("--seed", type=int, default=config.SEED)

    return p.parse_args()


def _build_agent_config(args) -> dict:
    """Build agent config dict from CLI args + per-algorithm defaults."""
    method = args.agent_method

    if method == "reinforce":
        c = dict(
            learning_rate=args.learning_rate or config.REINFORCE_LR,
            hidden_dims=args.hidden_dims or config.REINFORCE_HIDDEN_DIMS,
            entropy_coef=args.entropy_coef if args.entropy_coef is not None else config.REINFORCE_ENTROPY_COEF,
            max_grad_norm=config.REINFORCE_MAX_GRAD_NORM,
        )
    elif method == "reinforce_baseline":
        c = dict(
            learning_rate=args.learning_rate or config.BASELINE_POLICY_LR,
            value_lr=args.value_lr or config.BASELINE_VALUE_LR,
            hidden_dims=args.hidden_dims or config.BASELINE_HIDDEN_DIMS,
            entropy_coef=args.entropy_coef if args.entropy_coef is not None else config.BASELINE_ENTROPY_COEF,
            max_grad_norm=config.BASELINE_MAX_GRAD_NORM,
        )
    elif method == "actor_critic":
        c = dict(
            learning_rate=args.learning_rate or config.AC_POLICY_LR,
            value_lr=args.value_lr or config.AC_VALUE_LR,
            hidden_dims=args.hidden_dims or config.AC_HIDDEN_DIMS,
            entropy_coef=args.entropy_coef if args.entropy_coef is not None else config.AC_ENTROPY_COEF,
            gae_lambda=args.gae_lambda or config.AC_GAE_LAMBDA,
            max_grad_norm=config.AC_MAX_GRAD_NORM,
        )
    elif method == "trpo":
        c = dict(
            hidden_dims=args.hidden_dims or config.TRPO_HIDDEN_DIMS,
            value_lr=args.value_lr or config.TRPO_VALUE_LR,
            max_kl=args.max_kl or config.TRPO_MAX_KL,
            cg_iters=args.cg_iters or config.TRPO_CG_ITERS,
            backtrack_iters=config.TRPO_BACKTRACK_ITERS,
            backtrack_coef=config.TRPO_BACKTRACK_COEF,
            damping=args.damping or config.TRPO_DAMPING,
            gae_lambda=args.gae_lambda or config.TRPO_GAE_LAMBDA,
            entropy_coef=args.entropy_coef if args.entropy_coef is not None else config.TRPO_ENTROPY_COEF,
            max_grad_norm=config.TRPO_MAX_GRAD_NORM,
        )
    else:
        c = {}

    c["gamma"] = args.gamma
    c["max_action"] = args.max_v
    return c


def main():
    args = parse_args()

    setup_logger(log_file=config.LOG_FILE, level=config.LOG_LEVEL)

    log_dir = f"{args.log_dir}/{args.agent_method}"
    model_dir = f"{args.model_save_dir}/{args.agent_method}"

    env = DiffDriveEnv(
        max_steps=args.max_steps, dt=args.dt, wheelbase=args.wheelbase,
        max_v=args.max_v, obstacle_radius=args.obstacle_radius,
        goal_radius=args.goal_radius, reward_goal=args.reward_goal,
        reward_collision=args.reward_collision, reward_step=args.reward_step,
        progress_scale=args.progress_scale,
        spin_penalty=args.spin_penalty,
    )
    env.reset(seed=args.seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent_config = _build_agent_config(args)
    agent = create_agent(args.agent_method, obs_dim, act_dim, agent_config)

    if args.load_model:
        agent.load(args.load_model)
        log_message(f"Loaded model from {args.load_model}")

    log_message(f"=== Training: {args.agent_method} | {args.num_episodes} episodes ===")
    log_message(f"Agent config: {agent_config}")

    try:
        train(
            env=env, agent=agent,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            log_interval=args.log_interval,
            checkpoint_interval=args.checkpoint_interval,
            log_dir=log_dir,
            model_save_dir=model_dir,
        )
    except KeyboardInterrupt:
        log_message("Training interrupted.", level="WARNING")
    except Exception as e:
        log_message(f"Error: {e}", level="ERROR")
        raise


if __name__ == "__main__":
    main()

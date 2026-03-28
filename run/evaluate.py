"""
Evaluation and trajectory visualization.
Usage: python run/evaluate.py --load_model models/reinforce/best_model.pth --agent_method reinforce
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.environment import DiffDriveEnv
from src.agent import create_agent, AGENT_NAMES
from src.utils import config
from src.visualization.visualize import plot_multiple_trajectories, plot_all_on_one


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Diff Drive Navigation agent")
    p.add_argument("--load_model", type=str, required=True)
    p.add_argument("--agent_method", type=str, required=True, choices=AGENT_NAMES)
    p.add_argument("--num_episodes", type=int, default=config.EVAL_NUM_EPISODES)
    p.add_argument("--render", action="store_true", default=config.EVAL_RENDER)
    p.add_argument("--save_results", type=str, default="logs/eval_results.json")
    p.add_argument("--save_trajectories", type=str, default=None)
    p.add_argument("--save_overview", type=str, default=None)
    p.add_argument("--num_plot", type=int, default=3)
    p.add_argument("--seed", type=int, default=config.SEED)

    p.add_argument("--max_steps", type=int, default=config.MAX_STEPS)
    p.add_argument("--max_v", type=float, default=config.MAX_V)
    p.add_argument("--obstacle_radius", type=float, default=config.OBSTACLE_RADIUS)
    p.add_argument("--goal_radius", type=float, default=config.GOAL_RADIUS)
    p.add_argument("--reward_step", type=float, default=config.REWARD_STEP)
    p.add_argument("--progress_scale", type=float, default=config.PROGRESS_SCALE)
    p.add_argument("--spin_penalty", type=float, default=config.SPIN_PENALTY)
    return p.parse_args()


def evaluate(model, env, num_episodes=20, render=False):
    rewards, lengths, successes, collisions = [], [], [], []
    all_trajectories = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        last_terminated = False
        last_truncated = False

        while True:
            if render:
                env.render()
            action, _, _ = model.select_action(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                last_terminated = terminated
                last_truncated = truncated
                break

        success = bool(info.get("is_success", False))
        collision = last_terminated and not success

        rewards.append(total_reward)
        lengths.append(info["step_count"])
        successes.append(int(success))
        collisions.append(int(collision))
        all_trajectories.append({
            "trajectory": info["trajectory"],
            "goal": (float(info["goal"][0]), float(info["goal"][1])),
            "success": success,
            "collision": collision,
            "reward": total_reward,
            "length": info["step_count"],
        })

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "num_episodes": num_episodes,
        "trajectories": all_trajectories,
    }


def main():
    args = parse_args()

    env = DiffDriveEnv(
        max_steps=args.max_steps, max_v=args.max_v,
        obstacle_radius=args.obstacle_radius, goal_radius=args.goal_radius,
        reward_step=args.reward_step,
        progress_scale=args.progress_scale,
        spin_penalty=args.spin_penalty,
        render_mode="human" if args.render else None,
    )
    env.reset(seed=args.seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent_config = {"max_action": args.max_v, "gamma": config.GAMMA}
    agent = create_agent(args.agent_method, obs_dim, act_dim, agent_config)
    agent.load(args.load_model)

    print(f"Evaluating: {args.load_model} ({args.agent_method}) | {args.num_episodes} episodes")

    results = evaluate(agent, env, num_episodes=args.num_episodes, render=args.render)

    print(f"\n{'='*45}")
    print(f"  Results ({results['num_episodes']} episodes)")
    print(f"{'='*45}")
    print(f"  Mean reward:      {results['mean_reward']:>8.2f} +/- {results['std_reward']:.2f}")
    print(f"  Mean ep. length:  {results['mean_length']:>8.1f}")
    print(f"  Success rate:     {results['success_rate']:>8.1%}")
    print(f"  Collision rate:   {results['collision_rate']:>8.1%}")
    print(f"{'='*45}")

    if args.save_results:
        Path(args.save_results).parent.mkdir(parents=True, exist_ok=True)
        save_data = {k: v for k, v in results.items() if k != "trajectories"}
        with open(args.save_results, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"Results saved to {args.save_results}")

    trajectories = results["trajectories"]
    save_traj = args.save_trajectories or f"plots/{args.agent_method}_trajectories.png"
    save_over = args.save_overview or f"plots/{args.agent_method}_all_trajectories.png"

    Path(save_traj).parent.mkdir(parents=True, exist_ok=True)
    num_plot = min(args.num_plot, len(trajectories))

    if num_plot > 0:
        plot_multiple_trajectories(
            trajectories[:num_plot], obstacle_radius=args.obstacle_radius,
            title=f"{args.agent_method} — Example Trajectories",
            save_path=save_traj, show=False,
        )
        print(f"Trajectory plots saved to {save_traj}")

    plot_all_on_one(
        trajectories, obstacle_radius=args.obstacle_radius,
        title=f"{args.agent_method} — All {len(trajectories)} Trajectories "
              f"(Success: {results['success_rate']:.0%})",
        save_path=save_over, show=False,
    )
    print(f"Overview plot saved to {save_over}")

    env.close()


if __name__ == "__main__":
    main()

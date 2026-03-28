"""
Record evaluation episodes as GIF animations.
Usage: python run/record.py --load_model models/reinforce/best_model.pth --agent_method reinforce
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.environment import DiffDriveEnv
from src.agent import create_agent, AGENT_NAMES
from src.utils import config


def record_episode(agent, env, max_steps=500):
    """Run one episode, return list of RGB frames + metadata."""
    obs, info = env.reset()
    frames = [env.render()]
    total_reward = 0.0

    for _ in range(max_steps):
        action, _, _ = agent.select_action(obs, training=False)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        frames.append(env.render())
        if terminated or truncated:
            break

    success = bool(info.get("is_success", False))
    return frames, total_reward, info["step_count"], success


def frames_to_gif(frames, path, fps=20):
    """Save list of RGB numpy arrays as a GIF."""
    try:
        from PIL import Image
    except ImportError:
        print("Pillow not installed. Run: pip install Pillow")
        return

    images = [Image.fromarray(f) for f in frames if f is not None]
    if not images:
        print("No frames captured!")
        return

    images[0].save(
        path, save_all=True, append_images=images[1:],
        duration=1000 // fps, loop=0,
    )
    print(f"  Saved: {path} ({len(images)} frames)")


def parse_args():
    p = argparse.ArgumentParser(description="Record GIF of trained agent")
    p.add_argument("--load_model", type=str, required=True)
    p.add_argument("--agent_method", type=str, required=True, choices=AGENT_NAMES)
    p.add_argument("--num_episodes", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=config.MAX_STEPS)
    p.add_argument("--max_v", type=float, default=config.MAX_V)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--output_dir", type=str, default="gifs")
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    env = DiffDriveEnv(
        max_steps=args.max_steps, max_v=args.max_v,
        reward_step=config.REWARD_STEP,
        progress_scale=config.PROGRESS_SCALE,
        spin_penalty=config.SPIN_PENALTY,
        render_mode="rgb_array",
    )
    env.reset(seed=args.seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent_config = {"max_action": args.max_v, "gamma": config.GAMMA}
    agent = create_agent(args.agent_method, obs_dim, act_dim, agent_config)
    agent.load(args.load_model)

    print(f"Recording {args.num_episodes} episodes for {args.agent_method}...")

    for ep in range(1, args.num_episodes + 1):
        frames, reward, steps, success = record_episode(agent, env, args.max_steps)
        status = "success" if success else "fail"
        gif_path = f"{args.output_dir}/{args.agent_method}_ep{ep}_{status}.gif"
        frames_to_gif(frames, gif_path, fps=args.fps)
        print(f"  Episode {ep}: reward={reward:.1f}, steps={steps}, {status}")

    env.close()
    print("Done!")


if __name__ == "__main__":
    main()

"""
Record one horizontal-strip GIF: same environment seed for every algorithm,
so spawn / goal / initial heading match — fair visual comparison.

Usage:
  python run/record_compare.py --seed 42 \\
    reinforce:models/reinforce/best_model.pth \\
    reinforce_baseline:models/reinforce_baseline/best_model.pth \\
    actor_critic:models/actor_critic/best_model.pth \\
    trpo:models/trpo/best_model.pth \\
    ppo:models/ppo/best_model.pth
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.environment import DiffDriveEnv
from src.agent import create_agent, AGENT_NAMES
from src.utils import config


def parse_spec(s: str) -> tuple[str, str]:
    if ":" not in s:
        raise ValueError(f"Expected algo:path, got: {s}")
    name, path = s.split(":", 1)
    name = name.strip()
    path = path.strip()
    if name not in AGENT_NAMES:
        raise ValueError(f"Unknown agent {name!r}. Choose from {AGENT_NAMES}")
    return name, path


def record_episode_frames(agent, env: DiffDriveEnv, max_steps: int, seed: int):
    """Deterministic reset(seed) then rollout; return RGB frames list."""
    obs, _ = env.reset(seed=seed)
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


def pad_to_length(frames: list, target_len: int):
    """Repeat last frame so len == target_len."""
    if not frames:
        return []
    out = list(frames)
    last = frames[-1]
    while len(out) < target_len:
        out.append(last.copy() if hasattr(last, "copy") else np.array(last))
    return out


def hstack_row(
    frame_lists: list[list[np.ndarray]],
    labels: list[str],
    gap: int = 4,
    label_bar: int = 32,
    light: bool = False,
) -> np.ndarray:
    """One timestep: concatenate panels left-to-right with optional labels."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise RuntimeError("pip install Pillow")

    if light:
        bg = (255, 255, 255)
        text_fill = (30, 30, 30)
    else:
        bg = (30, 30, 35)
        text_fill = (240, 240, 240)

    panels = []
    h_max = 0
    for flist, label in zip(frame_lists, labels):
        img = Image.fromarray(flist[0])
        w, h = img.size
        h_max = max(h_max, h + label_bar)

        bar = Image.new("RGB", (w, label_bar), bg)
        draw = ImageDraw.Draw(bar)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
        except Exception:
            tw = len(label) * 8
        draw.text(((w - tw) / 2, 8), label, fill=text_fill, font=font)

        combined = Image.new("RGB", (w, h + label_bar), bg)
        combined.paste(bar, (0, 0))
        combined.paste(img, (0, label_bar))
        panels.append(combined)

    total_w = sum(p.size[0] for p in panels) + gap * (len(panels) - 1)
    out = Image.new("RGB", (total_w, h_max), bg)
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += p.size[0] + gap
    return np.asarray(out)


def build_comparison_gif(
    all_frames: list[list[np.ndarray]],
    labels: list[str],
    out_path: str,
    fps: int,
    gap: int = 4,
    light: bool = False,
):
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError("pip install Pillow")

    lengths = [len(f) for f in all_frames]
    T = max(lengths)
    padded = [pad_to_length(f, T) for f in all_frames]

    strip_frames = []
    for t in range(T):
        row = [p[t] for p in padded]
        strip_frames.append(hstack_row([[f] for f in row], labels, gap=gap, light=light))

    images = [Image.fromarray(f) for f in strip_frames]
    images[0].save(
        out_path, save_all=True, append_images=images[1:],
        duration=max(1, 1000 // fps), loop=0,
    )
    print(f"Saved: {out_path} ({len(images)} frames, {T} timesteps)")


def main():
    p = argparse.ArgumentParser(
        description="Side-by-side GIF: same seed per algorithm (fair comparison).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python run/record_compare.py --seed 7 "
               "reinforce:models/reinforce/best_model.pth ppo:models/ppo/best_model.pth",
    )
    p.add_argument(
        "specs", nargs="+",
        help="Entries as algo:path/to/model.pth (e.g. reinforce:models/reinforce/best_model.pth)",
    )
    p.add_argument("--seed", type=int, default=42, help="Same env reset(seed) for every algorithm")
    p.add_argument("--max_steps", type=int, default=config.MAX_STEPS)
    p.add_argument("--max_v", type=float, default=config.MAX_V)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--white", action="store_true", help="Use light/white background theme")
    p.add_argument("--screen_size", type=int, default=600, help="Render panel size in px")
    p.add_argument("--world_range", type=float, default=12.0, help="Half-extent of the rendered world")
    p.add_argument("--output", type=str, default="gifs/compare_algorithms.gif")
    args = p.parse_args()

    pairs = [parse_spec(s) for s in args.specs]
    Path(Path(args.output).parent).mkdir(parents=True, exist_ok=True)

    env = DiffDriveEnv(
        max_steps=args.max_steps, max_v=args.max_v,
        reward_step=config.REWARD_STEP,
        progress_scale=config.PROGRESS_SCALE,
        spin_penalty=config.SPIN_PENALTY,
        render_mode="rgb_array",
        light_theme=args.white,
    )
    env.render_screen_size = args.screen_size
    env.render_world_range = args.world_range
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    all_frames: list[list[np.ndarray]] = []
    labels: list[str] = []
    stats: list[str] = []

    for name, path in pairs:
        if not Path(path).exists():
            print(f"WARNING: missing model {path} — skip {name}")
            continue
        agent_cfg = {"max_action": args.max_v, "gamma": config.GAMMA}
        agent = create_agent(name, obs_dim, act_dim, agent_cfg)
        agent.load(path)

        frames, reward, steps, success = record_episode_frames(
            agent, env, args.max_steps, args.seed,
        )
        all_frames.append(frames)
        pretty = {
            "reinforce": "REINFORCE",
            "reinforce_baseline": "REINFORCE + Baseline",
            "actor_critic": "Actor-Critic",
            "trpo": "TRPO",
            "ppo": "PPO",
        }
        labels.append(pretty.get(name, name.replace("_", " ")))
        st = "ok" if success else "fail"
        stats.append(f"  {name}: reward={reward:.1f}, steps={steps}, {st}")
        print(f"{name}: reward={reward:.1f}, steps={steps}, success={success}")

    env.close()

    if not all_frames:
        print("No valid models loaded. Check paths.")
        return

    print("---")
    print("\n".join(stats))
    build_comparison_gif(all_frames, labels, args.output, fps=args.fps, light=args.white)
    print("Done.")


if __name__ == "__main__":
    main()

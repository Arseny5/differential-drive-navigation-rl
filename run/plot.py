"""
Plot training curves from CSV logs.
Supports single algorithm or comparison of multiple.

Usage:
    python run/plot.py                                        # all algorithms found in logs/
    python run/plot.py --csv_paths logs/reinforce/train_stats.csv logs/trpo/train_stats.csv
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from src.agent import AGENT_NAMES


def moving_average(data, window=50):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def load_csv(path: str) -> dict:
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return {
        "episodes": data[:, 0].astype(int),
        "rewards": data[:, 1],
        "lengths": data[:, 2],
        "successes": data[:, 3],
    }


COLORS = {
    "reinforce": "#E74C3C",
    "reinforce_baseline": "#3498DB",
    "actor_critic": "#2ECC71",
    "trpo": "#9B59B6",
    "ppo": "#E67E22",
}


def plot_comparison(datasets: dict[str, dict], save_dir: str = "plots", window: int = 50):
    """Plot reward, success rate, and episode length for one or more algorithms."""
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 16))

    for name, d in datasets.items():
        color = COLORS.get(name, "gray")
        episodes = d["episodes"]

        # Reward
        smoothed = moving_average(d["rewards"], window)
        offset = (len(episodes) - len(smoothed)) // 2
        ax1.plot(d["rewards"], alpha=0.08, color=color)
        ax1.plot(np.arange(len(smoothed)) + offset, smoothed, color=color, linewidth=2, label=name)

        # Success rate
        sr = moving_average(d["successes"], window=100)
        ax2.plot(np.arange(len(sr)) + 50, sr, color=color, linewidth=2, label=name)

        # Episode length
        ls = moving_average(d["lengths"], window)
        ax3.plot(np.arange(len(ls)) + offset, ls, color=color, linewidth=2, label=name)

    ax1.set_title("Reward per Episode", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Total Reward")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Success Rate (Rolling Window 100)", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Rate")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    ax3.set_title("Episode Length", fontsize=14, fontweight="bold")
    ax3.set_ylabel("Steps")
    ax3.set_xlabel("Episode")
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if len(datasets) == 1:
        name = list(datasets.keys())[0]
        save_path = os.path.join(save_dir, f"{name}_training_curves.png")
    else:
        save_path = os.path.join(save_dir, "comparison_training_curves.png")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {save_path}")
    plt.show()


def main():
    p = argparse.ArgumentParser(description="Plot training curves")
    p.add_argument("--csv_paths", type=str, nargs="*", default=None,
                   help="Explicit CSV paths (if not given, auto-discovers from logs/)")
    p.add_argument("--log_dir", type=str, default="logs")
    p.add_argument("--save_dir", type=str, default="plots")
    p.add_argument("--window", type=int, default=50)
    args = p.parse_args()

    datasets: dict[str, dict] = {}

    if args.csv_paths:
        for path in args.csv_paths:
            name = Path(path).parent.name
            if name == "logs":
                name = Path(path).stem
            datasets[name] = load_csv(path)
    else:
        for algo in AGENT_NAMES:
            csv = os.path.join(args.log_dir, algo, "train_stats.csv")
            if os.path.exists(csv):
                datasets[algo] = load_csv(csv)

    if not datasets:
        print("No training CSVs found. Run train.py first.")
        return

    print(f"Plotting: {list(datasets.keys())}")
    plot_comparison(datasets, save_dir=args.save_dir, window=args.window)


if __name__ == "__main__":
    main()

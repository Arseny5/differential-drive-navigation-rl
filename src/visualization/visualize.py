"""
Визуализация траекторий агента на 2D-плоскости.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_trajectory(
    trajectory: list[tuple[float, float]],
    goal: tuple[float, float],
    obstacle_center: tuple[float, float] = (0.0, 0.0),
    obstacle_radius: float = 2.0,
    title: str = "Agent Trajectory",
    ax: plt.Axes | None = None,
    show: bool = True,
    save_path: str | None = None,
) -> plt.Axes:
    """Одна траектория на плоскости с препятствием и целью."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    traj = np.array(trajectory)
    ax.plot(traj[:, 0], traj[:, 1], color="#4A90D9", linewidth=1.5, zorder=3)
    ax.scatter(traj[0, 0], traj[0, 1], color="#2ECC71", s=80, zorder=5, label="Start", edgecolors="black")
    ax.scatter(traj[-1, 0], traj[-1, 1], color="#E74C3C", s=80, zorder=5, label="End", edgecolors="black", marker="s")

    ax.scatter(goal[0], goal[1], color="#F39C12", s=120, zorder=5, marker="*", label="Goal", edgecolors="black")

    obstacle = patches.Circle(obstacle_center, obstacle_radius, color="#E74C3C", alpha=0.3, zorder=2)
    ax.add_patch(obstacle)
    obstacle_border = patches.Circle(obstacle_center, obstacle_radius, fill=False, edgecolor="#C0392B", linewidth=2, zorder=2)
    ax.add_patch(obstacle_border)

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return ax


def plot_multiple_trajectories(
    trajectories: list[dict],
    obstacle_radius: float = 2.0,
    title: str = "Evaluation Trajectories",
    save_path: str | None = None,
    show: bool = True,
):
    """
    Несколько траекторий на одном графике или на subplots.

    trajectories: list of dicts с ключами 'trajectory', 'goal', 'success'.
    """
    n = len(trajectories)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()

    for i, traj_data in enumerate(trajectories):
        ax = axes[i]
        traj = traj_data["trajectory"]
        goal = traj_data["goal"]
        success = traj_data.get("success", False)

        status = "SUCCESS" if success else "FAIL"
        color = "#2ECC71" if success else "#E74C3C"

        plot_trajectory(
            traj, goal,
            obstacle_radius=obstacle_radius,
            title=f"Episode {i+1} [{status}]",
            ax=ax, show=False,
        )
        ax.set_title(f"Episode {i+1} [{status}]", color=color, fontweight="bold")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()


def plot_all_on_one(
    trajectories: list[dict],
    obstacle_radius: float = 2.0,
    title: str = "All Trajectories",
    save_path: str | None = None,
    show: bool = True,
):
    """Все траектории на одном графике (обзорная визуализация)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    obstacle = patches.Circle((0, 0), obstacle_radius, color="#E74C3C", alpha=0.3, zorder=2)
    ax.add_patch(obstacle)
    obstacle_border = patches.Circle((0, 0), obstacle_radius, fill=False, edgecolor="#C0392B", linewidth=2, zorder=2)
    ax.add_patch(obstacle_border)

    cmap_success = plt.cm.Greens
    cmap_fail = plt.cm.Reds

    n_success = sum(1 for t in trajectories if t.get("success", False))
    n_fail = len(trajectories) - n_success
    s_idx, f_idx = 0, 0

    for traj_data in trajectories:
        traj = np.array(traj_data["trajectory"])
        goal = traj_data["goal"]
        success = traj_data.get("success", False)

        if success:
            color = cmap_success(0.4 + 0.5 * s_idx / max(1, n_success))
            s_idx += 1
        else:
            color = cmap_fail(0.4 + 0.5 * f_idx / max(1, n_fail))
            f_idx += 1

        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.0, alpha=0.7, zorder=3)
        ax.scatter(traj[0, 0], traj[0, 1], color=color, s=20, zorder=4, edgecolors="black", linewidths=0.5)
        ax.scatter(goal[0], goal[1], color="#F39C12", s=40, zorder=5, marker="*", edgecolors="black", linewidths=0.5)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="green", lw=2, label=f"Success ({n_success})"),
        Line2D([0], [0], color="red", lw=2, label=f"Fail ({n_fail})"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#F39C12", markersize=12, label="Goal"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

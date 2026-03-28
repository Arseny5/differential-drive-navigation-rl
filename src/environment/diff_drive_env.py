"""
Gymnasium-среда для навигации дифференциального привода.

Робот (x, y, θ) должен доехать до цели, избегая круглого препятствия в центре.

Кинематика:
    ẋ = (v_r + v_l) / 2 · cos(θ)
    ẏ = (v_r + v_l) / 2 · sin(θ)
    θ̇ = (v_r - v_l) / L
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DiffDriveEnv(gym.Env):
    """
    Дифференциальный привод на 2D плоскости.

    Наблюдение (8-мерный вектор):
        [x, y, cos(θ), sin(θ), goal_x, goal_y, dist_to_goal, dist_to_obstacle]

    Действие (непрерывное, 2D):
        [v_left, v_right] ∈ [-max_v, max_v]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        max_steps: int = 300,
        dt: float = 0.1,
        wheelbase: float = 0.5,
        max_v: float = 2.0,
        obstacle_radius: float = 2.0,
        goal_radius: float = 0.5,
        collision_radius: float = 0.25,
        reward_goal: float = 100.0,
        reward_collision: float = -100.0,
        reward_step: float = -0.05,
        progress_scale: float = 5.0,
        spin_penalty: float = 0.01,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.dt = dt
        self.wheelbase = wheelbase
        self.max_v = max_v
        self.obstacle_radius = obstacle_radius
        self.goal_radius = goal_radius
        self.collision_radius = collision_radius

        self.reward_goal = reward_goal
        self.reward_collision = reward_collision
        self.reward_step = reward_step
        self.progress_scale = progress_scale
        self.spin_penalty = spin_penalty

        # obs: [x, y, cos(θ), sin(θ), gx, gy, dist_goal, dist_obstacle]
        obs_high = np.array([10, 10, 1, 1, 10, 10, 20, 20], dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self.action_space = spaces.Box(
            low=-self.max_v, high=self.max_v, shape=(2,), dtype=np.float32
        )

        self._state = np.zeros(3, dtype=np.float64)  # x, y, θ
        self._goal = np.zeros(2, dtype=np.float64)
        self._step_count = 0
        self._prev_dist = 0.0
        self._trajectory: list[tuple[float, float]] = []
        self._is_success = False

    def _dist_sq(self, x: float, y: float) -> float:
        return float(x * x + y * y)

    def _sample_agent_pose(self) -> tuple[float, float, float]:
        """Agent spawn: Q3, x ∈ [-4, -2], y ∈ [-4, -2], outside collision shell."""
        min_r = self.obstacle_radius + self.collision_radius + 0.05
        min_sq = min_r * min_r
        for _ in range(300):
            ax = self.np_random.uniform(-4.0, -2.0)
            ay = self.np_random.uniform(-4.0, -2.0)
            if self._dist_sq(ax, ay) >= min_sq:
                atheta = self.np_random.uniform(-np.pi, np.pi)
                return ax, ay, atheta
        return -3.0, -3.0, self.np_random.uniform(-np.pi, np.pi)

    def _sample_goal(self) -> tuple[float, float]:
        """Goal spawn: Q1, x ∈ [2, 4], y ∈ [2, 4], outside obstacle disk."""
        r = self.obstacle_radius
        min_sq = r * r + 1e-3
        for _ in range(300):
            gx = self.np_random.uniform(2.0, 4.0)
            gy = self.np_random.uniform(2.0, 4.0)
            if self._dist_sq(gx, gy) >= min_sq:
                return gx, gy
        return 3.0, 3.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        ax, ay, atheta = self._sample_agent_pose()
        self._state = np.array([ax, ay, atheta], dtype=np.float64)

        gx, gy = self._sample_goal()
        self._goal = np.array([gx, gy], dtype=np.float64)

        self._step_count = 0
        self._prev_dist = self._dist_to_goal()
        self._trajectory = [(ax, ay)]
        self._is_success = False

        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.clip(action, -self.max_v, self.max_v)
        v_left, v_right = float(action[0]), float(action[1])

        x, y, theta = self._state
        v = (v_right + v_left) / 2.0
        omega = (v_right - v_left) / self.wheelbase

        x += v * np.cos(theta) * self.dt
        y += v * np.sin(theta) * self.dt
        theta += omega * self.dt
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        self._state = np.array([x, y, theta], dtype=np.float64)
        self._step_count += 1
        self._trajectory.append((x, y))

        dist_goal = self._dist_to_goal()
        dist_obs = self._dist_to_obstacle()

        terminated = False
        truncated = False
        self._is_success = False

        reward = self.reward_step
        reward += self.progress_scale * (self._prev_dist - dist_goal)
        reward -= self.spin_penalty * abs(v_right - v_left)

        d_collide = self.obstacle_radius + self.collision_radius
        if dist_obs < d_collide:
            reward += self.reward_collision
            terminated = True
            self._is_success = False
        elif dist_goal < self.goal_radius:
            reward += self.reward_goal
            terminated = True
            self._is_success = True
        elif self._step_count >= self.max_steps:
            truncated = True
            self._is_success = False

        self._prev_dist = dist_goal

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def _get_obs(self) -> np.ndarray:
        x, y, theta = self._state
        return np.array([
            x, y,
            np.cos(theta), np.sin(theta),
            self._goal[0], self._goal[1],
            self._dist_to_goal(),
            self._dist_to_obstacle(),
        ], dtype=np.float32)

    def _get_info(self) -> dict:
        return {
            "state": self._state.copy(),
            "goal": self._goal.copy(),
            "step_count": self._step_count,
            "dist_to_goal": self._dist_to_goal(),
            "dist_to_obstacle": self._dist_to_obstacle(),
            "trajectory": list(self._trajectory),
            "is_success": bool(self._is_success),
        }

    def _dist_to_goal(self) -> float:
        return float(np.linalg.norm(self._state[:2] - self._goal))

    def _dist_to_obstacle(self) -> float:
        return float(np.linalg.norm(self._state[:2]))

    def render(self):
        if self.render_mode is None:
            return None
        return self._render_frame()

    def _render_frame(self):
        try:
            import pygame
        except ImportError:
            return None

        screen_size = 600
        world_range = 8.0
        scale = screen_size / (2 * world_range)

        def world_to_screen(wx, wy):
            sx = int((wx + world_range) * scale)
            sy = int((world_range - wy) * scale)
            return sx, sy

        if not hasattr(self, "_pygame_screen"):
            pygame.init()
            if self.render_mode == "human":
                self._pygame_screen = pygame.display.set_mode((screen_size, screen_size))
                pygame.display.set_caption("Diff Drive Navigation")
            else:
                self._pygame_screen = pygame.Surface((screen_size, screen_size))
            self._pygame_clock = pygame.time.Clock()

        surf = self._pygame_screen
        surf.fill((25, 25, 30))

        for i in range(-int(world_range), int(world_range) + 1, 2):
            pygame.draw.line(surf, (40, 40, 45), world_to_screen(i, -world_range), world_to_screen(i, world_range))
            pygame.draw.line(surf, (40, 40, 45), world_to_screen(-world_range, i), world_to_screen(world_range, i))

        ox, oy = world_to_screen(0, 0)
        obs_r = int(self.obstacle_radius * scale)
        pygame.draw.circle(surf, (180, 50, 50), (ox, oy), obs_r)
        pygame.draw.circle(surf, (220, 70, 70), (ox, oy), obs_r, 2)

        gx, gy = world_to_screen(self._goal[0], self._goal[1])
        goal_r = max(4, int(self.goal_radius * scale))
        pygame.draw.circle(surf, (50, 220, 100), (gx, gy), goal_r)

        if len(self._trajectory) > 1:
            points = [world_to_screen(p[0], p[1]) for p in self._trajectory]
            pygame.draw.lines(surf, (100, 150, 255), False, points, 2)

        x, y, theta = self._state
        ax, ay = world_to_screen(x, y)
        agent_r = max(4, int(0.3 * scale))
        pygame.draw.circle(surf, (255, 220, 50), (ax, ay), agent_r)
        end_x = ax + int(agent_r * 1.8 * np.cos(-theta))
        end_y = ay + int(agent_r * 1.8 * np.sin(-theta))
        pygame.draw.line(surf, (255, 255, 255), (ax, ay), (end_x, end_y), 2)

        if self.render_mode == "human":
            pygame.display.flip()
            self._pygame_clock.tick(self.metadata["render_fps"])
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()

        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2)
            )
        return None

    def close(self):
        if hasattr(self, "_pygame_screen"):
            import pygame
            pygame.quit()
            del self._pygame_screen

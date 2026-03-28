# Differential Drive Navigation with Reinforcement Learning

Train an RL agent (differential drive robot) to navigate from a random start to a random goal on a 2D plane while avoiding a circular obstacle at the origin.

## Project Structure

```
diff-drive-nav/
├── src/
│   ├── agent/
│   │   ├── networks.py              — PolicyNetwork (Gaussian), ValueNetwork
│   │   ├── base_agent.py            — BaseAgent ABC + GAE helper
│   │   ├── reinforce.py             — Vanilla REINFORCE
│   │   ├── reinforce_baseline.py    — REINFORCE + learned baseline
│   │   ├── actor_critic.py          — A2C with GAE
│   │   └── trpo.py                  — TRPO (CG + line search)
│   ├── environment/
│   │   └── diff_drive_env.py        — Gymnasium environment
│   ├── training/
│   │   ├── trainer.py               — Training loop
│   │   └── logger.py                — Logging
│   ├── visualization/
│   │   └── visualize.py             — Trajectory plots (matplotlib)
│   └── utils/
│       └── config.py                — All hyperparameters
├── run/
│   ├── train.py                     — Train a single algorithm
│   ├── evaluate.py                  — Evaluate + visualize trajectories
│   ├── plot.py                      — Plot training curves (single or comparison)
│   └── compare.py                   — Train all 4 algorithms and compare
├── models/                          — Saved models
├── logs/                            — CSV metrics + training history
├── plots/                           — Generated figures
├── requirements.txt
└── README.md
```

## Environment

**Robot state:** `(x, y, θ)` — position and heading angle.

**Observation (8D):** `[x, y, cos(θ), sin(θ), goal_x, goal_y, dist_to_goal, dist_to_obstacle]`

**Action (2D, continuous):** `[v_left, v_right]` ∈ `[-2, 2]` — left and right wheel velocities.

**Differential drive kinematics:**

$$\dot{x} = \frac{v_r + v_l}{2} \cos\theta, \quad \dot{y} = \frac{v_r + v_l}{2} \sin\theta, \quad \dot{\theta} = \frac{v_r - v_l}{L}$$

**Spawn regions:**
- Agent: x ∈ [−4, −2], y ∈ [−4, −2]
- Goal: x ∈ [2, 4], y ∈ [2, 4]
- Obstacle: circle r = 2 at origin (0, 0)

Spawn rectangles are chosen so samples usually lie outside the obstacle; **invalid agent poses** (inside the collision shell) are rejected and resampled.

**Termination:** `terminated` on goal (`is_success=True`) or collision (`is_success=False`); `truncated` on max steps (`is_success=False`). Every `info` dict includes **`is_success`** (False while the episode is ongoing).

## Reward Function

Dense reward is **progress-only** plus small time/spin costs, so “farm shaping and never finish” is much harder than under heading / near-goal / near-obstacle bonuses. The sum of progress terms over an episode telescopes to roughly `progress_scale × (d₀ − d_T)`, which for this layout stays below the +100 success bonus when `progress_scale=5`.

| Component | Default | Rationale |
|-----------|---------|-----------|
| Step penalty | −0.05 | Shorter useless rollouts |
| Progress | `5.0 × (d_prev − d_goal)` | Reward only real motion toward the goal; oscillation cancels out |
| Spin penalty | `0.01 × \|v_r − v_l\|` | Mild discouragement of spinning in place |
| Goal reached | +100 | Dominant terminal bonus |
| Collision | −100 | Terminal failure |

## Algorithms

All four algorithms are implemented from scratch in PyTorch:

| Algorithm | Description |
|-----------|-------------|
| **REINFORCE** | Vanilla Monte-Carlo policy gradient with return normalization |
| **REINFORCE + Baseline** | REINFORCE with a learned value function V(s) as variance-reducing baseline |
| **Actor-Critic (A2C)** | Advantage Actor-Critic using GAE (bootstrapped advantage) |
| **TRPO** | Trust Region Policy Optimization — natural gradient via conjugate gradient, with KL-constrained line search |

All agents use a **Gaussian policy** (separate mean network + learnable log-std) for continuous control.

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train a single algorithm

```bash
python run/train.py --agent_method reinforce --num_episodes 10000
python run/train.py --agent_method reinforce_baseline --num_episodes 10000
python run/train.py --agent_method actor_critic --num_episodes 10000
python run/train.py --agent_method trpo --num_episodes 10000
```

### Train all four and compare

```bash
python run/compare.py --num_episodes 10000
```

This trains all four algorithms, evaluates each on 20 random episodes, prints a comparison table, and generates a combined plot at `plots/comparison_training_curves.png`.

### Plot training curves

```bash
python run/plot.py                     # auto-discovers all trained algorithms
python run/plot.py --csv_paths logs/reinforce/train_stats.csv logs/trpo/train_stats.csv
```

### Evaluate a trained model

```bash
python run/evaluate.py \
    --load_model models/actor_critic/best_model.pth \
    --agent_method actor_critic \
    --num_episodes 20
```

### Evaluate with Pygame rendering

```bash
python run/evaluate.py \
    --load_model models/actor_critic/best_model.pth \
    --agent_method actor_critic \
    --render
```

### GIFs (single algorithm)

```bash
python run/record.py \
    --load_model models/reinforce/best_model.pth \
    --agent_method reinforce \
    --num_episodes 3
```

### Side-by-side comparison GIF (same game, different algorithms)

Each algorithm runs with the **same** `env.reset(seed=…)`, so start pose, goal, and initial heading match. Frames are concatenated horizontally into one GIF (labels above each panel).

```bash
python run/record_compare.py --seed 42 \
    reinforce:models/reinforce/best_model.pth \
    reinforce_baseline:models/reinforce_baseline/best_model.pth \
    actor_critic:models/actor_critic/best_model.pth \
    trpo:models/trpo/best_model.pth \
    --output gifs/compare_algorithms.gif
```

## Generated Outputs

After training and evaluation, the following files are produced:

| File | Description |
|------|-------------|
| `logs/<method>/train_stats.csv` | Per-episode reward, length, success |
| `logs/<method>/training_history.json` | Full training metrics |
| `models/<method>/best_model.pth` | Best model (by rolling average reward) |
| `models/<method>/final_model.pth` | Model at end of training |
| `plots/comparison_training_curves.png` | Reward / success rate / length comparison |
| `plots/<method>_trajectories.png` | Example trajectory visualizations |
| `plots/<method>_all_trajectories.png` | All evaluation trajectories on one plot |
| `gifs/compare_algorithms.gif` | Side-by-side rollout (`record_compare.py`) |

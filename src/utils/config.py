"""
Configuration for the Diff-Drive Navigation project.

All parameters can be overridden via CLI arguments in run/train.py, run/evaluate.py.
"""

# ── Environment ──────────────────────────────────────────────────────
MAX_STEPS = 300
DT = 0.1
WHEELBASE = 0.5
MAX_V = 2.0
OBSTACLE_RADIUS = 2.0
GOAL_RADIUS = 0.5
COLLISION_RADIUS = 0.25

# ── Rewards (aligned with task: reach goal, avoid obstacle, no dithering) ─
REWARD_GOAL = 100.0
REWARD_COLLISION = -100.0
REWARD_STEP = -0.05
PROGRESS_SCALE = 5.0
SPIN_PENALTY = 0.01

# ── Training (shared) ───────────────────────────────────────────────
AGENT_METHOD = "reinforce_baseline"
NUM_EPISODES = 10000
GAMMA = 0.99
SEED = 42
LOG_INTERVAL = 50
CHECKPOINT_INTERVAL = 500

# ── REINFORCE ────────────────────────────────────────────────────────
REINFORCE_LR = 3e-4
REINFORCE_HIDDEN_DIMS = [128, 128]
REINFORCE_ENTROPY_COEF = 0.05
REINFORCE_MAX_GRAD_NORM = 0.5

# ── REINFORCE + Baseline ────────────────────────────────────────────
BASELINE_POLICY_LR = 3e-4
BASELINE_VALUE_LR = 1e-3
BASELINE_HIDDEN_DIMS = [128, 128]
BASELINE_ENTROPY_COEF = 0.02
BASELINE_MAX_GRAD_NORM = 0.5

# ── Actor-Critic (A2C + GAE) ────────────────────────────────────────
AC_POLICY_LR = 3e-4
AC_VALUE_LR = 1e-3
AC_HIDDEN_DIMS = [128, 128]
AC_ENTROPY_COEF = 0.02
AC_GAE_LAMBDA = 0.95
AC_MAX_GRAD_NORM = 0.5

# ── TRPO ─────────────────────────────────────────────────────────────
TRPO_HIDDEN_DIMS = [128, 128]
TRPO_VALUE_LR = 1e-3
TRPO_MAX_KL = 0.01
TRPO_CG_ITERS = 10
TRPO_BACKTRACK_ITERS = 10
TRPO_BACKTRACK_COEF = 0.5
TRPO_DAMPING = 0.1
TRPO_GAE_LAMBDA = 0.95
TRPO_ENTROPY_COEF = 0.01
TRPO_MAX_GRAD_NORM = 0.5

# ── Evaluation ───────────────────────────────────────────────────────
EVAL_NUM_EPISODES = 20
EVAL_RENDER = False
EVAL_MODEL_PATH = "models/best_model.pth"

# ── Paths ────────────────────────────────────────────────────────────
LOG_DIR = "logs"
MODEL_SAVE_DIR = "models"
PLOT_DIR = "plots"

# ── Logging ──────────────────────────────────────────────────────────
LOG_FILE = None
LOG_LEVEL = "INFO"

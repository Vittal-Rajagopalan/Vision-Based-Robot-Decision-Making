import time
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from training_logger import TrainingLogger

GRID_SIZE = 8
MAX_STEPS = 60
STARTING_ROBOT_POSITION = np.array([0, 0])
STARTING_COW_POSITION = np.array([4, 4])
STARTING_GOAL_POSITION = np.array([7, 7])
COW_MOVE_PROB = 0.7  # chance the cow moves each step; otherwise it stays put
ACTION_NAMES = {0: "right", 1: "up", 2: "left", 3: "down", 4: "stay"}

# Q-learning parameters
EPISODES = 15000
ALPHA = 0.1
GAMMA = 0.95
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.99995

# Evaluation
NUM_TRIALS = 100
SEED = 10101

# Files
MODEL_PATH = "q_table.npz"
LOG_PATH = "training_log.txt"
LOG_INTERVAL = 20


class FarmRobotEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.size = GRID_SIZE
        self.max_steps = MAX_STEPS
        self.steps = 0

        # Fail early if a starting position doesn't fit the grid
        for name, pos in (("robot", STARTING_ROBOT_POSITION),
                          ("cow", STARTING_COW_POSITION),
                          ("goal", STARTING_GOAL_POSITION)):
            if np.any(pos < 0) or np.any(pos >= self.size):
                raise ValueError(
                    f"Starting {name} position {pos} is outside the "
                    f"{self.size}x{self.size} grid (GRID_SIZE={GRID_SIZE})"
                )

        self._robot_loc = STARTING_ROBOT_POSITION.copy()
        self._cow_loc = STARTING_COW_POSITION.copy()
        self._goal_loc = STARTING_GOAL_POSITION.copy()

        self.action_space = gym.spaces.Discrete(5)

        self.observation_space = gym.spaces.Dict(
            {
                "robot": gym.spaces.Box(0, self.size - 1, shape=(2,), dtype=np.int32),
                "cow": gym.spaces.Box(0, self.size - 1, shape=(2,), dtype=np.int32),
                "goal": gym.spaces.Box(0, self.size - 1, shape=(2,), dtype=np.int32),
            }
        )

        # Action index -> (row, col) change
        self._action_to_direction = {
            0: np.array([0, 1]),   # right
            1: np.array([-1, 0]),  # up
            2: np.array([0, -1]),  # left
            3: np.array([1, 0]),   # down
            4: np.array([0, 0]),   # stay
        }

    def _get_obs(self):
        return {
            "robot": self._robot_loc.copy(),
            "cow": self._cow_loc.copy(),
            "goal": self._goal_loc.copy(),
        }

    def _get_info(self):
        return {"distance": self.distance(self._robot_loc, self._cow_loc)}

    def distance(self, loc1, loc2):
        """Chebyshev distance (used for 'too close')."""
        return np.max(np.abs(loc1 - loc2))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._robot_loc = STARTING_ROBOT_POSITION.copy()
        self._cow_loc = STARTING_COW_POSITION.copy()
        self._goal_loc = STARTING_GOAL_POSITION.copy()
        self.steps = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.steps += 1
        direction = self._action_to_direction[int(action)]

        self._robot_loc = np.clip(self._robot_loc + direction, 0, self.size - 1)
        self.move_cow()

        terminated = self.success()
        truncated = self.steps >= self.max_steps
        reward = self.reward_calculator()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def reward_calculator(self):
        if self.collision():
            return -10
        elif self.success():
            return 20
        elif self.too_close():
            return -3
        else:
            return -0.1

    def collision(self):
        return np.array_equal(self._robot_loc, self._cow_loc)

    def success(self):
        return np.array_equal(self._robot_loc, self._goal_loc)

    def too_close(self):
        return self.distance(self._robot_loc, self._cow_loc) == 1

    def move_cow(self):

        if self.np_random.random() >= COW_MOVE_PROB:
            return
        direction = int(self.np_random.integers(0, 4))
        self._cow_loc = np.clip(
            self._cow_loc + self._action_to_direction[direction], 0, self.size - 1
        )
    # ---------- visualisation ----------
    def _draw_grid(self):
        self.grid = np.zeros((self.size, self.size))
        self.grid[self._goal_loc[0], self._goal_loc[1]] = 1
        self.grid[self._cow_loc[0], self._cow_loc[1]] = 2
        self.grid[self._robot_loc[0], self._robot_loc[1]] = 3

    def render(self):
        self._draw_grid()
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.im = self.ax.imshow(self.grid, cmap="viridis", vmin=0, vmax=3)
        self.ax.set_xticks(range(self.size))
        self.ax.set_yticks(range(self.size))
        self.ax.grid(True)
        self.ax.set_title("Farm Robot Environment")
        plt.show(block=False)
        plt.pause(0.1)

    def updateVis(self):
        self._draw_grid()
        self.im.set_data(self.grid)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def get_pos(self):
        return self._robot_loc, self._cow_loc, self._goal_loc

    # ---------- baseline controller ----------
    def baseline_robot_mover(self):
        """Return an action index (0-4) using a simple hand-written heuristic."""
        best_action, best_score = 4, float("inf")

        for action, direction in self._action_to_direction.items():
            new_pos = np.clip(self._robot_loc + direction, 0, self.size - 1)

            goal_dist = np.sum(np.abs(self._goal_loc - new_pos))   # Manhattan
            cow_manhattan = np.sum(np.abs(self._cow_loc - new_pos))
            cow_cheb = self.distance(new_pos, self._cow_loc)

            score = goal_dist
            if cow_manhattan <= 1:
                score += 20
            elif cow_cheb == 1:
                score += 3
            if action == 4:
                score += 0.5

            if score < best_score:
                best_action, best_score = action, score

        return best_action


class Qlearning:
    def __init__(self, learning_rate, gamma, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reset_qtable()

    def update(self, state, action, reward, new_state, terminated):
        """Q(s,a) += lr * (target - Q(s,a)); no bootstrapping after the goal."""
        target = reward if terminated else reward + self.gamma * np.max(self.qtable[new_state, :])
        self.qtable[state, action] += self.learning_rate * (target - self.qtable[state, action])

    def reset_qtable(self):
        self.qtable = np.zeros((self.state_size, self.action_size))


class EpsilonGreedy:
    def __init__(self, epsilon, seed=None):
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def choose_action(self, action_space, state, qtable):
        if self.rng.uniform(0, 1) < self.epsilon:
            return action_space.sample()
        max_ids = np.where(qtable[state, :] == np.max(qtable[state, :]))[0]
        return int(self.rng.choice(max_ids))


def encode_coord(coord):
    """Converts [r, c] to a cell index 0..GRID_SIZE**2 - 1."""
    return coord[0] * GRID_SIZE + coord[1]


def get_state_index(obs):
    """Combines robot and cow cells into one integer (0..GRID_SIZE**4 - 1)."""
    return encode_coord(obs["robot"]) * GRID_SIZE * GRID_SIZE + encode_coord(obs["cow"])


def train(episodes=EPISODES):
    env = FarmRobotEnv()
    agent = Qlearning(ALPHA, GAMMA, state_size=GRID_SIZE ** 4, action_size=env.action_space.n)
    policy = EpsilonGreedy(epsilon=EPSILON)
    logger = TrainingLogger(LOG_PATH, interval=LOG_INTERVAL)

    print("🤖 Training the Q-Learning Agent...")
    for episode in range(1, episodes + 1):
        obs, info = env.reset()
        state = get_state_index(obs)
        terminated = truncated = False
        total_reward = 0

        while not (terminated or truncated):
            action = policy.choose_action(env.action_space, state, agent.qtable)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = get_state_index(next_obs)

            agent.update(state, action, reward, next_state, terminated)
            state = next_state
            total_reward += reward

        policy.epsilon = max(EPSILON_MIN, policy.epsilon * EPSILON_DECAY)
        logger.log_episode(episode, total_reward, success=terminated)

        if episode % 500 == 0:
            recent = np.mean(logger.rewards[-100:])
            print(f"Episode {episode}/{episodes} | Avg reward (last 100): {recent:.2f} "
                  f"| Epsilon: {policy.epsilon:.4f}")

    successes, total = logger.finish()
    np.savez(MODEL_PATH, qtable=agent.qtable, grid_size=GRID_SIZE)

    print(f"\n✅ Training complete. Successful episodes: {successes}/{total}")
    print(f"   Model saved to {MODEL_PATH}")
    print(f"   Log written to {LOG_PATH}")


def load_qtable():
    """Load the saved Q-table, or return None (after printing why) if unusable."""
    try:
        data = np.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"❌ No trained model found at {MODEL_PATH}. Train one first (option 1).")
        return None

    if int(data["grid_size"]) != GRID_SIZE:
        print(f"❌ Saved model is for a {int(data['grid_size'])}x{int(data['grid_size'])} grid, "
              f"but GRID_SIZE is {GRID_SIZE}. Retrain or change GRID_SIZE.")
        return None

    return data["qtable"]


def test():
    qtable = load_qtable()
    if qtable is None:
        return

    env = FarmRobotEnv()
    policy = EpsilonGreedy(epsilon=0.0)   # pure exploitation

    env.render()
    obs, info = env.reset()
    state = get_state_index(obs)
    terminated = truncated = False
    total_reward = 0

    while not (terminated or truncated):
        action = policy.choose_action(env.action_space, state, qtable)
        obs, reward, terminated, truncated, info = env.step(action)
        state = get_state_index(obs)
        total_reward += reward
        env.updateVis()
        time.sleep(0.3)

    if terminated:
        print(f"🎉 Reached the goal in {env.steps} steps. Total reward: {total_reward:.1f}")
    else:
        print(f"⏱️ Ran out of steps. Total reward: {total_reward:.1f}")

    plt.ioff()
    plt.show()


def run_trial(env, policy, qtable, seed):
    """Run one greedy episode with the trained Q-table.

    Returns (success, steps, collided), where collided is True if the robot
    landed on the cow at any point during the episode.
    """
    obs, info = env.reset(seed=seed)
    state = get_state_index(obs)
    terminated = truncated = False
    collided = False

    while not (terminated or truncated):
        action = policy.choose_action(env.action_space, state, qtable)
        obs, reward, terminated, truncated, info = env.step(action)
        state = get_state_index(obs)
        if env.collision():
            collided = True

    return terminated, env.steps, collided


def evaluate(num_trials=NUM_TRIALS):
    """Run many headless trials with the trained model and print summary stats."""
    qtable = load_qtable()
    if qtable is None:
        return

    env = FarmRobotEnv()
    policy = EpsilonGreedy(epsilon=0.0, seed=SEED)   # pure exploitation, reproducible tie-breaks

    successes = 0
    collision_trials = 0
    success_steps = []

    for trial in range(num_trials):
        success, steps, collided = run_trial(env, policy, qtable, seed=SEED + trial)
        successes += success
        collision_trials += collided
        if success:
            success_steps.append(steps)

    avg_steps = np.mean(success_steps) if success_steps else float("nan")

    print(f"\nResults over {num_trials} trials:")
    print(f"Success rate:        {100 * successes / num_trials:.1f}%")
    print(f"Collision rate:      {100 * collision_trials / num_trials:.1f}%")
    print(f"Avg. steps to goal:  {avg_steps:.1f}")


def main():
    print("Farm Robot")
    print("  1) Train a new model")
    print("  2) Run a test with the trained model")
    print(f"  3) Run {NUM_TRIALS} trials and show statistics")
    while True:
        choice = input("Choose 1, 2 or 3: ").strip()
        if choice == "1":
            train()
            break
        elif choice == "2":
            test()
            break
        elif choice == "3":
            evaluate()
            break
        print("Please enter 1, 2 or 3.")


if __name__ == "__main__":
    main()
import gymnasium as gym
import numpy as np

GRID_SIZE = 8
MAX_STEPS = 60
STARTING_ROBOT_POSITION = np.array([0, 0])
STARTING_COW_POSITION = np.array([4, 4])
STARTING_GOAL_POSITION = np.array([7, 7])
SEED = 10101
NUM_TRIALS = 100
COW_MOVE_PROB = 0.7  # chance the cow moves each step; otherwise it stays put

ACTION_NAMES = {0: "right", 1: "up", 2: "left", 3: "down", 4: "stay"}


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
        # Cow stays still 30% of the time, otherwise moves one cell in a
        # random direction (right/up/left/down). Uses the env's own RNG so
        # reset(seed=...) makes each trial reproducible.
        if self.np_random.random() >= COW_MOVE_PROB:
            return
        direction = int(self.np_random.integers(0, 4))
        self._cow_loc = np.clip(
            self._cow_loc + self._action_to_direction[direction], 0, self.size - 1
        )

    # ---------- baseline controller ----------
    def baseline_robot_mover(self):
        """Return an action index (0-4).

        Tries every action, scores the resulting robot position, picks the best:
          - closer to the goal is better
          - a cell the cow is on is heavily penalised
          - a cell adjacent to the cow is mildly penalised
        """
        best_action, best_score = 4, float("inf")

        for action, direction in self._action_to_direction.items():
            new_pos = np.clip(self._robot_loc + direction, 0, self.size - 1)

            goal_dist = np.max(np.abs(self._goal_loc - new_pos))
            cow_cheb = self.distance(new_pos, self._cow_loc)

            score = goal_dist
            if cow_cheb == 0:
                score += 20
            elif cow_cheb == 1:
                score += 3

            # Small tie-break: prefer actually moving over staying
            if action == 4:
                score += 0.5

            if score < best_score:
                best_action, best_score = action, score

        return best_action


def run_trial(env, seed):
    """Run one episode with the baseline controller.

    Returns (success, steps, collided), where collided is True if the robot
    landed on the cow at any point during the episode.
    """
    env.reset(seed=seed)
    collided = False

    while True:
        action = env.baseline_robot_mover()
        _, _, terminated, truncated, _ = env.step(action)
        if env.collision():
            collided = True
        if terminated or truncated:
            return terminated, env.steps, collided


if __name__ == "__main__":
    env = FarmRobotEnv()

    successes = 0
    collision_trials = 0
    success_steps = []

    for trial in range(NUM_TRIALS):
        success, steps, collided = run_trial(env, seed=SEED + trial)
        successes += success
        collision_trials += collided
        if success:
            success_steps.append(steps)

    avg_steps = np.mean(success_steps) if success_steps else float("nan")

    print(f"Success rate:        {100 * successes / NUM_TRIALS:.1f}%")
    print(f"Collision rate:      {100 * collision_trials / NUM_TRIALS:.1f}%")
    print(f"Avg. steps to goal:  {avg_steps:.1f}")
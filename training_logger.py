"""Writes training progress to a text file.

Every `interval` episodes it appends: episode number, that episode's reward,
and the average reward over the last `interval` episodes.
When training ends, `finish()` writes how many episodes were successful
(the robot reached the goal).
"""


class TrainingLogger:
    def __init__(self, path="training_log.txt", interval=20):
        self.path = path
        self.interval = interval
        self.rewards = []
        self.successes = 0
        with open(self.path, "w") as f:
            f.write("episode,reward,avg_reward_last_%d\n" % interval)

    def log_episode(self, episode, reward, success):
        """Call once per episode. `episode` is 1-based."""
        self.rewards.append(reward)
        if success:
            self.successes += 1

        if episode % self.interval == 0:
            recent = self.rewards[-self.interval:]
            avg = sum(recent) / len(recent)
            with open(self.path, "a") as f:
                f.write(f"{episode},{reward:.2f},{avg:.2f}\n")

    def finish(self):
        """Write the final success summary."""
        total = len(self.rewards)
        rate = 100 * self.successes / total if total else 0.0
        with open(self.path, "a") as f:
            f.write(f"\nSuccessful episodes: {self.successes} / {total} ({rate:.1f}%)\n")
        return self.successes, total

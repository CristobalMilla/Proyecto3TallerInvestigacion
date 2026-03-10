import numpy as np

class SimpleHandler:
    def __init__(self, num_persons, size, speed, change_chance=0.2):
        self.num_persons = num_persons
        self.size = size
        self.speed = speed
        self.change_chance = change_chance

        self.positions = np.random.uniform(-self.size, self.size, size=(num_persons, 2))
        self.directions = np.random.uniform(0, 2 * np.pi, size=num_persons)

        self.speed_factors = np.random.uniform(0.85, 1.10, size=num_persons)

        self.common_goal = self._new_common_goal()

    def _new_common_goal(self):

        margin = self.size * 0.15
        corners = np.array([
            [-self.size + margin, -self.size + margin],
            [-self.size + margin, self.size - margin],
            [self.size - margin, -self.size + margin],
            [self.size - margin, self.size - margin],
        ])
        idx = np.random.randint(0, len(corners))
        return corners[idx]


    def _normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def step(self, change_chance=None):
        chance = change_chance if change_chance is not None else self.change_chance

        for i in range(self.num_persons):

            dx = self.common_goal[0] - self.positions[i, 0]
            dy = self.common_goal[1] - self.positions[i, 1]

            angle_to_goal = np.arctan2(dy, dx)
            distance_to_goal = np.hypot(dx, dy)

            if np.random.random() < chance:
                self.directions[i] += np.random.uniform(-np.pi / 4, np.pi / 4)
            else:
                angle_diff = self._normalize_angle(angle_to_goal - self.directions[i])
                self.directions[i] += 0.18 * angle_diff + np.random.uniform(-np.pi / 20, np.pi / 20)

            step_speed = self.speed * self.speed_factors[i]

            #if distance_to_goal < self.size * 0.20:
            #    step_speed *= 0.75

            self.positions[i, 0] += step_speed * np.cos(self.directions[i])
            self.positions[i, 1] += step_speed * np.sin(self.directions[i])

            # Bounce logic updated for centered coordinates (-size to size)
            if self.positions[i, 0] <= -self.size or self.positions[i, 0] >= self.size:
                self.directions[i] = np.pi - self.directions[i]  # Reverse X
                self.positions[i, 0] = np.clip(self.positions[i, 0], -self.size, self.size)

            if self.positions[i, 1] <= -self.size or self.positions[i, 1] >= self.size:
                self.directions[i] = -self.directions[i]  # Reverse Y
                self.positions[i, 1] = np.clip(self.positions[i, 1], -self.size, self.size)

    def get_positions(self):
        return self.positions

    def get_common_goal(self):
        return self.common_goal.copy()
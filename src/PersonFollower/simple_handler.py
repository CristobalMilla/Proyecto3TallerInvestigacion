import numpy as np


class SimpleHandler:
    def __init__(self, num_persons, size, speed, change_chance=0.2, destination=None):
        self.num_persons = num_persons
        self.size = size
        self.speed = speed
        self.change_chance = change_chance

        # Persons (X, Y) - Spawned between -size and size
        self.positions = np.random.uniform(-self.size, self.size, size=(num_persons, 2))
        self.directions = np.random.uniform(0, 2 * np.pi, size=num_persons)

        # Destination always exists - set randomly if not provided
        if destination is not None:
            self.set_destination(destination)
        else:
            self._set_random_destination()

    def set_destination(self, destination):
        destination_array = np.asarray(destination, dtype=float)
        if destination_array.shape != (2,):
            raise ValueError("destination must be a 2D point: [x, y]")

        self.destination = np.clip(destination_array, -self.size, self.size)

    def _set_random_destination(self):
        self.destination = np.random.uniform(-self.size, self.size, size=2)

    def clear_destination(self):
        # Set a new random destination instead of None
        self._set_random_destination()

    def step(self, change_chance=None):
        # Use provided value or fallback to the class default
        chance = change_chance if change_chance is not None else self.change_chance

        for i in range(self.num_persons):
            # Calculate direction toward destination
            dx = self.destination[0] - self.positions[i, 0]
            dy = self.destination[1] - self.positions[i, 1]
            distance = np.hypot(dx, dy)

            if distance <= self.speed:
                self.positions[i] = self.destination.copy()
                continue

            # Set direction toward destination
            self.directions[i] = np.arctan2(dy, dx)
            
            # With probability 'chance', add random deviation
            if np.random.random() < chance:
                self.directions[i] += np.random.uniform(-np.pi / 4, np.pi / 4)

            self.positions[i, 0] += self.speed * np.cos(self.directions[i])
            self.positions[i, 1] += self.speed * np.sin(self.directions[i])

            # Bounce logic updated for centered coordinates (-size to size)
            if self.positions[i, 0] <= -self.size or self.positions[i, 0] >= self.size:
                self.directions[i] = np.pi - self.directions[i]  # Reverse X
                self.positions[i, 0] = np.clip(self.positions[i, 0], -self.size, self.size)

            if self.positions[i, 1] <= -self.size or self.positions[i, 1] >= self.size:
                self.directions[i] = -self.directions[i]  # Reverse Y
                self.positions[i, 1] = np.clip(self.positions[i, 1], -self.size, self.size)

    def get_positions(self):
        return self.positions
import numpy as np

class SimpleHandler:
    def __init__(self, num_persons, size, speed):
        self.num_persons = num_persons
        self.size = size
        self.speed = speed
    
        # Persons (X, Y)
        self.positions = np.random.uniform(0, self.size, size=(num_persons, 2))
        self.directions = np.random.uniform(0, 2 * np.pi, size=num_persons)

    def step(self, change_chance=0.2):
        for i in range(self.num_persons):
            if np.random.random() < change_chance:
                self.directions[i] += np.random.uniform(-np.pi/4, np.pi/4)

            self.positions[i, 0] += self.speed * np.cos(self.directions[i])
            self.positions[i, 1] += self.speed * np.sin(self.directions[i])

            # Bounce
            if self.positions[i, 0] <= 0 or self.positions[i, 0] >= self.size:
                self.directions[i] = np.pi - self.directions[i]
            if self.positions[i, 1] <= 0 or self.positions[i, 1] >= self.size:
                self.directions[i] = -self.directions[i]

    def get_positions(self):
        return self.positions
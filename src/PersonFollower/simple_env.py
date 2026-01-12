import numpy as np
import matplotlib.pyplot as plt
from PersonFollower.simple_handler import SimpleHandler

class SimpleEnvironment:
    def __init__(self, num_persons=5, size=50, person_speed=1.5, drone_speed=1.2):
        # Initialize the persons handler
        self.handler = SimpleHandler(num_persons, size, person_speed)
        
        self.size = size
        self.drone_speed = drone_speed
        # Drone starts in center
        self.drone_pos = np.array([size / 2.0, size / 2.0])
        
        # History for plotting
        self.person_hist = []
        self.drone_hist = []

    def run(self, steps=100):
        for _ in range(steps):
            # 1. Update persons
            self.handler.step()
            persons = self.handler.get_positions()
            
            # 2. Drone Logic: Calculate average (center of mass)
            avg_pos = np.mean(persons, axis=0)
            
            # 3. Move drone toward average
            vec = avg_pos - self.drone_pos
            dist = np.linalg.norm(vec)
            if dist > 0:
                self.drone_pos += (vec / dist) * self.drone_speed
            
            # Save state
            self.person_hist.append(persons.copy())
            self.drone_hist.append(self.drone_pos.copy())

    def plot_results(self):
        p_hist = np.array(self.person_hist)
        d_hist = np.array(self.drone_hist)
        
        plt.figure(figsize=(8, 8))
        
        # Plot each person's path
        num_p = p_hist.shape[1]
        for i in range(num_p):
            plt.plot(p_hist[:, i, 0], p_hist[:, i, 1], color='blue', alpha=0.2)
            
        # Plot drone path
        plt.plot(d_hist[:, 0], d_hist[:, 1], color='black', linewidth=2, label='Drone Path')
        plt.scatter(d_hist[0, 0], d_hist[0, 1], color='green', marker='X', s=100, label='Drone Start')
        plt.scatter(d_hist[-1, 0], d_hist[-1, 1], color='red', marker='X', s=100, label='Drone End')

        plt.xlim(0, self.size)
        plt.ylim(0, self.size)
        plt.title(f"Drone Tracking Average of {num_p} Persons")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    sim = SimpleEnvironment(num_persons=5, size=60)
    sim.run(steps=150)
    sim.plot_results()
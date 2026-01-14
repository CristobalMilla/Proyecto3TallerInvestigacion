import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from PersonFollower.simple_handler import SimpleHandler

class SimpleEnvironment(gym.Env):
    def __init__(self, num_persons=5, size=50, person_speed=1.5, drone_speed=1.2, change_chance=0.2, drone_spawn_mode="center"):
        super(SimpleEnvironment, self).__init__()
        self.num_persons = num_persons
        self.size = size
        self.drone_speed = drone_speed
        self.person_speed = person_speed
        self.change_chance = change_chance
        self.drone_spawn_mode = drone_spawn_mode

        # Define Action Space: Drone moves in 2D (Delta X, Delta Y)
        # We'll use a Box from -1 to 1, which we will scale by drone_speed
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Define Observation Space: Positions of all persons + drone position
        # Flat array: [drone_x, drone_y, p1_x, p1_y, p2_x, p2_y...]
        obs_shape = 2 + (num_persons * 2)
        self.observation_space = spaces.Box(low=0, high=size, shape=(obs_shape,), dtype=np.float32)

        self.handler = None
        self.drone_pos = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.handler = SimpleHandler(self.num_persons, self.size, self.person_speed, self.change_chance)
        
        if self.drone_spawn_mode == "random":
            self.drone_pos = np.random.uniform(0, self.size, size=2)
        else:
            self.drone_pos = np.array([self.size / 2.0, self.size / 2.0], dtype=np.float32)

        # Initialize history
        self.person_hist = []
        self.drone_hist = []
        self.distance_hist = [] # Track distances over time

        return self._get_obs(), {}

    def _get_obs(self):
        persons = self.handler.get_positions().flatten()
        return np.concatenate([self.drone_pos, persons]).astype(np.float32)

    def step(self, action):
        # 1. Move Persons
        self.handler.step()
        persons = self.handler.get_positions()

        # 2. Move Drone based on AI action
        self.drone_pos += action * self.drone_speed
        self.drone_pos = np.clip(self.drone_pos, 0, self.size)

        # Save state for plotting
        self.person_hist.append(persons.copy())
        self.drone_hist.append(self.drone_pos.copy())

        # 3. Calculate Reward (Negative distance to average position)
        avg_pos = np.mean(persons, axis=0)
        distance = np.linalg.norm(avg_pos - self.drone_pos)
        
        # Save distance for progress checking
        self.distance_hist.append(distance)
        
        reward = -distance 
        # 4. Check if done (simpler environments often use fixed step limits)
        terminated = False
        truncated = False # Handled by the SB3 wrapper usually

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        pass

    def plot_results(self):
        p_hist = np.array(self.person_hist)
        d_hist = np.array(self.drone_hist)
        dist_hist = np.array(self.distance_hist)
    
        # Create a figure with two subplots: one for map, one for distance
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
        # Plot 1: The Map (existing logic)
        num_p = p_hist.shape[1]
        for i in range(num_p):
            ax1.plot(p_hist[:, i, 0], p_hist[:, i, 1], color='blue', alpha=0.2)
        ax1.plot(d_hist[:, 0], d_hist[:, 1], color='black', linewidth=2, label='Drone Path')
        ax1.scatter(d_hist[0, 0], d_hist[0, 1], color='green', marker='X', s=100)
        ax1.set_xlim(0, self.size)
        ax1.set_ylim(0, self.size)
        ax1.set_title("Movement Map")
        ax1.legend()

        # Plot 2: Distance to Target over Time
        ax2.plot(dist_hist, color='purple', linewidth=2)
        ax2.set_title("Distance to Average Position Over Time")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Distance (Units)")
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        # Calculate and show average distance
        avg_dist = np.mean(dist_hist)
        ax2.axhline(avg_dist, color='red', linestyle='--', label=f'Avg: {avg_dist:.2f}')
        ax2.legend()

        plt.tight_layout()
        plt.show()
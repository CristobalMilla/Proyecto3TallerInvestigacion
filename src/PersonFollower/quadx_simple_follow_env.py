from __future__ import annotations
from typing import Any, Literal
import numpy as np
from gymnasium import spaces
from PyFlyt.gym_envs.quadx_envs.quadx_base_env import QuadXBaseEnv
from src.PersonFollower.simple_handler2 import SimpleHandler

class QuadXSimpleFollowEnv(QuadXBaseEnv):
    def __init__(
        self,
        num_persons: int = 3,
        size: float = 10.0,
        person_speed: float = 0.05,
        change_chance: float = 0.4, #prob de cambiar
        flight_mode: int = 0,
        max_duration_seconds: float = 10.0,
        agent_hz: int = 30,
        render_mode: None | Literal["human", "rgb_array"] = None,
        drone_spawn_mode: str = "center"
    ):
        # Initialize the base PyFlyt environment
        # start_pos is [x, y, z]
        super().__init__(
            start_pos=np.array([[0.0, 0.0, 1.5]]),
            flight_dome_size=size,
            max_duration_seconds=max_duration_seconds,
            agent_hz=agent_hz,
            render_mode=render_mode,
            flight_mode=flight_mode
        )

        self.drone_spawn_mode = drone_spawn_mode

        # Initialize your custom person handler
        # Note: PyFlyt uses coordinates centered at 0, so we adjust the logic slightly
        self.person_handler = SimpleHandler(
            num_persons=num_persons,
            size=size, # This is the radius/half-width
            speed=person_speed,
            change_chance=change_chance
        )

        self.prev_avg_person_pos = np.zeros(2, dtype=np.float64)
        self.avg_velocity_weight = 1.0

        # Observation space: Drone attitude + [relative average position] + [average velocity]
        # relative average position -> 3 values [dx, dy, dz]
        # average velocity         -> 3 values [vx, vy, vz]
        obs_shape = self.combined_space.shape[0] + 6
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64)

    def reset(self, *, seed: None | int = None, options: None | dict[str, Any] = dict()) -> tuple[np.ndarray, dict]:

        # Determine the start position for this episode
        if self.drone_spawn_mode == "random":
            # Random X and Y within the dome, fixed Z (altitude)
            random_xy = self.np_random.uniform(-self.flight_dome_size * 0.8, self.flight_dome_size * 0.8, size=(1, 2))
            start_pos = np.array([[random_xy[0, 0], random_xy[0, 1], 1.5]])
        else:
            start_pos = np.array([[0.0, 0.0, 1.5]])

        # Apply the new start position to the base class
        self.start_pos = start_pos

        # Reset PyFlyt physics
        super().begin_reset(seed, options)

        # Reset Person positions
        self.person_handler.positions = np.random.uniform(
            -self.flight_dome_size,
            self.flight_dome_size,
            size=(self.person_handler.num_persons, 2)
        )

        current_avg = np.mean(self.person_handler.get_positions(), axis=0)
        self.prev_avg_person_pos = current_avg.copy()

        # Initialize history for plotting
        self.person_hist = []
        self.drone_hist = []
        self.distance_hist = []

        super().end_reset()
        return self.state, self.info

    def compute_state(self) -> None:
        # 1. Update persons (pseudo-random movement)
        self.person_handler.step()
        persons_xy = self.person_handler.get_positions()

        # 2. Get Drone State from PyFlyt
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = super().compute_attitude()
        aux_state = super().compute_auxiliary()

        # Save history for plotting (CRITICAL: USE .copy())
        if hasattr(self, 'person_hist'):
            self.person_hist.append(persons_xy.copy())
            self.drone_hist.append(lin_pos[:2].copy())

        # 3. Compute average group position and velocity
        avg_person_pos = np.mean(persons_xy, axis=0)
        avg_person_vel = avg_person_pos - self.prev_avg_person_pos
        self.prev_avg_person_pos = avg_person_pos.copy()

        rel_avg_pos = np.array(
            [
                avg_person_pos[0] - lin_pos[0],
                avg_person_pos[1] - lin_pos[1],
                0.0 - lin_pos[2],
            ],
            dtype=np.float64,
        )

        weighted_avg_vel = np.array(
            [
                avg_person_vel[0] * self.avg_velocity_weight,
                avg_person_vel[1] * self.avg_velocity_weight,
                0.0,
            ],
            dtype=np.float64,
        )

        # 4. Combine into final observation
        # Choose attitude representation based on the base class setting
        if self.angle_representation == 0: # Euler
            attitude = np.concatenate([ang_vel, ang_pos, lin_vel, lin_pos, self.action, aux_state])
        else: # Quaternion
            attitude = np.concatenate([ang_vel, quaternion, lin_vel, lin_pos, self.action, aux_state])

        group_features = np.concatenate([rel_avg_pos, weighted_avg_vel])
        self.state = np.concatenate([attitude, group_features]).astype(np.float64)

    def compute_term_trunc_reward(self) -> None:
        # Check for collisions or out-of-bounds using PyFlyt base logic
        super().compute_base_term_trunc_reward()

        # Reward logic similar to simple_env2
        # Calculate distance to the average position of persons
        persons_xy = self.person_handler.get_positions()
        avg_person_pos = np.mean(persons_xy, axis=0)

        # Current drone position (from internal aviary)
        # We use lin_pos from state or get it directly
        drone_pos_3d = self.env.state(0)[3]
        drone_pos_2d = drone_pos_3d[:2]

        distance = np.linalg.norm(avg_person_pos - drone_pos_2d)
        self.distance_hist.append(distance)

        # Reward: 1 / (distance + 0.1)
        self.reward += 1.0 / (distance + 0.1)

        # Penalty for being too high or too low (keep drone at a reasonable altitude)
        # Drone starts at 1.5, let's keep it between 0.5 and 3.0
        z_pos = drone_pos_3d[2]
        if z_pos < 0.5 or z_pos > 3.0:
            self.reward -= 0.1

    def plot_results(self):
        import matplotlib.pyplot as plt
        p_hist = np.array(self.person_hist) # Shape: (steps, num_persons, 2)
        d_hist = np.array(self.drone_hist)  # Shape: (steps, 2)
        dist_hist = np.array(self.distance_hist)
        goal = self.person_handler.get_common_goal()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Overhead 2D Map
        # Iterate through EACH person (axis 1)
        num_steps = p_hist.shape[0]
        num_people = p_hist.shape[1]

        for i in range(num_people):
            # Extract all X and Y for person 'i' across all steps
            person_x = p_hist[:, i, 0]
            person_y = p_hist[:, i, 1]
            ax1.plot(person_x, person_y, color='blue', alpha=0.3, label=f'P{i}' if num_people < 5 else None)
            ax1.scatter(person_x[-1],person_y[-1], color='cyan', s=50, marker='o', label='Person End' if i == 0 else None)

        ax1.scatter(goal[0], goal[1], color='orange', marker='*', s=180, label='Common Goal')
        ax1.plot(d_hist[:, 0], d_hist[:, 1], color='black', linewidth=2, label='Drone Path')
        ax1.scatter(d_hist[0, 0], d_hist[0, 1], color='green', marker='X', s=100, label='Start')
        ax1.scatter(d_hist[-1, 0], d_hist[-1, 1], color='red', marker='X', s=100, label='End')

        limit = self.flight_dome_size
        ax1.set_xlim(-limit, limit)
        ax1.set_ylim(-limit, limit)
        ax1.set_title("Overhead Tracking Map (Physics Sim)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Distance to Average over Time
        ax2.plot(dist_hist, color='purple', linewidth=2)
        ax2.set_title("Distance to Average Person Position")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Distance")
        ax2.axhline(np.mean(dist_hist), color='red', linestyle='--', label=f'Avg: {np.mean(dist_hist):.2f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
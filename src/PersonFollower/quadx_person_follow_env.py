"""QuadX Waypoints Environment."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from gymnasium import spaces

from PyFlyt.gym_envs.quadx_envs.quadx_base_env import QuadXBaseEnv
from PersonFollower.person_handler import PersonHandler


class QuadXPersonFollowEnv(QuadXBaseEnv):
    """QuadX Person Following Environment.

    The drone must track multiple 'persons' moving pseudo-randomly on the ground.
    The drone's altitude is fixed to prevent vertical collisions as a 'solution'.
    """

    def __init__(
        self,
        # sparse_reward: bool = False,
        # use_yaw_targets: bool = False,
        # goal_reach_distance: float = 0.2,
        # goal_reach_angle: float = 0.1,
        num_persons: int = 4,
        flight_mode: int = 0,
        flight_dome_size: float = 5.0,
        max_duration_seconds: float = 10.0,
        angle_representation: Literal["euler", "quaternion"] = "quaternion",
        agent_hz: int = 30,
        render_mode: None | Literal["human", "rgb_array"] = None,
        render_resolution: tuple[int, int] = (480, 480),
    ):
        """__init__.

        Args:
            num_persons (int): number of persons in the environment.
            flight_mode (int): the flight mode of the UAV.
            flight_dome_size (float): size of the allowable flying area.
            max_duration_seconds (float): maximum simulation time of the environment.
            angle_representation (Literal["euler", "quaternion"]): can be "euler" or "quaternion".
            agent_hz (int): looprate of the agent to environment interaction.
            render_mode (None | Literal["human", "rgb_array"]): render_mode
            render_resolution (tuple[int, int]): render_resolution.

        """
        super().__init__(
            start_pos=np.array([[0.0, 0.0, 1.5]]), # Fixed starting height
            flight_mode=flight_mode,
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
        )

        # define persons
        self.persons = PersonHandler(
            num_persons=num_persons,
            flight_dome_size=flight_dome_size,
            np_random=self.np_random
        )

        # Define observation space
        # We use a Box instead of Sequence because the number of persons is fixed
        self.observation_space = spaces.Dict(
            {
                "attitude": self.combined_space,
                "target_deltas": spaces.Box(
                    low=-2 * flight_dome_size,
                    high=2 * flight_dome_size,
                    shape=(num_persons, 3),
                    dtype=np.float64,
                ),
            }
        )

    def reset(
        self, *, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> tuple[dict[Literal["attitude", "target_deltas"], np.ndarray], dict]:
        """Resets the environment.

        Args:
            seed: seed to pass to the base environment.
            options: None

        """
        super().begin_reset(seed, options)
        self.persons.reset(self.np_random)
        super().end_reset()

        return self.state, self.info

    def compute_state(self) -> None:
        """Computes the state of the current timestep.
        Persons move before the state is captured."""

        # 1. Update person positions
        self.persons.update()

        # 2. Capture drone physics
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = super().compute_attitude()
        aux_state = super().compute_auxiliary()

        # combine everything
        new_state = dict()
        if self.angle_representation == 0:
            new_state["attitude"] = np.concatenate(
                [
                    ang_vel,
                    ang_pos,
                    lin_vel,
                    lin_pos,
                    self.action,
                    aux_state,
                ],
                axis=-1,
            )
        else:
            new_state["attitude"] = np.concatenate(
                [
                    ang_vel,
                    quaternion,
                    lin_vel,
                    lin_pos,
                    self.action,
                    aux_state,
                ],
                axis=-1,
            )

        # 4. Get relative distances to the moving persons
        new_state["target_deltas"] = self.persons.get_relative_deltas(
            lin_pos
        )
        self.state = new_state

    def compute_term_trunc_reward(self) -> None:
        """Computes reward based on how well the drone is tracking the persons."""
        super().compute_base_term_trunc_reward()

        # Get distances to all persons from our target_deltas
        # deltas is shape (num_persons, 3)
        deltas = self.state["target_deltas"]
        distances = np.linalg.norm(deltas, axis=1)

        # Reward: 1 / (average_distance + small_constant)
        # This gives a high reward for being close and lower as it drifts away
        avg_dist = np.mean(distances)
        self.reward += 1.0 / (avg_dist + 0.1)

        # Penalty for high yaw rate to keep flight stable
        yaw_rate = abs(self.env.state(0)[0][2])
        self.reward -= 0.01 * yaw_rate ** 2

        # Optional: Truncate if everyone gets too far away (drone lost them)
        if np.all(distances > self.flight_dome_size * 0.5):
            self.reward -= 50.0
            self.truncation = True




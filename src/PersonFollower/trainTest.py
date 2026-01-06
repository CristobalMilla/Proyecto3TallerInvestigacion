import gymnasium as gym
import numpy as np
from PersonFollower.quadx_person_follow_env import QuadXPersonFollowEnv


def train():
    # Initialize our custom environment
    # We use render_mode="human" so you can see the drone (though persons are invisible)
    env = QuadXPersonFollowEnv(
        num_persons=2,
        render_mode="human",
        max_duration_seconds=10.0
    )

    print("Starting simulation test...")
    obs, info = env.reset()

    for step in range(1000):
        # Sample a random action (angular rates and thrust)
        action = env.action_space.sample()

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 30 == 0:
            avg_dist = np.mean(np.linalg.norm(obs["target_deltas"], axis=1))
            print(f"Step: {step} | Reward: {reward:.2f} | Avg Dist to Persons: {avg_dist:.2f}")

        if terminated or truncated:
            print("Episode finished. Resetting...")
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    train()
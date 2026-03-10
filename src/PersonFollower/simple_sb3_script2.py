import torch
import os
import numpy as np
from stable_baselines3 import PPO
from src.PersonFollower.quadx_simple_follow_env import QuadXSimpleFollowEnv

# 1. Optimize CPU usage
torch.set_num_threads(2)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"


def train():
    # Create the high-fidelity environment
    # flight_mode=1 uses position-based commands, which is easier to learn
    env = QuadXSimpleFollowEnv(
        num_persons=5,
        size=10.0,
        flight_mode=1,
        drone_spawn_mode="random",
        render_mode=None  # Set to "human" if you want to see the 3D drone
    )

    # 2. Initialize the Model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",
        learning_rate=0.0003
    )

    print("Starting training on QuadX Physics...")
    # Physics training takes longer; let's try 50,000 steps
    model.learn(total_timesteps=50000)

    # Save the model
    model.save("quadx_follower_model")
    print("Model saved.")

    # 3. Test the trained policy
    # We use a new env with "human" render if you want to watch the result
    test_env = QuadXSimpleFollowEnv(
        num_persons=5,
        size=10.0,
        flight_mode=1,
        drone_spawn_mode="random",
        render_mode=None
    )

    obs, _ = test_env.reset()
    print("Testing trained QuadX model for 200 steps...")
    print("Observation shape:", obs.shape)
    # [rel_avg_x, rel_avg_y, rel_avg_z, vel_x, vel_y, vel_z]
    print("Observation sample:", obs[-6:])
    for _ in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        if terminated or truncated:
            print("Episode ended early.")
            break

    print("Test finished. Opening plots...")
    test_env.plot_results()
    test_env.close()


if __name__ == "__main__":
    train()


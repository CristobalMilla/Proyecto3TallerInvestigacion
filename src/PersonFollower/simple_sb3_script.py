import torch
import os
import numpy as np
from stable_baselines3 import PPO
from PersonFollower.simple_env2 import SimpleEnvironment

# 1. Optimize CPU usage
torch.set_num_threads(2)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

def train():
    # Create the environment
    env = SimpleEnvironment(
        num_persons=3, 
        size=50, 
        drone_spawn_mode="random"
    )

    # 2. Initialize the Model (PPO is great for continuous control)
    # We force the device to 'cpu' as requested
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        device="cpu",
        learning_rate=0.0003
    )

    print("Starting training...")
    model.learn(total_timesteps=20000)

    # Save the model
    model.save("drone_follower_model")
    print("Model saved.")

    # 3. Quick Test
    obs, _ = env.reset()
    print("Testing trained model for 100 steps...")
    
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
    print("Test finished. Opening plot...")
    env.plot_results()

if __name__ == "__main__":
    train()


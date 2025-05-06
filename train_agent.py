import gymnasium as gym
import torch
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import CombinedExtractor
from splendor_env import SplendorEnv

# Register the custom environment if not already registered
try:
    gym.envs.registration.register(
        id="Splendor-v0",
        entry_point="splendor_env:SplendorEnv",
    )
except gym.error.Error:
    pass  # Already registered

# Create two models: one trainable, one fixed opponent
env = gym.make("Splendor-v0")

model_A = DQN("MultiInputPolicy", env, verbose=1)
model_B = DQN("MultiInputPolicy", env, verbose=0)

def self_play_train(model_A, model_B, total_episodes=100):
    for episode in range(total_episodes):
        print(f"\n[Episode {episode}] Starting...")
        obs, _ = env.reset()
        terminated, truncated = False, False
        current_player = 0  # 0 = A, 1 = B

        while not (terminated or truncated):
            if current_player == 0:
                action, _ = model_A.predict(obs, deterministic=False)
            else:
                action, _ = model_B.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            current_player = 1 - current_player

        print(f"[Episode {episode}] Finished. Starting learning...")
        model_A.learn(total_timesteps=1024, reset_num_timesteps=False)

        # Self-play model update
        if episode % 10 == 0:
            model_B.set_parameters(model_A.get_parameters())
            print(f"[Episode {episode}] Synced opponent model B with model A")

        # Evaluation
        if episode % 20 == 0:
            mean_reward, _ = evaluate_policy(model_A, env, n_eval_episodes=5)
            print(f"[Evaluation] Mean reward of model A: {mean_reward:.2f}")

    model_A.save("splendor_agent_A")
    model_B.save("splendor_agent_B")

# Run training loop
self_play_train(model_A, model_B, total_episodes=100)
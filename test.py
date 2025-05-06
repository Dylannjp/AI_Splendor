import gymnasium as gym
import torch
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import CombinedExtractor
from splendor_env import SplendorEnv
from stable_baselines3.common.env_util import make_vec_env

gym.envs.registration.register(
    id="Splendor-v0",
    entry_point="splendor_env:SplendorEnv",
)

env = gym.make("Splendor-v0")
model_A = DQN("MultiInputPolicy", env, verbose=1, tensorboard_log="./splendor_dqn_A/")
model_B = DQN("MultiInputPolicy", env, verbose=0)

def self_play_train(model_A, model_B, total_episodes=100):
    env = make_vec_env("Splendor-v0", n_envs=1)

    print("Env created")

    obs = env.reset()
    print("Env reset")

    for episode in range(total_episodes):
        print(f"Episode {episode} starting...")
        done, truncated = False, False
        while not (done or truncated):
            print("Predicting action...")
            action, _ = model_A.predict(obs, deterministic=False)
            print(f"Action predicted: {action}")
            obs, reward, done, info = env.step(action)
            print(f"Step done. Reward: {reward}, Done: {done}, Truncated: {truncated}")

        print(f"Episode {episode} done. Starting learning...")
        model_A.learn(total_timesteps=1000)
        print(f"Learning complete for episode {episode}")

self_play_train(model_A, model_B, total_episodes=100)
import gymnasium as gym
from stable_baselines3 import PPO
import splendor_env  # ensure env is registered

env = gym.make("Splendor-v0", render_mode="human")
model = PPO.load("splendor_ppo")

obs, _ = env.reset()
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
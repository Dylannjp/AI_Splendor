import gymnasium as gym
import numpy as np
from splendor_env import SplendorEnv
from stable_baselines3 import DQN


def train():
    # Create environment
    env = SplendorEnv()

    # Initialize two DQN agents: one for the player and one for the opponent
    agent = DQN("Multi", env, verbose=1, tensorboard_log="./dqn_tensorboard/")
    opponent = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./dqn_opponent_tensorboard/")

    total_episodes = 100000
    episode_rewards = []

    for episode in range(total_episodes):
        obs, _ = env.reset()  # Reset environment to start a new episode
        done = False
        episode_reward = 0
        while not done:
            # Player's turn (agent)
            action, _ = agent.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action, is_opponent=False)
            episode_reward += reward

            if terminated or truncated:
                done = True
                break

            # Opponent's turn
            action, _ = opponent.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action, is_opponent=True)
            episode_reward += reward

            if terminated or truncated:
                done = True
                break

        episode_rewards.append(episode_reward)
        if episode % 100 == 0:
            print(f"Episode {episode}/{total_episodes}, Reward: {episode_reward}")

    print("Training completed.")
    print(f"Average reward over episodes: {np.mean(episode_rewards)}")

if __name__ == "__main__":
    train()

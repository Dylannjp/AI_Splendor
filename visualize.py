import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(rewards, rolling_window=10, save_path=None):
    smoothed = np.convolve(rewards, np.ones(rolling_window)/rolling_window, mode='valid')
    plt.figure(figsize=(10, 5))
    plt.plot(smoothed)
    plt.title("Agent Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
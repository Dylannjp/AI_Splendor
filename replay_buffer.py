import random
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int):
        # capacity = maximum number of *games* (trajectories) to keep
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, trajectory):
        """
        trajectory: list of timesteps, each of the form
            (obs, root_policy, reward, to_play)
        where `root_policy` is the MCTS visit‐count distribution,
        `to_play` is which player’s turn it was, etc.
        """
        self.buffer.append(trajectory)

    def sample_batch(self, batch_size: int, unroll_steps: int):
        """
        Returns `batch_size` randomly sampled sub‐trajectories, each
        of length `unroll_steps+1`.  Each element is a dict of arrays:
            {
              "obs":    [B, unroll_steps+1, *obs_shape],
              "policy": [B, unroll_steps+1, action_space],
              "reward": [B, unroll_steps+1],
              "value":  [B, unroll_steps+1],   # if you have value targets
              "actions": [B, unroll_steps+1],
            }
        """
        # randomly pick full trajectories
        games = random.sample(self.buffer, k=batch_size)
        obs_keys = list(games[0][0][0].keys())  # assume all games have same obs keys
        batch = {"obs": {k: [] for k in obs_keys}, "policy": [], "reward": [], "actions": []}
        for game in games:
            # pick a random start within that game s.t. we can unroll `unroll_steps`
            start = random.randint(0, max(0, len(game) - (unroll_steps+1)))
            sub = game[start : start + (unroll_steps+1)]
            for k in obs_keys:
                # collect obs[k] over the subtrajectory
                batch["obs"][k].append([ step[0][k] for step in sub ])  # → list length T+1 of arrays
            batch["policy"].append([ step[1]   for step in sub ])  # [T+1, A]
            batch["reward"].append([ step[2]   for step in sub ])  # [T+1]
            batch["actions"].append([ step[3]   for step in sub ])  # [T+1]

        for k in obs_keys:
            arr = np.stack(batch["obs"][k], axis=0)  # [B, T+1, *shape_k]
            batch["obs"][k] = arr
        batch["policy"] = np.stack(batch["policy"], axis=0)  # [B, T+1, A]
        batch["reward"] = np.stack(batch["reward"], axis=0)  # [B, T+1]
        batch["actions"] = np.stack(batch["actions"], axis=0)  # [B, T+1]
        return batch

    def __len__(self):
        return len(self.buffer)

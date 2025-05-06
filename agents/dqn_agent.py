import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, obs_space, action_space, lr=1e-3, gamma=0.99):
        self.obs_space = obs_space
        self.action_space = action_space
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        flat_dim = sum(np.prod(space.shape) if hasattr(space, 'shape') else 1 for space in obs_space.values())
        self.model = DQNetwork(flat_dim, action_space.n).to(self.device)
        self.target_model = DQNetwork(flat_dim, action_space.n).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def act(self, obs, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_space.n - 1)
        obs_tensor = self._obs_to_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(obs_tensor)
        return int(torch.argmax(q_values, dim=1))

    def train_step(self, batch):
        obs, actions, rewards, next_obs, dones = batch
        obs = self._obs_to_tensor(obs)
        next_obs = self._obs_to_tensor(next_obs)

        q_vals = self.model(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_vals = self.target_model(next_obs).max(1)[0]
            targets = rewards + self.gamma * next_q_vals * (1 - dones)

        loss = nn.MSELoss()(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _obs_to_tensor(self, obs):
        flat = np.concatenate([np.array(val).flatten() for val in obs.values()])
        return torch.tensor(flat, dtype=torch.float32, device=self.device)
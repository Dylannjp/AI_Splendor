import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gymnasium import spaces

class MuZeroNet(nn.Module):
    def __init__(self, obs_space: spaces.Dict, action_space_size: int, hidden_size: int = 256):
        super().__init__()
        self.obs_keys  = sorted(obs_space.spaces.keys())
        self.obs_dim   = sum(int(np.prod(obs_space.spaces[k].shape)) for k in self.obs_keys)
        self.hidden_size      = hidden_size
        self.action_space_size = action_space_size
        self.device = torch.device("cuda")

        # Representation hθ
        self.repr_net = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        # Dynamics gθ
        self.dyn_fc = nn.Linear(hidden_size + action_space_size, hidden_size)
        self.dyn_r  = nn.Linear(hidden_size, 1)
        # Prediction fθ
        self.pred_fc = nn.Linear(hidden_size, hidden_size)
        self.pred_p  = nn.Linear(hidden_size, action_space_size)
        self.pred_v  = nn.Linear(hidden_size, 1)

    def _flatten_obs(self, obs_dict: dict):
        parts = []
        for k in self.obs_keys:
            data = obs_dict[k]
            if torch.is_tensor(data):
                t = data.detach().clone().to(self.device).float()
            else:
                t = torch.as_tensor(data, dtype=torch.float32, device=self.device)
            parts.append(t.reshape(t.shape[0], -1))
        # concat along the feature axis, giving [B, obs_dim]
        return torch.cat(parts, dim=1)

    def initial_state(self, obs_dict: dict):
        x = self._flatten_obs(obs_dict)
        return self.repr_net(x)

    def dynamics(self, prev_state: torch.Tensor, action_idx: torch.LongTensor):
        a_oh = F.one_hot(action_idx, self.action_space_size).float()
        x = torch.cat([prev_state, a_oh], dim=1)
        x = F.relu(self.dyn_fc(x))
        reward = self.dyn_r(x).squeeze(1)
        return x, reward

    def prediction(self, state: torch.Tensor):
        x = F.relu(self.pred_fc(state))
        policy = self.pred_p(x)
        value  = self.pred_v(x).squeeze(1)
        return policy, value

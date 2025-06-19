import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gymnasium import spaces

class MuZeroNet(nn.Module):
    def __init__(self, obs_space: spaces.Dict, action_space_size: int, hidden_size: int = 256):
        """
        obs_space: a gym.spaces.Dict of arrays; we'll record the total flatten dimension.
        action_space_size: number of possible actions (67 for Splendor).
        """
        super().__init__()
        # figure out flat obs dimension
        # assume every entry is a Box or MultiDiscrete that maps to an integer array
        self.obs_keys = sorted(obs_space.spaces.keys())
        self.obs_dim = sum(int(np.prod(obs_space.spaces[k].shape)) for k in self.obs_keys)

        self.action_space_size = action_space_size
        self.hidden_size = hidden_size

        # Representation network h_θ: obs → hidden state
        self.repr_net = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Dynamics network g_θ: (prev_state, action_onehot) → (next_state, reward)
        self.dyn_fc = nn.Linear(hidden_size + action_space_size, hidden_size)
        self.dyn_r  = nn.Linear(hidden_size, 1)

        # Prediction network f_θ: state → (policy_logits, value)
        self.pred_fc = nn.Linear(hidden_size, hidden_size)
        self.pred_p  = nn.Linear(hidden_size, action_space_size)
        self.pred_v  = nn.Linear(hidden_size, 1)

        # device will be set externally, e.g. net.to(device); we track for convenience
        self.device = torch.device("cpu")

    def to(self, *args, **kwargs):
        """Override so we can capture the device."""
        module = super().to(*args, **kwargs)
        # assume first arg is device or dtype
        if isinstance(args[0], torch.device):
            self.device = args[0]
        return module

    def _flatten_obs(self, obs_dict: dict) -> torch.Tensor:
        # obs_dict: keys → numpy arrays of shape (B, *shape)
        parts = []
        for k in self.obs_keys:
            arr = obs_dict[k]
            t = torch.as_tensor(arr, device=self.device)
            t = t.reshape(t.shape[0], -1).float()
            parts.append(t)
        return torch.cat(parts, dim=1)  # [B, self.obs_dim]

    def initial_state(self, obs_dict: dict) -> torch.Tensor:
        """
        obs_dict: a batch of env observations, each a dict field→array of shape (B,...).
        Returns: hidden_state [B, hidden_size]
        """
        x = self._flatten_obs(obs_dict)
        return self.repr_net(x)

    def dynamics(self, prev_state: torch.Tensor, action_idx: torch.LongTensor):
        """
        prev_state: [B, hidden_size]
        action_idx: [B] integers in [0, action_space_size)
        Returns:
          next_state: [B, hidden_size]
          reward:     [B]
        """
        # one-hot encode
        a_oh = F.one_hot(action_idx, num_classes=self.action_space_size).float()
        x = torch.cat([prev_state, a_oh.to(self.device)], dim=1)
        x = F.relu(self.dyn_fc(x))
        reward = self.dyn_r(x).squeeze(1)
        return x, reward

    def prediction(self, state: torch.Tensor):
        """
        state: [B, hidden_size]
        Returns:
          policy_logits: [B, action_space_size]
          value:         [B]
        """
        x = F.relu(self.pred_fc(state))
        policy = self.pred_p(x)
        value  = self.pred_v(x).squeeze(1)
        return policy, value

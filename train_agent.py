import time
import gymnasium as gym
import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

import pettingzoo.utils
from splendor_env import SplendorEnv  

class SB3ActionMaskWrapper(pettingzoo.utils.BaseWrapper, gym.Env):
    """Bridges your multi‐agent SplendorEnv → single‐agent Gym for SB3 + masking."""
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.observation_space = super().observation_space(self.possible_agents[0])["observation"]
        self.action_space      = super().action_space(self.possible_agents[0])
        obs_dict = super().observe(self.agent_selection)
        return obs_dict["observation"], {}

    def step(self, action):
        cur = self.agent_selection
        super().step(action)
        nxt = self.agent_selection
        obs_dict = super().observe(nxt)
        return (
            obs_dict["observation"],
            self._cumulative_rewards[cur],
            self.terminations[cur],
            self.truncations[cur],
            self.infos[cur],
        )

    def observe(self, agent):
        return super().observe(agent)["observation"]

    def action_mask(self):
        return super().observe(self.agent_selection)["action_mask"]


def mask_fn(env: SB3ActionMaskWrapper) -> np.ndarray:
    return env.action_mask()


def make_env(seed: int = 0):
    """Factory to build and reset the wrapped env so SB3 sees correct spaces."""
    # 1) Create & wrap the AEC env
    env = SB3ActionMaskWrapper(SplendorEnv(render_mode=None))
    # 2) Reset to initialize spaces
    env.reset(seed=seed)
    # 3) Wrap for action masking
    env = ActionMasker(env, mask_fn)
    return env


def train_splendor(steps: int = 50_000, seed: int = 0, save_path: str = "splendor_model"):
    # 1) Vectorize the mask‐wrapped Gym env
    venv = DummyVecEnv([lambda: make_env(seed)])
    venv.reset()

    # 2) Train with MaskablePPO
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        venv,
        verbose=1,
        seed=seed,
        tensorboard_log="./logs/splendor",
    )
    model.learn(total_timesteps=steps)

    # 3) Save
    ts = time.strftime("%Y%m%d-%H%M%S")
    model.save(f"{save_path}_{ts}")

    venv.close()


if __name__ == "__main__":
    train_splendor(steps=50_000)

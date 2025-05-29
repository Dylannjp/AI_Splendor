# play.py

from sb3_contrib import MaskablePPO
from splendor_env import SplendorEnv
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
import pettingzoo.utils
import gymnasium as gym
import numpy as np


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

    def action_masks(self):
        return super().observe(self.agent_selection)["action_mask"]


def mask_fn(env: SB3ActionMaskWrapper) -> np.ndarray:
    return env.action_masks()


def make_env(seed: int = 0):
    """Factory to build and reset the wrapped env so SB3 sees correct spaces."""
    # 1) Create & wrap the AEC env
    env = SB3ActionMaskWrapper(SplendorEnv(render_mode=None))
    # 2) Reset to initialize spaces
    env.reset(seed=seed)
    # 3) Wrap for action masking
    env = ActionMasker(env, mask_fn)
    return env


def main():
    model = MaskablePPO.load("splendor_model_20250528-200939.zip")
    venv = DummyVecEnv([lambda: make_env()])
    venv.reset()
    obs = venv.reset()    # obs_dict is a dict

    while True:
        # grab the *inner* env so you can query its mask() method
        mask = venv.envs[0].action_masks()       # shape: (n_actions,)

        # pass the array‐obs and mask to predict()
        action, _ = model.predict(obs,
                                  action_masks=mask,
                                  deterministic=True)
        # action is an array of shape (1,) → e.g. [17]

        # step the VecEnv (always arrays/batches)
        obs, rewards, dones, infos = venv.step(action)

        # render the inner env
        venv.envs[0].render()

        # if episode ended, reset
        if dones[0]:
            obs = venv.reset()

if __name__ == "__main__":
    main()

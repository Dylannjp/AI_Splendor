# train_and_eval_splendor.py

import re
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env

from splendor_env import SplendorEnv

if __name__ == "__main__":
    # 1) Start Ray.
    ray.init()

    # 2) Register our PettingZoo wrapped Splendor game.
    register_env(
        "splendor_pz",
        lambda cfg: PettingZooEnv(SplendorEnv(**cfg)),
    )

    # 3) Build the PPOConfig exactly like in the docs snippet:
    config = (
        PPOConfig()
        # point at our env
        .environment("splendor_pz", env_config={"num_players": 2})
        # flatten the nested, multi-agent dict into a vector
        .env_runners(
            env_to_module_connector=lambda env, *args, **kwargs:
                FlattenObservations(multi_agent=True),
            sample_timeout_s=1000000.0
        )
        # two policies p0/p1
        .multi_agent(
            policies={"p0", "p1"},
            policy_mapping_fn=lambda agent_id, ep: agent_id
        )
        # RL module (no LSTM, MLP heads)
        .rl_module(
            model_config=DefaultModelConfig(
                use_lstm=False,
                fcnet_hiddens=[256, 256],
                vf_share_layers=True,
            ),
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "p0": RLModuleSpec(),
                    "p1": RLModuleSpec(),
                }
            ),
        )
        # training hyperparams
        .training(
            lr=1e-4,
            train_batch_size=2000,
            vf_loss_coeff=0.005,
        )
        # evaluation setup
        .evaluation(
            evaluation_num_env_runners=1,
            evaluation_config={"explore": False},
            evaluation_interval=1,
            evaluation_duration_unit="episodes",
        )
    )

    # 4) Build the algorithm object
    algo = config.build_algo()

    # 5) Train for 20 iterations (or however many you like).
    for i in range(20):
        result = algo.train()
        print(f"== iteration {i} ==", 
              "reward_mean:", result["episode_reward_mean"],
              "timesteps_total:", result["timesteps_total"])

    # 6) Run evaluation
    eval_results = algo.evaluate()
    print(">>> Evaluation:", eval_results["evaluation"]["episode_reward_mean"])

    # 7) Save one final checkpoint
    ckpt_path = algo.save("my_splendor_checkpoints/")
    print("Last checkpoint saved to", ckpt_path)

    # 8) Clean up
    algo.stop()
    ray.shutdown()

import re
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from splendor_env import SplendorEnv

if __name__ == "__main__":
    ray.init()

    register_env("splendor_pz", lambda cfg: PettingZooEnv(SplendorEnv(**cfg)))

    # Build a PPOConfig just like your training script did:
    config = (
        PPOConfig()
        .environment("splendor_pz", env_config={"num_players": 2})
        .env_runners(
            env_to_module_connector=lambda env, *a, **k:
                FlattenObservations(multi_agent=True)
        )
        .multi_agent(
            policies={"p0", "p1"},
            policy_mapping_fn=lambda agent_id, _: re.sub(r"^player_", "p", agent_id),
        )
        .rl_module(
            model_config=DefaultModelConfig(use_lstm=False,
                                            fcnet_hiddens=[256, 256],
                                            vf_share_layers=True),
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={"p0": RLModuleSpec(), "p1": RLModuleSpec()}
            ),
        )
        .training(lr=1e-4, train_batch_size=2000, vf_loss_coeff=0.005)
        .framework("torch")
    )

    algo = config.build()
    algo.restore("/home/dylan/ray_results/PPO_2025-05-24_08-44-27/PPO_splendor_pz_a0d66_00000_0_2025-05-24_08-44-27/checkpoint_000000")

    # now run a proper evaluation
    results = algo.evaluate(
        evaluation_config={"explore": False},
        num_episodes=100
    )
    print("Mean reward over 100 eval games:", results["evaluation"]["episode_reward_mean"])    
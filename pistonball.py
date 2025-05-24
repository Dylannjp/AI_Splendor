import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.tune.registry import register_env
from pettingzoo.butterfly import pistonball_v6

if __name__ == "__main__":
    ray.init()

    register_env(
        "pistonball",
        lambda cfg: PettingZooEnv(pistonball_v6.env(n_pistons=20)),
    )

    config = (
        PPOConfig()
        .environment("pistonball")
        .env_runners(env_to_module_connector=FlattenObservations(multi_agent=True))
    )

    algo = config.build_algo()
    for _ in range(50):
        print(algo.train())
        
    algo.stop()

from ray.rllib.algorithms.ppo import PPOTrainer
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from pettingzoo.utils.wrappers import OrderEnforcingWrapper
from your_module import SplendorEnv
import numpy as np

# Rebuild and restore your trainer
trainer = PPOTrainer(config = (PPOConfig()
    .environment(env="splendor_pz")
    .framework("torch")
    .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    .resources(num_gpus=0)
    .env_runners(num_env_runners=2)
))
trainer.restore("ray_results/PPO_splendor_pz_2025-05-26_10-34-14p5fa_x02")
policy = trainer.get_policy()

# Wrap and reset
env = OrderEnforcingWrapper(SplendorEnv())

def evaluate_and_trace(n_episodes=5):
    for ep in range(1, n_episodes+1):
        print(f"\n=== Episode {ep} ===")
        obs = env.reset()
        done = {a: False for a in env.agents}
        step = 0

        while not all(done.values()):
            agent = env.agent_selection
            o, r, term, trunc, info = env.last()

            # 1) Compute action deterministically
            action, _, _ = policy.compute_single_action(o, explore=False)

            # 2) Log what you see and what you do
            print(f"Step {step:03d} | {agent} observes: {o['observation'][:10]}…")  # slice to keep it short
            print(f"         legal mask sum={o['action_mask'].sum()} → takes action {action}")

            # 3) (Optional) Render a human‐readable board
            env.render()

            # 4) Step
            env.step(action)

            # 5) Track reward & done
            print(f"         reward = {r:.3f}, done = {term or trunc}\n")
            done[agent] = term or trunc
            step += 1

        # Episode summary
        print(f"Episode {ep} finished in {step} steps.\n")

evaluate_and_trace(3)
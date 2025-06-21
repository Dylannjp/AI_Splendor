import copy
import torch
import numpy as np
from splendor_env import SplendorGymEnv
from muzero_net import MuZeroNet
from mcts import MCTS, MCTSConfig

def simulate_self_play(net, env, config: MCTSConfig, render: bool = False):
    """
    Runs one complete game where player 0 is controlled by MuZero+MCTS
    and player 1 uses the env’s default opponent policy.
    Logs each move with the acting player and the action taken.
    Returns the full trajectory of (obs, policy, reward, action, player).
    """
    obs, info = env.reset()
    legal_mask = info["legal_mask"]
    trajectory = []
    done = False

    # initial hidden state [1, H]
    obs_tensor = {
        k: torch.tensor(obs[k], dtype=torch.float32, device=net.device).unsqueeze(0)
        for k in net.obs_keys
    }
    hidden = net.initial_state(obs_tensor)

    turn = 0  # 0 = our net, 1 = opponent
    if render:
        print("=== Starting self-play ===")

    while not done:
        # Our turn: use MCTS + policy
        if turn == 0:
            # build MCTS from current hidden + mask
            mcts = MCTS(net, config, all_actions=env.all_actions)
            root = mcts.run(hidden, env.game)

            # extract policy = normalized visit counts
            visits = np.zeros(net.action_space_size, dtype=np.float32)
            for a, child in root.children.items():
                visits[a] = child.visit_count
            policy = visits / (visits.sum() + 1e-8)

            # pick best action
            a = int(visits.argmax())

            if render:
                env.render()
                print(f"[Player 0] → Action {a}: {env.all_actions[a]}")
                print(f"            (MCTS π[{a}]={policy[a]:.3f})\n")

            # step the environment
            next_obs, reward, done, _, info = env.step(a)

            trajectory.append((obs, policy, reward, a, 0))

            # advance the dynamics hidden state
            a_tensor = torch.tensor([a], device=net.device, dtype=torch.long)
            hidden, _ = net.dynamics(hidden, a_tensor)

            obs = next_obs
            legal_mask = info["legal_mask"]
            turn = 1  # now opponent

        # Opponent turn(s): env.step already runs all opponent moves internal to SplendorGymEnv,
        # but we can at least log that the opponent has acted (though we don't know exactly which).
        else:
            # Since the env.step call after our move already consumed all opponent moves,
            # we just note that control returns to us.
            if render:
                print("[Player 1] → (opponent moved — see board above)\n")
            turn = 0  # back to us

    # final board & result
    if render:
        env.render()
        print("=== Game over! reward =", reward)
    return trajectory

def main():
    env = SplendorGymEnv()
    net = MuZeroNet(env.observation_space, env.action_space.n)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # try loading a trained checkpoint
    try:
        net.load_state_dict(torch.load("muzero_splendor_final.pth", map_location=device))
        print("Loaded muzero_splendor_final.pth")
    except FileNotFoundError:
        print("No checkpoint found; running with untrained network.")

    # evaluation MCTS config (no exploration noise)
    eval_config = MCTSConfig(
        num_simulations=50,
        cpuct=1.0,
        discount=0.99,
        dirichlet_alpha=0.0,
        exploration_frac=0.0
    )

    # run & render one self-play game
    traj = simulate_self_play(net, env, eval_config, render=True)
    print(f"\nTrajectory length: {len(traj)}, final reward = {traj[-1][2]}")

if __name__ == "__main__":
    main()

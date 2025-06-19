import numpy as np
import torch
from torch import optim
from mcts import MCTS, MCTSConfig
from replay_buffer import ReplayBuffer
from muzero_net import MuZeroNet
import torch.nn.functional as F
from collections import namedtuple
import copy
from tqdm import trange


# Hyperparameters
Config = namedtuple("MCTSConfig", ["num_simulations", "cpuct", "discount", "dirichlet_alpha", "exploration_frac"])
mcts_config = MCTSConfig(num_simulations=125, cpuct=1.0, discount=0.99, dirichlet_alpha=0.15, exploration_frac=0.25)

def train_muzero(
    env,                # SplendorGymEnv instance
    net: MuZeroNet,     # your MuZero network
    buffer: ReplayBuffer,
    config=mcts_config,
    epochs=5000,       # total games to generate
    batch_size=32,
    unroll_steps=5,
    lr=1e-3,
    device="cuda"
):
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for game_idx in trange(epochs):
        # ==== (A) Self‐play to collect one full game trajectory ====

        root = None
        hidden = None 

        obs, info = env.reset()
        legal_mask = info["legal_mask"]  # [A] boolean mask of legal actions
        trajectory = []
        done = False

        # initial hidden state
        # flatten obs into tensor dict
        obs_tensor = {
            k: torch.tensor(obs[k][None], device=device, dtype=torch.float32)
            for k in net.obs_keys
        }
        hidden = net.initial_state(obs_tensor)  # now [1, H]


        while not done:
            # 1) Run MCTS on the current hidden state
            mcts = MCTS(net, config, all_actions=env.all_actions, root_game=copy.deepcopy(env.game))

            if root is not None:
                pass
            root = mcts.run(hidden, legal_mask)             # root node, children have visit counts

            visits = np.array([root.children[a].visit_count \
                            if a in root.children else 0
                            for a in range(net.action_space_size)], dtype=np.float32)
            visits /= visits.sum() if visits.sum() > 0 else 1.0
            a = np.argmax(visits)

            # 4) Step the environment
            next_obs, reward, done, truncated, info = env.step(a)
            trajectory.append((obs, visits, reward, a))

            root = root.children[a]  # move to the child node we just selected
            hidden = root.hidden_state  # use the hidden state from the MCTS root

            legal_mask = info["legal_mask"]  # update legal actions mask
            obs = next_obs

        # end of game
        buffer.add(trajectory)

        # ==== (B) Once buffer is “warm”, do gradient updates ====
        if len(buffer) >= batch_size:
            batch = buffer.sample_batch(batch_size, unroll_steps)
            # build torch targets and compute loss
            loss = muzero_loss(net, batch, config, device=device)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            optimizer.step()

        if (game_idx + 1) % 100 == 0:
            print(f"Game {game_idx+1}: buffer size={len(buffer)}, last loss={loss.item():.4f}")
        
        if (game_idx+1) % 500 == 0:
            val = evaluate(net, env, n_games=50)
            print(f"--> Eval win rate vs random: {val:.2f}")


    # final save
    torch.save(net.state_dict(), "muzero_splendor.pth")
    return net

def build_training_data(net, batch, config, device):
    """
    batch["obs"]: dict k→np.array of shape [B, T+1, *shape_k]
    batch["policy"]: [B, T+1, A]
    batch["reward"]: [B, T+1]
    batch["actions"]: [B, T+1]
    """
    obs_dict   = batch["obs"]
    pi_np      = batch["policy"]
    rewards_np = batch["reward"]
    actions_np = batch["actions"]
    B, T1, A   = pi_np.shape
    gamma      = config.discount


    # 1) initial obs at t=0
    obs0 = {
        k: torch.tensor(obs_dict[k][:,0], dtype=torch.float32, device=device)
        for k in net.obs_keys
    }
    # 2) compute s0
    with torch.no_grad():
        state0 = net.initial_state(obs0)  # [B, H]

    # 3) targets
    policy_target = torch.tensor(pi_np[:, :-1, :], dtype=torch.float32, device=device)  # [B, T, A]
    reward_target = torch.tensor(rewards_np[:,1:], dtype=torch.float32, device=device)   # [B, T]
    action_target = torch.tensor(actions_np[:,1:], dtype=torch.long,    device=device)   # [B, T]

    # 4) n‐step returns
    value_target = []
    for b in range(B):
        vt = []
        for t in range(T1-1):
            discounts = gamma**np.arange(0, T1-1-t)
            G = np.dot(discounts, rewards_np[b, t+1:])
            vt.append(G)
        value_target.append(vt)
    value_target = torch.tensor(value_target, dtype=torch.float32, device=device)  # [B, T]

    return policy_target, reward_target, value_target, action_target

def muzero_loss(net: MuZeroNet, batch: dict, config, device="cuda"):
    policy_target, reward_target, value_target, action_target = build_training_data(net, batch, config, device)
    B, T, A = policy_target.shape

    # initial hidden state
    obs0 = {
        k: torch.tensor(batch["obs"][k][:,0], device=device).float()
        for k in net.obs_keys
    }
    state = net.initial_state(obs0)  # [B, H]

    total_policy_loss = 0.0
    total_value_loss  = 0.0
    total_reward_loss = 0.0
    r_pred = None

    for t in range(T):
        # 1) prediction at s_t
        pi_logits, v_pred = net.prediction(state)            # [B, A], [B]
        total_policy_loss += -torch.sum(
            policy_target[:,t] * F.log_softmax(pi_logits, dim=1),
            dim=1
        ).mean()
        total_value_loss  += F.mse_loss(v_pred, value_target[:,t])

        # 2) reward from previous dynamics
        if t > 0:
            total_reward_loss += F.mse_loss(r_pred, reward_target[:,t-1])

        # 3) unroll one step via dynamics
        a_t = action_target[:,t]
        state, r_pred = net.dynamics(state, a_t)

    # average
    loss = (total_policy_loss + total_value_loss + total_reward_loss) / (T+1)
    return loss

def evaluate(net, env, n_games=50):
    wins = 0
    for _ in range(n_games):
        obs, info = env.reset()
        done = False
        hidden = net.initial_state({k: torch.tensor(obs[k][None]).float() for k in net.obs_keys})
        mask = info["legal_mask"]
        while not done:
            # simple MCTS or greedy policy
            mcts = MCTS(net, mcts_config, env.all_actions, copy.deepcopy(env.game))
            root = mcts.run(hidden, mask)
            a = max(root.children, key=lambda a: root.children[a].visit_count)
            obs, reward, done, _, info = env.step(a)
            mask = info["legal_mask"]
            action_idx = torch.tensor([a], device=hidden.device, dtype=torch.long)  # shape (1,)
            hidden, _ = net.dynamics(hidden, action_idx)
        if reward > 0:
            wins += 1
    print(wins)
    return wins / n_games
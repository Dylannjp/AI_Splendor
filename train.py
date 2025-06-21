import torch
from splendor_env import SplendorGymEnv
from muzero_net import MuZeroNet
from replay_buffer import ReplayBuffer
from train_muzero_parallel import train_muzero

def main():
    # 1) Create environment
    env = SplendorGymEnv()

    # 2) Build your MuZero network
    #    obs_space is a gym.spaces.Dict, action_space.n == 67
    net = MuZeroNet(env.observation_space, env.action_space.n)

    # 3) Create replay buffer (e.g. capacity for 1,000 games)
    buffer = ReplayBuffer(capacity=3)

    # 4) Optionally move to GPU if available
    device = torch.device("cuda")
    print(f"Training on {device}")

    # 5) Train
    trained_net = train_muzero(
        env,
        net=net,
        buffer=buffer,
        epochs=5,        # number of self‐play games
        batch_size=2,
        unroll_steps=5,
        lr=1e-3,
        device=device
    )

    # 6) Save final model (also saved inside train_muzero)
    torch.save(trained_net.state_dict(), "muzero_splendor_final.pth")
    print("Training complete. Model saved to muzero_splendor_final.pth")

if __name__ == "__main__":
    main()

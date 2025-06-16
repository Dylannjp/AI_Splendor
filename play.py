import numpy as np
from sb3_contrib import MaskablePPO
from splendor_env import SplendorEnv, ActionType

def print_legal_moves(env):
    """
    Print all currently legal moves (as “index : (ActionType, param)”).
    """
    agent = env.agent_selection
    obs_dict = env.observe(agent)
    mask = obs_dict["action_mask"]
    valid_indices = np.nonzero(mask)[0]

    print(f"\n→ It’s {agent}’s turn.  Legal moves:\n")
    for idx in valid_indices:
        print(f"   {idx:2d} : {env.all_actions[idx]}")
    return valid_indices

def human_turn(env):
    """
    Prompt the human to pick a legal index, re‐prompt until valid.
    Returns the chosen integer index.
    """
    valid = print_legal_moves(env)
    while True:
        choice = input("Enter move‐index: ").strip()
        if not choice.isdigit():
            print("  ✗ not a number, try again.")
            continue
        a = int(choice)
        if a in valid:
            return a
        print("  ✗ illegal index; pick one from:", valid[:10], "…")

def bot_turn(env, model):
    """
    Let the MaskablePPO model pick an action using action_mask.
    Returns the chosen integer index.
    """
    agent = env.agent_selection
    obs_dict = env.observe(agent)
    obs_vec = obs_dict["observation"]
    mask = obs_dict["action_mask"]
    action, _state = model.predict(obs_vec, action_masks=mask, deterministic=True)
    return int(action)

def main():
    # 1) Load your trained MaskablePPO checkpoint
    model = MaskablePPO.load("splendor_model_20250531-062222.zip")

    # 2) Create a fresh SplendorEnv (PettingZoo AEC)
    env = SplendorEnv(render_mode=None)
    obs, info = env.reset()  # Initialize

    # 3) Main loop: alternate between “player_0” (human) and “player_1” (bot)
    while True:
        env.render()
        current = env.agent_selection

        if current == "player_0":
            # --- Human’s turn ---
            chosen_idx = human_turn(env)

        else:
            # --- Bot’s turn ---
            chosen_idx = bot_turn(env, model)
            print(f"\n🤖 Bot ({current}) plays → {env.all_actions[chosen_idx]}")

        # 4) Perform step() (no unpacking!)
        env.step(chosen_idx)

        # 5) Immediately check for termination/truncation
        #    In AEC env, rewards sit in env.rewards[agent], terminations in env.terminations[…] or env.truncations[…]
        #    If any agent is terminated/truncated, the game is over by design.
        done = any(env.terminations.values()) or any(env.truncations.values())
        if done:
            # Identify winner
            # We gave +1 to winner, –1 to loser in SplendorEnv.step(…)
            r0 = env.rewards["player_0"]
            r1 = env.rewards["player_1"]
            if r0 > r1:
                print("\n🏆 You win!  (Player 0 got +1, Player 1 got –1)")
            else:
                print("\n💀 Bot wins!  (Player 1 got +1, Player 0 got –1)")
            break

        # 6) Otherwise, continue: env.agent_selection has already advanced to next
        #    (PettingZoo AEC’s agent_selector takes care of cycling)
        #    Now loop back and let the next agent act.

    env.close()


if __name__ == "__main__":
    main()

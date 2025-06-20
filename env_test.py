import numpy as np
from splendor_env import SplendorGymEnv, ActionType, normalize_card_key
import copy
import random
        
"""
def simulate_and_debug(env, n_episodes=1000, max_steps=100):
    stuck_details = []
    for ep in range(1, n_episodes+1):
        obs, info = env.reset()
        traj = []  # will hold one dict per step
        done = False
        for step in range(max_steps):
            player = env.game.current_player
            legal = env.game.legal_actions(player)
            legal_idxs = [i for i,a in enumerate(env.all_actions) if a in legal]
            if not legal_idxs:
                raise RuntimeError(f"No legal actions at step {step} (player {player})")
            choice = random.choice(legal_idxs)

            # record *before* applying the action
            traj.append({
                'step': step,
                'player': player,
                'action_idx': choice,
                'action': env.all_actions[choice],
                'legal_mask': info['legal_mask'].copy(),
                'obs': copy.deepcopy(obs),
            })

            obs, reward, done, trunc, info = env.step(choice)
            if done:
                break

        if not done:
            stuck_details.append({
                'episode': ep,
                'length': step+1,
                'traj': traj
            })
        if ep % 100 == 0:
            print(f"Completed {ep}/{n_episodes} episodes")
    return stuck_details

def debug_stuck(env, traj):
    print("=== DEBUG STUCK EPISODE ===")
    for rec in traj:
        t      = rec['step']
        player = rec['player']
        idx    = rec['action_idx']
        action = rec['action']
        obs    = rec['obs']
        mask   = rec['legal_mask']

        print(f" Step {t:4d}, player={player}, action_idx={idx}, action={action}")
        print(f"    board_gems: {obs['board_gems']}")
        print(f"    my_gems:    {obs['my_gems']}")
        print(f"    opp_gems:   {obs['opp_gems']}")
        print(f"    tier1 nonzeros: {np.nonzero(obs['tier1cards'])[0].tolist()}")
        print(f"    tier2 nonzeros: {np.nonzero(obs['tier2cards'])[0].tolist()}")
        print(f"    tier3 nonzeros: {np.nonzero(obs['tier3cards'])[0].tolist()}")
        print(f"    nobles:      {obs['nobles']}")
        print(f"    legal_mask:  {mask}\n")

def main():
    env = SplendorGymEnv()
    print("Running simulation to detect stuck episodes...")
    stuck = simulate_and_debug(env, n_episodes=1000, max_steps=100)
    if stuck:
        print(f"⚠️ {len(stuck)} episodes hit cap and never terminated")
        # dive into the first stuck trajectory:
        debug_stuck(env, stuck[0]['traj'])
        raise RuntimeError(f"{len(stuck)}/1000 episodes never terminated within 100 steps")
    else:
        print("✅ All episodes terminated cleanly!")

if __name__ == '__main__':
    main()
"""

def main():
    print("ws")
    env = SplendorGymEnv()
    print("Observation space:", env.observation_space)
    print("Action space: Discrete({})".format(env.action_space.n))

    # 1) RESET
    obs, info = env.reset()
    print("\n>>> After reset:")
    print(" legal_count:", np.sum(info["legal_mask"]))
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:18s} → shape {v.shape}, sample:", v.flatten()[:10])
        else:
            print(f"  {k:18s} → scalar {v}")

    assert env.observation_space.contains(obs), "OBS out of space!"

    # 2) RUN one full random-legal episode
    print("\n>>> Stepping randomly until done…")
    done = False
    total_steps = 0
    reward = None
    while not done and total_steps < 500:
        legal = env.game.legal_actions(env.agent_player)
        legal_idxs = [i for i, a in enumerate(env.all_actions) if a in legal]
        assert legal_idxs, "No legal actions available!?"
        choice = np.random.choice(legal_idxs)

        obs, reward, done, truncated, info = env.step(choice)
        total_steps += 1

        assert env.observation_space.contains(obs), f"Step {total_steps}: obs invalid!"
        if total_steps % 10 == 0:
            print(f"  step {total_steps:3d}, reward {reward}, legal_count {np.sum(info['legal_mask'])}")
            for k, v in obs.items():
                if isinstance(v, np.ndarray):
                    print(f"  {k:18s} → shape {v.shape}, sample:", v.flatten()[:10])
                else:
                    print(f"  {k:18s} → scalar {v}")

    print(f"\nEpisode finished in {total_steps} steps, final reward = {reward}")

    # 3) Face-up card marking test
    print("\n>>> Checking face-up marking at end:")
    for lvl, key, size in ((0, "tier1cards", 40), (1, "tier2cards", 30), (2, "tier3cards", 20)):
        arr = obs[key]
        faceup_tuples = [normalize_card_key(c) for c in env.game.board_cards[lvl] if c]
        for tpl in faceup_tuples:
            idx = env.card_index_map[lvl][tpl]
            assert arr[idx] == 1, f"Tier{lvl+1} template {tpl} should be marked face-up=1"
    print("  face-up check passed!")

    # 4) Reserved card marking (RESERVE_CARD)
    print("\n>>> Checking reserve-card (board) marking:")
    obs0, info0 = env.reset()
    legal0 = env.game.legal_actions(env.agent_player)
    reserve_card_idxs = [
        i for i, a in enumerate(env.all_actions)
        if a in legal0 and a[0] == ActionType.RESERVE_CARD
    ]
    assert reserve_card_idxs, "No RESERVE_CARD action legal on first turn!"
    obs1, _, _, _, _ = env.step(reserve_card_idxs[0])
    changed = (
        np.any(obs1["tier1cards"] == 2) or
        np.any(obs1["tier2cards"] == 2) or
        np.any(obs1["tier3cards"] == 2)
    )
    assert changed, "No card slot marked as 2 after RESERVE_CARD!"
    print("  reserve-card marking passed!")

    # 5) Reserved deck test (RESERVE_DECK)
    print("\n>>> Checking reserve-deck (hidden) effect:")
    obs0, info0 = env.reset()
    legal0 = env.game.legal_actions(env.agent_player)
    reserve_deck_idxs = [
        i for i, a in enumerate(env.all_actions)
        if a in legal0 and a[0] == ActionType.RESERVE_DECK
    ]
    if reserve_deck_idxs:
        obs1, _, _, _, _ = env.step(reserve_deck_idxs[0])
        assert obs1["my_reserved_count"] > 0, "Reserved count did not increase after RESERVE_DECK!"
        # Deck cards are hidden; no tier arrays should mark `== 2`
        assert not np.any(obs1["tier1cards"] == 2)
        assert not np.any(obs1["tier2cards"] == 2)
        assert not np.any(obs1["tier3cards"] == 2)
        print("  reserve-deck marking passed!")

    # 6) Buy card test
    print("\n>>> Checking buy and noble update:")
    env.reset()
    player = env.game.players[env.agent_player]
    for i, card in enumerate(env.game.board_cards[0]):
        if card and env.game.can_afford(player, card):
            idx = env.all_actions.index((ActionType.BUY_BOARD, (0, i)))
            obs2, _, _, _, _ = env.step(idx)
            tpl_key = normalize_card_key(card)
            card_idx = env.card_index_map[card.level][tpl_key]
            assert obs2["tier1cards"][card_idx] == 4, "Card should be marked as bought"
            print("  buy + board marking passed!")
            break

    print("\n✅ All tests passed successfully!")

if __name__ == "__main__":
        main()
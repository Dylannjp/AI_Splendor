import numpy as np
from splendor_env import SplendorGymEnv
def main():
    env = SplendorGymEnv()
    print("Observation space:", env.observation_space)
    print("Action space: Discrete({})".format(env.action_space.n))

    # 1) RESET
    obs, info = env.reset()
    print("\n>>> After reset:")
    print(" legal_count:", info["legal_count"])
    for k, v in obs.items():
        print(f"  {k:12s} → shape {np.shape(v)}, sample:", v.flatten()[:10])

    assert env.observation_space.contains(obs), "OBS out of space!"

    # 2) RUN one full random-legal episode
    print("\n>>> Stepping randomly until done…")
    done = False
    total_steps = 0
    reward = None
    while not done and total_steps < 200:
        # Always recompute legal moves from env.game
        legal = env.game.legal_actions(0)  # list of (ActionType, param)
        legal_idxs = [i for i, a in enumerate(env.all_actions) if a in legal]
        assert legal_idxs, "No legal actions available!?"
        choice = np.random.choice(legal_idxs)

        obs, reward, done, truncated, info = env.step(choice)
        total_steps += 1

        # verify obs validity each step
        assert env.observation_space.contains(obs), f"Step {total_steps}: obs invalid!"
        if total_steps % 10 == 0:
            print(f"  step {total_steps:3d}, reward {reward}, legal_count {info['legal_count']}")
            for k, v in obs.items():
                print(f"  {k:12s} → shape {np.shape(v)}, sample:", v.flatten()[:])

    print(f"\nEpisode finished in {total_steps} steps, final reward = {reward}")

    # 3) Inspect that face-up cards are marked as 1
    print("\n>>> Checking face-up marking at end:")
    for lvl, key, size in ((0, "tier1cards", 40), (1, "tier2cards", 30), (2, "tier3cards", 20)):
        arr = obs[key]
        faceup_tuples = [
            (c.level, c.bonus, c.VPs, tuple(c.cost))
            for c in env.game.board_cards[lvl] if c
        ]
        for tpl in faceup_tuples:
            idx = env.card_index_map[lvl][tpl]
            assert arr[idx] == 1, f"Tier{lvl+1} template {tpl} should be marked face-up=1"
    print("  face-up check passed!")

    # 4) Reserved/taken marking: pick one RESERVE action
    print("\n>>> Checking reserve/taken marking:")
    obs0, info0 = env.reset()
    legal0 = env.game.legal_actions(0)
    reserve_idxs = [
        i for i, a in enumerate(env.all_actions)
        if a in legal0 and a[0].name.startswith("RESERVE")
    ]
    assert reserve_idxs, "No RESERVE action legal on first turn!"
    choice = reserve_idxs[0]
    obs1, _, _, _, _ = env.step(choice)

    # Check that at least one card slot flipped to 2
    changed = (
        np.any(obs1["tier1cards"] == 2)
        or np.any(obs1["tier2cards"] == 2)
        or np.any(obs1["tier3cards"] == 2)
    )
    assert changed, "No card slot changed to 2 after reserve!"
    print("  reserve marking passed!")

    print("\n✅ All manual tests passed!")

if __name__ == "__main__":
    main()

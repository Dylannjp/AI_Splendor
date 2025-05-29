import numpy as np
from splendor_env import SplendorEnv
from game_logic.splendor_game import ActionType, COLOR_NAMES

def describe_action(move):
    typ, param = move
    if typ is ActionType.TAKE_DIFF:
        cols = [COLOR_NAMES[c] for c in param]
        return f"takes one each of {', '.join(cols)}"
    if typ is ActionType.TAKE_SAME:
        return f"takes two {COLOR_NAMES[param]} gems"
    if typ is ActionType.RESERVE_CARD:
        lvl, idx = param
        return f"reserves face-up card (tier {lvl+1}, slot {idx})"
    if typ is ActionType.RESERVE_DECK:
        lvl, _ = param
        return f"reserves face-down card from tier {lvl+1}"
    if typ is ActionType.BUY_BOARD:
        lvl, idx = param
        return f"buys face-up card (tier {lvl+1}, slot {idx})"
    if typ is ActionType.BUY_RESERVE:
        return "buys a reserved card"
    if typ is ActionType.DISCARD:
        return f"discards one {COLOR_NAMES[param]} gem"
    return typ.name

def print_state(game):
    print("  board gems  :", game.board_gems.tolist())
    for lvl in range(3):
        cards = []
        for c in game.board_cards[lvl]:
            row = [int(x) for x in (*c.cost, c.VPs, c.bonus)]
            cards.append(row)
        print(f"  tier {lvl+1} face-up:", cards)
    noble_reqs = [n.requirement.tolist() for n in game.nobles if n is not None]
    print("  nobles reqs :", noble_reqs)
    for i, p in enumerate(game.players):
        gems = p.gems.tolist()
        bonuses = p.bonuses.tolist()
        print(f"  P{i} gems     : {gems}  bonuses: {bonuses}  VP: {p.VPs}")
        reserved = [[int(x) for x in (*c.cost, c.VPs, c.bonus)] for c in p.reserved]
        print(f"     reserved  : {reserved}")
    print()

def main():
    env = SplendorEnv()
    obs, _ = env.reset()
    turn = 0

    # Pre-grab the full action list for mapping indices back to tuples
    all_actions = env.all_actions

    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        idx = env.agent_name_mapping[agent]
        game = env.unwrapped.game

        if termination or truncation:
            desc = "→ done"
            action_idx = None
        else:
            mask = obs["action_mask"]
            choices = np.nonzero(mask)[0]
            if len(choices) == 0:
                print(f"[ERROR] No legal moves for {agent} at turn {turn:03d}.")
                return
            action_idx = np.random.choice(choices)
            action_tuple = all_actions[action_idx]
            desc = describe_action(action_tuple)

        print(f"=== Turn {turn:03d} — {agent} {desc:<40} | reward={reward:.1f} ===")
        print_state(game)

        # Step only if there is an action to take
        env.step(action_idx)
        turn += 1

        if all(env.terminations[a] for a in env.agents):
            print("Game over.")
            break

    env.close()

if __name__ == "__main__":
    for _ in range(200):
        main()

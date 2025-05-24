import random
from splendor_game import SplendorGame, ActionType, COLOR_NAMES

def describe_action(action, card=None):
    typ, param = action
    if typ is ActionType.TAKE_DIFF:
        cols = [COLOR_NAMES[c] for c in param]
        return f"takes one each of {', '.join(cols)}"
    if typ is ActionType.TAKE_SAME:
        return f"takes two {COLOR_NAMES[param]} gems"
    if typ is ActionType.RESERVE_CARD:
        lvl, idx = param
        return f"reserves board card (tier {lvl+1}, slot {idx}) → {card}"
    if typ is ActionType.RESERVE_DECK:
        lvl, idx = param
        return f"reserves deck card (tier {lvl+1}, slot {idx}) → {card}"
    if typ is ActionType.BUY_BOARD:
        lvl, idx = param
        return f"buys board card (tier {lvl+1}, slot {idx}) → {card}"
    if typ is ActionType.BUY_RESERVE:
        return f"buys reserved card #{param} → {card}"
    if typ is ActionType.DISCARD:
        return f"discards one {COLOR_NAMES[param]} gem"
    return f"does {typ.name}"

def verbose_self_play(num_players=2, max_turns=200):
    random.seed(0)
    game = SplendorGame(num_players)
    game.setup_board()
    print("=== Starting Splendor self-play ===")
    for turn in range(max_turns):
        p = game.current_player
        legals = game.legal_actions(p)
        if not legals:
            print(f"Turn {turn:03d}: Player {p} has no legal moves; stopping.")
            break

        action = random.choice(legals)

        # record the card (if any) before modifying the state
        card = None
        if action[0] is ActionType.RESERVE_CARD or action[0] is ActionType.BUY_BOARD:
            lvl, idx = action[1]
            card = game.board_cards[lvl][idx]
        elif action[0] is ActionType.RESERVE_DECK:
            lvl, idx = action[1]
            card = game.decks[lvl][idx]
        elif action[0] is ActionType.BUY_RESERVE:
            idx = action[1]
            card = game.players[p].reserved[idx]

        before_vp = game.players[p].VPs
        game.step(action)
        after_vp = game.players[p].VPs

        desc = describe_action(action, card)

        # detect noble claim
        if action[0] in (ActionType.BUY_BOARD, ActionType.BUY_RESERVE):
            vp_gain = after_vp - before_vp
            if (card.VPs == 0) and (vp_gain == 3):
                desc += " and claims a noble (+3 VPs)"

        print(f"Turn {turn:03d}: Player {p} {desc}; now has {after_vp} points")

        if after_vp >= 15:
            print(f"🏆 Player {p} wins on turn {turn:03d}!")
            break

if __name__ == "__main__":
    verbose_self_play()

# Holy shit it actually works!
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

def dump_full_state(game):
    print(" Board gems:", game.board_gems)
    for i, p in enumerate(game.players):
        print(f" P{i} gems:", p.gems, "bonuses:", p.bonuses, "VPs:", p.VPs)4
        print(f"  reserved:", [(c.level, c.VPs) for c in p.reserved])
    print(" Board sizes:", [len(b) for b in game.board_cards])
    print(" Deck sizes :", [len(d) for d in game.decks])
    print(" Nobles left:", [n.requirement for n in game.nobles if n])

def verbose_self_play(num_players=2, max_turns=200, seed=None):
    if seed is not None:
        random.seed(seed)
    game = SplendorGame(num_players)
    print("=== Starting Splendor self-play ===")
    for turn in range(max_turns):
        
        p = game.current_player
        legals = game.legal_actions(p)
        if not legals:
            print(f"Turn {turn:03d}: Player {p} has no legal moves; stopping.")
            dump_full_state(game)
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

        if game.game_over:
            winner = game.decide_winner()
            print(f"player {winner} wins! game ends after {turn-1} turns!")
            break

if __name__ == "__main__":
    verbose_self_play()

# Holy shit it actually works!
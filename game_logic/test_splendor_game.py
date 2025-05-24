import numpy as np
import pytest

from splendor_game import SplendorGame, ActionType
from player import Player
from cards import Card, Noble

@pytest.fixture
def game():
    g = SplendorGame(num_players=2)
    # ensure boards are dealt
    g.setup_board()
    return g

def test_initial_setup(game):
    # 4 cards per tier on board
    for lvl in (0,1,2):
        assert len(game.board_cards[lvl]) == 4
    # correct number of nobles
    assert len(game.nobles) == 3  # players + 1
    # token pool sum
    assert game.board_gems.sum() == 4*5 + 5  # five colors + gold

def test_take_diff(game):
    p = game.players[0]
    # record starting tokens
    before = p.gems.copy()
    # pick a legal TAKE_DIFF action
    action = next(a for a in game.legal_actions(0) if a[0] is ActionType.TAKE_DIFF)
    game.step(action)
    # ensure player got gems and board lost them
    for color in action[1]:
        assert p.gems[color] == before[color] + 1
        assert game.board_gems[color] == 4 - (before[color] + 1)

def test_take_same(game):
    # artificially boost one color on board
    game.board_gems[2] = 4
    p = game.players[0]
    action = (ActionType.TAKE_SAME, 2)
    game.step(action)
    assert p.gems[2] == 2
    assert game.board_gems[2] == 2

def test_reserve_and_buy(game):
    p = game.players[0]
    # reserve a card from tier 0
    action_reserve = (ActionType.RESERVE_CARD, (0, 0))
    game.step(action_reserve)
    assert len(p.reserved) == 1
    # give player enough gems to afford it
    card = p.reserved[0]
    p.gems[:5] = card.cost  # ignore bonuses for now
    game.current_player = 0
    # buy from reserve
    action_buy = next(a for a in game.legal_actions(0) if a[0] is ActionType.BUY_RESERVE)
    game.step(action_buy)
    assert p.VPs == card.VPs
    assert p.bonuses[card.bonus] == 1

def test_discard(game):
    p = game.players[0]
    # give player 11 gems to force discard
    p.gems = np.array([3,3,3,2,0,0])
    # only DISCARD actions should be legal
    legals = game.legal_actions(0)
    assert all(a[0] is ActionType.DISCARD for a in legals)
    # perform one
    action = legals[0]
    game.step(action)
    assert p.gems.sum() == 10

def test_noble_visit(game):
    p = game.players[0]
    # manually give bonuses to satisfy first noble
    noble = game.nobles[0]
    p.bonuses = noble.requirement.copy()
    # trigger noble check
    game.handle_nobles(p)
    assert p.VPs == 3
    assert noble not in game.nobles

# passed everything, move on to playtest
# also chatgpt generated these tests, pretty cool. game worked smoothly afterwards

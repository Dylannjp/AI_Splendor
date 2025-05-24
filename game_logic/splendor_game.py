import copy
import numpy as np
from enum import Enum, auto
from itertools import combinations

from game_logic.player import Player
from game_logic.cards import Card, Noble
from game_logic.game_data import tier1_card_data, tier2_card_data, tier3_card_data, nobles_data


COLOR_NAMES = ['black', 'white', 'red', 'blue', 'green', 'gold']

class ActionType(Enum):
    BUY_BOARD    = auto()
    BUY_RESERVE  = auto()
    TAKE_DIFF    = auto()
    TAKE_SAME    = auto()
    RESERVE_CARD = auto()
    RESERVE_DECK = auto()
    DISCARD      = auto()

class SplendorGame:
    MAX_GEMS = 10

    def __init__(self, num_players=2):
        base = 4 # if num_players == 2 else 5 if num_players == 3 else 7 
        ## save for later, just 2 players for now.
        self.board_gems = np.array([base, base, base, base, base, 5], dtype=int)

        self.deck1 = []
        self.deck2 = []
        self.deck3 = []

        self.board1 = []
        self.board2 = []
        self.board3 = []

        self.nobles = []

        self.setup_nobles()
        self.setup_deck()
        self.setup_board()


        self.players = [Player() for _ in range(num_players)]
        self.current_player = 0


    def setup_deck(self):
        self.decks = [self.deck1, self.deck2, self.deck3]
        for lvl, data in [(0, tier1_card_data),
                          (1, tier2_card_data),
                          (2, tier3_card_data)]:
            cards = []
            for row in data:
                cost = row[:5]
                VPs = int(row[5])
                bonus = int(row[6])
                cards.append(Card(level=lvl, cost=cost, VPs=VPs, bonus=bonus))
            np.random.shuffle(cards)
            self.decks[lvl] = cards

    def setup_nobles(self):
        nobles = [Noble(req) for req in nobles_data]
        np.random.shuffle(nobles)
        self.nobles = nobles[:3]

    def setup_board(self):
        self.board1 = []
        self.board2 = []
        self.board3 = []
        self.board_cards = [self.board1, self.board2, self.board3]
        for level in (0, 1, 2):
            for _ in range(4):
                self.reveal_card(level)

    def reveal_card(self, level):
        card = self.decks[level].pop()
        self.board_cards[level].append(card)

    def refill_board(self):
        for level in (0,1,2):
            while len(self.board_cards[level]) < 4 and self.decks[level]:
                self.reveal_card(level)

    def legal_actions(self, p_idx):
        player = self.players[p_idx]
        total = int(player.gems.sum())

        if total > self.MAX_GEMS:
            return [(ActionType.DISCARD, c) for c in range(6) if player.gems[c] > 0]

        actions = []
        available_colors = [c for c in range(5) if self.board_gems[c] > 0]
        if len(available_colors) >= 3:
            # take any 3 distinct colors
            for combo in combinations(available_colors, 3):
                actions.append((ActionType.TAKE_DIFF, combo))
        elif len(available_colors) == 2:
            # only 2 colors left, allow taking those two
            for combo in combinations(available_colors, 2):
                actions.append((ActionType.TAKE_DIFF, combo))
        elif len(available_colors) == 1:
            # only 1 color left, allow taking one
            actions.append((ActionType.TAKE_DIFF, tuple(available_colors)))
        for c in range(5):
            if self.board_gems[c] >= 4:
                actions.append((ActionType.TAKE_SAME, c))
        if len(player.reserved) < 3:
            for level in (0,1,2):
                for idx in range(len(self.board_cards[level])):
                    actions.append((ActionType.RESERVE_CARD, (level, idx)))
            for level in (0,1,2):
                if self.decks[level] is not None:
                    actions.append((ActionType.RESERVE_DECK, (level, idx)))
        for level in (0,1,2):
            for idx, card in enumerate(self.board_cards[level]):
                if self.can_afford(player, card):
                    actions.append((ActionType.BUY_BOARD, (level, idx)))
        for idx, card in enumerate(player.reserved):
            if self.can_afford(player, card):
                actions.append((ActionType.BUY_RESERVE, idx))

        return actions

    def step(self, action):
        type, param = action
        player = self.players[self.current_player]

        if type is ActionType.TAKE_DIFF:
            self.take_gems(player, list(param))
        elif type is ActionType.TAKE_SAME:
            self.take_gems(player, [param,param])
        elif type is ActionType.RESERVE_CARD:
            level, idx = param
            self.reserve_card(player, level, idx)
        elif type is ActionType.RESERVE_DECK:
            level, idx = param
            self.reserve_card(player, level, idx = -1)
        elif type is ActionType.BUY_BOARD:
            level, idx = param
            self.buy_card(player, level, idx, from_reserve=False)
            self.handle_nobles(player)
        elif type is ActionType.BUY_RESERVE:
            res_idx = param
            self.buy_card(player, None, res_idx, from_reserve=True)
            self.handle_nobles(player)
        elif type is ActionType.DISCARD:
            color = param
            player.gems[color] -= 1
            self.board_gems[color] += 1
        else:
            raise ValueError(f"Unknown action {type}")

        if player.gems.sum() > self.MAX_GEMS:
            return

        self.refill_board()
        self.current_player = (self.current_player + 1) % len(self.players)

    def can_afford(self, player, card):
        # real cost after considering bonuses
        cost = card.cost - player.bonuses
        cost = np.clip(cost, 0, None)
        # calculate tokens needed after using colored gems
        needed = cost - player.gems[:5]
        needed = np.clip(needed, 0, None)
        # check if gold covers the rest
        return needed.sum() <= player.gems[5]

    def take_gems(self, player, colors):
        for c in colors:
            self.board_gems[c] -= 1
            player.gems[c] += 1

    def reserve_card(self, player, level, idx):
        if idx == -1:
            card = self.decks[level].pop()
        else:
            card = self.board_cards[level].pop(idx)
        player.reserved.append(card)
        if self.board_gems[5] > 0:
            self.board_gems[5] -= 1
            player.gems[5] += 1

    def buy_card(self, player, level, idx, from_reserve=False):
        if from_reserve:
            card = player.reserved.pop(idx)
        else:
            card = self.board_cards[level].pop(idx)
        cost = card.cost - player.bonuses
        cost = np.clip(cost, 0, None)
        for color in range(5):
            pay = min(cost[color], player.gems[color])
            player.gems[color] -= pay
            self.board_gems[color] += pay
            cost[color] -= pay
        gold = cost.sum()
        player.gems[5] -= gold
        self.board_gems[5] += gold
        player.bonuses[card.bonus] += 1
        player.VPs += card.VPs

    def handle_nobles(self, player):
        for noble in list(self.nobles):
            if all(player.bonuses >= noble.requirement):
                player.VPs += noble.VPs
                self.nobles.remove(noble)

import copy
import numpy as np
from enum import Enum, auto
from itertools import combinations

from game_logic.player import Player
from game_logic.cards import Card, Noble
from game_logic.game_data import tier1_card_data, tier2_card_data, tier3_card_data, nobles_data


COLOR_NAMES = ['black', 'white', 'red', 'blue', 'green', 'gold']

class ActionType(Enum):
    BUY_BOARD   = auto()
    BUY_RESERVE = auto()
    TAKE_DIFF   = auto()
    TAKE_SAME   = auto()
    RESERVE_CARD = auto()
    RESERVE_DECK = auto()
    DISCARD     = auto()
    PASS        = auto()

class SplendorGame:
    MAX_GEMS = 10

    def __init__(self, num_players=2):
        base = 4 # if num_players == 2 else 5 if num_players == 3 else 7
        self.board_gems = np.array([base, base, base, base, base, 5], dtype=int)

        self.decks = [[], [], []]  # decks for each level
        self.board_cards = [[], [], []]  # cards on the board for each level
        self.nobles = []

        self.setup_nobles()
        self.setup_deck()
        self.setup_board()

        self.players = [Player() for _ in range(num_players)]

        self.current_player = 0
        self.first_player = self.current_player
        self.final_turn = False
        self.game_over = False
        
        # Add the fixed list of all possible actions
        self.all_actions = self._get_all_possible_actions()
        self.action_to_idx = {
            action: idx for idx, action in enumerate(self.all_actions)
        }

    def setup_deck(self):
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
        self.nobles = nobles[:len(nobles_data)] # Keep all initially, show N+1
        self.nobles = nobles[:3] # Show N+1 nobles (3 for 2 players)

    def setup_board(self):
        for level in (0, 1, 2):
            for _ in range(4):
                self.board_cards[level].append(self.decks[level].pop())

    def refill_board(self):
        for level in range(3):
            for slot in range(4):
                if (slot < len(self.board_cards[level]) and self.board_cards[level][slot] is None and self.decks[level]):
                    self.board_cards[level][slot] = self.decks[level].pop()

    def _get_all_possible_actions(self):
        """Generates a fixed list of all 67 possible action types."""
        actions = []
        colors = list(range(5))

        # Take 3 Diff (10)
        actions.extend([(ActionType.TAKE_DIFF, combo) for combo in combinations(colors, 3)])
        # Take 2 Diff (10)
        actions.extend([(ActionType.TAKE_DIFF, combo) for combo in combinations(colors, 2)])
        # Take 1 Diff (5)
        actions.extend([(ActionType.TAKE_DIFF, (c,)) for c in colors]) # Ensure it's a tuple

        # Take 2 Same (5)
        actions.extend([(ActionType.TAKE_SAME, c) for c in colors])

        # Buy Board (12)
        actions.extend([(ActionType.BUY_BOARD, (lvl, slot)) for lvl in range(3) for slot in range(4)])

        # Buy Reserve (3)
        actions.extend([(ActionType.BUY_RESERVE, res_idx) for res_idx in range(3)])

        # Reserve Board (12)
        actions.extend([(ActionType.RESERVE_CARD, (lvl, slot)) for lvl in range(3) for slot in range(4)])

        # Reserve Deck (3) - FIX: Parameter is just the level
        actions.extend([(ActionType.RESERVE_DECK, (lvl, -1)) for lvl in range(3)])

        # Discard (6)
        actions.extend([(ActionType.DISCARD, c) for c in list(range(6))])
        
        # Pass (1) (Effecttively resigning because there's no way you win after this happens)
        actions.extend([(ActionType.PASS, None)])

        assert len(actions) == 67, f"Expected 67 actions, got {len(actions)}"
        return actions

    def legal_actions(self, p_idx):
        player = self.players[p_idx]
        total = int(player.gems.sum())

        if total > self.MAX_GEMS:
            return [(ActionType.DISCARD, c) for c in range(6) if player.gems[c] > 0] # Only non-gold

        actions = []
        available_colors = [c for c in range(5) if self.board_gems[c] > 0]

        # Take Diff - Generate *only* currently possible ones
        if len(available_colors) >= 3:
            for combo in combinations(available_colors, 3):
                actions.append((ActionType.TAKE_DIFF, combo))
        elif len(available_colors) == 2:
            for combo in combinations(available_colors, 2):
                actions.append((ActionType.TAKE_DIFF, combo))
        elif len(available_colors) == 1:
            actions.append((ActionType.TAKE_DIFF, tuple(available_colors)))

        # Take Same
        for c in range(5):
            if self.board_gems[c] >= 4:
                actions.append((ActionType.TAKE_SAME, c))

        # Reserve Card / Reserve Deck
        if len(player.reserved) < 3:
            for level in (0,1,2):
                for idx, card in enumerate(self.board_cards[level]):
                    if card is not None:
                        actions.append((ActionType.RESERVE_CARD, (level, idx)))
                if self.decks[level]: # Check if deck has cards
                    actions.append((ActionType.RESERVE_DECK, (level, -1))) # FIX: Use level only

        # Buy Board
        for level in (0,1,2):
            for idx, card in enumerate(self.board_cards[level]):
                if card and self.can_afford(player, card):
                    actions.append((ActionType.BUY_BOARD, (level, idx)))

        # Buy Reserve
        for idx, card in enumerate(player.reserved):
            if card and self.can_afford(player, card):
                actions.append((ActionType.BUY_RESERVE, idx))

        if len(actions) == 0:
            actions.append((ActionType.PASS, None))  # Allow passing if no other actions
            
        return actions

    def step(self, action):
        type, param = action
        #print("stepped")
        player = self.players[self.current_player]

        if type is ActionType.TAKE_DIFF:
            self.take_gems(player, list(param))
        elif type is ActionType.TAKE_SAME:
            self.take_gems(player, [param,param])
        elif type is ActionType.RESERVE_CARD:
            level, idx = param
            self.reserve_card(player, level, idx)
            self.refill_board()    
        elif type is ActionType.RESERVE_DECK:
            level, idx = param
            self.reserve_card(player, level, idx = -1)
            self.refill_board()
        elif type is ActionType.BUY_BOARD:
            level, idx = param
            self.buy_card(player, level, idx, from_reserve=False)
            self.refill_board()
            self.handle_nobles(player)
        elif type is ActionType.BUY_RESERVE:
            res_idx = param
            self.buy_card(player, None, res_idx, from_reserve=True)
            self.refill_board()
            self.handle_nobles(player)
        elif type is ActionType.DISCARD:
            color = param
            player.gems[color] -= 1
            self.board_gems[color] += 1
        elif type is ActionType.PASS:
            # No action needed, just pass the turn
            pass
        # If player must discard, don't advance turn
        if player.gems.sum() > self.MAX_GEMS:
            return
        
        if player.VPs >= 15 and not self.final_turn:
            self.final_turn = True

        self.current_player = (self.current_player + 1) % len(self.players)
        if self.final_turn and self.current_player == self.first_player:
            self.game_over = True

    def can_afford(self, player, card):
        cost = card.cost - player.bonuses
        cost = np.clip(cost, 0, None)
        needed = cost - player.gems[:5]
        needed = np.clip(needed, 0, None)
        return needed.sum() <= player.gems[5]

    def take_gems(self, player, colors):
        for c in colors:
            self.board_gems[c] -= 1
            player.gems[c] += 1

    def reserve_card(self, player, level, idx):
        if idx == -1:
            card = self.decks[level].pop()
        else:
            if idx >= len(self.board_cards[level]) or self.board_cards[level][idx] is None:
                return False
            card = self.board_cards[level][idx]
            self.board_cards[level][idx] = None

        player.reserved.append(card)
        if self.board_gems[5] > 0:
            self.board_gems[5] -= 1
            player.gems[5] += 1

    def buy_card(self, player, level, idx, from_reserve=False):
        if from_reserve:
            card = player.reserved.pop(idx) 
        else:
            if idx >= len(self.board_cards[level]) or self.board_cards[level][idx] is None:
                return False
            card = self.board_cards[level][idx]
            self.board_cards[level][idx] = None

        cost = card.cost - player.bonuses
        cost = np.clip(cost, 0, None)
        
        gold_needed = 0
        for color in range(5):
            pay = min(cost[color], player.gems[color])
            player.gems[color] -= pay
            self.board_gems[color] += pay
            cost[color] -= pay
            gold_needed += cost[color] # Remaining cost must be paid by gold

        player.gems[5] -= gold_needed
        self.board_gems[5] += gold_needed
        player.bonuses[card.bonus] += 1
        player.VPs += card.VPs

    def handle_nobles(self, player):
        # Only check N+1 nobles
        num_nobles_to_check = len(self.players) + 1
        available_nobles = [n for n in self.nobles if n is not None][:num_nobles_to_check]

        earned_nobles = []
        for noble in available_nobles:
            if all(player.bonuses >= noble.requirement):
                earned_nobles.append(noble)
        
        if earned_nobles:
            chosen_noble = earned_nobles[0]
            player.VPs += chosen_noble.VPs
            # Remove it from the main list so it can't be claimed again
            for i, n in enumerate(self.nobles):
                if n == chosen_noble:
                    self.nobles[i] = None # Mark as taken instead of removing, to keep indices stable
                    break
        
    def decide_winner(self):
        max_vp = max(player.VPs for player in self.players)
        max_bonus = max(player.bonuses.sum() for player in self.players)
        winner = [i for i, player in enumerate(self.players) if player.VPs == max_vp]
        
        if len(winner) == 1:
            print (f"Player {winner[0]} wins with {max_vp} VPs")
            return winner[0]

        max_bonus = max(self.players[i].bonuses.sum() for i in winner)
        bonus_winner = [i for i in winner if self.players[i].bonuses.sum() != max_bonus]
        if len(bonus_winner) == 1:
            print (f"Player {bonus_winner[0]} win with {max_vp} VPs and less bonuses")
            return bonus_winner[0]
        
        max_gems = max(self.players[i].gems.sum() for i in winner)
        gem_winner = [i for i in winner if self.players[i].gems.sum() != max_gems]
        if len(gem_winner) == 1:
            print (f"Player {gem_winner[0]} wins with {max_vp} VPs, less bonuses and less gems")
            return gem_winner[0]
        
        max_reserved = max(len(self.players[i].reserved) for i in winner)
        reserve_winner = [i for i in winner if len(self.players[i].reserved) != max_reserved]
        if len(reserve_winner) == 1:
            print(f"Player {reserve_winner[0]} wins with {max_vp} VPs, less bonuses, less gems and less reserved cards")
            return reserve_winner[0]
        
        later_turn_winner = [i in winner for i in range(len(self.players)) if i != self.first_player] # final very unlikely tiebreaker
        return later_turn_winner[0]

        

# No empty masks, consistent action lists! Environment is good now it seems.
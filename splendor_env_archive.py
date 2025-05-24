import gymnasium as gym
from gymnasium import spaces
import numpy as np
from itertools import combinations
from gymnasium.envs.registration import register

from game_logic.game_data import tier1_card_data, tier2_card_data, tier3_card_data, nobles_data

GEM_TYPES = ['black', 'white', 'red', 'blue', 'green', 'gold']
NUM_TOKENS = 6
MAX_TOKENS = 10

class SplendorEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.tier1_card_data = tier1_card_data
        self.tier2_card_data = tier2_card_data
        self.tier3_card_data = tier3_card_data
        self.nobles_data = nobles_data

        self.observation_space = spaces.Dict({
            "tier1cards": spaces.Box(low=0, high=5, shape=(40,), dtype=np.int32),
            "tier2cards": spaces.Box(low=0, high=5, shape=(30,), dtype=np.int32),
            "tier3cards": spaces.Box(low=0, high=5, shape=(20,), dtype=np.int32),
            # cards: 0 = in deck, 1 = in play, 2 = reserved by opponent, 3 = reserved by me, 4 = bought by opponent, 5 = bought by me
            "nobles": spaces.Box(low=0, high=3, shape=(10,), dtype=np.int32),
            # nobles: 0 = not in play, 1 = in play, 2 = acquired by opponent, 3 = acquired by me
            "board_gems": spaces.Box(low=0, high=5, shape=(6,), dtype=np.int32),
            # board_gems: 0 = black, 1 = white, 2 = red, 3 = blue, 4 = green, 5 = gold
            "my_gems": spaces.Box(low=0, high=10, shape=(6,), dtype=np.int32), # the amount of gems the agent has
            "opponent_gems": spaces.Box(low=0, high=10, shape=(6,), dtype=np.int32), # the amount of gems the opponent agent has
            "my_score": spaces.Discrete(23), # the amount of points the agent has, max is 24
            "opponent_score": spaces.Discrete(23), # the amount of points the opponent agent has, max is 24
        })

        self.step_count = 0
        self.max_steps = 200
        self.action_space = spaces.Discrete(42)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.tier1cards = np.zeros(40, dtype=np.int32)
        self.tier2cards = np.zeros(30, dtype=np.int32)
        self.tier3cards = np.zeros(20, dtype=np.int32)
        for n in range(4):
            self.reveal_cards(self.tier1cards)
            self.reveal_cards(self.tier2cards)
            self.reveal_cards(self.tier3cards)

        self.nobles = np.zeros(10, dtype=np.int32)
        noble_ids = np.random.choice(10, 3, replace=False)
        self.nobles[noble_ids] = 1

        self.board_gems = np.array([4, 4, 4, 4, 4, 5], dtype=np.int32)
        self.my_gems = np.zeros(6, dtype=np.int32)
        self.opponent_gems = np.zeros(6, dtype=np.int32)

        self.my_score = 0
        self.opponent_score = 0
        self.my_bonuses = np.zeros(5, dtype=np.int32)
        self.opponent_bonuses = np.zeros(5, dtype=np.int32)
        self.reserved = []

        self.done = False
        self.info = {}
        self.step_count = 0
        return self.observation(), self.info

    def observation(self):
        return {
            "tier1cards": self.tier1cards.copy(),
            "tier2cards": self.tier2cards.copy(),
            "tier3cards": self.tier3cards.copy(),
            "nobles": self.nobles.copy(),
            "board_gems": self.board_gems.copy(),
            "my_gems": self.my_gems.copy(),
            "opponent_gems": self.opponent_gems.copy(),
            "my_score": self.my_score,
            "opponent_score": self.opponent_score,
        }

    def reveal_cards(self, card_array): # flip a card from the deck into play
        deck_indices = np.where(card_array == 0)[0]
        if len(deck_indices) == 0: # no cards left in the deck
            return
    
        reveal_indices = np.random.choice(deck_indices, 1, replace=False)
        card_array[reveal_indices] = 1

    def take_3_different_gems(self, combo_id, is_opponent=False): # take 3 different gems from the board
        if is_opponent:
            gems = self.opponent_gems
        else:
            gems = self.my_gems
        board_gems = self.board_gems

        available_colors = [i for i in range(5) if board_gems[i] > 0]
        if len(available_colors) < 3: # if there aren't enough gems to take 3 different ones
            if combo_id >= len(available_colors):
                return -0.5
            gem = available_colors[combo_id]
            board_gems[gem] -= 1
            gems[gem] += 1
            return 0.0

        combos = list(combinations(available_colors, 3))
        if combo_id >= len(combos):
            return -0.5
        for gem in combos[combo_id]:
            board_gems[gem] -= 1
            gems[gem] += 1

        if is_opponent: # update gem pool (either player or opponent)
            self.opponent_gems = gems
        else:
            self.my_gems = gems

        return 0.0

    def take_2_same_gem(self, gem, is_opponent=False):
        if is_opponent:
            gems = self.opponent_gems
        else:
            gems = self.my_gems
        board_gems = self.board_gems

        if board_gems[gem] >= 4:
            board_gems[gem] -= 2
            gems[gem] += 2

            if is_opponent:
                self.opponent_gems = gems
            else:
                self.my_gems = gems

            return 0.0

        return -0.5

    def buy_card_from_board(self, card_id, is_opponent=False):
        # determine which tier the card belongs to
        if card_id < 4: # tier 1
            card_array, card_data = self.tier1cards, self.tier1_card_data
        elif card_id < 8:  # tier 2 
            card_array, card_data = self.tier2cards, self.tier2_card_data
            card_id -= 4  
        else:  # tier 3 
            card_array, card_data = self.tier3cards, self.tier3_card_data
            card_id -= 8
        revealed_cards = np.where(card_array == 1)[0]
        actual_card_index = revealed_cards[card_id] # index of the card the agent is trying to buy
        card_data = card_data[actual_card_index] # data of the card the agent is trying to buy

        if not self.can_afford(card_data, is_opponent): # can the agent afford the card?
            return -0.5

        self.pay_for_card(card_data, is_opponent)  # pay for the card
        
        if is_opponent:
            card_array[card_id] = 4  # Mark as bought by opponent
            self.opponent_score += card_data[0]
            self.opponent_bonuses[card_data[6]] += 1
        else:
            card_array[card_id] = 5  # Mark as bought by player
            self.my_score += card_data[0]
            self.my_bonuses[card_data[6]] += 1

        self.reveal_cards(card_array)
        self.check_nobles(is_opponent)
        return 3 + card_data[0]*3

    def reserve_card(self, card_id, is_opponent=False):
        # determine which tier the card belongs to
        if card_id < 4: # tier 1
            card_array, card_data = self.tier1cards, self.tier1_card_data
        elif card_id < 8:  # tier 2 
            card_array, card_data = self.tier2cards, self.tier2_card_data
            card_id -= 4  
        else:  # tier 3 
            card_array, card_data = self.tier3cards, self.tier3_card_data
            card_id -= 8

        revealed_cards = np.where(card_array == 1)[0]
        actual_card_index = revealed_cards[card_id] # index of the card the agent is trying to buy
        card_data = card_data[actual_card_index] # data of the card the agent is trying to buy

        if is_opponent:
            card_array[card_id] = 2 # reserved by opponent
        else:
            card_array[card_id] == 3 # reserved by player

        if self.board_gems[5] > 0: # gold gem for reserving a card, if available
            self.board_gems[5] -= 1
            if is_opponent:
                self.opponent_gems[5] += 1  # Opponent receives a gold gem
            else:
                self.my_gems[5] += 1  # Player receives a gold gem

        self.reveal_cards(card_array)

        return 2.0

    def buy_reserved_card(self, slot, is_opponent=False):

        reserved_cards = []  # List to store the indices of reserved cards

        if is_opponent: # search for reserved cards by the opponent (value = 2)
            for idx in range(len(self.tier1cards)):
                if self.tier1cards[idx] == 2:
                    reserved_cards.append(('tier1', idx))
            for idx in range(len(self.tier2cards)):
                if self.tier2cards[idx] == 2:
                    reserved_cards.append(('tier2', idx))
            for idx in range(len(self.tier3cards)):
                if self.tier3cards[idx] == 2:
                    reserved_cards.append(('tier3', idx))

        else: # search for reserved cards by the player (value = 3)
            for idx in range(len(self.tier1cards)):
                if self.tier1cards[idx] == 3:  
                    reserved_cards.append(('tier1', idx))
            for idx in range(len(self.tier2cards)):
                if self.tier2cards[idx] == 3:  
                    reserved_cards.append(('tier2', idx))
            for idx in range(len(self.tier3cards)):
                if self.tier3cards[idx] == 3: 
                    reserved_cards.append(('tier3', idx))

        if len(reserved_cards) <= slot:
            return -0.5
        tier, card_id = reserved_cards[slot] # get the tier and card id of the reserved card based on the slot

        # determine which tier and card data to use
        if tier == 'tier1':
            card_array = self.tier1cards
            card_data = self.tier1_card_data[card_id]
        elif tier == 'tier2':
            card_array = self.tier2cards
            card_data = self.tier2_card_data[card_id]
        else:
            card_array = self.tier3cards
            card_data = self.tier3_card_data[card_id]

        if not self.can_afford(card_data, is_opponent):
            return -0.5 

        self.pay_for_card(card_data, is_opponent)

        if is_opponent:            
            card_array[card_id] = 4 
            self.opponent_score += card_data[0]
            self.opponent_bonuses[card_data[6]] += 1
        else:
            card_array[card_id] = 5 
            self.my_score += card_data[0]
            self.my_bonuses[card_data[6]] += 1

        # Check if the agent has gained a noble after buying the card
        self.check_nobles(is_opponent)

        return 3.0 + card_data[0]*3  # Return reward for buying the card

    def can_afford(self, card_data, is_opponent=False):
        if is_opponent:
            gems = self.opponent_gems
            bonuses = self.opponent_bonuses
        else:
            gems = self.my_gems
            bonuses = self.my_bonuses

        for i in range(5):
            needed = max(0, card_data[1 + i] - bonuses[i])
            if needed > gems[i] + gems[5]:  # Include gold (index 5)
                return False
        return True

    def pay_for_card(self, card_data, is_opponent=False):
        if is_opponent:
            gems = self.opponent_gems
            bonuses = self.opponent_bonuses
        else:
            gems = self.my_gems
            bonuses = self.my_bonuses

        for i in range(5):  

            needed = max(0, card_data[1 + i] - bonuses[i]) # how many gems are needed after bonuses
            to_pay = min(gems[i], needed) 

            gems[i] -= to_pay
            self.board_gems[i] += to_pay
            needed -= to_pay

            if needed > 0: # if there are still gems needed, use gold gems
                gems[5] -= needed
                self.board_gems[5] += needed

        if is_opponent:
            self.opponent_gems = gems
        else:
            self.my_gems = gems

    def check_nobles(self, is_opponent=False):
        if is_opponent:
            bonuses = self.opponent_bonuses
        else:
            bonuses = self.my_bonuses

        for idx in range(len(self.nobles)):
            # if the noble is in play (status == 1), check if the agent has the required bonuses
            if self.nobles[idx] == 1:
                if all(bonuses[i] >= self.nobles_data[idx][i] for i in range(5)): # claim the noble
                    if self.render_mode:
                        print(f"Claimed noble {idx}!")

                    if is_opponent:
                        self.nobles[idx] = 2  # Mark noble as claimed by the opponent
                        self.opponent_score += 3  # Add points to the opponent's score
                    else:
                        self.nobles[idx] = 3  # Mark noble as claimed by the player
                        self.my_score += 3  # Add points to the player's score

    def handle_gem_discard(self, gems, total_gems, is_opponent=False):
        if is_opponent:
            gems = self.opponent_gems
        else:
            gems = self.my_gems

        excess = total_gems - 10
        for i in range(6):
            to_discard = min(excess, gems[i])
            gems[i] -= to_discard
            self.board_gems[i] += to_discard
            excess -= to_discard
            if excess == 0:
                break

        if is_opponent:
            self.opponent_gems = gems
        else:
            self.my_gems = gems

    def decode_action(self, action):
        if action < 10:
            return {"type": "take_3", "combo_id": action}
        elif action < 15:
            return {"type": "take_2", "gem": action - 10}
        elif action < 27:
            return {"type": "buy_board", "card_id": action - 15}
        elif action < 39:
            return {"type": "reserve", "card_id": action - 27}
        elif action < 42:
            return {"type": "buy_reserved", "slot": action - 39}
        return {"type": "invalid"}

    def step(self, action, is_opponent=False):
        if self.done:
            return self.observation(), 0.0, True, self.info

        action_info = self.decode_action(action)

        if action_info["type"] == "take_3":
            reward = self.take_3_different_gems(action_info["combo_id"], is_opponent=is_opponent)
        elif action_info["type"] == "take_2":
            reward = self.take_2_same_gem(action_info["gem"], is_opponent=is_opponent)
        elif action_info["type"] == "buy_board":
            reward = self.buy_card_from_board(action_info["card_id"], is_opponent=is_opponent)
        elif action_info["type"] == "reserve":
            reward = self.reserve_card(action_info["card_id"], is_opponent=is_opponent)
        elif action_info["type"] == "buy_reserved":
            reward = self.buy_reserved_card(action_info["slot"], is_opponent=is_opponent)

        # If it's the player's turn, check if they need to discard excess gems
        if not is_opponent and self.my_gems.sum() > 10:
            self.handle_gem_discard(self.my_gems, self.my_gems.sum(), is_opponent=is_opponent)

        # If it's the opponent's turn, check if they need to discard excess gems
        if is_opponent and self.opponent_gems.sum() > 10:
            self.handle_gem_discard(self.opponent_gems, self.opponent_gems.sum(), is_opponent=is_opponent)

        self.step_count += 1

        terminated = self.my_score >= 15 or self.opponent_score >= 15
        truncated = self.step_count >= 200
        return self.observation(), reward, terminated, truncated, self.info

    def render(self):
        pass

    def close(self):
        pass

# Register the environment
register(
    id="Splendor-v0",
    entry_point="splendor_env:SplendorEnv",
)

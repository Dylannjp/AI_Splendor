import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.registration import register
import itertools

from game_logic.game_data import tier1_card_data, tier2_card_data, tier3_card_data, nobles_data

GEM_TYPES = ['black', 'white', 'red', 'blue', 'green', 'gold']
NUM_TOKENS = 6
MAX_TOKENS = 10

class SplendorEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # setup cards
        self.tier1_card_data = tier1_card_data
        self.tier2_card_data = tier2_card_data
        self.tier3_card_data = tier3_card_data
        
        self.tier1cards = np.zeros(40, dtype=np.int32)
        self.tier2cards = np.zeros(30, dtype=np.int32)
        self.tier3cards = np.zeros(20, dtype=np.int32)

        # setup board to make card mechanics easier
        self.board1 = []
        self.board2 = []
        self.board3 = []
        self.update_board()
        self.reserved = []

        # setup nobles
        self.nobles_data = nobles_data
        self.nobles = np.zeros(10, dtype=np.int32)
        noble_ids = np.random.choice(10, 3, replace=False)
        self.nobles[noble_ids] = 1

        # setup gems
        self.board_gems = np.array([4, 4, 4, 4, 4, 5], dtype=np.int32)
        self.my_gems = np.zeros(6, dtype=np.int32)
        self.opponent_gems = np.zeros(6, dtype=np.int32)

        # setup scores and bonuses
        self.my_score = 0
        self.opponent_score = 0
        self.my_bonuses = np.zeros(5, dtype=np.int32)
        self.opponent_bonuses = np.zeros(5, dtype=np.int32)

        # setup helper variables
        self.binary_lists_with_sum_of_3 = list(filter(lambda binary_list: sum(binary_list) == 3, itertools.product([0, 1], repeat=5)))
        self.binary_lists_with_sum_of_2 = list(filter(lambda binary_list: sum(binary_list) == 2, itertools.product([0, 1], repeat=5)))

        # setup actions
        self.all_actions = []
        self.create_all_actions()
        self.valid_actions=[0]*len(self.all_actions)

        self.turns = 0
        self.max_turns = 200

        self.observation_space = spaces.Dict({
            "tier1cards": spaces.Box(low=0, high=5, shape=(40,), dtype=np.int32),
            "tier2cards": spaces.Box(low=0, high=5, shape=(30,), dtype=np.int32),
            "tier3cards": spaces.Box(low=0, high=5, shape=(20,), dtype=np.int32),
            # cards: 0 = in deck, 1 = in play, 2 = reserved by opponent, 3 = reserved by me, 4 = bought by opponent, 5 = bought by me
            "nobles": spaces.Box(low=0, high=3, shape=(10,), dtype=np.int32),
            # nobles: 0 = not in play, 1 = in play, 2 = acquired by opponent, 3 = acquired by me
            "board_gems": spaces.Box(low=0, high=5, shape=(6,), dtype=np.int32),
            # board_gems: 0 = black, 1 = white, 2 = red, 3 = blue, 4 = green, 5 = gold
            "my_gems": spaces.Box(low=0, high=5, shape=(6,), dtype=np.int32), # the amount of gems the agent has
            "opponent_gems": spaces.Box(low=0, high=5, shape=(6,), dtype=np.int32), # the amount of gems the opponent agent has
            "my_score": spaces.Discrete(24), # the amount of points the agent has, max is 24
            "opponent_score": spaces.Discrete(24), # the amount of points the opponent agent has, max is 24
            "valid_actions": spaces.Discrete(42)
        })
        self.action_space = spaces.Discrete()
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # setup cards
        self.tier1cards = np.zeros(40, dtype=np.int32)
        self.tier2cards = np.zeros(30, dtype=np.int32)
        self.tier3cards = np.zeros(20, dtype=np.int32)

        # setup board to make card mechanics easier
        self.board1 = []
        self.board2 = []
        self.board3 = []
        self.update_board()

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
        return reveal_indices
    
    def update_board(self):
        while len(self.board1) < 4:
              if len(self.tier1cards == 0):
                  break
              index = self.reveal_cards(self.tier1cards)
              self.board1.append(self.tier1cards[index],index)
        while len(self.board2) < 4:
              if len(self.tier2cards == 0):
                  break
              index = self.reveal_cards(self.tier2cards)
              self.board2.append(self.tier2cards[index],index)
        while len(self.board3) < 4:
              if len(self.tier3cards == 0):
                  break
              index = self.reveal_cards(self.tier3cards)
              self.board3.append(self.tier3cards[index],index)

    def get_takable_gems(self): # get all possible combinations of gems that can be taken
        doable_3gems = [] # possible combinations for 3 gems
        for comb in self.binary_lists_with_sum_of_3:
            if  min([self.board_gems[i] - comb[i] for i in range(len(comb))]) >=0:
                doable_3gems.append(comb)

        doable_2gems = [] # possible combinations for 2 gems
        for comb in self.binary_lists_with_sum_of_2:
            if  min([self.board_gems[i] - comb[i] for i in range(len(comb))]) >=0:
                doable_2gems.append(comb)    

        doable_1gems = [] # possible combinations for 1 gem
        for comb in [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]]:
            if  min([self.board_gems[i] - comb[i] for i in range(len(comb))]) >=0:
                doable_1gems.append(comb)      

        gem_buy = []
        
        gem_buy.extend(doable_3gems)
        
        if sum(1 for g in self.board_gems if g == 0) == 3: # if there are 3 gems with 0 tokens, meaning you can only take 2 gem types
            gem_buy.extend(doable_2gems)
        
        if sum(1 for g in self.board_gems if g == 0) == 4: # if there are 4 gems with 0 tokens, meaning you can only take 1 gem type
            gem_buy.extend(doable_1gems)

        for x in range(len(self.my_gems[:5])): # if the supply of gems is at least 4 we can take 2 of the same gems.
                list_empty = [0,0,0,0,0]
                if self.board_gems[x] > 3:
                    list_empty[x] = 2
                    if list_empty != [0,0,0,0,0]:
                        gem_buy.append(list_empty)

        return [('take', x) for x in gem_buy]

    def get_purchasable_cards(self):
        card_buy=[]
        tier_card_data = [self.tier1_card_data, self.tier2_card_data, self.tier3_card_data]
        for board_index, board in enumerate([self.board_1, self.board_2, self.board_3]):
            card.data = tier_card_data[board_index]
            for card, index in board:
                over = 0
                for i in range(4): # iterate through each gem
                    current_card = card.data[index]
                    over += max(0, current_card[i] - self.my_bonuses[i])
                if over <= self.my_gems[5]: # can the gold gem cover the excess
                    card_buy.append(board_index, index) # we append a tuple containing board tier and index of the card
        return  [('buy card', x) for x in card_buy]

    def get_reservable_cards(self):
        reserve_cards = []
        if len(self.player_reserved_cards) < 3: # if the player has less than 3 reserved cards
            for board_index, board in enumerate([self.board_1, self.board_2, self.board_3]):
                    for card, index in board:
                            reserve_cards.append(board_index, index) # we append a tuple containing board tier and index of the card
        return [('reserve', x) for x in reserve_cards] 
    
    def get_purchasable_reserved(self):
        buyable_reserved_cards = []
        tier_card_data = [self.tier1_card_data, self.tier2_card_data, self.tier3_card_data]
        for card, index, tier in self.reserved:
            card_data = tier_card_data[tier]
            current_card = card_data[index]
            over = 0
            for i in range(4):
                over += max(0, current_card[i] - self.my_bonuses[i])

            if over <= self.gems[5]: # can the gold gem cover the excess
                buyable_reserved_cards.append(self.player_reserved_cards.index(card)) # returns index in reserved cards 0-2
        return [('buy reserved', x) for x in buyable_reserved_cards]

    def get_discardable_gems(self):
        discardable_gems = []
        if sum(self.my_gems) == 13:
            discard_3gems = [] # possible combinations to discard 3 gems
            for comb in self.binary_lists_with_sum_of_3:
                if  min([self.my_gems[i] - comb[i] for i in range(len(comb))]) >=0:
                    discard_3gems.append(comb)
            discardable_gems.extend(discard_3gems)
        if sum(self.my_gems) == 12:
            discard_2gems = [] # possible combinations to discard 2 gems
            for comb in self.binary_lists_with_sum_of_2:
                if  min([self.my_gems[i] - comb[i] for i in range(len(comb))]) >=0:
                    discard_2gems.append(comb)
            discardable_gems.extend(discard_2gems)
        if sum(self.my_gems) == 11:
            discard_1gem = [] # possible combinations to discard 1 gem
            for comb in [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]]:
                if  min([self.my_gems[i] - comb[i] for i in range(len(comb))]) >=0:
                    discard_1gem.append(comb) 
            discardable_gems.extend(discard_1gem)     

        return discardable_gems

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

    def discard_gems(self):
        discardable_gems = self.get_discardable_gems()
        excess = self.my_gems - 10
        for i in range(6):
            to_discard = min(excess, gems[i])
            gems[i] -= to_discard
            self.board_gems[i] += to_discard
            excess -= to_discard
            if excess == 0:
                break

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

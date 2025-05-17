import gymnasium as gym
from gymnasium import spaces
import numpy as np
from itertools import combinations
from gymnasium.envs.registration import register

from game_data import tier1_card_data, tier2_card_data, tier3_card_data, nobles_data

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

    def reveal_cards(self, card_array):
        deck_indices = np.where(card_array == 0)[0]
        reveal_indices = np.random.choice(deck_indices, 1, replace=False)
        card_array[reveal_indices] = 1

    def take_3_different_gems(self, combo_id, is_opponent=False):
        """
        The player or opponent takes 3 different gems from the board.
        Args:
        - combo_id: the combination index to pick gems
        - is_opponent: flag to indicate if the action is for the opponent
        """
        # Select correct gem pool based on the player or opponent
        if is_opponent:
            gems = self.opponent_gems
        else:
            gems = self.my_gems
        board_gems = self.board_gems

        available_colors = [i for i in range(5) if board_gems[i] > 0]
        if len(available_colors) < 3:
            if combo_id >= len(available_colors):
                return -1.0
            gem = available_colors[combo_id]
            board_gems[gem] -= 1
            gems[gem] += 1
            return 0.0

        combos = list(combinations(available_colors, 3))
        if combo_id >= len(combos):
            return -1.0
        for gem in combos[combo_id]:
            board_gems[gem] -= 1
            gems[gem] += 1

        # Update the correct gem pool (either player or opponent)
        if is_opponent:
            self.opponent_gems = gems
        else:
            self.my_gems = gems

        return 0.0

    def take_2_same_gem(self, gem, is_opponent=False):
        """
        The player or opponent takes 2 of the same gem from the board.
        Args:
        - gem: the gem type being taken
        - is_opponent: flag to indicate if the action is for the opponent
        """
        if is_opponent:
            gems = self.opponent_gems
        else:
            gems = self.my_gems
        board_gems = self.board_gems

        if board_gems[gem] >= 4:
            board_gems[gem] -= 2
            gems[gem] += 2

            # Update the correct gem pool (either player or opponent)
            if is_opponent:
                self.opponent_gems = gems
            else:
                self.my_gems = gems

            return 0.0

        return -1.0

    def buy_card_from_board(self, card_id, is_opponent=False):
        if card_id < 40:
            card_array, card_data = self.tier1cards, self.tier1_card_data[card_id]
        elif card_id < 70:
            card_array, card_data = self.tier2cards, self.tier2_card_data[card_id - 40]
            card_id -= 40
        else:
            card_array, card_data = self.tier3cards, self.tier3_card_data[card_id - 70]
            card_id -= 70

        if card_array[card_id] != 1 and card_array[card_id] != 4:  # 4 means bought by opponent
            return -1.0

        if not self.can_afford(card_data, is_opponent):
            return -1.0

        self.pay_for_card(card_data, is_opponent)  # pay for the card
        
        # Update the card status to bought by the respective player
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
        return 1.0 + card_data[0]

    def reserve_card(self, card_id, is_opponent=False):
        if card_id < 40:
            card_array = self.tier1cards
        elif card_id < 70:
            card_array = self.tier2cards
            card_id -= 40
        else:
            card_array = self.tier3cards
            card_id -= 70

        # Check if the card is available to be reserved (must be in play, not reserved or bought)
        if card_array[card_id] != 1:  # 1 means the card is in play and not reserved or bought
            return -1.0

        # Mark the card as reserved by the player or opponent
        card_array[card_id] = 3 if not is_opponent else 2  # 3 = reserved by player, 2 = reserved by opponent

        # Give the player or opponent a gold gem for reserving the card
        if self.board_gems[5] > 0:
            self.board_gems[5] -= 1
            if is_opponent:
                self.opponent_gems[5] += 1  # Opponent receives a gold gem
            else:
                self.my_gems[5] += 1  # Player receives a gold gem

        return 0.5

    def buy_reserved_card(self, card_id, is_opponent=False):
        # Determine which tier the card belongs to
        if card_id < 40:
            card_array = self.tier1cards
            card_data = self.tier1_card_data[card_id]
        elif card_id < 70:
            card_array = self.tier2cards
            card_data = self.tier2_card_data[card_id - 40]
            card_id -= 40
        else:
            card_array = self.tier3cards
            card_data = self.tier3_card_data[card_id - 70]
            card_id -= 70

        if is_opponent: # check if card has been reserved properly
            if card_array[card_id] != 2:
                return -1.0
        else:
            if card_array[card_id] != 3:
                return -1.0

        # Ensure the agent (player or opponent) can afford the card
        if not self.can_afford(card_data, is_opponent):
            return -1.0  # Not enough gems to afford the card

        # Pay for the reserved card (whether player or opponent)
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

        return 1.0 + card_data[0]  # Return reward for buying the card

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
            # Calculate how many gems of the current type are needed, considering the bonuses
            needed = max(0, card_data[1 + i] - bonuses[i])
            to_pay = min(gems[i], needed)  # Pay as much as possible from the available gems

            # Deduct gems from the agent's collection and add them to the board
            gems[i] -= to_pay
            self.board_gems[i] += to_pay
            needed -= to_pay  # Reduce the remaining needed gems

            # If there are still gems left to pay, use gold (gem index 5)
            if needed > 0:
                gems[5] -= needed  # Pay using gold
                self.board_gems[5] += needed  # Add the gold back to the board's gem supply

        # After the transaction, check if the agent's gems need to be updated
        if is_opponent:
            self.opponent_gems = gems  # Update the opponent's gem count
        else:
            self.my_gems = gems  # Update the player's gem count

    def check_nobles(self, is_opponent=False):
        # Determine which bonuses to check based on whether it's the opponent or the player
        if is_opponent:
            bonuses = self.opponent_bonuses
        else:
            bonuses = self.my_bonuses

        # Loop through each noble
        for idx, status in enumerate(self.nobles):
            # If the noble is in play (status == 1), check if the agent has the required bonuses
            if status == 1 and all(bonuses[i] >= self.nobles_data[idx][i] for i in range(5)):
                # Noble has been claimed
                if self.render_mode:
                    print(f"Claimed noble {idx}!")

                if is_opponent:
                    self.nobles[idx] = 2  # Mark noble as claimed by the opponent
                    self.opponent_score += 3  # Add points to the opponent's score
                else:
                    self.nobles[idx] = 3  # Mark noble as claimed by the player
                    self.my_score += 3  # Add points to the player's score

    def handle_gem_discard(self, gems, total_gems, is_opponent=False):
        """
        Handles the discarding of excess gems if total gems exceed the limit (10).
        Args:
        - gems: the gem pool for either the player or the opponent (self.my_gems or self.opponent_gems)
        - total_gems: the total number of gems the agent (player or opponent) holds
        """
        excess = total_gems - 10
        for i in range(6):
            to_discard = min(excess, gems[i])
            gems[i] -= to_discard
            self.board_gems[i] += to_discard
            excess -= to_discard
            if excess == 0:
                break

        # Update the correct gem pool (either player or opponent)
        if is_opponent:
            self.opponent_gems = gems  # Update the player's gems
        else:
            self.my_gems = gems  # Update the opponent's gems

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
        reward = -1.0

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

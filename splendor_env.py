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
            "nobles": spaces.MultiDiscrete([4] * 10),
            "board_gems": spaces.Box(low=0, high=5, shape=(6,), dtype=np.int32),
            "agent_gems": spaces.Box(low=0, high=10, shape=(6,), dtype=np.int32),
            "opponent_gems": spaces.Box(low=0, high=10, shape=(6,), dtype=np.int32),
            "my_score": spaces.Discrete(23),
            "opponent_score": spaces.Discrete(23),
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
        self._reveal_cards(self.tier1cards, 4)
        self._reveal_cards(self.tier2cards, 4)
        self._reveal_cards(self.tier3cards, 4)

        self.nobles = np.zeros(10, dtype=np.int32)
        noble_ids = np.random.choice(10, 3, replace=False)
        self.nobles[noble_ids] = 1

        self.board_gems = np.array([4, 4, 4, 4, 4, 5], dtype=np.int32)
        self.agent_gems = np.zeros(6, dtype=np.int32)
        self.opponent_gems = np.zeros(6, dtype=np.int32)

        self.my_score = 0
        self.opponent_score = 0
        self.my_bonuses = np.zeros(5, dtype=np.int32)
        self.reserved = []

        self.done = False
        self.info = {}
        self.step_count = 0
        return self.observation(), self.info

    def _reveal_cards(self, card_array, num_to_reveal):
        deck_indices = np.where(card_array == 0)[0]
        reveal_indices = np.random.choice(deck_indices, num_to_reveal, replace=False)
        card_array[reveal_indices] = 1

    def observation(self):
        return {
            "tier1cards": self.tier1cards.copy(),
            "tier2cards": self.tier2cards.copy(),
            "tier3cards": self.tier3cards.copy(),
            "nobles": self.nobles.copy(),
            "board_gems": self.board_gems.copy(),
            "agent_gems": self.agent_gems.copy(),
            "opponent_gems": self.opponent_gems.copy(),
            "my_score": self.my_score,
            "opponent_score": self.opponent_score,
        }

    def _take_3_different_gems(self, combo_id):
        available_colors = [i for i in range(5) if self.board_gems[i] > 0]
        if len(available_colors) < 3:
            if combo_id >= len(available_colors):
                return -1.0
            gem = available_colors[combo_id]
            self.board_gems[gem] -= 1
            self.agent_gems[gem] += 1
            return 0.2

        combos = list(combinations(available_colors, 3))
        if combo_id >= len(combos):
            return -1.0
        for gem in combos[combo_id]:
            self.board_gems[gem] -= 1
            self.agent_gems[gem] += 1
        return 0.2

    def _take_2_same_gem(self, gem):
        if self.board_gems[gem] >= 4:
            self.board_gems[gem] -= 2
            self.agent_gems[gem] += 2
            return 0.2
        return -1.0

    def _buy_card_from_board(self, card_id):
        if card_id < 40:
            card_array, card_data = self.tier1cards, self.tier1_card_data[card_id]
        elif card_id < 70:
            card_array, card_data = self.tier2cards, self.tier2_card_data[card_id - 40]
            card_id -= 40
        else:
            card_array, card_data = self.tier3cards, self.tier3_card_data[card_id - 70]
            card_id -= 70

        if card_array[card_id] != 1 or not self._can_afford(card_data):
            return -1.0

        self._pay_for_card(card_data)
        card_array[card_id] = 5
        self.my_score += card_data[0]
        self.my_bonuses[card_data[6]] += 1
        self._check_nobles()
        return 1.0

    def _reserve_card(self, card_id):
        if len(self.reserved) >= 3:
            return -1.0

        if card_id < 40:
            card_array = self.tier1cards
        elif card_id < 70:
            card_array = self.tier2cards
            card_id -= 40
        else:
            card_array = self.tier3cards
            card_id -= 70

        if card_array[card_id] != 1:
            return -1.0

        card_array[card_id] = 3
        self.reserved.append((card_array, card_id))

        if self.board_gems[5] > 0:
            self.board_gems[5] -= 1
            self.agent_gems[5] += 1

        return 0.5

    def _buy_reserved_card(self, slot):
        if slot >= len(self.reserved):
            return -1.0

        card_array, card_id = self.reserved[slot]
        if card_array[card_id] != 3:
            return -1.0

        if card_array is self.tier1cards:
            card_data = self.tier1_card_data[card_id]
        elif card_array is self.tier2cards:
            card_data = self.tier2_card_data[card_id]
        else:
            card_data = self.tier3_card_data[card_id]

        if not self._can_afford(card_data):
            return -1.0

        self._pay_for_card(card_data)
        card_array[card_id] = 5
        self.my_score += card_data[0]
        self.my_bonuses[card_data[6]] += 1
        self.reserved.pop(slot)
        self._check_nobles()
        return 1.0

    def _can_afford(self, card_data):
        for i in range(5):
            needed = max(0, card_data[1 + i] - self.my_bonuses[i])
            if needed > self.agent_gems[i] + self.agent_gems[5]:
                return False
        return True

    def _pay_for_card(self, card_data):
        for i in range(5):
            needed = max(0, card_data[1 + i] - self.my_bonuses[i])
            to_pay = min(self.agent_gems[i], needed)
            self.agent_gems[i] -= to_pay
            self.board_gems[i] += to_pay
            needed -= to_pay
            if needed > 0:
                self.agent_gems[5] -= needed
                self.board_gems[5] += needed

    def _check_nobles(self):
        for idx, status in enumerate(self.nobles):
            if status == 1 and all(self.my_bonuses[i] >= self.nobles_data[idx][i] for i in range(5)):
                self.nobles[idx] = 3
                self.my_score += 3
                if self.render_mode:
                    print(f"Claimed noble {idx}!")

    def _handle_gem_discard(self, total_gems):
        # Simplified logic for now; later add discard interaction
        excess = total_gems - 10
        for i in range(6):
            to_discard = min(excess, self.agent_gems[i])
            self.agent_gems[i] -= to_discard
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

    def step(self, action):
        if self.done:
            return self.observation(), 0.0, True, self.info

        action_info = self.decode_action(action)
        reward = -1.0

        if action_info["type"] == "take_3":
            reward = self._take_3_different_gems(action_info["combo_id"])
        elif action_info["type"] == "take_2":
            reward = self._take_2_same_gem(action_info["gem"])
        elif action_info["type"] == "buy_board":
            reward = self._buy_card_from_board(action_info["card_id"])
        elif action_info["type"] == "reserve":
            reward = self._reserve_card(action_info["card_id"])
        elif action_info["type"] == "buy_reserved":
            reward = self._buy_reserved_card(action_info["slot"])

        if self.agent_gems.sum() > 10:
            self._handle_gem_discard(self.agent_gems.sum())

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

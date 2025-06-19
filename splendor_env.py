from gymnasium import spaces
import gymnasium as gym
import numpy as np
import random

from game_logic.splendor_game import SplendorGame

class SplendorGymEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, opponent_policy=None):
        super().__init__()
        self.num_players = 2
        self.MAX_GEMS = SplendorGame.MAX_GEMS
        
        self.card_templates = {}
        self.card_index_map = {}
        self._build_card_templates()

        # Simplified observation space
        self.observation_space = spaces.Dict({
            # Cards: 0 = in deck, 1 = in play , 2 = taken or reserved

            "tier1cards": spaces.Box(low=0, high=2, shape=(40,), dtype=np.int8),
            "tier2cards": spaces.Box(low=0, high=2, shape=(30,), dtype=np.int8),
            "tier3cards": spaces.Box(low=0, high=2, shape=(20,), dtype=np.int8),
            
            # Nobles: 0 = not in play, 1 = in play, 2 = claimed
            "nobles":    spaces.Box(low=0, high=2, shape=(3,), dtype=np.int8),

            # Gem counts, 0-4 colored gems, 0-5 gold gems
            "board_gems":    spaces.MultiDiscrete([5]*5 + [6]),
            "my_gems":       spaces.MultiDiscrete([5]*5 + [6]),
            "opp_gems":      spaces.MultiDiscrete([5]*5 + [6]),

            # Bonuses
            "my_bonuses":  spaces.MultiDiscrete([15] * 5),
            "opp_bonuses": spaces.MultiDiscrete([15] * 5),

            # Scores
            "my_score":      spaces.Discrete(25),
            "opp_score":     spaces.Discrete(25),
        })

        # Action space: one of the 67 fixed action tuples
        temp_game = SplendorGame(self.num_players)
        self.all_actions = temp_game.all_actions
        self.action_space = spaces.Discrete(len(self.all_actions)) 

        # How to choose the opponent's move (defaults to random)
        self.opponent_policy = opponent_policy or (lambda game, p: random.choice(game.legal_actions(p)))

        self.game = None

    def _build_card_templates(self):
        """Create a stable list of `(level,bonus,VPs,cost_tuple)` keys + a lookup map."""
        game = SplendorGame(self.num_players)
        # collect every card object from decks + face-up
        for lvl in (0, 1, 2):
            cards = list(game.decks[lvl]) + [c for c in game.board_cards[lvl] if c]
            # sort by (bonus, VPs, cost tuple) to create a stable ordering
            sorted_cards = sorted(cards, key=lambda c: (c.bonus, c.VPs, tuple(c.cost)))
            self.card_templates[lvl] = sorted_cards
            # build reverse map from key → position
            self.card_index_map[lvl] = {
                (c.level, c.bonus, c.VPs, tuple(c.cost)): i
                for i, c in enumerate(sorted_cards)
            }

    def reset(self, seed=None, options=None):
        self.game = SplendorGame(self.num_players)
        if not hasattr(self, "_mask_buf"):
            self.mask_buf = np.zeros(len(self.all_actions), dtype=np.int8)
        mask = self.mask_buf
        mask.fill(0)  # reset mask buffer
        for a in self.game.legal_actions(0):
            mask[self.all_actions.index(a)] = 1   
        return self._make_obs(), {"legal_mask": mask}

    def step(self, action_idx):
        assert 0 <= action_idx < len(self.all_actions), "action_idx must be in range of action space"
        action = self.all_actions[action_idx]

        legal = self.game.legal_actions(0)
        assert action in legal, f"Illegal action {action} at this state!"
        
        self.game.step(action)

        # 2) Opponent plays until it's our turn again or game ends
        while not self.game.game_over and self.game.current_player != 0:
            opp_act = self.opponent_policy(self.game, 1)
            self.game.step(opp_act)

        mask = self.mask_buf
        mask.fill(0)  # reset mask buffer
        for a in self.game.legal_actions(0):
            mask[self.all_actions.index(a)] = 1

        # 3) Build observation, reward, done
        obs = self._make_obs()
        terminated = self.game.game_over
        truncated = False

        if terminated:
            winner = self.game.decide_winner()
            reward = 1.0 if winner == 0 else -1.0
        else:
            reward = 0.0

        return obs, reward, terminated, truncated, {"legal_mask": mask}

    def _make_obs(self):
        """Construct the simplified Dict observation from self.game."""
        o = {
            "tier1cards": np.zeros(40, dtype=np.int8),
            "tier2cards": np.zeros(30, dtype=np.int8),
            "tier3cards": np.zeros(20, dtype=np.int8),
            "nobles"    : np.zeros(3,  dtype=np.int8),
            "board_gems": self.game.board_gems.copy(),
            "my_gems"   : self.game.players[0].gems.copy(),
            "opp_gems"  : self.game.players[1].gems.copy(),
            "my_bonuses": self.game.players[0].bonuses.copy(),
            "opp_bonuses": self.game.players[1].bonuses.copy(),
            "my_score"  : np.int8(self.game.players[0].VPs),
            "opp_score" : np.int8(self.game.players[1].VPs),
        }

        # Nobles: first 3 slots
        for i, n in enumerate(self.game.nobles[:3]):
            o["nobles"][i] = 2 if n is None else 1

        # Cards: mark 1=face-up, 2=reserved or bought
        for lvl, key in ((0, "tier1cards"), (1, "tier2cards"), (2, "tier3cards")):
            # face-up
            for c in (c for c in self.game.board_cards[lvl] if c):
                idx = self.card_index_map[lvl][(c.level, c.bonus, c.VPs, tuple(c.cost))]
                o[key][idx] = 1

            # reserved by either player
            for pid in (0, 1):
                for c in self.game.players[pid].reserved:
                    if c.level == lvl:
                        idx = self.card_index_map[lvl][(c.level, c.bonus, c.VPs, tuple(c.cost))]
                        o[key][idx] = 2

        # Infer “bought” cards: any template not on board or in deck → 2
        for lvl, key in ((0, "tier1cards"), (1, "tier2cards"), (2, "tier3cards")):
            # build set of key-tuples for deck + face-up
            deck_keys  = {(c.level, c.bonus, c.VPs, tuple(c.cost)) for c in self.game.decks[lvl]}
            board_keys = {(c.level, c.bonus, c.VPs, tuple(c.cost)) for c in self.game.board_cards[lvl] if c}

            for tpl_key, idx in self.card_index_map[lvl].items():
                if o[key][idx] == 0 and tpl_key not in deck_keys and tpl_key not in board_keys:
                    # neither in deck nor face-up → must have been bought
                    o[key][idx] = 2

        return o

    def render(self, mode="human"):
        print("=== Splendor ===")
        print(" Board gems:", self.game.board_gems.tolist())
        for i,p in enumerate(self.game.players):
            who = "Player 0" if i==0 else "Player 1"
            print(f" {who} Gems:", p.gems.tolist(),
                  "Bonuses:", p.bonuses.tolist(),
                  "VPs:", p.VPs,
                  "Reserved:", [(c.level, c.VPs) for c in p.reserved])
        print(" Face-up board sizes:", [len(b) for b in self.game.board_cards])
        print(" Deck sizes:", [len(d) for d in self.game.decks])
        print("---------------")
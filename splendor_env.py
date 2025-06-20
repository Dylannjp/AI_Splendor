from gymnasium import spaces
import gymnasium as gym
import numpy as np
import random

from game_logic.splendor_game import SplendorGame, ActionType

def normalize_card_key(card):
    return (
        int(card.level),
        int(card.bonus),
        int(card.VPs),
        tuple(int(x) for x in card.cost)
    )

class SplendorGymEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, opponent_policy=None):
        super().__init__()
        self.num_players = 2    
        self.agent_player = -1
        self.opponent_player = -1
        self.MAX_GEMS = SplendorGame.MAX_GEMS
        
        self.card_templates = {}
        self.card_index_map = {}
        self._build_card_templates()

        self.observation_space = spaces.Dict({
            "tier1cards": spaces.Box(low=0, high=4, shape=(40,), dtype=np.int8),
            "tier2cards": spaces.Box(low=0, high=4, shape=(30,), dtype=np.int8),
            "tier3cards": spaces.Box(low=0, high=4, shape=(20,), dtype=np.int8),
            "nobles":    spaces.Box(low=0, high=2, shape=(3,), dtype=np.int8),
            "board_gems":    spaces.MultiDiscrete([5]*5 + [6]),
            "my_gems":       spaces.MultiDiscrete([5]*5 + [6]),
            "opp_gems":      spaces.MultiDiscrete([5]*5 + [6]),
            "my_bonuses":  spaces.MultiDiscrete([15] * 5),
            "opp_bonuses": spaces.MultiDiscrete([15] * 5),
            "my_score":      spaces.Discrete(25),
            "opp_score":     spaces.Discrete(25),
            "my_reserved_count": spaces.Discrete(4),
            "opp_reserved_count": spaces.Discrete(4),
        })

        temp_game = SplendorGame(self.num_players)
        self.all_actions = temp_game.all_actions
        self.action_space = spaces.Discrete(len(self.all_actions)) 
        self.opponent_policy = opponent_policy or (lambda game, p: random.choice(game.legal_actions(p)))
        self.game = None
        self.obs_cache = None

    def _build_card_templates(self):
        game = SplendorGame(self.num_players)
        for lvl in (0, 1, 2):
            cards = list(game.decks[lvl]) + [c for c in game.board_cards[lvl] if c]
            sorted_cards = sorted(cards, key=lambda c: (c.bonus, c.VPs, tuple(c.cost)))
            self.card_templates[lvl] = sorted_cards
            self.card_index_map[lvl] = {
                normalize_card_key(c): i for i, c in enumerate(sorted_cards)
            }

    def reset(self, seed=None, options=None, agent_player=0):
        self.game = SplendorGame(self.num_players)
        self.agent_player = agent_player
        self.opponent_player = 1 - agent_player
        self.obs_cache = self.make_obs()
        self.mask_buf = np.zeros(len(self.all_actions), dtype=np.int8)
        for a in self.game.legal_actions(self.agent_player):
            self.mask_buf[self.all_actions.index(a)] = 1
        return self.obs_cache.copy(), {"legal_mask": self.mask_buf}

    def step(self, action_idx):
        assert 0 <= action_idx < len(self.all_actions)
        action = self.all_actions[action_idx]
        legal = self.game.legal_actions(self.agent_player)
        assert action in legal

        action = self.action_to_card(action)
        self.game.step(action[:2])
        obs = self.obs_cache.copy()
        self.update_obs_delta(obs, action, self.agent_player)

        while not self.game.game_over and self.game.current_player != self.agent_player:
            opp_act = self.opponent_policy(self.game, self.opponent_player)
            opp_act = self.action_to_card(opp_act)
            self.game.step(opp_act[:2])
            self.update_obs_delta(obs, opp_act, self.opponent_player)

        self.obs_cache = obs.copy()
        self.mask_buf.fill(0)
        for a in self.game.legal_actions(self.agent_player):
            self.mask_buf[self.all_actions.index(a)] = 1

        pass_idx = self.all_actions.index((ActionType.PASS, None))
        if self.mask_buf.sum() == 1 and self.mask_buf[pass_idx] == 1:
            # print("Agent has no legal moves, passing turn.")
            terminated = True
            reward = -1.0
            return self.obs_cache.copy(), reward, terminated, False, {"legal_mask": self.mask_buf}

        terminated = self.game.game_over
        reward = 1.0 if terminated and self.game.decide_winner() == self.agent_player else -1.0 if terminated else 0.0

        return self.obs_cache, reward, terminated, False, {"legal_mask": self.mask_buf}

    def update_obs_delta(self, obs, action, player):
        actor = self.game.players[player]
        is_agent = (player == self.agent_player)

        # action -> action_type, param, card

        # Gems + board always updated
        obs["board_gems"] = self.game.board_gems.copy()
        obs["my_gems" if is_agent else "opp_gems"] = actor.gems.copy()

        if action[0] == ActionType.RESERVE_CARD:
            card = action[2]
            if card:
                tpl_key = normalize_card_key(card)
                idx = self.card_index_map[card.level][tpl_key]
                tier_key = f"tier{card.level+1}cards"
                obs[tier_key][idx] = 2 if is_agent else 3

                # Refresh face-up cards on this tier
                current = {normalize_card_key(c) for c in self.game.board_cards[card.level] if c}
                for i, tpl in enumerate(self.card_index_map[card.level]):
                    if tpl in current:
                        obs[tier_key][i] = 1
                    elif obs[tier_key][i] == 1:
                        obs[tier_key][i] = 0

            # Only reserved count needs update
            obs["my_reserved_count" if is_agent else "opp_reserved_count"] = len(actor.reserved)

        elif action[0] == ActionType.RESERVE_DECK:
            # Hidden card, only reserved count
            obs["my_reserved_count" if is_agent else "opp_reserved_count"] = len(actor.reserved)

        elif action[0] in [ActionType.BUY_BOARD, ActionType.BUY_RESERVE]:
            card = action[2]
            if card:
                tpl_key = normalize_card_key(card)
                idx = self.card_index_map[card.level][tpl_key]
                tier_key = f"tier{card.level+1}cards"
                obs[tier_key][idx] = 4

                # Refresh face-up cards on this tier
                current = {normalize_card_key(c) for c in self.game.board_cards[card.level] if c}
                for i, tpl in enumerate(self.card_index_map[card.level]):
                    if tpl in current:
                        obs[tier_key][i] = 1
                    elif obs[tier_key][i] == 1:
                        obs[tier_key][i] = 0

            if is_agent:
                obs["my_score"] = actor.VPs
                obs["my_bonuses"] = actor.bonuses.copy()
                obs["my_reserved_count"] = len(actor.reserved)
            else:
                obs["opp_score"] = actor.VPs
                obs["opp_bonuses"] = actor.bonuses.copy()
                obs["opp_reserved_count"] = len(actor.reserved)

            # Update nobles
            for i, noble in enumerate(self.game.nobles[:3]):
                obs["nobles"][i] = 2 if noble is None else 1

    def make_obs(self):
        my = self.game.players[self.agent_player]
        opp = self.game.players[self.opponent_player]
        obs = {
            "tier1cards": np.zeros(40, dtype=np.int8),
            "tier2cards": np.zeros(30, dtype=np.int8),
            "tier3cards": np.zeros(20, dtype=np.int8),
            "nobles": np.array([1 if n is not None else 2 for n in self.game.nobles[:3]], dtype=np.int8),
            "board_gems": self.game.board_gems.copy(),
            "my_gems": my.gems.copy(),
            "opp_gems": opp.gems.copy(),
            "my_bonuses": my.bonuses.copy(),
            "opp_bonuses": opp.bonuses.copy(),
            "my_score": my.VPs,
            "opp_score": opp.VPs,
            "my_reserved_count": len(my.reserved),
            "opp_reserved_count": len(opp.reserved),
        }
        for lvl, key in ((0, "tier1cards"), (1, "tier2cards"), (2, "tier3cards")):
            for c in self.game.board_cards[lvl]:
                if c:
                    tpl_key = normalize_card_key(c)
                    idx = self.card_index_map[lvl][tpl_key]
                    obs[key][idx] = 1
        return obs

    def action_to_card(self, action):
        type, param = action
        card = None
        if type == ActionType.BUY_BOARD or type == ActionType.RESERVE_CARD:
            level, idx = param
            card = self.game.board_cards[level][idx] if idx >= 0 else None
        elif type == ActionType.BUY_RESERVE:
            card = self.game.players[self.game.current_player].reserved[param]
        elif type == ActionType.RESERVE_DECK and self.game.decks[param[0]]:
            card = self.game.decks[param[0]][-1]
        return action + (card,)

    def render(self, mode="human"):
        print("=== Splendor ===")
        print(" Board gems:", self.game.board_gems.tolist())
        for i, p in enumerate(self.game.players):
            who = "Player 0" if i == 0 else "Player 1"
            print(f" {who} Gems:", p.gems.tolist(),
                  "Bonuses:", p.bonuses.tolist(),
                  "VPs:", p.VPs,
                  "Reserved:", [(c.level, c.VPs) for c in p.reserved])
        print(" Face-up board sizes:", [len(b) for b in self.game.board_cards])
        print(" Deck sizes:", [len(d) for d in self.game.decks])
        print("---------------")

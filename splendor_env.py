# splendor_env.py

import numpy as np
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.env import AECEnv
from gymnasium import spaces

from game_logic.splendor_game import SplendorGame, ActionType

class SplendorEnv(AECEnv):
    # must use this exact key for CaptureStdoutWrapper
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()  # <— initialize AECEnv internals
        self.num_players = 2

        # an upper‐bound on legal actions/day
        self.max_per_turn = 65

        # force this so CaptureStdoutWrapper doesn’t complain
        self.render_mode = "human"

        # --- agent naming/mapping ---
        self.agents             = [f"player_{i}" for i in range(num_players)]
        self.possible_agents    = list(self.agents)
        self.agent_name_mapping = dict(zip(self.agents, range(num_players)))

        # --- action spaces ---
        self.action_spaces = {
            a: spaces.Discrete(self.max_per_turn)
            for a in self.agents
        }

        # --- observation space ---
        obs = {
            "board_gems":      spaces.Box(0.0, 5.0, (6,), dtype=np.float32),
            "board_cards":     spaces.Box(0.0, 7.0, (3, 4, 7), dtype=np.float32),
            "nobles":          spaces.Box(0.0, 4.0, (num_players + 1, 5), dtype=np.float32),
            "player_gems":     spaces.Box(0.0, 5.0, (num_players, 6), dtype=np.float32),
            "player_bonuses":  spaces.Box(0.0, 7.0, (num_players, 5), dtype=np.float32),
            "player_reserved": spaces.Box(0.0, 7.0, (num_players, 3, 7), dtype=np.float32),
            "current_player":  spaces.Discrete(num_players),
            "legal_mask":      spaces.MultiBinary(self.max_per_turn),
        }
        self.observation_spaces = {a: spaces.Dict(obs) for a in self.agents}

        # placeholders
        self.game = None


    def reset(self, seed=None, return_info=False, options=None):

        # 2) now build our game, agent selector, and zero out everything
        self.game            = SplendorGame(self.num_players)
        self._selector       = agent_selector(self.agents)
        self.agent_selection = self._selector.next()

        self.rewards             = {a: 0. for a in self.agents}
        self.terminations        = {a: False for a in self.agents}
        self.truncations         = {a: False for a in self.agents}
        self.infos               = {a: {}   for a in self.agents}
        self._cumulative_rewards = {a: 0. for a in self.agents}  # <— dict, not method!

        first_agent = self.agents[self.game.current_player]
        self.agent_selection = first_agent

        # and return our very first observation
        ob = self.observe(self.agent_selection)
        return (ob, self.infos) if return_info else ob


    def step(self, action):
        agent = self.agent_selection
        idx   = self.agent_name_mapping[agent]
        legal = self.game.legal_actions(idx)
        
        if action is not None and action < len(legal):
            self.game.step(legal[action])

        # now **sync** env’s agent_selection to the game
        next_agent         = self.agents[self.game.current_player]
        self.agent_selection = next_agent

        # check terminal
        done = any(p.VPs >= 15 for p in self.game.players)
        if done:
            winner = max(range(self.num_players),
                         key=lambda i: self.game.players[i].VPs)
            for a in self.agents:
                self.terminations[a] = True
                self.truncations[a]  = False
                self.rewards[a] = 1.0 if self.agent_name_mapping[a] == winner else -1.0

        # let AECEnv push rewards through
        self._accumulate_rewards()
    

    def observe(self, agent):
        idx = self.agent_name_mapping[agent]
        gp  = self.game

        # 1) gems
        board_gems = gp.board_gems.copy().astype(np.float32)

        # 2) board cards
        board_cards = np.zeros((3, 4, 7), dtype=np.float32)
        for lvl in range(3):
            for slot, card in enumerate(gp.board_cards[lvl]):
                board_cards[lvl, slot] = np.array([
                    *card.cost, card.VPs, card.bonus
                ], dtype=np.float32)

        # 3) nobles
        full_nobles = np.zeros((self.num_players+1, 5), dtype=np.float32)
        for i, n in enumerate(gp.nobles):
            full_nobles[i] = n.requirement
        nobles = full_nobles

        # 4) player gems & bonuses
        player_gems    = np.stack([p.gems    for p in gp.players], axis=0).astype(np.float32)
        player_bonuses = np.stack([p.bonuses for p in gp.players], axis=0).astype(np.float32)

        # 5) reserved
        player_reserved = np.zeros((self.num_players, 3, 7), dtype=np.float32)
        for pi, p in enumerate(gp.players):
            for ri, card in enumerate(p.reserved[:3]):
                player_reserved[pi, ri] = np.array([
                    *card.cost, card.VPs, card.bonus
                ], dtype=np.float32)

        # 6) legal mask
        mask = np.zeros(self.max_per_turn, dtype=np.uint8)
        mask[: len(gp.legal_actions(idx))] = 1

        return {
        "board_gems":      board_gems,
        "board_cards":     board_cards,
        "nobles":          nobles,
        "player_gems":     player_gems,
        "player_bonuses":  player_bonuses,
        "player_reserved": player_reserved,
        "current_player":  np.array(idx, dtype=np.int32),
        "legal_mask":      mask,
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]
    def action_space(self, agent):
        return self.action_spaces[agent]

    def render(self, mode="human"):
        print(self.game)

    def close(self):
        pass


import numpy as np
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.env import AECEnv
from pettingzoo.test import api_test # Keep commented for training
from gymnasium import spaces

from game_logic.splendor_game import SplendorGame

class SplendorEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "splendor_v0"}

    def __init__(self, render_mode=None):
        super().__init__()
        self.num_players = 2
        self.max_per_turn = 66
        self.render_mode = render_mode

        self.agents = [f"player_{i}" for i in range(self.num_players)]
        self.possible_agents = list(self.agents)
        self.agent_name_mapping = dict(zip(self.agents, range(self.num_players)))

        # Define observation space dimensions
        total_dim = 6 + (3 * 4 * 7) + ((self.num_players + 1) * 5) + (2 * 6) + (2 * 5) + (2 * 3 * 7) + 1
        obs_space = spaces.Box(0.0, 23.0, (total_dim,), dtype=np.float32)
        mask_space = spaces.MultiBinary(self.max_per_turn)

        self.action_spaces = {a: spaces.Discrete(self.max_per_turn) for a in self.agents}
        self.observation_spaces = {
            a: spaces.Dict({
                "observation": obs_space,
                "action_mask": mask_space
            }) for a in self.agents
        }

        # Get the fixed action list (requires a game instance)
        temp_game = SplendorGame(self.num_players)
        self.all_actions = temp_game.all_actions
        self.game = None

        self._selector = agent_selector(self.agents)
        self.rewards = {a: 0. for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._cumulative_rewards = {a: 0. for a in self.agents}
        self.agent_selection = None

    def reset(self, seed=None, options=None):
        self.game = SplendorGame(self.num_players)
        self.agents = self.possible_agents[:]
        self._selector.reinit(self.agents)
        self.agent_selection = self._selector.next()

        self.rewards = {a: 0. for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._cumulative_rewards = {a: 0. for a in self.agents}

        obs = self.observe(self.agent_selection)
        return obs, self.infos[self.agent_selection] # obs, info

    def observe(self, agent):
        idx = self.agent_name_mapping[agent]
        gp = self.game

        board_gems = gp.board_gems.copy().astype(np.float32)
        board_cards = np.zeros((3, 4, 7), dtype=np.float32)
        for lvl in range(3):
            for slot, card in enumerate(gp.board_cards[lvl]):
                 if card:
                    board_cards[lvl, slot] = np.array([*card.cost, card.VPs, card.bonus], dtype=np.float32)

        full_nobles = np.zeros((self.num_players + 1, 5), dtype=np.float32)
        nobles_to_show = [n for n in gp.nobles if n is not None][:self.num_players + 1]
        for i, n in enumerate(nobles_to_show):
             full_nobles[i] = n.requirement
        nobles = full_nobles

        player_gems = np.stack([p.gems for p in gp.players], axis=0).astype(np.float32)
        player_bonuses = np.stack([p.bonuses for p in gp.players], axis=0).astype(np.float32)

        player_reserved = np.zeros((self.num_players, 3, 7), dtype=np.float32)
        for pi, p in enumerate(gp.players):
            for ri, card in enumerate(p.reserved):
                if card:
                    player_reserved[pi, ri] = np.array([*card.cost, card.VPs, card.bonus], dtype=np.float32)

        feats = [
            board_gems.flatten(),
            board_cards.flatten(),
            nobles.flatten(),
            player_gems.flatten(),
            player_bonuses.flatten(),
            player_reserved.flatten(),
            np.array([gp.players[idx].VPs], dtype=np.float32),
        ]
        obs_vector = np.concatenate(feats).astype(np.float32)
        
        legal_actions_list = gp.legal_actions(idx)
        mask = np.zeros(self.max_per_turn, dtype=np.int8)
        legal_actions_set = set(legal_actions_list)

        for i, action in enumerate(self.all_actions):
            if action in legal_actions_set:
                mask[i] = 1

        expected_dim = self.observation_spaces[agent]["observation"].shape[0]
        if len(obs_vector) != expected_dim:
             obs_vector = np.pad(obs_vector, (0, expected_dim - len(obs_vector)), 'constant')

        return {
            "observation": obs_vector,
            "action_mask": mask,
        }

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        idx = self.agent_name_mapping[agent]
        
        current_mask = self.observe(agent)["action_mask"]

        if action < len(current_mask) and current_mask[action] == 1:
            action_to_take = self.all_actions[action]
            print(f"Agent {agent} chose action {action_to_take} (legal).")
            self.game.step(action_to_take)
        else:
            print(f"CRITICAL WARNING: Agent {agent} chose illegal action {action}! Masking FAILED. Taking first legal.")
            current_mask = self.observe(agent)["action_mask"]
            legal_actions_list = self.game.legal_actions(idx)
            legal_indices = [i for i, m in enumerate(current_mask) if m]
            print(f"[ERROR]  Agent {agent} chose ILLEGAL action idx={action}!")
            print(f"         Mask sum={current_mask.sum()} (legal indices={legal_indices})")
            print(f"         legal_actions_list (raw) = {legal_actions_list}")
            legal_actions_list = self.game.legal_actions(idx)
            if legal_actions_list:
                self.game.step(legal_actions_list[0])
            else:
                 print(f"Error: No legal actions for agent {agent} on illegal action step.")
                 self.terminations = {a: True for a in self.agents}
                 return

        done = any(p.VPs >= 15 for p in self.game.players)
        for ag in self.agents:
            self.rewards[ag] = 0

        if done:
            print(f"Game over! Player {idx} reached 15 VPs.")
            winner_score = -1
            winner_agent = None
            for ag in self.agents:
                p_idx = self.agent_name_mapping[ag]
                score = self.game.players[p_idx].VPs
                if score > winner_score:
                    winner_score = score
                    winner_agent = ag
            
            for ag in self.agents:
                self.terminations[ag] = True
                self.rewards[ag] = 1.0 if ag == winner_agent else -1.0
                self.infos[ag] = {} # Clear infos on termination
        else:
             # If not done, set next agent (unless player needs to discard)
             if self.game.players[idx].gems.sum() <= self.game.MAX_GEMS:
                 self.agent_selection = self._selector.next()
             # else: keep current player (they need to discard)
             
             # Update infos for *all* agents
             for ag in self.agents:
                 if not self.terminations[ag]:
                    # Generate legal actions for the *next* agent if turn changes,
                    # or current agent if they need to discard.
                    current_player_idx = self.agent_name_mapping[self.agent_selection]
                    self.infos[ag] = {"legal_moves": self.game.legal_actions(current_player_idx)}
                 else:
                    self.infos[ag] = {}

        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()
            
    # ... (Keep render, close, observation_space, action_space) ...
    def render(self): pass
    def close(self): pass
    def observation_space(self, agent): 
        return self.observation_spaces[agent]
    def action_space(self, agent): 
        return self.action_spaces[agent]
    
# env = SplendorEnv()
# api_test(env, num_cycles=1000)
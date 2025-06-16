import numpy as np
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.env import AECEnv
from pettingzoo.test import api_test # Keep commented for training
from gymnasium import spaces
from game_logic.splendor_game import SplendorGame, ActionType

class SplendorEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "splendor_v0"}

    def __init__(self, render_mode=None):
        super().__init__()
        self.num_players = 2
        self.max_per_turn = 67
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
        for action in legal_actions_list:
            mask[gp.action_to_idx[action]] = 1

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
        player = self.game.players[idx]
        current_mask = self.observe(agent)["action_mask"]
        pass_idx = self.game.action_to_idx[(ActionType.PASS, None)]

        if action == pass_idx: # if action is to pass, just resign you're cooked
            print(f"Agent {agent} chose to pass (resign).")
            for ag in self.agents:
                self.terminations[ag] = True
                if ag == agent:
                    self.rewards[ag] = -1.0
                else:
                    self.rewards[ag] = +1.0
                self.infos[ag] = {}
            self._accumulate_rewards()
            return
        
        if action < len(current_mask) and current_mask[action] == 1:
            action_to_take = self.all_actions[action]
            # print(f"Agent {agent} chose action {action_to_take} (legal).")

            if action_to_take[0] is ActionType.BUY_RESERVE:
                res_idx = action_to_take[1]
                # grab a reference to that reserved card *before* stepping
                reserved_card = player.reserved[res_idx]
            else:
                reserved_card = None
                
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
                action_to_take = legal_actions_list[0]
                self.game.step(action_to_take)
            else:
                 print(f"Error: No legal actions for agent {agent} on illegal action step.")
                 self.terminations = {a: True for a in self.agents}
                 return
            
        if action_to_take[0] is ActionType.BUY_BOARD:
            lvl, _ = action_to_take[1]
            if lvl == 1:
                self.rewards[agent] += 0.075  # <-- tier bonus here
            elif lvl == 2:
                self.rewards[agent] += 0.15
        elif action_to_take[0] is ActionType.BUY_RESERVE:
            res_idx = action_to_take[1]

            # 2) Tier bonus (exactly the same numbers you used in BUY_BOARD):
            if reserved_card.level == 1:
                self.rewards[agent] += 0.075    # Tier 2  +0.075
            elif reserved_card.level == 2:
                self.rewards[agent] += 0.15    # Tier 3  +0.150

        # elif action_to_take[0] is ActionType.RESERVE_CARD:
        #     lvl, _ = action_to_take[1]
        #     if lvl == 1:
        #         self.rewards[agent] += 0.03  # <-- tier bonus here
        #     if lvl == 2:
        #         self.rewards[agent] += 0.06
        elif action_to_take[0] is ActionType.DISCARD:
            self.rewards[agent] -= 0.075  # <-- discard bonus

        done = self.game.game_over
        if done: 
            winner = self.game.decide_winner()
            winner_agent = self.agents[winner]
            for ag in self.agents:
                self.terminations[ag] = True
                self.rewards[ag] = 1.0 if ag == winner_agent else -1.0
            self.infos = {ag: {} for ag in self.agents}
        else:
            # normal turn‐advance logic…
            if self.game.players[idx].gems.sum() <= self.game.MAX_GEMS:
                self.agent_selection = self._selector.next()
            for ag in self.agents:
                if not self.terminations[ag]:
                    cur_idx = self.agent_name_mapping[self.agent_selection]
                    self.infos[ag] = {"legal_moves": self.game.legal_actions(cur_idx)}
                else:
                    self.infos[ag] = {}
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()
            
    def render(self):
        print(f"  board gems  : {self.game.board_gems.tolist()}")

        for lvl in range(3):
            cards = []
            for c in self.game.board_cards[lvl]:
                row = [int(x) for x in (*c.cost, c.VPs, c.bonus)]
                cards.append(row)
            print(f"  tier {lvl+1} face-up: {cards}")

        noble_reqs = [n.requirement.tolist() for n in self.game.nobles if n is not None]
        print(f"  nobles reqs : {noble_reqs}")

        # 5) For each player, print gems, bonuses, VP, and their reserved list
        for i, p in enumerate(self.game.players):
            gems    = p.gems.tolist()
            bonuses = p.bonuses.tolist()
            print(f"  P{i} gems     : {gems}  bonuses: {bonuses}  VP: {p.VPs}")
            reserved = [[int(x) for x in (*c.cost, c.VPs, c.bonus)] for c in p.reserved]
            print(f"     reserved  : {reserved}")
        print()
    def close(self): pass
    def observation_space(self, agent): 
        return self.observation_spaces[agent]
    def action_space(self, agent): 
        return self.action_spaces[agent]
    
# env = SplendorEnv()
# api_test(env, num_cycles=1000)
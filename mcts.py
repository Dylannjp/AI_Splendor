from collections import namedtuple
import math
import torch
import numpy as np
from collections import namedtuple


MCTSConfig = namedtuple("MCTSConfig", [
    "num_simulations",
    "cpuct",
    "discount",
    "dirichlet_alpha",
    "exploration_frac"
])

class Node:
    def __init__(self, prior):
        self.visit_count  = 0
        self.prior        = prior
        self.value_sum    = 0
        self.children     = {}  # action -> Node
        self.hidden_state = None
        self.reward       = 0
        self.game_state   = None  # store the game state for reference

    def expanded(self):
        return bool(self.children)

    def value(self):
        return self.value_sum / (1 + self.visit_count)

class MCTS:
    def __init__(self, net, config: MCTSConfig, all_actions, root_game):
        self.net    = net
        self.config = config
        self.all_actions = all_actions
        self.root_game = root_game

    def run(self, root_hidden, legal_mask):
        # 1) Initialize root
        root = Node(prior=1.0)  # root is always player 0's turn
        root.hidden_state = root_hidden
        root.game_state = self.root_game.clone()  # store the game state for reference
        # get initial policy & value
        pi, v = self.net.prediction(root_hidden)
        root.value_sum += v.item()
        root.visit_count += 1

        for a, p in enumerate(pi[0]):
            if legal_mask[a]: 
                child = Node(prior=p.item())
                gs = root.game_state.clone()  # clone the game state for each action
                gs.step(self.all_actions[a])
                child.game_state = gs
                child.hidden_state = None
                root.children[a] = child
        
        # add Dirichlet noise for exploration at the root
        if self.config.dirichlet_alpha > 0:
            actions = list(root.children)
            noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(actions))
            for a, n in zip(actions, noise):
                root.children[a].prior = (
                    root.children[a].prior * (1 - self.config.exploration_frac)
                    + n * self.config.exploration_frac
                )

        # 2) Simulations
        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]
            selected_actions = []  # keep track of actions taken in this simulation

            # a) Selection
            while node.expanded():
                # find best action by UCB:
                best_a = max(
                node.children.keys(),
                key=lambda a: self._ucb_score(node, node.children[a])
                )
                selected_actions.append(best_a)  # remember the action taken
                node = node.children[best_a]
                search_path.append(node)
                action_taken = best_a    # remember which action you just followed

            # b) Expansion & Evaluation
            parent = search_path[-2]
            action_taken = selected_actions[-1]

            gs = parent.game_state.clone()  # clone the game state for the action
            gs.step(self.all_actions[action_taken])

            a_tensor = torch.tensor([action_taken], device=self.net.device)
            next_hidden, reward = self.net.dynamics(parent.hidden_state, a_tensor)
            next_hidden = next_hidden.detach()  # detach to avoid backprop through dynamics

            node.hidden_state = next_hidden
            node.reward = reward.item()
            node.game_state = gs

            pi2, v2 = self.net.prediction(next_hidden)

            legal_next = gs.legal_actions(gs.current_player)
            mask_next  = np.zeros(len(self.all_actions), dtype=bool)
            for act in legal_next:
                idx = self.all_actions.index(act)
                mask_next[idx] = True

            # 2) only expand those indices
            for a2, p2 in enumerate(pi2[0]):
                if not mask_next[a2]:
                    continue

                # now safe to simulate it
                child = Node(prior=p2.item())
                gs2   = gs.clone()
                gs2.step(self.all_actions[a2])   # this will never IndexError now
                child.game_state = gs2
                node.children[a2] = child

            # c) Backup
            self._backpropagate(search_path, v2.item())

        return root

    def _ucb_score(self, parent, child):
        pb_c = math.log((parent.visit_count + self.config.cpuct + 1) / self.config.cpuct) + 1
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
        return child.value() + pb_c * child.prior
    
    def _backpropagate(self, path, value):
        for node in reversed(path):
            node.value_sum   += value
            node.visit_count += 1
            value = node.reward + self.config.discount * value

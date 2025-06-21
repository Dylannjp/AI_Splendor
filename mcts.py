import math
import copy
import numpy as np
import torch
from typing import Dict, Sequence, Optional, List, Any
from collections import namedtuple

MCTSConfig = namedtuple("MCTSConfig", [
    "num_simulations",
    "cpuct",
    "discount",
    "dirichlet_alpha",
    "exploration_frac"
])

class Node:
    def __init__(self, prior: float, env_state: Any):
        # UCB stats
        self.visit_count: int    = 0
        self.prior: float        = prior
        self.value_sum: float    = 0.0

        # latent & reward
        self.hidden_state: Optional[torch.Tensor] = None
        self.reward: float       = 0.0

        # SplendorGame clone (for mask queries only)
        self.env_state: Any      = env_state

        # children: action index → Node
        self.children: Dict[int, "Node"] = {}

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def expanded(self) -> bool:
        return bool(self.children)

class MCTS:
    def __init__(
        self,
        net: Any,
        config: Any,
        all_actions: Sequence[Any],
    ):
        self.net         = net
        self.config      = config
        self.all_actions = all_actions

    def run(self, root_hidden: torch.Tensor, root_env: Any) -> Node:
        """
        root_hidden: latent state from net.initial_state(obs)
        root_env:    a SplendorGame instance at the current public state
        """
        # 1) Root inference
        policy_logits, value = self.net.prediction(root_hidden)
        root = Node(prior=1.0, env_state=copy.deepcopy(root_env))
        root.hidden_state = root_hidden
        root.value_sum    = value.item()
        root.visit_count  = 1

        # 2) Seed root children using real env legality
        legal = set(root.env_state.legal_actions(root.env_state.current_player))
        priors = torch.softmax(policy_logits[0], dim=-1)
        for a, p in enumerate(priors):
            if self.all_actions[a] in legal:
                child_env = copy.deepcopy(root.env_state)
                root.children[a] = Node(prior=p.item(), env_state=child_env)

        self.add_dirichlet_noise(root)

        # 3) Simulations
        for _ in range(self.config.num_simulations):
            node, path = root, [root]

            # -- Selection --
            while node.expanded():
                action, node = self.select_child(node)
                path.append(node)

            parent = path[-2]

            # -- Expansion & rollout in latent space --
            action_idx = torch.tensor([action], device=parent.hidden_state.device)
            h, r        = self.net.dynamics(parent.hidden_state, action_idx)
            node.hidden_state = h.detach()
            node.reward       = r.item()
            pl2, v2           = self.net.prediction(h)
            node.value_sum    = v2.item()
            node.visit_count  = 1

            # -- Advance the env clone for legality only --
            child_env      = copy.deepcopy(parent.env_state)
            child_env.step(self.all_actions[action])
            node.env_state = child_env

            # -- Seed grandchildren --
            legal2  = set(child_env.legal_actions(child_env.current_player))
            priors2 = torch.softmax(pl2[0], dim=-1)
            for a2, p2 in enumerate(priors2):
                if self.all_actions[a2] in legal2:
                    node.children[a2] = Node(prior=p2.item(),
                                             env_state=copy.deepcopy(child_env))

            # -- Backup --
            self.backpropagate(path, v2.item(), self.config.discount)

        return root

    def add_dirichlet_noise(self, root: Node):
        alpha, frac = self.config.dirichlet_alpha, self.config.exploration_frac
        if frac <= 0 or alpha <= 0:
            return
        num = len(root.children)
        if num == 0:
            return
        noise = np.random.dirichlet([alpha] * num)
        for (a, child), n in zip(root.children.items(), noise):
            child.prior = child.prior * (1 - frac) + n * frac

    def select_child(self, node: Node):
        best_score = -float('inf')
        best_a, best_c = None, None
        for a, c in node.children.items():
            ucb = (
                c.value
                + self.config.cpuct * c.prior * math.sqrt(node.visit_count)
                  / (1 + c.visit_count)
            )
            if ucb > best_score:
                best_score, best_a, best_c = ucb, a, c
        return best_a, best_c

    def backpropagate(self, path: List[Node], value: float, discount: float):
        for n in reversed(path):
            n.value_sum   += value
            n.visit_count += 1
            value = n.reward + discount * value

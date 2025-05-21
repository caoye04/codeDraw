from .node import MCTSNode, INF
from .config import MCTSConfig
from env.base_env import BaseGame

from model.linear_model_trainer import NumpyLinearModelTrainer
import numpy as np


class PUCTMCTS:
    def __init__(self, init_env:BaseGame, model: NumpyLinearModelTrainer, config: MCTSConfig, root:MCTSNode=None):
        self.model = model
        self.config = config
        self.root = root
        if root is None:
            self.init_tree(init_env)
        self.root.cut_parent()
        self.has_print_priors = False
    
    def init_tree(self, init_env:BaseGame):
        env = init_env.fork()
        obs = env.observation
        self.root = MCTSNode(
            action=None, env=env, reward=0
        )
        # compute and save predicted policy
        child_prior, _ = self.model.predict(env.compute_canonical_form_obs(obs, env.current_player))
        self.root.set_prior(child_prior)
    
    def get_subtree(self, action:int):
        # return a subtree with root as the child of the current root
        # the subtree represents the state after taking action
        if self.root.has_child(action):
            new_root = self.root.get_child(action)
            return PUCTMCTS(new_root.env, self.model, self.config, new_root)
        else:
            return None
    
    def puct_action_select(self, node: MCTSNode):
        legal_actions = np.where(node.action_mask == 1)[0]  
        puct_scores = np.zeros(len(legal_actions))
        parent_visits = np.sum(node.child_N_visit) + 1  

        for i, action in enumerate(legal_actions):
            if node.child_N_visit[action] == 0:
                q_value = 0
            else:
                q_value = node.child_V_total[action] / node.child_N_visit[action]
            exploration = self.config.C * node.child_priors[action] * np.sqrt(parent_visits) / (1 + node.child_N_visit[action])
            puct_scores[i] = q_value + exploration
        return legal_actions[np.argmax(puct_scores)]


    def backup(self, node:MCTSNode, value):
        current_node = node
        current_value = value
        
        while current_node is not None:
            if current_node.parent is not None:
                parent = current_node.parent
                action = current_node.action
                parent.child_N_visit[action] += 1
                parent.child_V_total[action] += current_value
            current_node = current_node.parent
            current_value = -current_value 
    
    def pick_leaf(self):
        # select the leaf node to expand
        # the leaf node is the node that has not been expanded
        # create and return a new node if game is not ended
        current_node = self.root
        while not current_node.done:
            legal_actions = np.where(current_node.action_mask == 1)[0]
            unexplored_actions = [action for action in legal_actions if current_node.child_N_visit[action]==0]
            if unexplored_actions:
                action = np.random.choice(unexplored_actions)
            else:
                action = self.puct_action_select(current_node)
            if not current_node.has_child(action):
                current_node.add_child(action)
                return current_node.get_child(action)
            else:
                current_node = current_node.get_child(action)
        return current_node
    
    def get_policy(self, node:MCTSNode = None):
        if node is None:
            node = self.root
        
        legal_actions = np.where(node.action_mask == 1)[0]
        policy = np.zeros(node.n_action)
    
        total_visit = np.sum(node.child_N_visit[legal_actions])

        for action in legal_actions:
            policy[action] = node.child_N_visit[action] / total_visit
        return policy

    def search(self):
        for _ in range(self.config.n_search):
            leaf = self.pick_leaf()
            value = 0
            if leaf.done:
                value = leaf.reward
            else:
                policy,value = self.model.predict(
                    leaf.env.compute_canonical_form_obs(leaf.env.observation, leaf.env.current_player)
                )
                leaf.set_prior(policy)
            self.backup(leaf, value)
            
        return self.get_policy(self.root)
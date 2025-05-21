from .node import MCTSNode, INF
from .config import MCTSConfig
from env.base_env import BaseGame

import numpy as np

class UCTMCTSConfig(MCTSConfig):
    def __init__(
        self,
        n_rollout:int = 1,
        *args, **kwargs
    ):
        MCTSConfig.__init__(self, *args, **kwargs)
        self.n_rollout = n_rollout


class UCTMCTS:
    def __init__(self, init_env:BaseGame, config: UCTMCTSConfig, root:MCTSNode=None):
        self.config = config
        self.root = root
        if root is None:
            self.init_tree(init_env)
        self.root.cut_parent()
    
    def init_tree(self, init_env:BaseGame):
        # initialize the tree with the current state
        # fork the environment to avoid side effects
        env = init_env.fork()
        self.root = MCTSNode(
            action=None, env=env, reward=0,
        )
    
    def get_subtree(self, action:int):
        # return a subtree with root as the child of the current root
        # the subtree represents the state after taking action
        if self.root.has_child(action):
            new_root = self.root.get_child(action)
            return UCTMCTS(new_root.env, self.config, new_root)
        else:
            return None
    
    def uct_action_select(self, node:MCTSNode) -> int:
        # select the best action based on UCB when expanding the tree
        
        if node.done:
            return -1
        
        valid_actions = np.where(node.action_mask == 1)[0]
        best_action = -1
        best_value = -float('inf')

        for action in valid_actions:
            if node.child_N_visit[action] == 0:
                return action
        
            exploitation = node.child_V_total[action] / (node.child_N_visit[action]+0.000001)
            exploration = self.config.C * np.sqrt(np.log(sum(node.child_N_visit)) / node.child_N_visit[action])
            ucb_value = exploitation + exploration
            
            if ucb_value > best_value:
                best_value = ucb_value
                best_action = action
        
        return best_action


    def backup(self, node:MCTSNode, value:float) -> None:
        # backup the value of the leaf node to the root
        # update N_visit and V_total of each node in the path
        
        current = node

        while current is not None and current.parent is not None:
            action = current.action
            current.parent.child_N_visit[action] += 1
            current.parent.child_V_total[action] += value
            
            value = -value
            current = current.parent
            
    
    def rollout(self, node:MCTSNode) -> float:
        # simulate the game until the end
        # return the reward of the game
        # NOTE: the reward should be convert to the perspective of the current player!
        
        env = node.env.fork()
        player_perspective = env.current_player 
        total_reward = 0

        for _ in range(self.config.n_rollout):
            sim_env = env.fork()
            while not sim_env.ended:
                valid_actions = np.where(sim_env.action_mask == 1)[0]
                if len(valid_actions) == 0:
                    break
                action = np.random.choice(valid_actions)
                _, step_reward, done = sim_env.step(action)
            
            final_reward = step_reward
            if sim_env.current_player != player_perspective:
                final_reward = -final_reward
            total_reward += final_reward

        if player_perspective == -1:
            defensive_bias = 2.05
            total_reward = total_reward * (1 + defensive_bias)
        
        return total_reward / self.config.n_rollout

    
    def pick_leaf(self) -> MCTSNode:
        # select the leaf node to expand
        # the leaf node is the node that has not been expanded
        # create and return a new node if game is not ended

        current = self.root
        
        while not current.done:
            valid_actions = np.where(current.action_mask == 1)[0]
            unexpanded_actions = [a for a in valid_actions if not current.has_child(a)]
            
            if unexpanded_actions:
                action = np.random.choice(unexpanded_actions)
                child = current.add_child(action)
                return child
            
            action = self.uct_action_select(current)
            if action == -1:
                break
                
            current = current.get_child(action)
        
        return current


    
    def get_policy(self, node:MCTSNode = None) -> np.ndarray:
        # return the policy of the tree(root) after the search
        # the policy conmes from the visit count of each action 
        
        if node is None:
            node = self.root
        
        valid_actions = np.where(node.action_mask == 1)[0]
        policy = np.zeros(node.n_action)
        
        total_visits = sum(node.child_N_visit[a] for a in valid_actions)
        
        if total_visits > 0:
            for action in valid_actions:
                policy[action] = node.child_N_visit[action] / total_visits
        else:
            for action in valid_actions:
                policy[action] = 1.0 / len(valid_actions)
        
        return policy      
        

    def search(self):
        # search the tree for n_search times
        # eachtime, pick a leaf node, rollout the game (if game is not ended) 
        #   for n_rollout times, and backup the value.
        # return the policy of the tree after the search
        for _ in range(self.config.n_search):
            leaf = self.pick_leaf()
            value = 0
            if leaf.done:
                value = leaf.reward
            else:
                value = self.rollout(leaf)

            self.backup(leaf, value)

        return self.get_policy(self.root)
from env.base_env import BaseGame
from .base_player import BasePlayer

from mcts.puct_mcts import PUCTMCTS, MCTSConfig as PUCTMCTSConfig
from model.net_trainer import ModelTrainer

import numpy as np

class PUCTPlayer():
    def __init__(self, config:PUCTMCTSConfig = None, model:ModelTrainer = None, deterministic:bool = False) -> None:
        if config is None:
            config = PUCTMCTSConfig()
        self.config = config
        self.model = model
        self.mcts = None
        self.deterministic = deterministic
    
    def __str__(self):
        return "PUCT Player"

    def clear(self):
        self.mcts = None
    
    def init(self, init_game):
        self.mcts = PUCTMCTS(init_game, self.model, self.config)

    def play(self, state:BaseGame):
        if self.mcts is None: # if the tree is not initialized, initialize it
            self.init(state)
        policy = self.mcts.search()
        if self.deterministic:
            action = np.argmax(policy)
        else:
            action = np.random.choice(len(policy), p=policy)
        # print(policy)
        self.mcts = self.mcts.get_subtree(action)
        return action
    
    def opp_play(self, action):
        # reuse the subtree if the opponent takes the action
        # that is in the current tree
        if self.mcts is None:
            return
        if self.mcts.root.has_child(action):
            self.mcts = self.mcts.get_subtree(action)
        else:
            self.mcts = None
        return
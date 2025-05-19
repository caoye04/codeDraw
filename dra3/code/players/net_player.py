from env.base_env import BaseGame
from .base_player import BasePlayer

from model.net_trainer import ModelTrainer

import numpy as np

class NuralNetPlayer():
    def __init__(self, net:ModelTrainer) -> None:
        self.net = net
    
    def __str__(self):
        return "Nural Net Player"

    def play(self, state:BaseGame, deterministic:bool = False):
        policy = self.net.predict(state.compute_canonical_form_obs(state.observation, state.current_player))
        if deterministic:
            action = np.argmax(policy)
        else:
            action = np.random.choice(len(policy), p=policy)
        return action
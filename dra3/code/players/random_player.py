from env.base_env import BaseGame
from .base_player import BasePlayer
import numpy as np

class RandomPlayer(BasePlayer):
    def __init__(self):
        pass

    def __str__(self):
        return "Random Player"

    def play(self, state:BaseGame):
        valid_moves = state.action_mask
        a = np.random.choice(valid_moves.nonzero()[0])
        return a
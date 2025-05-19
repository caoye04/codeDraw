from env.base_env import BaseGame
from .base_player import BasePlayer
import numpy as np

class HumanPlayer(BasePlayer):
    def __init__(self):
        pass
    
    def __str__(self):
        return "Human Player"

    def play(self, state:BaseGame):
        valid = state.action_mask
        while True:
            print("valid moves:", valid.nonzero()[0])
            a = int(input())
            if valid[a]:
                break
            else:
                print('Invalid')
        return a
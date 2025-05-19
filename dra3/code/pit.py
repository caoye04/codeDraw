from env import *
from players import *
from mcts.uct_mcts import UCTMCTSConfig
from tqdm import trange, tqdm

import numpy as np

def print_devide_line(n=50):
    print("--"*n)

def pit(game:BaseGame, player1:BasePlayer, player2:BasePlayer, log_output:bool=True):
    game.reset()
    if log_output:
        print(f"start playing {type(game)}")
        print_devide_line()
    reward = 0
    
    for player in [player1, player2]:
        if player.__class__.__name__ == 'UCTPlayer':
            player.clear()
            
    while True:
        a1 = player1.play(game)
        _, reward, done = game.step(a1)
        if player2.__class__.__name__ == 'UCTPlayer':
            player2.opp_play(a1)
        if log_output:
            print(f"Player 1 ({player1}) move: {a1}")
            print(game.to_string())
            print_devide_line()
        if done:
            break
        a2 = player2.play(game)
        _, reward, done = game.step(a2)
        if player1.__class__.__name__ == 'UCTPlayer':
            player1.opp_play(a2)
        if log_output:
            print(f"Player 2 ({player2}) move: {a2}")
            print(game.to_string())
            print_devide_line()
        if done:
            reward *= -1
            break
    if log_output:
        if reward == 1:
            print(f"Player 1 ({player1}) win")
        elif reward == -1:
            print(f"Player 2 ({player2}) win")
        else:
            print("Draw")
    return reward
        
def multi_match(game:BaseGame, player1:BasePlayer, player2:BasePlayer, n_match=100):
    print(f"Player 1:{player1}  Player 2:{player2}")
    n_p1_win, n_p2_win, n_draw = 0, 0, 0
    T = trange(n_match)
    for _ in T:
        reward = pit(game, player1, player2, log_output=False)
        if reward == 1:
            n_p1_win += 1
        elif reward == -1:
            n_p2_win += 1
        else:
            n_draw += 1
        T.set_description_str(f"P1 win: {n_p1_win} ({n_p1_win}) P2 win: {n_p2_win} ({n_p2_win}) Draw: {n_draw} ({n_draw})")        
    print(f"Player 1 ({player1}) win: {n_p1_win} ({n_p1_win/n_match*100:.2f}%)")
    print(f"Player 2 ({player2}) win: {n_p2_win} ({n_p2_win/n_match*100:.2f}%)")
    print(f"Draw: {n_draw} ({n_draw/n_match*100:.2f}%)")
    print(f"Player 1 not lose: {n_p1_win+n_draw} ({(n_p1_win+n_draw)/n_match*100:.2f}%)")
    print(f"Player 2 not lose: {n_p2_win+n_draw} ({(n_p2_win+n_draw)/n_match*100:.2f}%)")
    return n_p1_win, n_p2_win, n_draw
        
        
def search_best_C():
    from matplotlib import pyplot as plt
    p2nl = []
    cs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.5, 5.0]
    n_match = 100
    for c in cs:
        config = UCTMCTSConfig()
        config.C = c
        config.n_rollout = 7
        config.n_search = 64
        player1 = AlphaBetaPlayer()
        player2 = UCTPlayer(config, deterministic=True)
        game = TicTacToeGame()
        p1w, p2w, drw = multi_match(game, player1, player2, n_match=n_match)
        p2nl.append((p2w+drw)/n_match)
    plt.plot(cs, p2nl)
    plt.savefig('p2nl.png')
        
if __name__ == '__main__':
    #####################
    # Modify code below #
    #####################
    
    # set seed to reproduce the result
    # np.random.seed(0)
        
    # game = GobangGame(3, 3) # Tic-Tac-Toe game
    game = GobangGame(5, 4) # 5x5 Gobang (4 stones in line to win) 
    # game = GoGame(7)
    
    # config for UCT-MCTS
    config = UCTMCTSConfig()
    # config.C = 0.5
    # config.C = 0.7
    config.C = 1.0
    # config.C = 2.5
    # config.C = 5.0
    config.n_rollout = 7
    # config.n_rollout = 13
    config.n_search = 64
    # config.n_search = 400
    
    # player initialization    
    # player1 = HumanPlayer()
    player1 = RandomPlayer()
    # player1 = AlphaBetaPlayer()
    # player1 = AlphaBetaHeuristicPlayer()
    # player1 = UCTPlayer(config, deterministic=True)
    
    # player2 = HumanPlayer()
    # player2 = RandomPlayer()
    # player2 = AlphaBetaPlayer()
    # player2 = AlphaBetaHeuristicPlayer()
    player2 = UCTPlayer(config, deterministic=True)
    
    # single match
    # pit(game, player1, player2)
    
    multi_match(game, player1, player2, n_match=5)
    multi_match(game, player2, player1, n_match=5)
    # multi_match(game, player1, player2, n_match=100)
    
    #####################
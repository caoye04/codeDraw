from env import *
from players import *
from mcts.puct_mcts import MCTSConfig as PUCTMCTSConfig
from tqdm import trange, tqdm
from multiprocessing import Process
from torch.distributed.elastic.multiprocessing import Std, start_processes

from model.net_trainer import ModelTrainer, ModelTrainingConfig
from model.example_net import *
from model.linear_model import NumpyLinearModel
from model.linear_model_trainer import NumpyLinearModelTrainer

import logging
logger = logging.getLogger(__name__)

import numpy as np

def log_devide_line(n=50):
    logger.info("--"*n)

def pit(game:BaseGame, player1:PUCTPlayer, player2:PUCTPlayer, log_output:bool=False):
    game.reset()
    if log_output:
        logger.info(f"start playing {type(game)}")
        log_devide_line()
    reward = 0
    
    for player in [player1, player2]:
        if hasattr(player, "clear"):
            player.clear()
            
    while True:
        a1 = player1.play(game)
        _, reward, done = game.step(a1)
        if hasattr(player2, "opp_play"):
            player2.opp_play(a1)
        if log_output:
            logger.info(f"Player 1 ({player1}) move: {a1}")
            logger.info(game.to_string())
            log_devide_line()
        if done:
            break
        a2 = player2.play(game)
        _, reward, done = game.step(a2)
        if hasattr(player1, "opp_play"):
            player1.opp_play(a2)
        if log_output:
            logger.info(f"Player 2 ({player2}) move: {a2}")
            logger.info(game.to_string())
            log_devide_line()
        if done:
            reward *= -1
            break
    if log_output:
        if reward == 1:
            logger.info(f"Player 1 ({player1}) win")
        elif reward == -1:
            logger.info(f"Player 2 ({player2}) win")
        else:
            logger.info("Draw")
    # print(game.observation, reward)
    return reward

def multi_match(game:BaseGame, player1:PUCTPlayer, player2:PUCTPlayer, n_match=100, disable_tqdm=False):
    assert n_match % 2 == 0 and n_match > 1, "n_match should be an even number greater than 1"
    n_p1_win, n_p2_win, n_draw = 0, 0, 0
    for _ in trange(n_match//2, disable=disable_tqdm):
        reward = pit(game, player1, player2, log_output=False)
        if reward == 1:
            n_p1_win += 1
        elif reward == -1:
            n_p2_win += 1
        else:
            n_draw += 1
    first_play = n_p1_win, n_p2_win, n_draw
    for _ in trange(n_match//2, disable=disable_tqdm):
        reward = pit(game, player2, player1, log_output=False)
        if reward == 1:
            n_p2_win += 1
        elif reward == -1:
            n_p1_win += 1
        else:
            n_draw += 1
    latter_play = n_p1_win - first_play[0], n_p2_win - first_play[1], n_draw - first_play[2]
    return (n_p1_win, n_p2_win, n_draw), first_play, latter_play

if __name__ == '__main__':
    
    # env = GoGame(7, obs_mode="extra_feature") # "extra_feature" only compatible with MynetModel, LinearModel, MLPNet
    env = GoGame(7)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    MLP_mcts_config = PUCTMCTSConfig(
        n_search=240, 
        temperature=1.0, 
        C=1.0,
    )
    MLP_model_config = BaseNetConfig(
        linear_hidden=[256, 128]
    )
    
    MLP_net = MLPNet(env.observation_size, env.action_space_size, MLP_model_config, device=device)
    MLP_net = ModelTrainer(env.observation_size, env.action_space_size, MLP_net, ModelTrainingConfig())
    MLP_net.load_checkpoint("checkpoint/mlp_7x7_3layers_exfeat_1", "best.pth.tar")
    MLP_puct_player = PUCTPlayer(MLP_mcts_config, MLP_net, deterministic=True)
    
    
    
    Mynet_mcts_config = PUCTMCTSConfig(
        n_search=240, 
        temperature=1.0, 
        C=1.0,
    )
    Mynet_model_config = BaseNetConfig(
        linear_hidden=[128, 128],
        num_channels=32
    )
    
    Mynet_net = MyNet(env.observation_size, env.action_space_size, Mynet_model_config, device=device)
    Mynet_net = ModelTrainer(env.observation_size, env.action_space_size, Mynet_net, ModelTrainingConfig())
    Mynet_net.load_checkpoint("checkpoint/mynet", "best.pth.tar")
    Mynet_puct_player = PUCTPlayer(Mynet_mcts_config, Mynet_net, deterministic=True)
    
    player1_name = "MLP_net"
    player1 = MLP_puct_player
    player2_name = "My_net"
    player2 = Mynet_puct_player
    
    n_match = 30
    
    results, first_play_results, second_play_results = \
        multi_match(env, player1, player2, n_match)
    
    n_p1_win, n_p2_win, n_draw = results
    
    print(f"Player 1 ({player1_name}) win: {n_p1_win} ({n_p1_win/n_match*100:.2f}%)")
    print(f"Player 2 ({player2_name}) win: {n_p2_win} ({n_p2_win/n_match*100:.2f}%)")
    print(f"Draw: {n_draw} ({n_draw/n_match*100:.2f}%)")
    print(f"Player 1 not lose: {n_p1_win+n_draw} ({(n_p1_win+n_draw)/n_match*100:.2f}%)")
    print(f"Player 2 not lose: {n_p2_win+n_draw} ({(n_p2_win+n_draw)/n_match*100:.2f}%)")
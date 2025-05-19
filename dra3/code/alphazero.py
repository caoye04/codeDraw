from env import *
from env.base_env import *
from torch.nn import Module
from model.net_trainer import ModelTrainer, ModelTrainingConfig
from model.example_net import *
from model.linear_model import NumpyLinearModel
from model.linear_model_trainer import NumpyLinearModelTrainer
from mcts import puct_mcts
from draw_elo import draw as draw_elo_curve

import numpy as np
import random
import torch
import copy, os, json
from tqdm import tqdm
from random import shuffle
from players import PUCTPlayer, RandomPlayer, AlphaBetaPlayer

from pit_puct_mcts import multi_match

import logging
logger = logging.getLogger(__name__)


class AlphaZeroConfig():
    def __init__(
        self, 
        n_train_iter:int=300,
        n_match_train:int=20,
        n_match_update:int=20,
        n_match_eval:int=20,
        max_queue_length:int=8000,
        update_threshold:float=0.501,
        n_search:int=200, 
        temperature:float=1.0, 
        C:float=1.0,
        checkpoint_path:str="checkpoint"
    ):
        self.n_train_iter = n_train_iter
        self.n_match_train = n_match_train
        self.max_queue_length = max_queue_length
        self.n_match_update = n_match_update
        self.n_match_eval = n_match_eval
        self.update_threshold = update_threshold
        self.n_search = n_search
        self.temperature = temperature
        self.C = C
        
        self.checkpoint_path = checkpoint_path

class AlphaZero:
    def __init__(self, env:BaseGame, net:ModelTrainer, config:AlphaZeroConfig):
        self.env = env
        self.net = net
        self.last_net = net.copy()
        self.config = config
        self.mcts_config = puct_mcts.MCTSConfig(
            C=config.C, 
            n_search=config.n_search, 
            temperature=config.temperature
        )
        self.mcts_config.with_noise = False
        self.mcts = None
        self.train_eamples_queue = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
    
    def execute_episode(self):
        train_examples = []
        env = self.env.fork()
        state = env.reset()
        config = copy.copy(self.mcts_config)
        config.with_noise = True
        mcts = puct_mcts.PUCTMCTS(env, self.net, config)
        while True:
            player = env.current_player
            # MCTS self-play
            ########################
            # TODO: your code here #
            policy = None # compute policy with mcts
            
            symmetries = get_symmetries(state, policy) # rotate&flip the data&policy
            train_examples += [(x[0], x[1], player) for x in symmetries]
            
            pass # choose a action accroding to policy
            done = False # apply the action to env
            if done:
                pass # record all data
                # tips: use env.compute_canonical_form_obs to transform the observation into BLACK's perspective
            
            pass # update mcts (you can use get_subtree())
            ########################
    
    def evaluate(self, show_log:bool=True):
        player = PUCTPlayer(self.mcts_config, self.net, deterministic=True)
        # baseline_player = AlphaBetaPlayer()
        baseline_player = RandomPlayer()
        result = multi_match(self.env, player, baseline_player, self.config.n_match_eval)
        if show_log:
            logger.info(f"[EVALUATION RESULT]: win{result[0][0]}, lose{result[0][1]}, draw{result[0][2]}")
            logger.info(f"[EVALUATION RESULT]:(first)  win{result[1][0]}, lose{result[1][1]}, draw{result[1][2]}")
            logger.info(f"[EVALUATION RESULT]:(second) win{result[2][0]}, lose{result[2][1]}, draw{result[2][2]}")
        return result
    
    def learn(self):
        self.net.save_checkpoint(folder=self.config.checkpoint_path, filename='best.pth.tar')
        self.net.save_checkpoint(folder=self.config.checkpoint_path, filename=f'iter_{0:04d}.pth.tar')
        for iter in range(1, self.config.n_train_iter + 1):
            logger.info(f"------ Start Self-Play Iteration {iter} ------")
            
            # collect new examples
            T = tqdm(range(self.config.n_match_train), desc="Self Play")
            cnt = ResultCounter()
            for _ in T:
                episode = self.execute_episode()
                self.train_eamples_queue += episode
                cnt.add(episode[0][-1], 1)
            logger.info(f"[NEW TRAIN DATA COLLECTED]: {str(cnt)}")
            
            # pop old examples
            if len(self.train_eamples_queue) > self.config.max_queue_length:
                self.train_eamples_queue = self.train_eamples_queue[-self.config.max_queue_length:]
            
            # shuffle examples for training
            train_data = copy.copy(self.train_eamples_queue)
            shuffle(train_data)
            logger.info(f"[TRAIN DATA SIZE]: {len(train_data)}")
            
            # save current net to last_net
            self.net.save_checkpoint(folder=self.config.checkpoint_path, filename='temp.pth.tar')
            self.net.save_checkpoint(folder=self.config.checkpoint_path, filename=f'iter_{iter:04d}.pth.tar')
            self.last_net.load_checkpoint(folder=self.config.checkpoint_path, filename='temp.pth.tar')
            
            # train current net
            self.net.train(train_data)
            
            # evaluate current net
            env = self.env.fork()
            env.reset()
            
            last_mcts_player = PUCTPlayer(self.mcts_config, self.last_net, deterministic=True)
            current_mcts_player = PUCTPlayer(self.mcts_config, self.net, deterministic=True)
            
            result = multi_match(self.env, last_mcts_player, current_mcts_player, self.config.n_match_update)[0]
            # win_rate = result[1] / sum(result)
            total_win_lose = result[0] + result[1]
            win_rate = result[1] / total_win_lose if total_win_lose > 0 else 1
            logger.info(f"[EVALUATION RESULT]: currrent_win{result[1]}, last_win{result[0]}, draw{result[2]}; win_rate={win_rate:.3f}")
            
            if win_rate > self.config.update_threshold:
                self.net.save_checkpoint(folder=self.config.checkpoint_path, filename='best.pth.tar')
                logger.info(f"[ACCEPT NEW MODEL]")
                self.evaluate()
            else:
                self.net.load_checkpoint(folder=self.config.checkpoint_path, filename='temp.pth.tar')
                logger.info(f"[REJECT NEW MODEL]")

    def round_robin(self, K:int=20, window_size:int=None):
        assert K % 2 == 0
        
        all_checkpoints = [i for i in os.listdir(self.config.checkpoint_path) if i.endswith('.pth.tar') and i.startswith('iter_')]
        all_checkpoints.sort(key=lambda x:int(x.split('_')[-1].split('.')[0]))
        print("Test all checkpoints:", all_checkpoints)
        
        results = []
        if window_size is None:
            window_size = len(all_checkpoints)
        tasks = []
        for i, ckpt_i in enumerate(all_checkpoints[1:]):
            for ckpt_j in all_checkpoints[max(0, i-window_size+1):i+1]:
                tasks.append([ckpt_i, ckpt_j])
        print(tasks)
        for ckpt_i, ckpt_j in tqdm(tasks, desc="Evaluate all checkpoints with each other"):
            self.net.load_checkpoint(folder=self.config.checkpoint_path, filename=ckpt_i)
            self.last_net.load_checkpoint(folder=self.config.checkpoint_path, filename=ckpt_j)
            
            current_mcts_player = PUCTPlayer(self.mcts_config, self.net, deterministic=True)
            last_mcts_player = PUCTPlayer(self.mcts_config, self.last_net, deterministic=True)
            
            result, first_result, second_result = \
                multi_match(self.env, current_mcts_player, last_mcts_player, K)
            results.append({
                    "p1":ckpt_i, "p2":ckpt_j,
                    "p1_win":first_result[0], "p2_win":first_result[1], "draw":first_result[2]
                })
            results.append({
                    "p1":ckpt_j, "p2":ckpt_i,
                    "p1_win":second_result[1], "p2_win":second_result[0], "draw":second_result[2]
                })
        for ckpt_i in tqdm(all_checkpoints, desc="Evaluate all checkpoints with baseline"):
            self.net.load_checkpoint(folder=self.config.checkpoint_path, filename=ckpt_i)
            result, first_result, second_result = self.evaluate(show_log=False)
            results.append({
                    "p1":ckpt_i, "p2":"baseline",
                    "p1_win":first_result[0], "p2_win":first_result[1], "draw":first_result[2]
                })
            results.append({
                    "p1":"baseline", "p2":ckpt_i,
                    "p1_win":second_result[1], "p2_win":second_result[0], "draw":second_result[2]
                })
        logger.info("[round_robin] All done.")
        return results

if __name__ == "__main__":
    from env import *
    import torch
    
    MASTER_SEED = 0
    random.seed(MASTER_SEED)
    np.random.seed(MASTER_SEED)
    torch.manual_seed(MASTER_SEED)
    
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler("log.txt")
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    # MLP Config
    config = AlphaZeroConfig(
        n_train_iter=30,
        n_match_train=20,
        n_match_update=20,
        n_match_eval=20,
        max_queue_length=80000,
        update_threshold=0.501,
        n_search=240, 
        temperature=1.0, 
        C=1.0,
        checkpoint_path="checkpoint/mlp_7x7_3layers_exfeat"
    )
    model_training_config = ModelTrainingConfig(
        epochs=15,
        batch_size=128,
        lr=0.001,
        dropout=0.3,
    )
    model_config = BaseNetConfig(
        linear_hidden=[256, 128]
    )
    
    
    # Linear Config
    # config = AlphaZeroConfig(
    #     n_train_iter=30,
    #     n_match_train=20,
    #     n_match_update=10,
    #     n_match_eval=10,
    #     max_queue_length=80000,
    #     update_threshold=0.001,
    #     n_search=240, 
    #     temperature=1.0, 
    #     C=1.0,
    #     checkpoint_path="checkpoint/linear_7x7_exfeat_norm_1"
    # )
    # model_training_config = ModelTrainingConfig(
    #     epochs=10,
    #     batch_size=128,
    #     lr=0.0001,
    #     weight_decay=0.001
    # )
    # model_config = BaseNetConfig()
    
    assert config.n_match_update % 2 == 0
    assert config.n_match_eval % 2 == 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # env = GoGame(7, obs_mode="extra_feature") # "extra_feature" only compatible with NumpyLinearModel, LinearModel, MLPNet
    env = GoGame(7)
    
    # Deep Neural Network
    # net = MyNet(env.observation_size, env.action_space_size, model_config, device=device)
    net = MLPNet(env.observation_size, env.action_space_size, model_config, device=device)
    net = ModelTrainer(env.observation_size, env.action_space_size, net, model_training_config)
    
    # Numpy Linear Model
    # net = NumpyLinearModel(env.observation_size, env.action_space_size, model_config, device=device)
    # net = NumpyLinearModelTrainer(env.observation_size, env.action_space_size, net, model_training_config)
    
    alphazero = AlphaZero(env, net, config)
    alphazero.learn()
    
    # Evaluate and calculate elo score of each checkpoint
    results = alphazero.round_robin(20, window_size=5) # increase window size if available
    match_data_path = os.path.join(config.checkpoint_path, "eval_results_temp.json")
    with open(match_data_path, "w") as f:
        json.dump(results, f)
    draw_elo_curve(match_data_path, os.path.join(config.checkpoint_path, "elo.png"))
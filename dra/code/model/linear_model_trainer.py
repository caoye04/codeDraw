import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from torch import nn

import torch
import torch.optim as optim

from env.base_env import BaseGame
from .linear_model import NumpyLinearModel

import logging
logger = logging.getLogger(__name__)

# TODO: Refactor this later (dcy11011)
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
    
class NumpyModelTrainingConfig:
    def __init__(
        self, lr:float=0.0007, 
        epochs:int=20, 
        batch_size:int=128, 
        weight_decay:float=0,
    ):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay

class AverageMeter(object):

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class NumpyLinearModelTrainer():
    def __init__(self, observation_size:tuple[int, int], action_size:int, model:NumpyLinearModel, config:NumpyModelTrainingConfig=None):
        self.model = model
        if type(observation_size) is int:
            observation_size = [observation_size]
        self.observation_size = observation_size
        self.action_size = action_size
        if config is None:
            config = NumpyModelTrainingConfig()
        self.config = config
    
    @property
    def device(self):
        return self.model.device
    
    @device.setter
    def device(self, value):
        self.model.device = value
        
    def to(self, device):
        self.model.to(device)
        self.model.device = device
        return self
    
    def copy(self):
        return NumpyLinearModelTrainer(
            self.observation_size, 
            self.action_size, 
            self.model.__class__(self.observation_size, self.action_size, self.model.config, self.model.device, self.model.base_function))
    
    def train(self, examples):
        t = tqdm(range(self.config.epochs), desc='Training Numpy Linear Model')
        for epoch in t:
            # logger.info('EPOCH ::: ' + str(epoch + 1))
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / self.config.batch_size)

            for bc in range(batch_count):
                sample_ids = np.random.randint(len(examples), size=self.config.batch_size)
                observations, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                observations = np.array(observations).astype(np.float64)
                target_pis = np.array(pis).astype(np.float64)
                target_vs = np.array(vs).astype(np.float64).reshape(-1, 1)

                # compute output
                out_pi, out_v = self.model.forward(observations)
                l_pi = self.model.loss_pi(out_pi, target_pis)
                l_v = self.model.loss_v(out_v, target_vs)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), observations.size)
                v_losses.update(l_v.item(), observations.size)
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses, progress=f"{bc}/{batch_count}")

                # compute gradient and do SGD step
                self.model.learn(self.config.lr, observations, target_pis, out_pi, target_vs)
                if self.config.weight_decay is not None:
                    self.model.weight_decay(self.config.weight_decay)

    def predict(self, observation:np.ndarray):
        """
        board: np array with board
        """
        # preparing input
        observation = observation.astype(np.float64)
        observation = observation.reshape(1, *self.observation_size)
        pi, v = self.model.forward(observation)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            logger.warning("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.makedirs(folder, exist_ok=True)
        else:
            logger.debug("Checkpoint Directory exists. ")
        self.model.save(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        self.model.load(filepath)

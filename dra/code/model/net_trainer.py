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

import logging
logger = logging.getLogger(__name__)

# TODO: Refactor this later (dcy11011)
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
    
class ModelTrainingConfig:
    def __init__(
        self, lr:float=0.0007, 
        dropout:float=0.3, 
        epochs:int=20, 
        batch_size:int=128, 
        weight_decay:float=0,
    ):
        self.lr = lr
        self.dropout = dropout
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


class ModelTrainer():
    def __init__(self, observation_size:tuple[int, int], action_size:int, net:nn.Module, config:ModelTrainingConfig=None):
        self.net = net
        if type(observation_size) is int:
            observation_size = [observation_size]
        self.observation_size = observation_size
        self.action_size = action_size
        if config is None:
            config = ModelTrainingConfig()
        self.config = config
    
    @property
    def device(self):
        return self.net.device
    
    @device.setter
    def device(self, value):
        self.net.device = value
        
    def to(self, device):
        self.net.to(device)
        self.net.device = device
        return self
    
    def copy(self):
        return ModelTrainer(self.observation_size, self.action_size, self.net.__class__(self.observation_size, self.action_size, self.net.config, self.net.device))
    
    def train(self, examples):
        optimizer = optim.Adam(self.net.parameters(), weight_decay=self.config.weight_decay)
        t = tqdm(range(self.config.epochs), desc='Training Net')
        for epoch in t:
            # logger.info('EPOCH ::: ' + str(epoch + 1))
            self.net.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / self.config.batch_size)

            for bc in range(batch_count):
                sample_ids = np.random.randint(len(examples), size=self.config.batch_size)
                observations, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                observations = torch.FloatTensor(np.array(observations).astype(np.float64)).to(self.net.device)
                target_pis = torch.FloatTensor(np.array(pis)).to(self.net.device)
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64)).to(self.net.device)

                # compute output
                out_pi, out_v = self.net(observations)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), observations.size(0))
                v_losses.update(l_v.item(), observations.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses, progress=f"{bc}/{batch_count}")

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, observation):
        """
        board: np array with board
        """
        # preparing input
        observation = torch.FloatTensor(observation.astype(np.float64)).to(self.net.device)
        observation = observation.view(1, *self.observation_size)
        self.net.eval()
        with torch.no_grad():
            pi, v = self.net(observation)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            logger.warning("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.makedirs(folder, exist_ok=True)
        else:
            logger.debug("Checkpoint Directory exists. ")
        torch.save({
            'state_dict': self.net.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise BaseException("No model in path {}".format(filepath))
        map_location = self.net.device
        checkpoint = torch.load(filepath, map_location=map_location)
        self.net.load_state_dict(checkpoint['state_dict'])

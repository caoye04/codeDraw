import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functools import reduce

from env.base_env import BaseGame

class BaseNetConfig:
    def __init__(
        self, 
        num_channels:int = 256,
        dropout:float = 0.3,
        linear_hidden:list[int] = [64, 32],
    ):
        self.num_channels = num_channels
        self.linear_hidden = linear_hidden
        self.dropout = dropout
        
class MLPNet(nn.Module):
    def __init__(self, observation_size:tuple[int, int], action_space_size:int, config:BaseNetConfig, device:torch.device='cpu'):
        super().__init__()
        self.config = config
        self.device = device
        input_dim = observation_size[0] * observation_size[1] if len(observation_size) == 2 else observation_size[0]
        self.layer1 = nn.Linear(input_dim, config.linear_hidden[0])
        self.layer2 = nn.Linear(config.linear_hidden[0], config.linear_hidden[1])
        
        self.policy_head = nn.Linear(config.linear_hidden[1], action_space_size)
        self.value_head = nn.Linear(config.linear_hidden[1], 1)
        self.relu = nn.ReLU()
        self.to(device)

    def forward(self, x: torch.Tensor):
        #                                                         x: batch_size x board_x x board_y
        x = x.view(x.size(0), -1) # reshape tensor to 1d vectors, x.size(0) is batch size
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        pi = self.policy_head(x)
        v = self.value_head(x)
        return F.log_softmax(pi, dim=1), torch.tanh(v)


class LinearModel(nn.Module):
    def __init__(self, observation_size:tuple[int, int], action_space_size:int, config:BaseNetConfig, device:torch.device='cpu'):
        super(LinearModel, self).__init__()
        
        self.action_size = action_space_size
        self.config = config
        self.device = device
        
        observation_size = reduce(lambda x, y: x*y , observation_size, 1)
        self.l_pi = nn.Linear(observation_size, action_space_size)
        self.l_v  = nn.Linear(observation_size, 1)
        self.to(device)
    
    def forward(self, s: torch.Tensor):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(s.shape[0], -1)                                # s: batch_size x (board_x * board_y)
        pi = self.l_pi(s)
        v = self.l_v(s)
        return F.log_softmax(pi, dim=1), torch.tanh(v)

class MyNet(nn.Module):
    def __init__(self, observation_size: tuple[int, int], action_space_size: int,
                 config: BaseNetConfig, device: torch.device = 'cpu') -> None:
        super().__init__()
        
        self.config = config
        self.device = device
        
        # 构建卷积网络层
        channels = config.num_channels
        self.conv_stack = nn.ModuleList([
            nn.Conv2d(1, channels, 3, padding=1),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.Conv2d(channels, channels, 3, padding=1)
        ])
        
        # 用于残差连接的直通路径
        self.identity_path = nn.Conv2d(1, channels, kernel_size=1)
        
        # 计算展平后的特征大小
        flatten_size = observation_size[0] * observation_size[1] * channels
        
        # 构建全连接层
        hidden_sizes = config.linear_hidden
        self.dense_layers = nn.ModuleList([
            nn.Linear(flatten_size, hidden_sizes[0]),
            nn.Linear(hidden_sizes[0], hidden_sizes[1])
        ])
        
        # 输出层
        self.policy_output = nn.Linear(hidden_sizes[1], action_space_size)
        self.value_output = nn.Linear(hidden_sizes[1], 1)
        
        # 防过拟合层
        self.regularization = nn.Dropout(config.dropout)
        
        # 将模型移至指定设备
        self.to(device)
    
    def forward(self, state_input):
        # 增加通道维度
        x = state_input.unsqueeze(1)
        
        # 保存残差路径
        skip_connection = self.identity_path(x)
        
        # 卷积层处理
        conv1_out = F.relu(self.conv_stack[0](x))
        conv2_out = F.relu(self.conv_stack[1](conv1_out))
        conv3_out = self.conv_stack[2](conv2_out)
        
        # 残差合并
        merged_features = F.relu(conv3_out + skip_connection)
        
        # 特征展平
        batch_size = merged_features.size(0)
        flattened = merged_features.reshape(batch_size, -1)
        
        # 全连接层处理
        fc1_out = F.relu(self.dense_layers[0](flattened))
        fc1_regularized = self.regularization(fc1_out)
        fc2_out = F.relu(self.dense_layers[1](fc1_regularized))
        fc2_regularized = self.regularization(fc2_out)
        
        # 输出头
        policy_logits = self.policy_output(fc2_regularized)
        value_pred = self.value_output(fc2_regularized)
        
        # 返回策略分布和状态评估
        return F.log_softmax(policy_logits, dim=1), torch.tanh(value_pred)

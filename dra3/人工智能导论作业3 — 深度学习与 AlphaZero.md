# 人工智能导论作业3 — 深度学习与 AlphaZero



### 一、问题背景

在之前的作业中，我们实现了 AlphaZero 的训练流程，并尝试用线性模型学习围棋的策略和价值函数。但是，简单的线性模型并不足以建模复杂的围棋问题，我们需要引入深度神经网络来解决围棋问题。

### 二、任务目标

任务目标 本次作业中，我们将继续在 $7 \times 7$ 围棋问题上探索 AlphaZero 算法的能力。下发的代码文件中，已经实现了一个简单的全连接网络的示例（`model／example＿net．py：MLPNet`，你需要先利用示例网络运行 AlphaZero 训练，然后参考示例网络，自己设计并实现一个深度网络，测试训练效果。提交时请删除＊．so，＊．pyd 和＊／build／等临时文件和训练过程的 ckeckpoint，仅提交代码和 1 个最好的模型参数文件，本题的文字报告请和其他题目写在同一个文档中提交。

### 三、具体任务

1．将之前作业中完成的代码填入对应位置，运行训练，并汇报使用 MLP 模型的 AlphaZero 算法训练过程中对 Random Player 的胜率，和训练过程的 elo 分数曲线图。

2．参考示例代码，设计并实现一个不一样的深度模型，要求至少需要使用一个卷积层处理二维棋盘特征。请绘制网络结构图，并简要说明设计的理由。

3．使用自己设计的深度模型，运行训练，并汇报训练过程中 AlphaZero 算法对 Random Player的胜率以及 elo 分数曲线图。要求训练过程中，对 Random Player 的胜率至少有一次不低于 $90 \%$ 。

4．修改 `pit＿puct＿mcts．py`，加载训练后的 MLP 模型和自己设计的模型进行对弈，汇报对局的胜率。



### 四、提示与注意点

1. 参数选择：完成上述题目时，可以自由选择适合你的情况的参数进行训练，但过于不合理的参数设置可能会导致扣分（若报告未说明实验使用的参数，则会以提交的代码为准）。
2. 文件大小限制：提交的模型参数文件大小不能大于 32 MB ，且只能提交 1 个你认为效果最好的模型参数文件（有特殊情况请与助教提前沟通）。
3. 在` model／example＿net．py `中预留了一个 MyNet 类用于实现你自己设计的深度模型。你也可以将其重命名为合适的名字。
4. 模型设计不是越复杂越好，过于复杂或者参数量过大的模型可能导致训练缓慢，容易过拟合。
5. 卷积层的实现你可能会用到` torch．nn．Conv2D`回和 `torch．nn．BatchNorm2D`。
6. 推荐使用并行脚本`alphazero＿parallel．py`进行训练，并根据实验环境实际情况，使用合适并行数（主函数中 N＿WORKER 变量控制）。训练时间可能较长，建议提前评估合理安排。
7. 本次作业下发的训练期本默认会覆盖同一保存路径下的文件，训练时请做好备份，或确保使用了不同的保存路径。



### 五、项目结构与重要文件内容

### example_net

```py
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
    def __init__(self, observation_size:tuple[int, int], action_space_size:int, config:BaseNetConfig, device:torch.device='cpu'):
        super().__init__()
        self.config = config
        self.device = device
        self.to(device)
        ########################
        # TODO: your code here #
        ########################
    
    def forward(self, s: torch.Tensor):
        ########################
        # TODO: your code here #
        return None, None
        ########################
```

### 示例答案1：

```py
class MyNet(nn.Module):
    def __init__(self, observation_size:tuple[int, int], action_space_size:int, config:BaseNetConfig, device:torch.device='cpu'):
        super().__init__()
        self.config = config
        self.device = device
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        conv_output_size = observation_size[0] * observation_size[1] * 32
        
        self.fc1 = nn.Linear(conv_output_size, config.linear_hidden[0])
        self.fc2 = nn.Linear(config.linear_hidden[0], config.linear_hidden[1])
        
        self.policy_head = nn.Linear(config.linear_hidden[1], action_space_size)
        self.value_head = nn.Linear(config.linear_hidden[1], 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)
        self.to(device)
    
    def forward(self, s: torch.Tensor):
        s = s.unsqueeze(1)
        
        x = self.relu(self.bn1(self.conv1(s)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        pi = self.policy_head(x)
        v = self.value_head(x)
        
        return F.log_softmax(pi, dim=1), torch.tanh(v)
```

### 示例答案2：

```py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyNet(nn.Module):
    def __init__(self, observation_size: tuple[int, int], action_space_size: int,
                 config: BaseNetConfig, device: torch.device = 'cpu') -> None:
        super().__init__()
        self.config = config
        self.device = device
        
        self.convLayer1 = nn.Conv2d(1, config.num_channels, kernel_size=3, stride=1, padding=1)
        self.convLayer2 = nn.Conv2d(config.num_channels, config.num_channels, kernel_size=3, stride=1, padding=1)
        self.convLayer3 = nn.Conv2d(config.num_channels, config.num_channels, kernel_size=3, stride=1, padding=1)
        
        self.shortcut = nn.Conv2d(1, config.num_channels, kernel_size=1)
        
        conv_out_size = observation_size[0] * observation_size[1] * config.num_channels
        self.fc1 = nn.Linear(conv_out_size, config.linear_hidden[0])
        self.fc2 = nn.Linear(config.linear_hidden[0], config.linear_hidden[1])
        self.policy_head = nn.Linear(config.linear_hidden[1], action_space_size)
        self.value_head = nn.Linear(config.linear_hidden[1], 1)
        self.dropout = nn.Dropout(config.dropout)
        
        self.to(device)
    
    def forward(self, s):
        s = s.unsqueeze(1)
        
        residual = self.shortcut(s)
        
        out = F.relu(self.convLayer1(s))
        out = F.relu(self.convLayer2(out))
        out = self.convLayer3(out)
        
        out = F.relu(out + residual)
        
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        
        pi = self.policy_head(out)
        v = self.value_head(out)
        
        return F.log_softmax(pi, dim=1), torch.tanh(v)
```


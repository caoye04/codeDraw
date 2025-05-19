# Homework1：UCT MCTS

## 快速开始

### 环境准备

更建议使用Linux系统完成作业，但MacOs和Windows也可以完成作业。
请先确保系统已经安装了C编译器（GCC/Clang，如不确定就先试试后续步骤），然后执行以下命令：

```bash
pip install -r requirements.txt
```


### 编译游戏环境

将下面的`***`替换成`go`, `gobang`和`tictactoe`并分别执行：

```bash
cd env/***/
python setup.py build_ext --inplace
```

编译成功后会在`env/***`路径下生成对应的`.so`文件（Windows系统则为`.pyd`文件），如产生警告不会影响正常运行。编译遇到疑难问题请联系助教。

如要重新编译，请先删除已编译好的`.so/pyd`文件和`build`临时目录再进行编译，或者执行下面的指令强制重新生成：

```bash
python setup.py build_ext --inplace --force
```

你也可以直接执行 `compile_all_env.sh` 脚本重新编译所有环境。



### 对局

`/players`中定义了使用不同算法的游戏玩家，其中包括一个支持你手动操作的游戏玩家。`pit.py`中包含了对局的脚本，你可以执行以下命令测试AlphaBeta Search和随机策略对局的效果：

```bash
python -m pit
```

你可以修改`pit.py`中`if __name__=='__main__'`后的内容，更改对局双方的类型，测试单次对局和多次对局。


## 如何实现MCTS

### 环境交互

游戏环境（`env`）是维护游戏的核心逻辑、管理棋局状态、并将复杂的操作抽象成统一接口的对象。通过环境的封装，你可以轻松地对游戏进行操作、读取状态。方便起见，与我们将环境封装成了强化学习中常用的`gym`环境的风格。为了提高执行的效率，棋局的核心逻辑使用C++编写，通过cython与python部分代码对接。通过`env/***/setup.py`脚本可以编译这些环境。在使用这些环境前必须先进行编译。

以下是对环境使用方法的简要介绍：

* 你可以将一个环境（`env`）看做一个棋盘对象。在使用环境进行对局时，你首先需要使用`reset()`方法重置环境，然后让对局双方轮流使用`step()`方法采取动作，根`step()`的返回值判断游戏是否结束和胜者。
* 选择动作前，你可以使用`action_space_size`属性获取当前的棋盘有多少个可能的动作（动作从0开始编号），再通过环境的`action_mask`属性获取当前哪些状态是合法的（该属性会返回一个长度为`action_space_size`的`numpy`数组，其中每一位为1则表示该动作当前合法，反之为非法）。非法动作会被环境拒绝执行并抛出异常。

* 在棋盘类游戏中，棋盘的格子按照从上到下、从左到右的顺序从0开始编号，动作编号和棋盘格子编号对应，采取`i`号动作即意味着在`i`号格子落子（参考：`/env/tictactoe/tictactoe_env.py:19-22`）。游戏环境会在落子后自动切换游戏玩家，且默认黑子先手，你不需要指定你是黑子还是白子。这也意味着你需要注意不要在你的代码中让同色玩家连续落子，这会被环境识别为两方先后落子。

* 环境的`step()`方法可以执行一步落子，其只需要一个参数，即动作（格子）的编号。`step()`返回一个三元组，分别代表动作后棋盘的状态（`numpy`二维数组表示）、动作的奖励（浮点值）、游戏是否结束(布尔值)。游戏结束时，第三个返回值会为`True`，第二个返回值如为`1`则表示**当前采取了这一步动作的玩家**胜利,`-1`反之，平局则返回一个非0的极小值（`/env/base_env.py:10`）。如果游戏没有结束，奖励值为0。

* **请使用`fork()` 方法复制一个环境**。该方法能安全地拷贝当前的游戏局面，保证局面状态的信息完整的、彻底的、高效的复制。如果直接使用赋值语句复制环境，因为python中仅仅拷贝了对象的指针，你对新环境的修改也会修改到旧的环境，而在树搜索中，我们往往需要创建一个局面的互不干扰副本，用以探索不同的操作的后续结果。
* 如果你不关心当前的棋盘，只关心最终的胜负（奖励值），在`step()`时可以设置`return_obs=False`以跳过棋盘的拷贝，加快执行速度。

* 游戏结束后，如果要重新开始游戏，请再次调用`reset()`方法。

更多环境的属性以及使用方法，请参考`/env/base_env.py`的定义以及`/other_algo/alpha_beta_search.py`中的用法。


### MCTS实现

对于UCT-MCTS你需要补全`/mcts/uct_mcts.py`中所有标记了`TODO`的位置。你可以参考[A Survey of Monte Carlo Tree Search Methods](https://repository.essex.ac.uk/4117/1/MCTS-Survey.pdf)中对UCT的描述。

对于PUCT-MCTS你需要补全`/mcts/puct_mcts.py`中所有标记了`TODO`的位置。

在实现上，PUCT和UCT的最大区别是选择节点时，前者使用了PUCB公式代替UCB公式，并且不使用rollout评估节点的价值，而是引入了价值模型对节点的价值进行预测。除此之外，二者的代码大部分是相通的。关于PUCT的实现可以参考[Supplementary Materials](https://www.science.org/doi/suppl/10.1126/science.aar6404/suppl_file/aar6404-silver-sm.pdf)。

### 线性模型

你需要补全`model/linear_model.py`中所有标记了`TODO`的位置。该文件中，定义了基于`Numpy`的线性模型。该模型中，分别有两套用于计算策略和价值的模型参数，二者分开训练，互不相干。

需要注意，模型的输入是1个batch的observation，也就是说，$B$个observation会被拼合成一个形状为$B\times O$的矩阵作为输入（$B$是batch size， $O$是observation的维度）。你应该参考代码中已经实现的部分，尽量使用`Numpy`提供的函数进行计算，并且尽量对整个batch的数据整体进行计算，而尽量不要使用python `for`循环遍历batch中的每一个样本和样本的每一个维度，否则可能导致网络计算变得非常慢。

`model/linear_model.py`中提供了一个函数`check_grad`用来帮助判断梯度的计算是否正确。你可以直接在项目根目录下执行`python -m model.linear_model `来验证梯度计算是否正确，正常情况下输出的最大误差不会超过$0.000001$。请注意，这个方法只能判断你的梯度是否和损失函数是对应的，无法验证损失函数的实现是否是正确的。

### 深度模型

你需要补全`model/example_net.py`中的`MyNet`类，实现一个自己设计的深度网络（当然，你也可以换个类名，也可以新建一个文件来写）。

我们提供了一个简单的MLP网络和pytorch实现的线性模型作为参考。

要更换训练的网络类型，只需要修改训练脚本（`alphazero.py`和`alphazero_parallel.py`）主函数中创建网络的相关代码即可。

**输入特征** 需要注意，第二次作业中，围棋游戏默认的observation是手工设计的特征（除棋盘外，还拼接了多种统计量），而第三次作业中，使用的是默认特征（仅有一个$n\times n$的二维棋盘特征）。使用何种特征是在围棋环境创建时指定的(`env = GoGame(7, obs_mode="extra_feature")`创建环境对应的是前者，不填写`obs_mode`参数则是后者）

### MCTS训练

我们的训练采用类AlphaGoZero的形式，其大致的步骤如下：

1. PUCT-MCTS自我博弈（self-play），收集训练数据；
2. 提取搜索过程中得到的（observation，policy，reward）数据，将数据统一处理成黑方视角，并对棋盘进行旋转、反转增广数据；
3. 对策略和价值网络进行训练；
4. 评估训练后，PUCT-MCTS对战训练前的版本的胜率，如果胜率超过阈值，则保留新的参数，否则丢弃；如果保留了新的参数，评估并记录对战基线方法的胜率；
5. 回到1继续执行

其中，训练使用的损失函数可以参照 [A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play](https://www.science.org/doi/10.1126/science.aar6404)公式1：
$$
l=(z-v)^2-\pi^T\log p + c\|\theta\|^2
$$
其中，$(z-v)^2$是value loss，$-\pi^T\log p$是policy loss，$c\|\theta\|^2$是正则化系数，在我们的框架中用weight decay实现，本次作业中不需要考虑。

你需要根据提示补全`alphazero.py`和`alphazero_parallel.py`中标记`TODO`的内容。二者需要补全的部分是相同的，在其中一个中完成补全后，可以把相关代码复制到另一个中，但是需要修改返回训练数据的部分。（`alphazero.py`中直接return即可，`alphazero_parallel.py`在循环结束后进行进程间通信传递数据）

### 调试和运行训练

建议使用`alphazero.py`进行调试，该代码文件实现了单进程版本的训练流程，而`alphazero_parallel.py`则实现了多进程版本的训练。在你完成调试后，你可以使用后者更快地进行训练。请注意根据实际情况调整`alphazero_parallel.py`中控制进程数的`N_WORKER`参数。使用深度模型的训练，建议使用后者，否则训练时间可能会很长。

经测试，默认参数下MLP模型可以在10次迭代内收敛到95%以上对Random胜率。但该结果会受实现差异、参数差异和随机种子的影响，仅供参考。

### 模型能力评估

训练结束后，会将训练过程中产生的所有checkpoint进行互相对弈、和baseline方法对弈（默认为Random Player），得到的对弈结果会用于计算所有checkpoint的[elo分数](https://en.wikipedia.org/wiki/Elo_rating_system)，并绘制曲线图，相关结果会保存在输出目录（`AlphaZeroConfig.checkpoint_path`）下。elo分数可以大致反映算法的围棋水平，但由于测试方式的限制，该结果不能作为围棋能力的绝对衡量，只能作为相对能力的参考。

训练结束后，还可以使用`pit_puct_mcts.py`加载两个不同的模型进行对轮对弈，计算胜率。现有的代码仅供参考，请根据自己的实际需要修改（注意模型参数、加载路径等要与训练时一致）。

### 特别提醒

**训练脚本启动时，默认会覆盖同一保存路径下的`best.pth.tar`！请注意备份，或确保没有使用已经存在的路径作为输出路径。**


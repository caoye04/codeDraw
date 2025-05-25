# HW4 - Report 推理和语言模型

> 曹烨 2021012167 软23 caoye541@gmail.com

---

[toc]

---

## T1：贝叶斯网络

考虑以下含有 4 个变量 $A, B, C, D$ 的贝叶斯网络。相应的条件概率表如下。
（a）计算 $p(-a,-b,-c,+d)$ 。
（b）计算 $p(+b)$ 。
（c）计算 $p(-a \mid+b)$ 。
（d）计算并比较 $p(+b)$ 和 $p(+b \mid-d)$ 两者的值是否相等，并从随机变量独立的角度进行解释。
（e）计算并比较 $p(-a \mid+c)$ 和 $p(-a \mid+c,-d)$ 两者的值是否相等，并从随机变量条件独立的角度进行解释。

![](pic/1.png)



---

## T2：变量消除法

在如下图所示的贝叶斯网络中，我们希望通过变量消除法计算 $P(B=b \mid W=w)$ 。初始因子为 $P(e), P(b), P(a \mid b, e), P(h \mid a), P(w \mid a)$ 。

![](pic/2.png)

（a）假设消除顺序为 $E \rightarrow H \rightarrow A$ 。请写出每一步产生的新因子的表达式并列出剩余的因子。
（b）请说明如何使用剩余的因子计算 $P(B=b \mid W=w)$ 。
（c）因子大小是变量消除法计算复杂度的关键因素。例如，假设所有的变量都是二元变量，则因子 $P(H \mid A)$ 的大小是 2 ，它有 $2^2$ 种取值需要维护；由于 $W$ 已经被观测，因子 $P(W \mid A)$ 的大小是 1 ，它只有 $2^1$ 种取值需要维护。在以上变量消除的过程中，最大的因子是哪个，它有多少种取值需要维护？简要说明若改用顺序 $E \rightarrow A \rightarrow H$ ，最大中间因子的维度是否会更小、相同或更大，并说明理由。





---

## T3：朴素贝叶斯

一位心理学家做了一个关于＂幸福＂的调查。每个受访者提供一个向量，其元素 1 或 0 分别对应于他们对某一问题回答＂是＂或＂否＂。该向量的属性为

$$
\mathbf{x}=\text { (rich, married, healthy). }
$$


例如，回答 $(1,0,1)$ 表示受访者＂富有＂、＂未婚＂、＂健康＂。此外，每个受访者如果对自己的生活感到＂满意＂，就给出一个 $y=1$ 的数值；如果＂不满意＂，就给出 $y=0$ 。

心理学家一共收到了 9 份问卷，声称对自己的生活感到＂满意＂的人给出的问卷结果为：

$$
(1,1,1),(0,0,1),(1,1,0),(1,0,1) ;
$$
而对于＂不满意＂的人，则是：$(0,0,0),(1,0,0),(0,0,1),(0,1,0),(0,0,0)$ 。
基于以上数据，使用朴素贝叶斯分类器（不带 Laplacian smoothing），
（a）一个＂不富有＂、＂已婚＂、＂健康＂的人感到＂满意＂的概率是多少？
（b）一个＂不富有＂、＂已婚＂的人感到＂满足＂的概率是多少？（也就是说，我们不知道他们是否 ＂健康＂）





---

## T4：编程题：语言模型

> 在本题中，你将基于简化版本的 GPT 语言模型框架，深人理解语言模型的关键结构模块的实现原理和设计理念。我们已经完成了基础的模型定义，包括嵌人层、位置编码、注意力机制、前馈网络以及残差连接，你需要实现模型中的关键计算模块或替换其中的部分模块。
>
> 注意：本题主要考察代码实现的正确性。如果你的设备性能受限导致训练时间过长，你可以适当调整训练参数以缩短整体训练时间。

### 4.1. Gradient Accumulation

> 在语言模型的训练过程中，使用较大的 batch size 通常有助于提高训练稳定性、减少梯度方差并加速收敛。但在显存有限的条件下，大 batch size 并不总是可行。为此，我们可以采用梯度累积（Gradient Accumulation）技术，通过多次小 batch 的前向－反向传播来模拟大 batch 的效果。设累积步数为 $k$ ，小 batch 的损失函数为 $\mathcal{L}_t$ ，则每次更新的累积损失为：
>
> $$
> \mathcal{L}_{\text {accum }}=\frac{1}{k} \sum_{i=1}^k \mathcal{L}_{t+i}
> $$
>
>
> 每次更新模型参数时反向传播的梯度为：
>
> $$
> \nabla \theta=\nabla_\theta\left(\mathcal{L}_{\text {accum }}\right)=\frac{1}{k} \sum_{i=1}^k \nabla_\theta \mathcal{L}_{t+i}
> $$
>
>
> 请你修改现有的训练代码，使其支持梯度累积机制，每隔若干步进行一次模型参数更新，模拟较大 batch size 的效果。





### 4.2. 实现因果自注意力机制 (Causal Self-Attention)

> 你需要手动实现 GPT 模型中的核心计算模块：Causal Multi－Head Self－Attention，不能借助 nn ．MultiheadAttention 等已有实现。模块输人特征与维度约定如下：
>
> 令输人为一个三维张量 $X \in \mathbb{R}^{B \times L \times C}$ ，其中 $B$ 为 batch 大小，$L$ 为序列长度，$C$ 为嵌人维度，满足 $C=h \cdot d$ ，即头数乘以每个头的维度大小。计算步骤如下：
>
> **1．QKV 映射（带偏置）：**
>
> 使用一组共享线性变换对输人进行查询 $(Q)$ 、键 $(K)$ 和值 $(V)$ 的映射：
> $$
> Q=X W^Q+b^Q, \quad K=X W^K+b^K, \quad V=X W^V+b^V
> $$
> 其中 $W^Q, W^K, W^V \in \mathbb{R}^{C \times C}, b^Q, b^K, b^V \in \mathbb{R}^C$ 。
>
> **2．重构为多头形式：**
> 将 $Q, K, V$ reshape 为多头表示：
> $$
> Q, K, V \in \mathbb{R}^{B \times h \times L \times d}
> $$
>
> **3．缩放点积注意力：**
>
> 计算注意力得分：
> $$
> A=\frac{Q K^{\top}}{\sqrt{d}} \in \mathbb{R}^{B \times h \times L \times L}
> $$
>
> **4．应用 Causal Mask：**
>
> 使用下三角掩码 $\mathbf{M} \in\{0,1\}^{L \times L}$ 限制未来信息访问，确保模型生成过程只依赖于过去的信息：
> $$
> A_{i, j}= \begin{cases}A_{i, j}, & \text { if } j \leq i \\ -\infty, & \text { if } j>i\end{cases}
> $$
>
> **5．使用 Softmax 归一化注意力得分：**
> $$
> A=\operatorname{Softmax}(A) \in \mathbb{R}^{B \times h \times L \times L}
> $$
>
> **6．上下文表示计算：**
>
> 将注意力得分与值向量加权求和：
> $$
> Z=A V \in \mathbb{R}^{B \times h \times L \times d}
> $$
>
> **7．头合并与线性投影：**
>
> 将所有头拼接在一起，并使用一个线性层将张量映射回原始维度：
> $$
> Y=\operatorname{Concat}\left(Z_1, \ldots, Z_h\right) W^O+b^O \in \mathbb{R}^{B \times L \times C}
> $$
>
> 其中 $W^O \in \mathbb{R}^{C \times C}, b^O \in \mathbb{R}^C$ 。
>
> 你需要依照上述计算流程，补全 CausalSelfAttention 类的代码。实现要求：
>
> - 使用 torch．matmul 或＠运算符完成注意力机制的计算。
> - 手动构造 causal mask（可调用 torch．tril）。
>
> 使用补全后的代码进行模型训练，汇报训练过程中 Training Loss 和 Val Loss 的变化过程。



### 4.3. 位置编码的实现与对比

> 在我们提供的代码中，使用了可学习的位置编码（nn．Embedding）。近年来，旋转位置编码（Rotary Positional Embedding，RoPE）作为一种结构更为精巧的相对位置编码方法被广泛采用，尤其在 LLaMA 等模型架构中表现良好。在本题中，你需要实现使用 RoPE 作为位置编码的 Transformer 模型进行训练，并汇报 Training Loss 和 Val Loss 的变化过程。
>
> RoPE 的形式化定义如下：给定一对输人向量 $x=\left[x_0, x_1, \ldots, x_{d-1}\right] \in \mathbb{R}^d$ ，设维度 $d$ 为偶数，将其视为 $d / 2$ 个二维向量对：
>
> $$
> \mathbf{x}^{(i)}=\left[\begin{array}{c}
> x_{2 i} \\
> x_{2 i+1}
> \end{array}\right] \quad \text { for } i=0,1, \ldots, \frac{d}{2}-1
> $$
>
>
> 对于第 $i$ 对位置向量，在位置 $p$ 上定义旋转频率参数：
>
> $$
> \theta_i=\frac{1}{10000^{2 i / d}}
> $$
>
>
> 对应的二维旋转矩阵为：
>
> $$
> R\left(p, \theta_i\right)=\left[\begin{array}{cc}
> \cos \left(p \cdot \theta_i\right) & -\sin \left(p \cdot \theta_i\right) \\
> \sin \left(p \cdot \theta_i\right) & \cos \left(p \cdot \theta_i\right)
> \end{array}\right]
> $$
>
>
> 将每个二维向量应用该旋转操作：
>
> $$
> \mathbf{x}_p^{(i)}=R\left(p, \theta_i\right) \cdot \mathbf{x}^{(i)}
> $$
>
>
> 最终，位置编码后的向量 $x_p^{\prime}$ 为各个旋转结果拼接而成：
>
> $$
> x_p^{\prime}=\left[\mathbf{x}_p^{(0)} ; \mathbf{x}_p^{(1)} ; \ldots ; \mathbf{x}_p^{(d / 2-1)}\right]
> $$
>
>
> 对于注意力机制中的 $Q, K \in \mathbb{R}^{B \times h \times L \times d}$ ，将上述旋转分别应用于每个位置 $p \in\{0, \ldots, T-1\}$上的表示。 $V$ 不做变换。
>
> 注意力权重仍然按以下方式计算：
>
> $$
> \operatorname{Attention}(Q, K, V)=\operatorname{Softmax}\left(\frac{Q^{\prime} K^{\prime T}}{\sqrt{d}}\right) V
> $$
>
> 其中 $Q^{\prime}, K^{\prime}$ 是经过 RoPE 旋转的位置相关表示。
> 请实现使用 RoPE 作为位置编码的模型，并汇报 Training Loss 和 Val Loss 的变化过程。注意，在该模型中，原本的位置编码需要被移除，且 RoPE 的实现应当被集成在 CausalSelfAttention类的实现中。

### 4.4. 实现语言模型的采样函数

在我们给出的代码中，已经实现了语言模型的 top－k 采样策略，即每次生成新的 token 时，从概率大小前 k 的 token 中进行采样。还有一种所谓的 top－p 采样策略，每次只从累积概率超过阈值 $p$ 的最小单词集合中进行随机采样。其形式化定义如下。

设当前时刻模型输出的 token 概率分布为：

$$
P=\operatorname{Softmax}(z) \in \mathbb{R}^{|V|}
$$

其中 $z \in \mathbb{R}^{|V|}$ 是 logits 向量，$|V|$ 是词表大小。

我们将所有 token 按概率从大到小排序，记排序后 token 的索引为：

$$
\pi=\operatorname{argsort}(P), \quad \text { 使得 } P_{\pi_1} \geq P_{\pi_2} \geq \cdots \geq P_{\pi_{|V|}}
$$


选取最小的前 $k$ 个 token，使得它们的概率累计和不小于阈值 $p$ ：

$$
\sum_{i=1}^k P_{\pi_i} \geq p, \quad \text { 且 } \quad \sum_{i=1}^{k-1} P_{\pi_i}<p
$$


最终仅在这些 token 上归一化进行采样：

$$
\hat{P}_i=\left\{\begin{array}{ll}
\frac{P_i}{\sum_{j \in S} P_j}, & i \in S \\
0, & i \notin S
\end{array} \quad \text { 其中 } S=\left\{\pi_1, \ldots, \pi_k\right\}\right.
$$


请参考现有模型中的 generate 函数，实现使用 top－p 策略的采样函数，并在报告中附上使用两种策略生成的文本片段。





## 相关代码

### 1. 项目结构

```
code/
├── .git/
├── config/
│   └── train_wikitext.py
├── data/
│   └── wikitext_large/
│       ├── test.bin
│       ├── train.bin
│       └── val.bin
├── .DS_Store
├── .gitignore
├── configurator.py
├── model.py
├── model_RoPE.py
├── README.md
├── sample.py
└── train.py
```

### 2. configurator.py

```py
import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if '=' not in arg:
        # assume it's the name of a config file
        assert not arg.startswith('--')
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        # assume it's a --key=value argument
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                # attempt to eval it it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            # ensure the types match ok
            assert type(attempt) == type(globals()[key])
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")

```

### 3. model_RoPE.py

```py

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    block_size: int = 256
    vocab_size: int = 50304 
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = True

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # TODO: Implement the CausalSelfAttention class with RoPE positional Embedding
        # Attributes that could possibly be used: config.n_embd, config.n_head, config.dropout, config.bias
        

    def forward(self, x):
        # shape of x: B, L, C
        # shape of output: B, L, C
        # TODO: Implement the CausalSelfAttention class

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        idx: Tensor of shape (B, T)
        max_new_tokens: number of tokens to generate
        temperature: sampling temperature
        top_k: top-k filtering (int)
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

```

### 4. model.py

```py
import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    block_size: int = 256
    vocab_size: int = 50304 
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = True

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # TODO: Implement the CausalSelfAttention class
        # Attributes that could possibly be used: config.n_embd, config.n_head, config.dropout, config.bias
        

    def forward(self, x):
        # shape of x: B, L, C
        # shape of output: B, L, C
        # TODO: Implement the CausalSelfAttention class

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        idx: Tensor of shape (B, T)
        max_new_tokens: number of tokens to generate
        temperature: sampling temperature
        top_k: top-k filtering (int)
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.no_grad()
    def generate_with_top_p(self, idx, max_new_tokens, temperature=1.0, top_p=0.9):
        """
        Generate text using top-k and/or top-p (nucleus) sampling.

        Args:
            idx: Tensor of shape (B, T)
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: top-k filtering (int)
            top_p: top-p (nucleus) sampling (float, in [0, 1])
        """
        pass
        # TODO: Implement text generation with top-p (nucleus) sampling.
        # top_p: top-p (nucleus) sampling (float, in [0, 1])

```

### 5. README

```md
## Training
~~~
python train.py config/train_wikitext.py
~~~

Note: you can modify training configurations based on your device.
Modify Line 10 in train.py to switch between different model architectures.
By default we use cpu to train the model. If you want to use a gpu, please modify Line 51 of train.py.

## Sampling
~~~
python sample.py --out_dir=YOUR_MODEL_DIR_PATH
~~~
Modify Line 9 in sample.py to switch between different model architectures.
```

### 6. sample.py

```py
"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model_my_implementation_rope import ModelConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 5 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 42
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = ModelConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')

```

### 7. train.py

```py
import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch

from model_complete import ModelConfig, GPT

# ----------------------------- Configuration ----------------------------------
# Default values for GPT-2 (124M) training on WikiText
# I/O
out_dir = 'out-wikitext'
eval_interval = 250
log_interval = 10
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch' 

# Data
dataset = 'wikitext'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024

# Model Architecture
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.2
bias = True

# Optimizer Settings
learning_rate = 6e-4
max_iters = 20000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning Rate Schedule
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 20000
min_lr = 6e-5

# System
device = 'cpu' # you can set device to 'cuda' if you are using a gpu
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True

# Override config from CLI/config file
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}

# ----------------------------- Initialization ----------------------------------
seed_offset = 0

tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(42 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

# ----------------------------- Dataset Loader ----------------------------------
data_dir = os.path.join('data', dataset)

def get_batch(split):
    data_path = os.path.join(data_dir, f'{split}.bin')
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)

# ----------------------------- Model Initialization ----------------------------
iter_num = 0
best_val_loss = 1e9
meta_path = os.path.join(data_dir, 'meta.pkl')

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=50304, dropout=dropout)

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    gptconf = ModelConfig(**model_args)
    model = GPT(gptconf)

elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    checkpoint = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device)
    for k in model_args:
        if k in checkpoint['model_args']:
            model_args[k] = checkpoint['model_args'][k]
    model = GPT(ModelConfig(**model_args))
    state_dict = checkpoint['model']
    for k in list(state_dict):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model.to(device)
raw_model = model

# scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
# you could use mixed precision training if you are familar with it
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

# ----------------------------- Evaluation and Training --------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0

while iter_num <= max_iters:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                print(f"saving checkpoint to {out_dir}")
                torch.save({
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # TODO: Implement the gradient accumulation process
    # accumulate the gradient with several forward-backward process, then call optimizer.step() to update model parameters

    t1 = time.time()
    if iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            print(f"iter {iter_num}: loss {lossf:.4f}, time {(t1 - t0)*1000:.2f}ms")
    t0 = t1

    iter_num += 1
    local_iter_num += 1

```


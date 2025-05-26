你好！我是计算机系的一名学生，最近在完成一门人工智能课程作业。

其要求、代码结构、相关要求均如下所示。

我已经完成了第一题和第二题和第三题的要求！并且自己设计了实验，得出了比较好的实验结论啦。

我现在希望你能帮我看看第四题应该怎么写？

按理来说应该只需要改TODO部分的代码，我希望你从一个程序员的视角，帮我一步一步地完成第四题的要求

并且帮我设计一下实验并讲一讲预期结果





## 编程题：语言模型

> 在本题中，你将基于简化版本的 GPT 语言模型框架，深人理解语言模型的关键结构模块的实现原理和设计理念。我们已经完成了基础的模型定义，包括嵌人层、位置编码、注意力机制、前馈网络以及残差连接，你需要实现模型中的关键计算模块或替换其中的部分模块。
>
> 注意：本题主要考察代码实现的正确性。如果你的设备性能受限导致训练时间过长，你可以适当调整训练参数以缩短整体训练时间。

### 4.1. 梯度累积（Gradient Accumulation）

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

> 在我们给出的代码中，已经实现了语言模型的 top－k 采样策略，即每次生成新的 token 时，从概率大小前 k 的 token 中进行采样。还有一种所谓的 top－p 采样策略，每次只从累积概率超过阈值 $p$ 的最小单词集合中进行随机采样。其形式化定义如下。
>
> 设当前时刻模型输出的 token 概率分布为：
>
> $$
> P=\operatorname{Softmax}(z) \in \mathbb{R}^{|V|}
> $$
>
> 其中 $z \in \mathbb{R}^{|V|}$ 是 logits 向量，$|V|$ 是词表大小。
>
> 我们将所有 token 按概率从大到小排序，记排序后 token 的索引为：
>
> $$
> \pi=\operatorname{argsort}(P), \quad \text { 使得 } P_{\pi_1} \geq P_{\pi_2} \geq \cdots \geq P_{\pi_{|V|}}
> $$
>
>
> 选取最小的前 $k$ 个 token，使得它们的概率累计和不小于阈值 $p$ ：
>
> $$
> \sum_{i=1}^k P_{\pi_i} \geq p, \quad \text { 且 } \quad \sum_{i=1}^{k-1} P_{\pi_i}<p
> $$
>
>
> 最终仅在这些 token 上归一化进行采样：
>
> $$
> \hat{P}_i=\left\{\begin{array}{ll}
> \frac{P_i}{\sum_{j \in S} P_j}, & i \in S \\
> 0, & i \notin S
> \end{array} \quad \text { 其中 } S=\left\{\pi_1, \ldots, \pi_k\right\}\right.
> $$
>
> 请参考现有模型中的 generate 函数，实现使用 top－p 策略的采样



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
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_size = config.n_embd // config.n_head
        
        # QKV线性映射层
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # 输出投影层
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # dropout层
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # 注册一个下三角矩阵作为缓存，用于causal mask
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )
        
        # RoPE 相关参数，使用全部维度进行旋转(确保维度是偶数)
        assert self.head_size % 2 == 0, f"head_size must be even for RoPE, got {self.head_size}"
        self.rotary_dim = self.head_size
        
        # 预计算旋转频率
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim))
        self.register_buffer("inv_freq", inv_freq)
    

        # TODO: Implement the CausalSelfAttention class with RoPE positional Embedding
        # Attributes that could possibly be used: config.n_embd, config.n_head, config.dropout, config.bias
        

    def apply_rotary_pos_emb(self, q, k, seq_len):
        # 应用 RoPE 旋转位置编码
        position_ids = torch.arange(seq_len, device=q.device).float()
        freqs = torch.outer(position_ids, self.inv_freq)
        cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
        sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat([-x2, x1], dim=-1)
        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin
        return q_rot, k_rot

    def forward(self, x):
        
        # x的形状: (batch_size, seq_len, n_embd)
        batch_size, seq_length, n_embd = x.size()
        
        # 1. 计算QKV
        qkv = self.c_attn(x)
        
        # 2. 拆分并重塑为多头形式,将每部分重塑为多头形式并转置
        q, k, v = qkv.chunk(3, dim=2)
        q = q.view(batch_size, seq_length, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.n_head, self.head_size).transpose(1, 2)
        
        # 3. 应用 RoPE 到 Q 和 K
        q, k = self.apply_rotary_pos_emb(q, k, seq_length)
        
        # 4. 计算缩放点积注意力
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # 5. 应用causal mask
        mask = self.mask[:, :, :seq_length, :seq_length]
        att = att.masked_fill(mask == 0, float('-inf'))
        
        # 6. 应用softmax归一化
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # 7. 计算上下文表示：注意力加权的值
        y = att @ v
        
        # 8. 合并多头
        y = y.transpose(1, 2).contiguous()
        y = y.view(batch_size, seq_length, n_embd)
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        
        return y
    
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
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_size = config.n_embd // config.n_head
        
        # QKV线性映射层（使用单个线性层同时计算Q、K、V）
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # 输出投影层
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # dropout层
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # 注册一个下三角矩阵作为缓存，用于causal mask
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):

        # x的形状: (batch_size, seq_len, n_embd)
        batch_size, seq_length, n_embd = x.size()
        
        # 1. 计算QKV (batch, seq_len, 3*n_embd)
        qkv = self.c_attn(x)
        
        # 2. 拆分并重塑为多头形式，将整个张量分成三部分：Q, K, V
        q, k, v = qkv.chunk(3, dim=2)
        
        # 将每部分重塑为多头形式并转置
        q = q.view(batch_size, seq_length, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.n_head, self.head_size).transpose(1, 2)
        
        # 3. 计算缩放点积注意力
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # 4. 应用causal mask，确保位置i只能关注位置j≤i
        mask = self.mask[:, :, :seq_length, :seq_length]
        att = att.masked_fill(mask == 0, float('-inf'))
        
        # 5. 应用softmax归一化
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # 6. 计算上下文表示：注意力加权的值
        y = att @ v
        
        # 7. 合并多头
        y = y.transpose(1, 2).contiguous()
        y = y.view(batch_size, seq_length, n_embd)
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        
        return y

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
import logging

import numpy as np
import torch

from model import ModelConfig, GPT


# ----------------------------- Configuration ----------------------------------
# Default values for GPT-2 (124M) training on WikiText
# I/O
out_dir = 'out-wikitext'
eval_interval = 50
log_interval = 50
eval_iters = 50
eval_only = False
always_save_checkpoint = True
init_from = 'scratch' 

# Data
dataset = 'wikitext'
gradient_accumulation_steps = 5 * 8
batch_size = 8
block_size = 128

# Model Architecture
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.2
bias = True

# Optimizer Settings
learning_rate = 6e-4
max_iters = 2000
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
device = 'cuda' # you can set device to 'cuda' if you are using a gpu
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False

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
        # print(f"Evaluating on {split}...")
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            # if k % 50 == 0:  # 每50次迭代输出一次
                # print(f"  Evaluation iter {k}/{eval_iters}")
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
        # print(f"Finished {split} evaluation, loss: {out[split]:.4f}") 
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
                # print(f"saving checkpoint to {out_dir}")
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


    # 开始前先将梯度清零
    optimizer.zero_grad()
    
    # 累积多个小batch的梯度
    for micro_step in range(gradient_accumulation_steps):
        if micro_step > 0:
            X, Y = get_batch('train')
        
        logits, loss = model(X, Y)
        loss = loss / gradient_accumulation_steps
        
        loss.backward()
    
    # 梯度裁剪防止梯度爆炸
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # 更新参数
    optimizer.step()

    t1 = time.time()
    if iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accumulation_steps
        # if local_iter_num >= 5:
        #     print(f"iter {iter_num}: loss {lossf:.4f}, time {(t1 - t0)*1000:.2f}ms")
    t0 = t1

    iter_num += 1
    local_iter_num += 1

```

### 8. config/train_wikitext.py

```py
out_dir = 'out-wikitext'
eval_interval = 50 
eval_iters = 50
log_interval = 50 

always_save_checkpoint = False

wandb_log = False 
wandb_project = 'wikitext_large'
wandb_run_name = 'mini-gpt'

dataset = 'wikitext_large'
gradient_accumulation_steps = 8
batch_size = 8
block_size = 128

n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.2

learning_rate = 1e-3 
max_iters = 2000
lr_decay_iters = 2000
min_lr = 1e-4 
beta2 = 0.99 

warmup_iters = 100 


```





## 实验数据记录

```
// 实验一：梯度累积实验
// 参数设置（电脑性能不太好，把参数都调低了）
// eval_interval = 50
// eval_iters = 50
// log_interval = 50
// batch_size = 8
// block_size = 128
// n_layer = 4
// n_head = 4
// n_embd = 256
// gradient_accumulation_steps = 1 or 8

// 实验1：梯度累积实验(naive)
// 参数设置：gradient_accumulation_steps=1

step 0: train loss 10.8687, val loss 10.8612
step 50: train loss 8.6614, val loss 8.9377
step 100: train loss 9.5972, val loss 10.0618
step 150: train loss 10.5328, val loss 10.9315
step 200: train loss 10.6483, val loss 11.0833
step 250: train loss 10.8252, val loss 11.2894
step 300: train loss 10.8880, val loss 11.3613
step 350: train loss 10.9829, val loss 11.3378
step 400: train loss 11.0517, val loss 11.4744
step 450: train loss 11.2346, val loss 11.6310
step 500: train loss 11.1710, val loss 11.4906
step 550: train loss 11.3033, val loss 11.7154
step 600: train loss 11.5041, val loss 11.8897
step 650: train loss 11.7044, val loss 12.1023
step 700: train loss 11.6033, val loss 11.9441
step 750: train loss 11.5800, val loss 11.9206
step 800: train loss 11.7217, val loss 12.0807
step 850: train loss 11.8720, val loss 12.2190
step 900: train loss 11.7532, val loss 12.0825
step 950: train loss 11.7374, val loss 12.0701
step 1000: train loss 11.8302, val loss 12.1709
step 1050: train loss 11.8813, val loss 12.1523
step 1100: train loss 11.8738, val loss 12.1829
step 1150: train loss 12.0026, val loss 12.4131
step 1200: train loss 12.0670, val loss 12.3772
step 1250: train loss 12.1074, val loss 12.4054
step 1300: train loss 12.1954, val loss 12.5282
step 1350: train loss 12.1704, val loss 12.3990
step 1400: train loss 12.1784, val loss 12.4475
step 1450: train loss 12.4197, val loss 12.6230
step 1500: train loss 12.3797, val loss 12.6665
step 1550: train loss 12.4507, val loss 12.7866
step 1600: train loss 12.4796, val loss 12.7887
step 1650: train loss 12.4841, val loss 12.8198
step 1700: train loss 12.5191, val loss 12.8368
step 1750: train loss 12.5671, val loss 12.9021
step 1800: train loss 12.6172, val loss 12.9194
step 1850: train loss 12.7098, val loss 12.9565
step 1900: train loss 12.7207, val loss 12.9662
step 1950: train loss 12.7523, val loss 13.0501
step 2000: train loss 12.7623, val loss 13.0877

// 实验1：梯度累积实验(梯度累积)
// 参数设置：gradient_accumulation_steps=8
step 0: train loss 10.8687, val loss 10.8612
step 50: train loss 7.7211, val loss 7.3998
step 100: train loss 6.6847, val loss 5.9085
step 150: train loss 6.3893, val loss 5.6637
step 200: train loss 6.2696, val loss 5.5327
step 250: train loss 6.1367, val loss 5.4427
step 300: train loss 6.1035, val loss 5.3920
step 350: train loss 6.0469, val loss 5.3374
step 400: train loss 5.9526, val loss 5.3672
step 450: train loss 5.9053, val loss 5.2776
step 500: train loss 5.8906, val loss 5.3069
step 550: train loss 5.8820, val loss 5.3083
step 600: train loss 5.8300, val loss 5.3190
step 650: train loss 5.7692, val loss 5.2925
step 700: train loss 5.8003, val loss 5.2605
step 750: train loss 5.7436, val loss 5.2619
step 800: train loss 5.7139, val loss 5.2769
step 850: train loss 5.6579, val loss 5.2022
step 900: train loss 5.6612, val loss 5.2253
step 950: train loss 5.6659, val loss 5.2594
step 1000: train loss 5.6437, val loss 5.2367
step 1050: train loss 5.6053, val loss 5.1830
step 1100: train loss 5.6543, val loss 5.1824
step 1150: train loss 5.6068, val loss 5.1869
step 1200: train loss 5.6151, val loss 5.1794
step 1250: train loss 5.5847, val loss 5.2259
step 1300: train loss 5.5835, val loss 5.2234
step 1350: train loss 5.5644, val loss 5.1748
step 1400: train loss 5.5977, val loss 5.1808
step 1450: train loss 5.5623, val loss 5.1863
step 1500: train loss 5.5510, val loss 5.1744
step 1550: train loss 5.5249, val loss 5.1796
step 1600: train loss 5.5298, val loss 5.1436
step 1650: train loss 5.5045, val loss 5.1450
step 1700: train loss 5.5185, val loss 5.1488
step 1750: train loss 5.5168, val loss 5.1154
step 1800: train loss 5.4770, val loss 5.1791
step 1850: train loss 5.5181, val loss 5.1240
step 1900: train loss 5.4775, val loss 5.1478
step 1950: train loss 5.4770, val loss 5.1090
step 2000: train loss 5.4998, val loss 5.0990
```

```
// 实验二：因果自注意力机制实验
// 参数设置（电脑性能不太好，把参数都调低了）
// eval_interval = 50
// eval_iters = 50
// log_interval = 50
// batch_size = 8
// block_size = 128
// n_layer = 4
// n_head = 4
// n_embd = 256
// gradient_accumulation_steps = 8


// 实验2：自注意力机制（naive）
step 0: train loss 10.8687, val loss 10.8612
step 50: train loss 7.7211, val loss 7.3998
step 100: train loss 6.6847, val loss 5.9085
step 150: train loss 6.3893, val loss 5.6637
step 200: train loss 6.2696, val loss 5.5327
step 250: train loss 6.1367, val loss 5.4427
step 300: train loss 6.1035, val loss 5.3920
step 350: train loss 6.0469, val loss 5.3374
step 400: train loss 5.9526, val loss 5.3672
step 450: train loss 5.9053, val loss 5.2776
step 500: train loss 5.8906, val loss 5.3069
step 550: train loss 5.8820, val loss 5.3083
step 600: train loss 5.8300, val loss 5.3190
step 650: train loss 5.7692, val loss 5.2925
step 700: train loss 5.8003, val loss 5.2605
step 750: train loss 5.7436, val loss 5.2619
step 800: train loss 5.7139, val loss 5.2769
step 850: train loss 5.6579, val loss 5.2022
step 900: train loss 5.6612, val loss 5.2253
step 950: train loss 5.6659, val loss 5.2594
step 1000: train loss 5.6437, val loss 5.2367
step 1050: train loss 5.6053, val loss 5.1830
step 1100: train loss 5.6543, val loss 5.1824
step 1150: train loss 5.6068, val loss 5.1869
step 1200: train loss 5.6151, val loss 5.1794
step 1250: train loss 5.5847, val loss 5.2259
step 1300: train loss 5.5835, val loss 5.2234
step 1350: train loss 5.5644, val loss 5.1748
step 1400: train loss 5.5977, val loss 5.1808
step 1450: train loss 5.5623, val loss 5.1863
step 1500: train loss 5.5510, val loss 5.1744
step 1550: train loss 5.5249, val loss 5.1796
step 1600: train loss 5.5298, val loss 5.1436
step 1650: train loss 5.5045, val loss 5.1450
step 1700: train loss 5.5185, val loss 5.1488
step 1750: train loss 5.5168, val loss 5.1154
step 1800: train loss 5.4770, val loss 5.1791
step 1850: train loss 5.5181, val loss 5.1240
step 1900: train loss 5.4775, val loss 5.1478
step 1950: train loss 5.4770, val loss 5.1090
step 2000: train loss 5.4998, val loss 5.0990

// 实验2：自注意力机制（自注意力版本）
step 0: train loss 10.8662, val loss 10.8542
step 50: train loss 7.5865, val loss 7.1151
step 100: train loss 6.6412, val loss 5.8539
step 150: train loss 6.3711, val loss 5.5990
step 200: train loss 6.1624, val loss 5.4743
step 250: train loss 5.9518, val loss 5.3351
step 300: train loss 5.9141, val loss 5.2737
step 350: train loss 5.7503, val loss 5.2125
step 400: train loss 5.6557, val loss 5.1563
step 450: train loss 5.6166, val loss 5.0761
step 500: train loss 5.5589, val loss 4.9830
step 550: train loss 5.5321, val loss 4.9942
step 600: train loss 5.4571, val loss 4.9575
step 650: train loss 5.3777, val loss 4.9157
step 700: train loss 5.3466, val loss 4.9017
step 750: train loss 5.3151, val loss 4.8989
step 800: train loss 5.2337, val loss 4.8804
step 850: train loss 5.2252, val loss 4.8674
step 900: train loss 5.1698, val loss 4.7963
step 950: train loss 5.1331, val loss 4.7871
step 1000: train loss 5.0937, val loss 4.7383
step 1050: train loss 5.0598, val loss 4.7787
step 1100: train loss 5.0431, val loss 4.7035
step 1150: train loss 5.0246, val loss 4.7914
step 1200: train loss 5.0566, val loss 4.7793
step 1250: train loss 4.9954, val loss 4.7369
step 1300: train loss 4.9862, val loss 4.7073
step 1350: train loss 4.9738, val loss 4.7048
step 1400: train loss 4.9081, val loss 4.7076
step 1450: train loss 4.9727, val loss 4.6432
step 1500: train loss 4.9203, val loss 4.5844
step 1550: train loss 4.9268, val loss 4.6486
step 1600: train loss 4.8944, val loss 4.6228
step 1650: train loss 4.9005, val loss 4.6196
step 1700: train loss 4.8782, val loss 4.6330
step 1750: train loss 4.8147, val loss 4.6198
step 1800: train loss 4.8108, val loss 4.5844
step 1850: train loss 4.8373, val loss 4.6096
step 1900: train loss 4.8561, val loss 4.5650
step 1950: train loss 4.8106, val loss 4.5878
step 2000: train loss 4.8221, val loss 4.5844
```

```
// 实验三：位置编码实现与对比
// 参数设置（RoPE的效果会在长序列的情况下体现更好，我们稍微调大了一些参数），并将参数放置在了'train_wikitext_long.py'
// eval_interval = 50
// eval_iters = 50
// log_interval = 50
// batch_size = 8
// block_size = 256
// n_layer = 8
// n_head = 8
// n_embd = 512
// gradient_accumulation_steps = 8
// max_iters = 5000 (之前两个实验是2000，我们调大了)

// 实验3：位置编码(naive版本)
step 0: train loss 10.9455, val loss 10.9393
step 50: train loss 6.9422, val loss 6.1181
step 100: train loss 6.3389, val loss 5.6308
step 150: train loss 6.0167, val loss 5.3877
step 200: train loss 5.8298, val loss 5.2515
step 250: train loss 5.7321, val loss 5.1689
step 300: train loss 5.5972, val loss 5.0427
step 350: train loss 5.4711, val loss 4.9975
step 400: train loss 5.3834, val loss 4.9059
step 450: train loss 5.2072, val loss 4.8584
step 500: train loss 5.1690, val loss 4.8475
step 550: train loss 5.0770, val loss 4.6871
step 600: train loss 4.9966, val loss 4.7322
step 650: train loss 4.9436, val loss 4.6504
step 700: train loss 4.8544, val loss 4.6145
step 750: train loss 4.7970, val loss 4.5346
step 800: train loss 4.7999, val loss 4.5473
step 850: train loss 4.7277, val loss 4.4899
step 900: train loss 4.6679, val loss 4.4068
step 950: train loss 4.6435, val loss 4.4515
step 1000: train loss 4.5691, val loss 4.4060
step 1050: train loss 4.5461, val loss 4.3444
step 1100: train loss 4.5618, val loss 4.4044
step 1150: train loss 4.5230, val loss 4.3773
step 1200: train loss 4.4589, val loss 4.3113
step 1250: train loss 4.4076, val loss 4.3543
step 1300: train loss 4.3561, val loss 4.3246
step 1350: train loss 4.3843, val loss 4.3034
step 1400: train loss 4.3101, val loss 4.3189
step 1450: train loss 4.3308, val loss 4.2536
step 1500: train loss 4.2461, val loss 4.2929
step 1550: train loss 4.2189, val loss 4.2083
step 1600: train loss 4.2372, val loss 4.2740
step 1650: train loss 4.2541, val loss 4.3002
step 1700: train loss 4.1463, val loss 4.2354
step 1750: train loss 4.2132, val loss 4.2163
step 1800: train loss 4.1521, val loss 4.2311
step 1850: train loss 4.1513, val loss 4.1745
step 1900: train loss 4.0683, val loss 4.1788
step 1950: train loss 4.0894, val loss 4.1860
step 2000: train loss 4.0983, val loss 4.1668
step 2050: train loss 4.0741, val loss 4.1577
step 2100: train loss 4.0147, val loss 4.1232
step 2150: train loss 4.0283, val loss 4.0747
step 2200: train loss 3.9467, val loss 4.0794
step 2250: train loss 3.9866, val loss 4.1750
step 2300: train loss 3.8887, val loss 4.1293
step 2350: train loss 3.9563, val loss 4.0918
step 2400: train loss 3.9160, val loss 4.0921
step 2450: train loss 3.8699, val loss 4.1425
step 2500: train loss 3.9941, val loss 4.0736
step 2550: train loss 3.8926, val loss 4.0992
step 2600: train loss 3.8815, val loss 4.0890
step 2650: train loss 3.8377, val loss 4.0955
step 2700: train loss 3.8966, val loss 4.0574
step 2750: train loss 3.8170, val loss 4.0415
step 2800: train loss 3.8542, val loss 4.0491
step 2850: train loss 3.8319, val loss 4.0531
step 2900: train loss 3.8269, val loss 4.0670
step 2950: train loss 3.7905, val loss 4.0091
step 3000: train loss 3.7817, val loss 4.0031
step 3050: train loss 3.7596, val loss 4.1076
step 3100: train loss 3.7587, val loss 4.0176
step 3150: train loss 3.7896, val loss 3.9710
step 3200: train loss 3.7083, val loss 3.9989
step 3250: train loss 3.7531, val loss 4.0281
step 3300: train loss 3.6688, val loss 3.9744
step 3350: train loss 3.7595, val loss 3.9656
step 3400: train loss 3.7041, val loss 4.0241
step 3450: train loss 3.7197, val loss 3.9633
step 3500: train loss 3.7104, val loss 3.9638
step 3550: train loss 3.6615, val loss 3.9822
step 3600: train loss 3.6893, val loss 4.0166
step 3650: train loss 3.6431, val loss 3.9692
step 3700: train loss 3.6705, val loss 3.9320
step 3750: train loss 3.6507, val loss 3.9832
step 3800: train loss 3.6378, val loss 3.9518
step 3850: train loss 3.6349, val loss 3.9343
step 3900: train loss 3.6565, val loss 3.9158
step 3950: train loss 3.5764, val loss 3.9777
step 4000: train loss 3.6038, val loss 3.9689
step 4050: train loss 3.6143, val loss 3.9454
step 4100: train loss 3.6278, val loss 3.9259
step 4150: train loss 3.6197, val loss 3.9279
step 4200: train loss 3.5717, val loss 3.9459
step 4250: train loss 3.5220, val loss 3.9271
step 4300: train loss 3.6055, val loss 3.9949
step 4350: train loss 3.5935, val loss 3.9140
step 4400: train loss 3.5356, val loss 3.9206
step 4450: train loss 3.5806, val loss 3.9726
step 4500: train loss 3.5661, val loss 3.9143
step 4550: train loss 3.5447, val loss 3.9244
step 4600: train loss 3.5922, val loss 3.8771
step 4650: train loss 3.5541, val loss 3.8494
step 4700: train loss 3.5456, val loss 3.9636
step 4750: train loss 3.5464, val loss 3.9547
step 4800: train loss 3.5855, val loss 3.9049
step 4850: train loss 3.5383, val loss 3.9214
step 4900: train loss 3.5396, val loss 3.9005
step 4950: train loss 3.5125, val loss 3.8836
step 5000: train loss 3.5618, val loss 3.8967


//实验3：位置编码（RoPE版本)
step 0: train loss 10.9134, val loss 10.8852
step 50: train loss 6.8501, val loss 6.0032
step 100: train loss 6.1983, val loss 5.5790
step 150: train loss 5.8455, val loss 5.2754
step 200: train loss 5.6569, val loss 5.1338
step 250: train loss 5.4524, val loss 5.0028
step 300: train loss 5.3144, val loss 4.9076
step 350: train loss 5.2020, val loss 4.8459
step 400: train loss 5.1265, val loss 4.7520
step 450: train loss 4.9618, val loss 4.6859
step 500: train loss 4.9484, val loss 4.6452
step 550: train loss 4.8678, val loss 4.6019
step 600: train loss 4.7936, val loss 4.5169
step 650: train loss 4.7583, val loss 4.5237
step 700: train loss 4.7325, val loss 4.4976
step 750: train loss 4.6471, val loss 4.4500
step 800: train loss 4.6244, val loss 4.4284
step 850: train loss 4.5508, val loss 4.4235
step 900: train loss 4.5627, val loss 4.4031
step 950: train loss 4.4933, val loss 4.3828
step 1000: train loss 4.4569, val loss 4.3675
step 1050: train loss 4.4472, val loss 4.3388
step 1100: train loss 4.3431, val loss 4.2489
step 1150: train loss 4.3457, val loss 4.3582
step 1200: train loss 4.2828, val loss 4.2172
step 1250: train loss 4.2678, val loss 4.2813
step 1300: train loss 4.2434, val loss 4.2074
step 1350: train loss 4.2331, val loss 4.2139
step 1400: train loss 4.2215, val loss 4.3104
step 1450: train loss 4.1614, val loss 4.2397
step 1500: train loss 4.2170, val loss 4.1798
step 1550: train loss 4.1910, val loss 4.2154
step 1600: train loss 4.1248, val loss 4.1915
step 1650: train loss 4.0961, val loss 4.1622
step 1700: train loss 4.0992, val loss 4.1380
step 1750: train loss 4.0479, val loss 4.2179
step 1800: train loss 4.0153, val loss 4.1364
step 1850: train loss 3.9903, val loss 4.1494
step 1900: train loss 3.9807, val loss 4.1418
step 1950: train loss 3.9307, val loss 4.1074
step 2000: train loss 3.9817, val loss 4.1442
step 2050: train loss 3.9301, val loss 4.0428
step 2100: train loss 3.8849, val loss 4.1044
step 2150: train loss 3.9086, val loss 4.0404
step 2200: train loss 3.8788, val loss 4.0825
step 2250: train loss 3.8975, val loss 4.0911
step 2300: train loss 3.8657, val loss 4.0686
step 2350: train loss 3.8465, val loss 4.0094
step 2400: train loss 3.8695, val loss 4.0312
step 2450: train loss 3.8104, val loss 4.0029
step 2500: train loss 3.8167, val loss 4.0262
step 2550: train loss 3.7770, val loss 4.0910
step 2600: train loss 3.7879, val loss 4.0022
step 2650: train loss 3.7501, val loss 4.0000
step 2700: train loss 3.7758, val loss 3.9675
step 2750: train loss 3.7816, val loss 4.0192
step 2800: train loss 3.7053, val loss 4.0351
step 2850: train loss 3.7316, val loss 4.0450
step 2900: train loss 3.7107, val loss 3.9708
step 2950: train loss 3.6584, val loss 3.9544
step 3000: train loss 3.6833, val loss 3.9919
step 3050: train loss 3.6640, val loss 3.9803
step 3100: train loss 3.6613, val loss 4.0053
step 3150: train loss 3.6478, val loss 3.9594
step 3200: train loss 3.6396, val loss 3.9496
step 3250: train loss 3.6617, val loss 3.9530
step 3300: train loss 3.6457, val loss 3.9573
step 3350: train loss 3.5653, val loss 3.9896
step 3400: train loss 3.6229, val loss 3.9104
step 3450: train loss 3.6030, val loss 3.9324
step 3500: train loss 3.6062, val loss 3.9169
step 3550: train loss 3.5364, val loss 3.9296
step 3600: train loss 3.5964, val loss 3.8600
step 3650: train loss 3.5540, val loss 3.9112
step 3700: train loss 3.5714, val loss 3.9278
step 3750: train loss 3.5658, val loss 3.9385
step 3800: train loss 3.5548, val loss 3.8757
step 3850: train loss 3.5622, val loss 3.8629
step 3900: train loss 3.5308, val loss 3.8786
step 3950: train loss 3.5629, val loss 3.9325
step 4000: train loss 3.5001, val loss 3.9054
step 4050: train loss 3.5362, val loss 3.8887
step 4100: train loss 3.5174, val loss 3.8705
step 4150: train loss 3.4808, val loss 3.8663
step 4200: train loss 3.5641, val loss 3.8966
step 4250: train loss 3.4897, val loss 3.8764
step 4300: train loss 3.4565, val loss 3.8998
step 4350: train loss 3.4854, val loss 3.8928
step 4400: train loss 3.4614, val loss 3.8238
step 4450: train loss 3.4492, val loss 3.8877
step 4500: train loss 3.4556, val loss 3.8548
step 4550: train loss 3.4939, val loss 3.8246
step 4600: train loss 3.4199, val loss 3.8818
step 4650: train loss 3.4584, val loss 3.8324
step 4700: train loss 3.4490, val loss 3.8885
step 4750: train loss 3.4572, val loss 3.9002
step 4800: train loss 3.4471, val loss 3.8529
step 4850: train loss 3.4132, val loss 3.8817
step 4900: train loss 3.4462, val loss 3.8659
step 4950: train loss 3.4515, val loss 3.9219
step 5000: train loss 3.4169, val loss 3.8175
```


"""
feed_forward.py —— 前馈神经网络

作用：每个词的向量经过 Attention 处理、"融合了上下文信息"之后，
      再单独过一个小型的两层全连接网络，做进一步的非线性变换。

设计要点（对应设计文档 3.5 节）：
    - 结构很简单：Linear(d_model, d_ff) -> ReLU -> Dropout -> Linear(d_ff, d_model)
    - d_ff 通常比 d_model 大很多（论文里 d_model=512, d_ff=2048），
      可以理解成"先把维度打开变大，让模型有更多空间做非线性变换，再压缩回原来的维度"
    - 这个变换是"逐位置(position-wise)"独立进行的：句子里第 1 个词和第 5 个词
      在这一步的计算互不影响，只有前面的 Attention 层才会让不同位置的词互相"交流"。
"""

import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """
    输入/输出形状都是 (batch_size, seq_len, d_model)，中间维度会先放大到 d_ff 再压缩回来。
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        参数：
            d_model: 词向量维度，需要和模型其他部分一致
            d_ff: 中间隐藏层维度，论文里是 d_model 的 4 倍左右(512 -> 2048)
            dropout: 两层线性层之间的 dropout 比例
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear2(self.dropout(self.linear1(x).relu()))

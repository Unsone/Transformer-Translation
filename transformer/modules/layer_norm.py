"""
layer_norm.py —— 残差连接与归一化

作用：让深层网络能够训练稳定收敛的"稳定器"，论文里每个子层
      (Self-Attention、Cross-Attention、Feed Forward)后面都会接这个
      （对应设计文档 3.6 节）。

两个技术合在一起用：
    1. 残差连接 (Residual Connection)：输出 = 子层(输入) + 输入。
       好处：梯度可以直接"抄近道"传回浅层，不容易梯度消失，深层网络更容易训练。
    2. Layer Normalization：对每个词自己的向量（沿最后一维）做归一化(均值0方差1)，
       让每一层的输入分布更稳定，训练收敛更快。

本文件采用和原论文一致的 **Post-Norm** 结构：先算子层，再残差相加，最后做归一化：
    输出 = LayerNorm(x + Dropout(Sublayer(x)))

封装成 SublayerConnection 类，这样在 encoder.py / decoder.py 里可以写得很简洁：
    x = sublayer_connection(x, lambda x: self_attention(x, x, x, mask))

补充说明（进阶内容，初学阶段可以跳过）：
    后来很多改进版 Transformer（比如 GPT 系列)常用 **Pre-Norm** 结构
    (先归一化再算子层：x + Dropout(Sublayer(LayerNorm(x))))，训练更稳定、更容易调参。
    如果你把项目跑通之后想做实验对比，可以回来改这个文件试试 Pre-Norm 的效果。
"""

import torch.nn as nn


class SublayerConnection(nn.Module):
    """
    "子层 + 残差 + LayerNorm" 的通用封装。

    使用方式（sublayer 传一个函数进来，函数的输入输出形状要一致）：
        connection = SublayerConnection(d_model, dropout)
        x = connection(x, lambda x: multi_head_attention(x, x, x, mask))
        x = connection(x, feed_forward)
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        """
        参数：
            d_model: 词向量维度，LayerNorm 是沿这一维做归一化的
            dropout: 子层输出之后、残差相加之前的 dropout 比例
        """
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        参数：
            x: (batch, seq_len, d_model)，这一层的输入
            sublayer: 一个函数（或者说一个可调用对象），输入输出形状都是 (batch, seq_len, d_model)，
                      可以是 MultiHeadAttention 或 PositionwiseFeedForward 的调用

        返回：
            (batch, seq_len, d_model)，形状不变
        """
        return self.norm(x + self.dropout(sublayer(x)))

"""
positional_encoding.py —— 位置编码

作用：给每个位置的词向量"打上位置标签"，让模型知道词的先后顺序。

设计要点（对应设计文档 3.2 节）：
    - Attention 机制本身是并行处理所有词的，不知道谁在前谁在后，
      所以需要额外注入位置信息，"我爱你"和"你爱我"才不会被模型看成一样的东西。
    - 用论文里固定的正弦/余弦公式生成一张位置编码表，不需要训练，
      所以用 register_buffer 存起来（会随模型 .to(device) 一起搬到 GPU，
      但不会出现在 optimizer 要更新的参数列表里）。
    - 用法：位置编码是"加"到词向量上的，不是拼接。
        x = embedding(tokens)              # (batch, seq_len, d_model)
        x = positional_encoding(x)         # 形状不变，但每个位置的向量里
                                            # 混入了"我在第几个位置"的信息
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    位置编码层。

    输入/输出形状都是 (batch_size, seq_len, d_model)，只是给输入加上了位置信息。
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        参数：
            d_model: 词向量维度，必须和 embedding.py 里的 d_model 一致
            max_len: 位置编码最多支持多长的句子，只要比训练/推理时用到的最大句子长度大即可，
                     对应 config.py 里的 max_len
            dropout: 在"词向量 + 位置编码"之后加一层 dropout，防止过拟合
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 提前算好一张 (max_len, d_model) 的位置编码表 pe，
        # pe[pos, i] 表示"第 pos 个位置"在"第 i 个维度"上的编码值。
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)

        # div_term 是公式里的 10000^(2i/d_model)，这里用 exp+log 是为了数值稳定，
        # 等价于 1 / (10000 ** (2i / d_model))
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度用 cos

        pe = pe.unsqueeze(0)  # 加一个 batch 维度 -> (1, max_len, d_model)，方便后面广播相加

        # register_buffer：不是可学习参数，但要随模型一起保存/搬设备
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        # 只取位置编码表里前 seq_len 个位置，加到词向量上
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

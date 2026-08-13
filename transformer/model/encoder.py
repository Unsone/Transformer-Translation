"""
encoder.py —— 编码器

作用：把源语言句子(比如英文)编码成一系列"富含上下文信息"的向量表示
      （对应设计文档 4.1 节）。

结构分两层：
    EncoderLayer：单层编码器，内部有两个子层
        1. Multi-Head Self-Attention（自己关注自己）
        2. Feed Forward
       每个子层都包一层 SublayerConnection（残差 + LayerNorm）
    Encoder：把 EncoderLayer 堆叠 N 层（论文里 N=6），
             每层结构相同但参数不共享，前一层输出是后一层输入
"""

import torch.nn as nn

from ..modules.attention import MultiHeadAttention
from ..modules.feed_forward import PositionwiseFeedForward
from ..modules.layer_norm import SublayerConnection


class EncoderLayer(nn.Module):
    """
    单层编码器。

    输入/输出形状都是 (batch, src_len, d_model)，形状不变，
    只是每个位置的向量"融合了更多上下文信息"。
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        参数：
            d_model: 词向量维度
            num_heads: 多头注意力的头数
            d_ff: 前馈网络中间层维度
            dropout: 各子层里统一使用的 dropout 比例
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # 两个子层各配一个 SublayerConnection（负责残差 + LayerNorm）
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)

    def forward(self, x, src_mask):
        """
        参数：
            x: (batch, src_len, d_model)，上一层的输出（或者第一层时，是 embedding+位置编码的结果）
            src_mask: (batch, 1, 1, src_len)，源语言的 padding mask，来自 modules/masks.py
                      的 create_padding_mask，屏蔽 <pad> 位置，不参与 self-attention 计算
        """
        # 子层 1：Self-Attention —— Q/K/V 都来自 x 自己，所以叫"自"注意力
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, src_mask))

        # 子层 2：Feed Forward
        x = self.sublayer2(x, self.feed_forward)

        return x


class Encoder(nn.Module):
    """
    编码器整体：堆叠 N 层 EncoderLayer。

    输入：embedding + 位置编码之后的源语言张量，形状 (batch, src_len, d_model)
    输出：形状不变，但每个位置的向量已经融合了整个源语言句子的上下文信息，
          这个输出会被 decoder.py 里的 Cross-Attention 用到（也叫 "memory"）
    """

    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        参数：
            num_layers: 堆叠几层 EncoderLayer，论文里是 6
            其余参数和 EncoderLayer 一致，会原样传给每一层
        """
        super().__init__()
        # 用 nn.ModuleList 而不是普通 Python list，这样 nn.Module 才能正确追踪到
        # 每一层里的参数（普通 list 存 nn.Module 的话，参数不会被自动注册，训练不到）
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return x
"""
decoder.py —— 解码器

作用：结合 Encoder 的输出("memory")和已经生成的目标语言部分句子，
      一步步预测下一个词（对应设计文档 4.2 节）。

结构分两层：
    DecoderLayer：单层解码器，内部有三个子层（比 EncoderLayer 多一个）
        1. Masked Multi-Head Self-Attention（只能看自己前面的词，靠 look-ahead mask 实现）
        2. Multi-Head Cross-Attention（Q 来自 Decoder 自己，K/V 来自 Encoder 的输出 memory，
           这一步就是"翻译时去参考原文"，是两种语言真正对齐的地方）
        3. Feed Forward
       每个子层同样包一层 SublayerConnection
    Decoder：把 DecoderLayer 堆叠 N 层
"""

import torch.nn as nn

from ..modules.attention import MultiHeadAttention
from ..modules.feed_forward import PositionwiseFeedForward
from ..modules.layer_norm import SublayerConnection


class DecoderLayer(nn.Module):
    """
    单层解码器。

    输入/输出形状都是 (batch, tgt_len, d_model)。
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        参数含义和 EncoderLayer 一致，见 encoder.py
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)   # 子层1用
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)  # 子层2用
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)  # 子层3用

        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.sublayer3 = SublayerConnection(d_model, dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        参数：
            x: (batch, tgt_len, d_model)，目标语言这一层的输入
            memory: (batch, src_len, d_model)，Encoder 的输出，Cross-Attention 要用到
            src_mask: (batch, 1, 1, src_len)，源语言 padding mask，
                      Cross-Attention 时用来屏蔽源语言里的 <pad> 位置
            tgt_mask: (batch, 1, tgt_len, tgt_len)，目标语言的组合 mask
                      (padding mask AND look-ahead mask，来自 modules/masks.py 的
                      create_decoder_mask)，Self-Attention 时用来同时屏蔽 <pad> 和"未来的词"
        """
        # 子层 1：Masked Self-Attention —— Q/K/V 都来自 x 自己，但用 tgt_mask 挡住未来
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, tgt_mask))

        # 子层 2：Cross-Attention —— Q 来自 x(目标语言)，K/V 来自 memory(源语言的编码结果)
        #         这一步让 Decoder 在生成每个词时，能去"查阅"原文里最相关的部分
        x = self.sublayer2(x, lambda x: self.cross_attn(x, memory, memory, src_mask))

        # 子层 3：Feed Forward
        x = self.sublayer3(x, self.feed_forward)

        return x


class Decoder(nn.Module):
    """
    解码器整体：堆叠 N 层 DecoderLayer。

    输入：embedding + 位置编码之后的目标语言张量，形状 (batch, tgt_len, d_model)，
          以及 Encoder 的输出 memory
    输出：形状 (batch, tgt_len, d_model)，会被 transformer.py 里最后的 Linear 层
          映射成对目标词表的预测得分
    """

    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x
"""
transformer.py —— 完整模型

作用：把 Embedding、位置编码、Encoder、Decoder、输出层拼装成一个完整的、
      能直接调用的模型类，这是 scripts/train.py 真正会 import 的类
      （对应设计文档 4.3 节）。

对外接口(和 transformer/training/trainer.py 里的约定完全对应)：
    logits = model(src, tgt, src_mask, tgt_mask)
    输入：
        src:      (batch, src_len)        源语言 id 序列
        tgt:      (batch, tgt_len)        目标语言 id 序列(训练时传 tgt_input，即 tgt[:, :-1])
        src_mask: (batch, 1, 1, src_len)              源语言 padding mask
        tgt_mask: (batch, 1, tgt_len, tgt_len)        目标语言组合 mask
    输出：
        logits:   (batch, tgt_len, tgt_vocab_size)    每个位置对目标词表的预测得分(未过 softmax)
"""

import torch.nn as nn

from ..modules.embedding import Embeddings
from ..modules.positional_encoding import PositionalEncoding
from .decoder import Decoder
from .encoder import Encoder


class Transformer(nn.Module):
    """
    完整的 Encoder-Decoder Transformer。

    典型用法：
        model = Transformer(src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab),
                             d_model=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1)
        logits = model(src, tgt_input, src_mask, tgt_mask)
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        """
        参数：
            src_vocab_size: 源语言词表大小，即 len(src_vocab)（来自 transformer/data/vocab.py）
            tgt_vocab_size: 目标语言词表大小
            d_model: 词向量维度，论文默认 512
            num_heads: 多头注意力头数，论文默认 8
            num_layers: Encoder / Decoder 各堆叠几层，论文默认 6
            d_ff: 前馈网络中间层维度，论文默认 2048
            dropout: 全模型统一使用的 dropout 比例，论文默认 0.1
            max_len: 位置编码支持的最大句子长度，要比训练/推理时最长的句子还要长
        """
        super().__init__()

        if src_vocab_size <= 0:
            raise ValueError("src_vocab_size 必须大于 0")
        if tgt_vocab_size <= 0:
            raise ValueError("tgt_vocab_size 必须大于 0")
        if d_model <= 0:
            raise ValueError("d_model 必须大于 0")
        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("num_heads 必须大于 0，且 d_model 必须能被 num_heads 整除")
        if num_layers <= 0:
            raise ValueError("num_layers 必须大于 0")
        if d_ff <= 0:
            raise ValueError("d_ff 必须大于 0")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1) 区间")
        if max_len <= 0:
            raise ValueError("max_len 必须大于 0")

        # 源语言和目标语言分别用独立的 Embedding，因为是两种不同的语言、不同的词表，
        # 参数不共享（如果是同语言的任务，比如摘要生成，可以考虑共享）
        self.src_embed = Embeddings(src_vocab_size, d_model)
        self.tgt_embed = Embeddings(tgt_vocab_size, d_model)

        # 位置编码是固定公式算出来的，中英文可以共用同一个 PositionalEncoding 实例，
        # 因为它不含可学习参数，这里为了清晰起见还是分开创建，互不影响
        self.src_pos = PositionalEncoding(d_model, max_len, dropout)
        self.tgt_pos = PositionalEncoding(d_model, max_len, dropout)

        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)

        # 最后一个线性层，把 Decoder 输出的 d_model 维向量，映射成"对每个目标语言词的得分"
        # 注意：这里不接 softmax，因为 training/loss.py 里用的 nn.CrossEntropyLoss
        # 内部自带 softmax，如果这里再手动做一次 softmax 会导致计算错误
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        self._init_parameters()

    def _init_parameters(self):
        """
        用 Xavier 初始化所有维度大于 1 的参数(主要是各个 Linear 层的权重矩阵)。

        为什么要做这一步：nn.Linear 等层默认的随机初始化方式，不一定适合较深的网络，
        Xavier 初始化会根据输入/输出的维度自动调整初始值的范围，
        让训练一开始的梯度大小比较合理，是原论文和大多数 Transformer 实现的标准做法。
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        """
        只跑 Encoder 部分，返回 memory。

        推理阶段(scripts/translate.py)会用到：源语言句子只需要编码一次，
        然后 Decoder 可以反复调用、一个词一个词地生成，不需要每生成一个词就重新编码源语言。
        """
        x = self.src_pos(self.src_embed(src))  # (batch, src_len, d_model)
        return self.encoder(x, src_mask)

    def decode(self, tgt, memory, src_mask, tgt_mask):
        """
        只跑 Decoder 部分，输入已经算好的 memory，返回 Decoder 最后一层的输出
        （还没过 generator，即还不是词表得分）。
        """
        x = self.tgt_pos(self.tgt_embed(tgt))  # (batch, tgt_len, d_model)
        return self.decoder(x, memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        完整的前向传播：源语言编码 -> 目标语言解码 -> 映射到词表得分。

        返回：
            (batch, tgt_len, tgt_vocab_size)
        """
        memory = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, memory, src_mask, tgt_mask)
        logits = self.generator(decoder_output)
        return logits

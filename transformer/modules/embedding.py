"""
embedding.py —— 词嵌入层

作用：把"词的编号(一个整数)"变成"一个有意义的向量(一串浮点数)"。

设计要点（对应设计文档 3.1 节）：
    - 本质是对 nn.Embedding 的封装，多做了一步：乘上 sqrt(d_model) 做缩放。
      这是论文 3.4 节提到的细节：词向量和位置编码相加之前，先把词向量放大，
      让词向量的数值量级和位置编码差不多，训练更稳定。这一步很容易被忽略，
      如果你发现模型收敛得比预期慢，可以回来检查这里有没有漏掉。
    - 中文和英文需要各自独立的 Embeddings 实例（词表不同、参数不共享），
      在 transformer/model/transformer.py 里会分别创建两份。
"""

import math

import torch.nn as nn


class Embeddings(nn.Module):
    """
    词嵌入层。

    输入：形状 (batch_size, seq_len) 的整数张量（词的 id）
    输出：形状 (batch_size, seq_len, d_model) 的浮点张量
    """

    def __init__(self, vocab_size: int, d_model: int):
        """
        参数：
            vocab_size: 词表大小，即 len(vocab)（vocab 来自 transformer/data/vocab.py）
            d_model: 词向量维度，需要和整个模型其他部分（attention、ffn等）保持一致，
                     通常在 config.py 里统一配置，比如 512
        """
        super().__init__()
        self.lut = nn.Embedding(vocab_size, d_model)  # lut = look-up table，即查找表
        self.d_model = d_model

    def forward(self, x):
        # x: (batch_size, seq_len)  ->  (batch_size, seq_len, d_model)
        return self.lut(x) * math.sqrt(self.d_model)

"""
attention.py —— 注意力机制

这是整个 Transformer 最核心的部分（对应设计文档 3.4 节），建议多花时间理解。

核心思想一句话：对句子里的每个词，去"查询(Query)"句子里所有词的"内容(Key/Value)"，
                算出"这个词应该更关注谁"的权重，再按权重把所有词的信息加权求和。

本文件包含两部分：
    1. scaled_dot_product_attention：最底层的注意力计算公式
    2. MultiHeadAttention：把 1 切成多个"头"并行算，再拼接起来

这个 MultiHeadAttention 类会在三个地方被复用（这也是为什么要单独封装成通用模块）：
    - Encoder 内部的 Self-Attention        (Q/K/V 都来自源语言句子自己)
    - Decoder 内部的 Masked Self-Attention (Q/K/V 都来自目标语言句子自己，但加了 look-ahead mask)
    - Decoder 内部的 Cross-Attention       (Q 来自目标语言，K/V 来自 Encoder 的输出)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(query, key, value, mask=None, dropout: nn.Dropout = None):
    """
    缩放点积注意力，对应公式：Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V

    参数：
        query: (batch, num_heads, q_len, d_k)
        key:   (batch, num_heads, k_len, d_k)
        value: (batch, num_heads, k_len, d_k)
        mask:  形状能广播到 (batch, num_heads, q_len, k_len) 的布尔张量，
               True 表示"可以看"，False 表示"要屏蔽"（约定见 masks.py）
        dropout: 可选，对注意力权重做 dropout（论文里的做法）

    返回：
        output: (batch, num_heads, q_len, d_k)   —— 加权求和后的结果
        attn:   (batch, num_heads, q_len, k_len)  —— 注意力权重矩阵，可以拿来画图分析模型关注了哪里
    """
    d_k = query.size(-1)

    # 第一步：QK^T，算出每个 query 位置和每个 key 位置的"相关性得分"
    # (batch, num_heads, q_len, d_k) x (batch, num_heads, d_k, k_len) -> (batch, num_heads, q_len, k_len)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 第二步：如果有 mask，把不该看的位置的得分设成一个很小的数（比如 -1e9），
    # 这样接下来做 softmax 时，这些位置的权重会趋近于 0
    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-1e9"))  # noqa: E712

    # 第三步：softmax，把得分变成"权重"（每一行加起来等于 1）
    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    # 第四步：按权重把所有位置的 value 加权求和
    output = torch.matmul(p_attn, value)

    return output, p_attn


class MultiHeadAttention(nn.Module):
    """
    多头注意力：不是只算一次 Attention，而是把 Q/K/V 切成多个"头"独立算，
    再把结果拼接起来，让模型能同时从多个不同角度关注句子里的关系。
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        参数：
            d_model: 词向量维度，比如 512
            num_heads: 头数，比如 8。要求 d_model 必须能整除 num_heads，
                       因为每个头分到的维度 d_k = d_model / num_heads
            dropout: 对注意力权重做 dropout 的比例
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        # 四个线性层：分别把输入投影成 Q、K、V，以及最后把多头拼接结果再投影一次
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # 存一下最近一次的注意力权重，方便调试/可视化时查看模型在关注哪里
        self.attn_weights = None

    def forward(self, query, key, value, mask=None):
        """
        参数：
            query: (batch, q_len, d_model)
            key:   (batch, k_len, d_model)
            value: (batch, k_len, d_model)
            mask:  形状 (batch, 1, q_len, k_len) 或 (batch, 1, 1, k_len) 的布尔张量，
                   会自动广播到每个头上，见 masks.py 里对应的生成函数

        返回：
            (batch, q_len, d_model)，形状和输入的 query 一致
        """
        batch_size = query.size(0)

        # 1. 线性投影，然后 reshape 成多头形式
        #    (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        q = self.linear_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.linear_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.linear_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2. 每个头独立做 scaled dot-product attention
        #    mask 形状是 (batch, 1, q_len, k_len)，中间的 1 会自动广播到 num_heads 维度
        x, self.attn_weights = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.dropout)

        # 3. 把多个头的结果拼接回去
        #    (batch, num_heads, q_len, d_k) -> (batch, q_len, num_heads, d_k) -> (batch, q_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # 4. 最后再过一次线性层，让模型学习怎么组合多个头的信息
        return self.linear_out(x)

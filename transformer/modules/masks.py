"""
masks.py —— 掩码生成

作用：告诉 Attention 机制"哪些位置不能看"。这是 Transformer 里最容易搞混、
      但又非常重要的部分（对应设计文档 3.3 节），务必理解清楚再往下写。

约定：本文件生成的所有 mask，取值都是布尔型，**True 表示"可以看/允许注意力关注"**，
      **False 表示"不能看/需要屏蔽"**。attention.py 里会按这个约定使用 mask
      （把 False 的位置的注意力得分设成一个很小的数，softmax 后权重接近 0）。

两种 mask，作用完全不同：

    ① Padding Mask（填充掩码）
       原因：dataset.py 里为了拼 batch，把短句子用 <pad> 补齐了，
             这些补出来的位置没有意义，不该参与注意力计算。
       用在：Encoder 的 self-attention、Decoder 的 cross-attention（因为
             cross-attention 的 K/V 来自 encoder 输出，同样要屏蔽源语言里的 pad）。

    ② Look-ahead Mask（前瞻掩码 / 因果掩码）
       原因：训练时 Decoder 是"一次性"看到整个目标句子的（Teacher Forcing），
             但翻译本质是从左到右一个词一个词生成的，不能让模型在预测第 i 个词时
             偷看第 i+1、i+2...个词的"答案"。
       用在：只用在 Decoder 的 self-attention。

Decoder 的 self-attention 实际上要同时满足"不能看 pad"和"不能看未来"，
所以 create_decoder_mask 把两者合并（逻辑与）。
"""

import torch


def create_padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    生成 padding mask。

    参数：
        seq: 形状 (batch_size, seq_len) 的整数张量（词的 id），
             来自 dataset.py 里 collate_fn 产出的 "src" 或 "tgt"
        pad_idx: <pad> 对应的 id，即 transformer/data/vocab.py 里的 PAD_IDX

    返回：
        形状 (batch_size, 1, 1, seq_len) 的布尔张量，True 表示"这个位置是真实词，可以关注"。
        之所以是 4 维，是为了能直接广播到 attention 里 (batch, num_heads, q_len, k_len)
        形状的注意力得分矩阵上——中间两个维度会被广播机制自动扩展。
    """
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


def create_look_ahead_mask(size: int, device=None) -> torch.Tensor:
    """
    生成 look-ahead mask（下三角矩阵）。

    参数：
        size: 目标句子长度 tgt_len
        device: 生成的张量放在哪个设备上（cpu/cuda），建议传入和输入数据相同的 device

    返回：
        形状 (size, size) 的布尔张量。第 i 行第 j 列表示"第 i 个位置是否可以看第 j 个位置"，
        取值规则：j <= i 时为 True（可以看自己和前面的词），j > i 时为 False（不能看未来的词）。

    例：size=4 时返回：
        [[ True, False, False, False],
         [ True,  True, False, False],
         [ True,  True,  True, False],
         [ True,  True,  True,  True]]
    """
    mask = torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))
    return mask


def create_decoder_mask(tgt_seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    生成 Decoder self-attention 专用的组合 mask = padding mask AND look-ahead mask。

    参数：
        tgt_seq: 形状 (batch_size, tgt_len) 的整数张量，Decoder 的输入
                 （注意：训练时这是 tgt[:, :-1]，见 trainer.py 里的说明）
        pad_idx: <pad> 对应的 id

    返回：
        形状 (batch_size, 1, tgt_len, tgt_len) 的布尔张量，
        两个条件必须同时满足（同时为 True）才允许关注。
    """
    tgt_len = tgt_seq.size(1)

    padding_mask = create_padding_mask(tgt_seq, pad_idx)  # (batch, 1, 1, tgt_len)
    look_ahead_mask = create_look_ahead_mask(tgt_len, device=tgt_seq.device)  # (tgt_len, tgt_len)

    # 两者做逻辑与，利用广播机制自动对齐成 (batch, 1, tgt_len, tgt_len)
    combined_mask = padding_mask & look_ahead_mask
    return combined_mask

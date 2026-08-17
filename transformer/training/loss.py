"""训练使用的损失函数。"""

import torch
import torch.nn as nn

from transformer.data.vocab import PAD_IDX


def sequence_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, pad_idx: int = PAD_IDX) -> torch.Tensor:
    """计算忽略 padding 的逐 token 交叉熵。

    `logits` 的形状为 ``(batch, sequence, vocab_size)``，`targets` 为
    ``(batch, sequence)``。
    """
    if logits.ndim != 3 or targets.ndim != 2:
        raise ValueError("logits 必须为 3 维，targets 必须为 2 维")
    if logits.shape[:2] != targets.shape:
        raise ValueError("logits 与 targets 的 batch 和序列长度必须一致")
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=pad_idx
    )

"""优化器创建函数。"""

import torch


def create_optimizer(
    model: torch.nn.Module, learning_rate: float = 3e-4, weight_decay: float = 1e-4
) -> torch.optim.Optimizer:
    """为 Transformer 创建 AdamW 优化器。"""
    if learning_rate <= 0:
        raise ValueError("learning_rate 必须大于 0")
    if weight_decay < 0:
        raise ValueError("weight_decay 不能小于 0")
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def create_linear_warmup_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int = 0
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """创建线性 warmup 调度器；``warmup_steps=0`` 时不使用调度器。"""
    if warmup_steps < 0:
        raise ValueError("warmup_steps 不能小于 0")
    if warmup_steps == 0:
        return None

    def warmup_factor(step: int) -> float:
        return min(1.0, (step + 1) / warmup_steps)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_factor)

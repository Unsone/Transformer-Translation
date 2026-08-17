"""checkpoint 的保存格式校验、加载和训练状态恢复。"""

from pathlib import Path
from typing import Any

import torch

from transformer.model import Transformer


REQUIRED_KEYS = {"model_state_dict", "optimizer_state_dict", "model_config", "src_vocab", "tgt_vocab"}


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    """加载受本项目信任来源生成的 checkpoint。

    checkpoint 含有自定义 ``Vocab`` 对象，故需使用 ``weights_only=False``。
    不要加载来源不可信的 PyTorch checkpoint。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    missing = REQUIRED_KEYS - checkpoint.keys()
    if missing:
        raise ValueError(f"checkpoint 缺少必要字段：{', '.join(sorted(missing))}")
    return checkpoint


def load_model_from_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[Transformer, dict[str, Any]]:
    """恢复模型与关联元数据（词表、配置、训练历史）。"""
    checkpoint = load_checkpoint(path, device)
    model = Transformer(
        src_vocab_size=len(checkpoint["src_vocab"]),
        tgt_vocab_size=len(checkpoint["tgt_vocab"]),
        **checkpoint["model_config"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def restore_optimizer(optimizer: torch.optim.Optimizer, checkpoint: dict[str, Any]) -> None:
    """恢复 checkpoint 中的优化器状态，以继续训练。"""
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

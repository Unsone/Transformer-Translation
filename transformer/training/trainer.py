"""Transformer 的训练、验证和 checkpoint 保存逻辑。"""

from pathlib import Path
import time
from typing import Any

import torch

from transformer.data.vocab import PAD_IDX
from transformer.modules.masks import create_decoder_mask, create_padding_mask

from .loss import sequence_cross_entropy


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        max_grad_norm: float | None = None,
    ):
        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError("max_grad_norm 必须大于 0")
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm

    def _prepare_batch(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        src = batch["src"].to(self.device)
        tgt = batch["tgt"].to(self.device)
        if tgt.size(1) < 2:
            raise ValueError("目标序列至少需要包含 <sos> 和 <eos>")
        tgt_input, tgt_expected = tgt[:, :-1], tgt[:, 1:]
        src_mask = create_padding_mask(src, PAD_IDX)
        tgt_mask = create_decoder_mask(tgt_input, PAD_IDX)
        return src, tgt_input, tgt_expected, src_mask, tgt_mask

    def train_step(self, batch: dict[str, torch.Tensor]) -> float:
        self.model.train()
        src, tgt_input, tgt_expected, src_mask, tgt_mask = self._prepare_batch(batch)
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(src, tgt_input, src_mask, tgt_mask)
        loss = sequence_cross_entropy(logits, tgt_expected)
        if not torch.isfinite(loss):
            raise FloatingPointError("训练 loss 非有限值")
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss.item()

    @torch.no_grad()
    def validate(self, dataloader) -> float:
        self.model.eval()
        losses = []
        for batch in dataloader:
            src, tgt_input, tgt_expected, src_mask, tgt_mask = self._prepare_batch(batch)
            logits = self.model(src, tgt_input, src_mask, tgt_mask)
            losses.append(sequence_cross_entropy(logits, tgt_expected).item())
        if not losses:
            raise ValueError("验证 DataLoader 为空")
        return sum(losses) / len(losses)

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 1,
        start_epoch: int = 0,
        on_epoch_end=None,
    ) -> list[dict[str, float | int | None]]:
        if epochs <= 0:
            raise ValueError("epochs 必须大于 0")
        history = []
        for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            start_time = time.perf_counter()
            losses = [self.train_step(batch) for batch in train_loader]
            if not losses:
                raise ValueError("训练 DataLoader 为空")
            val_loss = self.validate(val_loader) if val_loader is not None else None
            record = {
                "epoch": epoch,
                "train_loss": sum(losses) / len(losses),
                "val_loss": val_loss,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "seconds": time.perf_counter() - start_time,
                "gpu_peak_memory_mib": (
                    torch.cuda.max_memory_allocated(self.device) / 1024**2 if self.device.type == "cuda" else None
                ),
            }
            history.append(record)
            if on_epoch_end is not None:
                on_epoch_end(record, history)
        return history

    def save_checkpoint(self, path: str | Path, **metadata: Any) -> None:
        """保存可用于后续推理或恢复训练的训练状态。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
                **metadata,
            },
            path,
        )

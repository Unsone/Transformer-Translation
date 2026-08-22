"""训练英译中 Transformer。

示例：uv run python scripts/train.py --epochs 10
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformer.config import ModelConfig, TrainingConfig
from transformer.data.dataset import get_dataloader
from transformer.model import Transformer
from transformer.training import (
    Trainer,
    create_linear_warmup_scheduler,
    create_optimizer,
    load_checkpoint,
    load_model_from_checkpoint,
    restore_optimizer,
)


def parse_args() -> argparse.Namespace:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description="训练一个英译中 Transformer")
    parser.add_argument("--data-path", type=Path, default=defaults.data_path)
    parser.add_argument("--checkpoint-path", type=Path, default=defaults.checkpoint_path)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--max-examples", type=int, default=defaults.max_examples)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--resume", type=Path, help="从已有 checkpoint 恢复模型和优化器状态")
    parser.add_argument("--warmup-steps", type=int, default=defaults.warmup_steps)
    parser.add_argument("--max-grad-norm", type=float, default=defaults.max_grad_norm)
    parser.add_argument("--save-every", type=int, default=defaults.save_every)
    parser.add_argument("--log-path", type=Path, help="训练日志文件路径；默认写入 checkpoint 同目录")
    model_defaults = ModelConfig()
    parser.add_argument("--d-model", type=int, default=model_defaults.d_model)
    parser.add_argument("--num-heads", type=int, default=model_defaults.num_heads)
    parser.add_argument("--num-layers", type=int, default=model_defaults.num_layers)
    parser.add_argument("--d-ff", type=int, default=model_defaults.d_ff)
    parser.add_argument("--dropout", type=float, default=model_defaults.dropout)
    parser.add_argument("--max-len", type=int, default=model_defaults.max_len)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_config = TrainingConfig(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_examples=args.max_examples,
        seed=args.seed,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        save_every=args.save_every,
    )
    if training_config.save_every <= 0:
        raise ValueError("save_every 必须大于 0")
    log_path = args.log_path or training_config.checkpoint_path.with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
    )
    logger = logging.getLogger(__name__)
    model_config = ModelConfig(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len,
    )
    train_loader, val_loader, src_vocab, tgt_vocab = get_dataloader(
        str(training_config.data_path),
        batch_size=training_config.batch_size,
        min_freq=training_config.min_freq,
        max_examples=training_config.max_examples,
        val_ratio=training_config.val_ratio,
        test_ratio=training_config.test_ratio,
        seed=training_config.seed,
    )
    if args.resume:
        model, resumed_checkpoint = load_model_from_checkpoint(args.resume, device)
        if len(src_vocab) != len(resumed_checkpoint["src_vocab"]) or len(tgt_vocab) != len(resumed_checkpoint["tgt_vocab"]):
            raise ValueError("当前数据生成的词表与待恢复 checkpoint 不一致")
        model_config = ModelConfig(**resumed_checkpoint["model_config"])
        previous_history = resumed_checkpoint.get("history", [])
    else:
        model = Transformer(len(src_vocab), len(tgt_vocab), **model_config.to_dict())
        previous_history = []
    optimizer = create_optimizer(model, training_config.learning_rate, training_config.weight_decay)
    scheduler = create_linear_warmup_scheduler(optimizer, training_config.warmup_steps)
    trainer = Trainer(model, optimizer, device, scheduler, training_config.max_grad_norm)
    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        restore_optimizer(trainer.optimizer, checkpoint)
        if trainer.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    logger.info("device=%s, src_vocab=%s, tgt_vocab=%s", device, len(src_vocab), len(tgt_vocab))

    best_val_loss = min(
        (record["val_loss"] for record in previous_history if record.get("val_loss") is not None), default=float("inf")
    )

    def checkpoint_metadata(history):
        return {
            "model_config": model_config.to_dict(),
            "training_config": training_config.to_dict(),
            "src_vocab": src_vocab,
            "tgt_vocab": tgt_vocab,
            "history": history,
            "epoch": history[-1]["epoch"],
        }

    def save_epoch(record, new_history):
        nonlocal best_val_loss
        history = previous_history + new_history
        if record["epoch"] % training_config.save_every == 0:
            epoch_path = training_config.checkpoint_path.with_name(
                f"{training_config.checkpoint_path.stem}-epoch{record['epoch']}{training_config.checkpoint_path.suffix}"
            )
            trainer.save_checkpoint(epoch_path, **checkpoint_metadata(history))
            trainer.save_checkpoint(training_config.checkpoint_path, **checkpoint_metadata(history))
            logger.info("checkpoint saved: %s", epoch_path)
        if record["val_loss"] is not None and record["val_loss"] < best_val_loss:
            best_val_loss = record["val_loss"]
            best_path = training_config.checkpoint_path.with_name(
                f"{training_config.checkpoint_path.stem}-best{training_config.checkpoint_path.suffix}"
            )
            trainer.save_checkpoint(best_path, **checkpoint_metadata(history))
            logger.info("best checkpoint saved: %s", best_path)
        logger.info(
            "epoch=%s train_loss=%.4f val_loss=%s lr=%.6g seconds=%.2f",
            record["epoch"],
            record["train_loss"],
            f"{record['val_loss']:.4f}" if record["val_loss"] is not None else "n/a",
            record["learning_rate"],
            record["seconds"],
        )

    new_history = trainer.fit(
        train_loader,
        val_loader,
        training_config.epochs,
        start_epoch=len(previous_history),
        on_epoch_end=save_epoch,
    )
    history = previous_history + new_history
    if history[-1]["epoch"] % training_config.save_every != 0:
        trainer.save_checkpoint(training_config.checkpoint_path, **checkpoint_metadata(history))
    logger.info("latest checkpoint saved to %s", training_config.checkpoint_path)


if __name__ == "__main__":
    main()

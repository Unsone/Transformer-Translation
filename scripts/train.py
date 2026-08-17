"""训练英译中 Transformer。

示例：uv run python scripts/train.py --epochs 10
"""

import argparse
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
from transformer.training import Trainer, create_optimizer, load_checkpoint, load_model_from_checkpoint, restore_optimizer


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
    )
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
    trainer = Trainer(
        model,
        create_optimizer(model, training_config.learning_rate, training_config.weight_decay),
        device,
    )
    if args.resume:
        restore_optimizer(trainer.optimizer, load_checkpoint(args.resume, device))
    print(f"device={device}, src_vocab={len(src_vocab)}, tgt_vocab={len(tgt_vocab)}")
    new_history = trainer.fit(train_loader, val_loader, training_config.epochs)
    for record in new_history:
        record["epoch"] += len(previous_history)
    history = previous_history + new_history
    for record in history:
        print(
            f"epoch={record['epoch']} train_loss={record['train_loss']:.4f} "
            f"val_loss={record['val_loss']:.4f}"
        )
    trainer.save_checkpoint(
        training_config.checkpoint_path,
        model_config=model_config.to_dict(),
        training_config=training_config.to_dict(),
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        history=history,
        epoch=history[-1]["epoch"],
    )
    print(f"checkpoint saved to {training_config.checkpoint_path}")


if __name__ == "__main__":
    main()

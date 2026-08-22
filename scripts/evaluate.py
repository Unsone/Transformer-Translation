"""在验证集上计算 token loss 与 perplexity。"""

import argparse
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformer.data.dataset import get_dataloader
from transformer.training import Trainer, create_optimizer, load_model_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 Transformer checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-path", type=Path, default=PROJECT_ROOT / "data" / "cmn.txt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    _, val_loader, test_loader, _, _ = get_dataloader(
        str(args.data_path),
        batch_size=args.batch_size,
        max_examples=args.max_examples,
        src_vocab=checkpoint["src_vocab"],
        tgt_vocab=checkpoint["tgt_vocab"],
        seed=args.seed,
        return_test_loader=True,
    )
    dataloader = val_loader if args.split == "validation" else test_loader
    if dataloader is None:
        raise ValueError("所选数据切分为空；请提供至少 10 条样本")
    trainer = Trainer(model, create_optimizer(model), device)
    loss = trainer.validate(dataloader)
    print(f"{args.split}_loss={loss:.4f}")
    print(f"perplexity={math.exp(loss):.4f}")


if __name__ == "__main__":
    main()

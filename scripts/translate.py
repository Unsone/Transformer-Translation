"""加载 checkpoint 并将一条英文句子贪心翻译为中文。"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformer.training import load_model_from_checkpoint
from transformer.translation import greedy_translate


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 checkpoint 翻译英文句子")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("sentence", help="待翻译的英文句子")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    translation = greedy_translate(
        model, args.sentence, checkpoint["src_vocab"], checkpoint["tgt_vocab"], device, args.max_tokens
    )
    print(translation)


if __name__ == "__main__":
    main()

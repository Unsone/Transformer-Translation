"""checkpoint 加载与贪心翻译的最小测试。"""

import tempfile
import unittest
from pathlib import Path

import torch

from transformer.data.vocab import Vocab
from transformer.model import Transformer
from transformer.training import Trainer, create_optimizer, load_model_from_checkpoint
from transformer.translation import greedy_translate


class InferenceSmokeTest(unittest.TestCase):
    def test_load_checkpoint_and_translate(self):
        src_vocab = Vocab.build([["hello"]])
        tgt_vocab = Vocab.build([["你", "好"]])
        model_config = {"d_model": 8, "num_heads": 2, "num_layers": 1, "d_ff": 16, "dropout": 0.0, "max_len": 16}
        model = Transformer(len(src_vocab), len(tgt_vocab), **model_config)
        trainer = Trainer(model, create_optimizer(model), torch.device("cpu"))

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pt"
            trainer.save_checkpoint(
                path,
                model_config=model_config,
                src_vocab=src_vocab,
                tgt_vocab=tgt_vocab,
                epoch=1,
                history=[],
            )
            loaded_model, checkpoint = load_model_from_checkpoint(path)
            result = greedy_translate(
                loaded_model, "hello", checkpoint["src_vocab"], checkpoint["tgt_vocab"], torch.device("cpu"), 4
            )

        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()

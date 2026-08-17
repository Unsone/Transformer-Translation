"""训练闭环的最小冒烟测试。"""

import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from transformer.data.dataset import TranslationDataset, collate_fn
from transformer.model import Transformer
from transformer.training import Trainer, create_optimizer


class TrainingSmokeTest(unittest.TestCase):
    def test_fit_and_save_checkpoint(self):
        dataset = TranslationDataset(
            src_ids_list=[[1, 4, 2], [1, 5, 6, 2]],
            tgt_ids_list=[[1, 7, 2], [1, 8, 9, 2]],
        )
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        model = Transformer(10, 12, d_model=8, num_heads=2, num_layers=1, d_ff=16, dropout=0.0)
        trainer = Trainer(model, create_optimizer(model, learning_rate=1e-3), torch.device("cpu"))

        history = trainer.fit(loader, loader, epochs=1)
        self.assertEqual(len(history), 1)
        self.assertIsNotNone(history[0]["val_loss"])

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "model.pt"
            trainer.save_checkpoint(checkpoint, epoch=1)
            saved = torch.load(checkpoint, weights_only=False)
        self.assertEqual(saved["epoch"], 1)
        self.assertIn("model_state_dict", saved)
        self.assertIn("optimizer_state_dict", saved)


if __name__ == "__main__":
    unittest.main()

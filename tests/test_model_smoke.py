"""完整 Transformer 的最小可运行性测试。"""

import unittest

import torch
import torch.nn as nn

from transformer.model import Transformer
from transformer.modules.masks import create_decoder_mask, create_padding_mask


class TransformerSmokeTest(unittest.TestCase):
    def test_forward_loss_and_backward(self):
        src = torch.tensor([[1, 4, 2, 0], [1, 5, 6, 2]], dtype=torch.long)
        tgt_input = torch.tensor([[1, 7, 2], [1, 8, 2]], dtype=torch.long)
        tgt_expected = torch.tensor([[7, 2, 0], [8, 2, 0]], dtype=torch.long)

        model = Transformer(10, 12, d_model=8, num_heads=2, num_layers=1, d_ff=16, dropout=0.0)
        logits = model(
            src,
            tgt_input,
            create_padding_mask(src, pad_idx=0),
            create_decoder_mask(tgt_input, pad_idx=0),
        )

        self.assertEqual(logits.shape, (2, 3, 12))
        loss = nn.CrossEntropyLoss(ignore_index=0)(
            logits.reshape(-1, logits.size(-1)), tgt_expected.reshape(-1)
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.generator.weight.grad)
        self.assertTrue(torch.isfinite(model.generator.weight.grad).all())

    def test_invalid_model_arguments_raise_value_error(self):
        with self.assertRaises(ValueError):
            Transformer(10, 12, d_model=8, num_heads=3)


if __name__ == "__main__":
    unittest.main()

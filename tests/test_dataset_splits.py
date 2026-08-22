"""数据集切分、词表隔离与长度过滤测试。"""

import random
import tempfile
import unittest
from pathlib import Path

from transformer.data.dataset import get_dataloader


class DatasetSplitTest(unittest.TestCase):
    def _write_pairs(self, directory: str, count: int = 10) -> Path:
        path = Path(directory) / "pairs.txt"
        path.write_text(
            "\n".join(f"unique{i}\t句{i}" for i in range(count)), encoding="utf-8"
        )
        return path

    def test_split_is_reproducible_and_vocab_uses_train_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_pairs(temp_dir)
            result1 = get_dataloader(
                str(path), batch_size=2, val_ratio=0.2, test_ratio=0.2, seed=7, return_test_loader=True
            )
            result2 = get_dataloader(
                str(path), batch_size=2, val_ratio=0.2, test_ratio=0.2, seed=7, return_test_loader=True
            )

        train1, val1, test1, src_vocab, _ = result1
        train2, val2, test2, _, _ = result2
        self.assertEqual(len(train1.dataset), 6)
        self.assertEqual(len(val1.dataset), 2)
        self.assertEqual(len(test1.dataset), 2)
        self.assertEqual(train1.dataset.src_ids_list, train2.dataset.src_ids_list)
        self.assertEqual(val1.dataset.src_ids_list, val2.dataset.src_ids_list)
        self.assertEqual(test1.dataset.src_ids_list, test2.dataset.src_ids_list)

        indices = list(range(10))
        random.Random(7).shuffle(indices)
        held_out = indices[:4]
        for index in held_out:
            self.assertNotIn(f"unique{index}", src_vocab.word2idx)

    def test_length_filter_removes_overlong_pairs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pairs.txt"
            path.write_text("short\t短\nthis is too long\t很长的句子", encoding="utf-8")
            train, val, test, _, _ = get_dataloader(
                str(path),
                max_src_len=1,
                val_ratio=0.0,
                test_ratio=0.0,
                return_test_loader=True,
            )

        self.assertEqual(len(train.dataset), 1)
        self.assertIsNone(val)
        self.assertIsNone(test)


if __name__ == "__main__":
    unittest.main()

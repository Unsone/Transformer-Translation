"""翻译指标测试。"""

import unittest

from transformer.metrics import corpus_bleu


class MetricsTest(unittest.TestCase):
    def test_perfect_translation_has_full_bleu(self):
        reference = [list("今天天气很好")]
        self.assertAlmostEqual(corpus_bleu(reference, reference), 100.0)

    def test_unrelated_translation_scores_lower(self):
        reference = [list("今天天气很好")]
        hypothesis = [list("完全不同内容")]
        self.assertLess(corpus_bleu(reference, hypothesis), 50.0)


if __name__ == "__main__":
    unittest.main()

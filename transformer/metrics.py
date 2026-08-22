"""无需额外依赖的翻译质量指标。"""

import math
from collections import Counter
from typing import Sequence


def _ngrams(tokens: Sequence[str], order: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1))


def corpus_bleu(references: Sequence[Sequence[str]], hypotheses: Sequence[Sequence[str]], max_order: int = 4) -> float:
    """计算单参考译文的平滑语料级 BLEU，返回范围为 0--100。"""
    if len(references) != len(hypotheses):
        raise ValueError("references 与 hypotheses 数量必须一致")
    if not references:
        raise ValueError("至少需要一条参考译文")
    if max_order <= 0:
        raise ValueError("max_order 必须大于 0")

    matches = [0] * max_order
    totals = [0] * max_order
    reference_length = 0
    hypothesis_length = 0
    for reference, hypothesis in zip(references, hypotheses):
        reference_length += len(reference)
        hypothesis_length += len(hypothesis)
        for order in range(1, max_order + 1):
            hypothesis_ngrams = _ngrams(hypothesis, order)
            reference_ngrams = _ngrams(reference, order)
            totals[order - 1] += sum(hypothesis_ngrams.values())
            matches[order - 1] += sum(
                min(count, reference_ngrams[ngram]) for ngram, count in hypothesis_ngrams.items()
            )

    if hypothesis_length == 0:
        return 0.0
    # 加一平滑保证短句仍可获得可解释的分数。
    precisions = [(matched + 1) / (total + 1) for matched, total in zip(matches, totals)]
    brevity_penalty = min(1.0, math.exp(1 - reference_length / hypothesis_length))
    return 100 * brevity_penalty * math.exp(sum(math.log(value) for value in precisions) / max_order)

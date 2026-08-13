"""
transformer.model 子包

把 transformer/modules/ 里的小零件，按照论文的 Encoder-Decoder 架构拼装起来：

    encoder.py       EncoderLayer(单层) + Encoder(堆叠 N 层)
    decoder.py       DecoderLayer(单层) + Decoder(堆叠 N 层)
    transformer.py   完整的 Seq2Seq Transformer，scripts/train.py 会直接用这里的 Transformer 类

依赖关系是单向的：transformer.py -> encoder.py / decoder.py -> modules/ 里的零件，
不会出现互相 import 的情况。
"""

from .encoder import Encoder, EncoderLayer
from .decoder import Decoder, DecoderLayer
from .transformer import Transformer

__all__ = [
    "Encoder",
    "EncoderLayer",
    "Decoder",
    "DecoderLayer",
    "Transformer",
]
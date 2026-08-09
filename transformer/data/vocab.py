from collections import Counter
from typing import Iterable, List

# 关于特殊 token 的约定
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

class Vocab:

    # 创建 Vocab，并维护附带的两个索引字典。
    # 这里两个字典都只初始化了 4 个特殊 token。
    def __init__(self):
        self.word2idx = {
            PAD_TOKEN: PAD_IDX,
            SOS_TOKEN: SOS_IDX,
            EOS_TOKEN: EOS_IDX,
            UNK_TOKEN: UNK_IDX,
        }
        # 使用 items方法获取 word2idx 字典的 key-value 列表，然后对应映射到 idx2word 字典。
        # in 前面的 word 和 idx 与 items，这里的 word 和 idx 和前面冒号两边的 word 和 idx 本质相同，按名字对应。
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    # 类方法，可以使用 cls 关键字直接获取类定义本身。
    @classmethod
    # Iterable 的意思是可迭代对象，任何迭代对象都可以作为参数。
    # 这里的 tokenized_sentences 就是一个可迭代对象，里面的每个元素都是一个 List[str]，即一个句子被切分成的 token 列表。
    # min_freq 表示最小词频，小于这个词频的词会被忽略。
    def build(cls, tokenized_sentences: Iterable[List[str]], min_freq: int = 1) -> "Vocab":

        # 创建 Vocab 对象
        vocab = cls()
        # 创建计数器对象，用来统计词频，对于每个 token 列表，更新计数器的计数。
        counter = Counter()
        # update 方法会更新计数器的计数，传入的参数是 token 列表 sentence，从而实现统计每一个 token 在全部数据中出现的次数。
        for sentence in tokenized_sentences:
            counter.update(sentence)

        # 先使用 sorted 方法对计数器统计所得的 List 进行按频率递降的排序，再依次给出词和索引进行循环。
        for word, freq in sorted(counter.items(), key=lambda x: -x[1]):
            # 如果词频小于 min_freq，则跳过该词。
            if freq < min_freq:
                continue
            # 如果该词错误地再次出现，则跳过该循环。
            # 为什么这个逻辑可以处理错误？正常来说，经过 counter 统计之后，token 之间互不相同，不可能再次出现。
            # 但是这里就是再次出现了，只能说明训练数据出现了预设的四个特殊 token 之中的内容，此时把这一轮跳过即可。
            if word in vocab.word2idx:
                continue
            # 具体实现 token 添加到 word2idx 和 idx2word 字典中，索引是当前字典长度。
            new_idx = len(vocab.word2idx)
            vocab.word2idx[word] = new_idx
            vocab.idx2word[new_idx] = word
        return vocab

    # 编码
    def encode(self, tokens: List[str], add_sos_eos: bool = True) -> List[int]:

        # 使用 get 方法获取到 word2idx 字典中每个 token 对应的索引，如果没有找到，则返回 UNK_IDX。
        ids = [self.word2idx.get(tok, UNK_IDX) for tok in tokens]
        # 添加 SOS 和 EOS。
        if add_sos_eos:
            ids = [SOS_IDX] + ids + [EOS_IDX]
        # 返回索引列表。
        return ids

    # 解码
    def decode(self, ids: List[int], remove_special: bool = True) -> List[str]:

        # 创建一个特殊索引集合，用于过滤掉特殊索引。只有输入 remove_special 为 True 时，才会过滤掉特殊索引。
        # 如果 remove_special 为 False，则 special_ids 为空集合，不会过滤掉任何索引。
        special_ids = {PAD_IDX, SOS_IDX, EOS_IDX} if remove_special else set()
        # 创建一个空列表，用于存储解码后的 token。
        tokens = []
        # 遍历索引列表，如果索引在特殊索引集合中，则跳过该索引；否则，将索引对应的 token 添加到 tokens 中。
        for idx in ids:
            if idx in special_ids:
                continue
            tokens.append(self.idx2word.get(idx, UNK_TOKEN))
        # 返回 token 列表，也就是一句话。
        return tokens

    def __len__(self) -> int:
        return len(self.word2idx)

    def __repr__(self) -> str:
        return f"Vocab(size={len(self)})"
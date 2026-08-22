import random
from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .tokenizer import tokenize
from .vocab import PAD_IDX, Vocab

# 从给定路径读取数据文件，并返回 (英文, 中文) 列表
def read_cmn_file(path: str, max_examples: int = None) -> List[Tuple[str, str]]:

    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            en_sentence, zh_sentence = parts[0], parts[1]
            pairs.append((en_sentence, zh_sentence))

            if max_examples is not None and len(pairs) >= max_examples:
                break
    # 返回 (英文句子字符串, 中文句子字符串) 列表
    return pairs


# 创建 TranslationDataset 类，继承自 torch.utils.data.Dataset。
# 希望把数据封装成 Pytorch 给出的标准数据集格式 DataLoader。
class TranslationDataset(Dataset):

    # __init__ 方法是类的初始化方法。
    # src_ids_list / tgt_ids_list: 源语言和目标语言语料的数字 id 列表。
    # __init__ 方法为 TranslationDataset 类的实例 self 添加这两个属性。
    def __init__(self, src_ids_list: List[List[int]], tgt_ids_list: List[List[int]]):
        assert len(src_ids_list) == len(tgt_ids_list), "源语言和目标语言的样本数量必须一致"
        self.src_ids_list = src_ids_list
        self.tgt_ids_list = tgt_ids_list

    def __len__(self) -> int:
        return len(self.src_ids_list)

    # __getitem__ 方法是类的索引访问方法。
    # 当你写 dataset[idx] 时，实际上就是调用了 dataset.__getitem__(self, idx)。
    # 这里返回的是一个元组 (src_ids, tgt_ids)，分别是源语言和目标语言的数字 id 张量。
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # tensor 方法用于创建张量，参数 dtype 表示数据类型，这里设为 torch.long，本质上是 int64 类型。
        # 张量创建过程中，程序首先读取 List 的内容，发现 List 是一个包含一定数目 int 的列表，
        # 随即在 Heap 上划分一段内存空间并把 List 的内容拷贝到内存里，
        # 在这之后，PyTorch 会创建一个 Tensor Header，把内存首地址、形状、步长、设备等信息打包成一个 Python 可见的对象，最后返回。
        src_ids = torch.tensor(self.src_ids_list[idx], dtype=torch.long)
        tgt_ids = torch.tensor(self.tgt_ids_list[idx], dtype=torch.long)
        return src_ids, tgt_ids


# pad_sequence 把长短不一的句子强行拉成一样长，好让它们能叠在一起，组成一个矩阵，这样才能塞进 GPU 做批量并行计算。
def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> dict:

    # *batch 解包样本列表，zip 将原本按样本聚合的 (src, tgt) 重组为按语言聚合的 (srcs, tgts)。
    # 即 src_batch 存放所有源句张量，tgt_batch 存放所有目标句张量，便于后续统一做 padding。
    # 把每个 batch 里所有句子的英文（src）扔进一堆，把所有句子的中文（tgt）扔进另一堆。
    src_batch, tgt_batch = zip(*batch)
    # 分别把这两堆长短不一的句子，用 pad_sequence 强行拉齐成整齐的矩阵。
    # 这个过程得出的矩阵，长等于 batch 的大小，宽等于最长的句子的长度。也就是把句子的数字序列一行一行写出来组成矩阵。
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    # 返回一个字典，包含 src_padded 和 tgt_padded 两个张量。
    return {"src": src_padded, "tgt": tgt_padded}


def get_dataloader(
    path: str,
    batch_size: int = 32,
    min_freq: int = 1,
    max_examples: int = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    src_vocab: Vocab = None,
    tgt_vocab: Vocab = None,
    seed: int = 42,
    max_src_len: int = None,
    max_tgt_len: int = None,
    return_test_loader: bool = False,
):
    """
    把 read_cmn_file -> tokenize -> Vocab.build -> TranslationDataset -> DataLoader
    这一整条流水线串起来，是 scripts/train.py 真正会调用的函数。

    参数：
        path: cmn.txt 路径
        batch_size: 每个 batch 多少条样本
        min_freq: 建词表时的词频阈值，见 vocab.py 里的说明
        max_examples: 调试时可以只用一小部分数据，传 None 表示用全部数据
        val_ratio: 切多少比例的数据出来当验证集（0~1 之间）
        test_ratio: 切多少比例的数据出来当测试集（0~1 之间）
        shuffle: 训练集是否打乱
        src_vocab / tgt_vocab: 如果已经建好词表（比如推理阶段要复用训练时的词表），
                                可以直接传入，不再重新构建
        seed: 数据切分使用的随机种子，保证切分可复现
        max_src_len / max_tgt_len: 过滤 token 数超过此限制的句对，None 表示不限制
        return_test_loader: 为 True 时额外返回 test_loader

    返回：
        默认返回 train_loader, val_loader, src_vocab, tgt_vocab；
        return_test_loader=True 时返回 train_loader, val_loader, test_loader, src_vocab, tgt_vocab
    """
    # 1. 读文件，得到 (英文, 中文) 句对
    pairs = read_cmn_file(path, max_examples=max_examples)
    if len(pairs) == 0:
        raise ValueError(f"没有从 {path} 读取到任何数据，请检查文件路径和文件格式")

    if not 0 <= val_ratio < 1 or not 0 <= test_ratio < 1 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio 和 test_ratio 必须位于 [0, 1) 且两者之和小于 1")

    # 2. 切词。约定：源语言(src)=英文，目标语言(tgt)=中文，
    #    即任务方向是"英译中"；如果想做反过来的"中译英"，
    #    在这里把 en_tokens / zh_tokens 的角色对调即可，其余代码不用改。
    tokenized_pairs = [(tokenize(en, lang="en"), tokenize(zh, lang="zh")) for en, zh in pairs]
    tokenized_pairs = [
        (en_tokens, zh_tokens)
        for en_tokens, zh_tokens in tokenized_pairs
        if en_tokens
        and zh_tokens
        and (max_src_len is None or len(en_tokens) <= max_src_len)
        and (max_tgt_len is None or len(zh_tokens) <= max_tgt_len)
    ]
    if not tokenized_pairs:
        raise ValueError("过滤后没有可用句对；请检查长度限制或数据内容")

    # 3. 先随机切分，再只用训练集建词表，避免验证/测试语料泄漏进词表。
    indices = list(range(len(tokenized_pairs)))
    random.Random(seed).shuffle(indices)
    val_size = int(len(indices) * val_ratio)
    test_size = int(len(indices) * test_ratio)
    val_indices = indices[:val_size]
    test_indices = indices[val_size : val_size + test_size]
    train_indices = indices[val_size + test_size :]
    train_pairs = [tokenized_pairs[i] for i in train_indices]
    val_pairs = [tokenized_pairs[i] for i in val_indices]
    test_pairs = [tokenized_pairs[i] for i in test_indices]
    if not train_pairs:
        raise ValueError("训练集为空；请减小 val_ratio/test_ratio 或提供更多数据")

    # 4. 建词表（如果外部没有传入现成的词表）
    if src_vocab is None:
        src_vocab = Vocab.build([en_tokens for en_tokens, _ in train_pairs], min_freq=min_freq)
    if tgt_vocab is None:
        tgt_vocab = Vocab.build([zh_tokens for _, zh_tokens in train_pairs], min_freq=min_freq)

    def numericalize(pairs_to_encode):
        return (
            [src_vocab.encode(en_tokens) for en_tokens, _ in pairs_to_encode],
            [tgt_vocab.encode(zh_tokens) for _, zh_tokens in pairs_to_encode],
        )

    # 5. token 转数字 id，并自动加上 <sos>/<eos>
    train_src, train_tgt = numericalize(train_pairs)
    val_src, val_tgt = numericalize(val_pairs)
    test_src, test_tgt = numericalize(test_pairs)

    train_dataset = TranslationDataset(train_src, train_tgt)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )

    val_loader = None
    if val_src:
        val_dataset = TranslationDataset(val_src, val_tgt)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    test_loader = None
    if test_src:
        test_dataset = TranslationDataset(test_src, test_tgt)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    if return_test_loader:
        return train_loader, val_loader, test_loader, src_vocab, tgt_vocab
    return train_loader, val_loader, src_vocab, tgt_vocab


# ----------------------------------------------------------------------
# 运行 `python -m transformer.data.dataset` 检查正确性
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import os

    # 假设 cmn.txt 放在项目根目录的 data/ 文件夹下，按需修改这个路径
    demo_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cmn.txt")

    train_loader, val_loader, src_vocab, tgt_vocab = get_dataloader(
        demo_path, batch_size=4, max_examples=100
    )

    print("英文词表大小:", len(src_vocab))
    print("中文词表大小:", len(tgt_vocab))

    batch = next(iter(train_loader))
    print("src shape:", batch["src"].shape)
    print("tgt shape:", batch["tgt"].shape)
    print("src[0]:", batch["src"][0].tolist())
    print("tgt[0]:", batch["tgt"][0].tolist())

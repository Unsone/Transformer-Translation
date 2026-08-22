"""使用训练完成的 Transformer 执行贪心英译中。"""

import torch

from transformer.data.tokenizer import tokenize_en
from transformer.data.vocab import EOS_IDX, SOS_IDX
from transformer.modules.masks import create_decoder_mask, create_padding_mask


@torch.no_grad()
def greedy_decode_ids(model, src: torch.Tensor, device: torch.device, max_tokens: int = 64) -> list[int]:
    """对单条已数值化源句执行贪心解码，返回包含特殊 token 的 id 序列。"""
    if max_tokens <= 0:
        raise ValueError("max_tokens 必须大于 0")
    if src.ndim == 1:
        src = src.unsqueeze(0)
    if src.ndim != 2 or src.size(0) != 1:
        raise ValueError("src 必须是形状为 (seq_len,) 或 (1, seq_len) 的单条源句")
    src = src.to(device)
    src_mask = create_padding_mask(src, pad_idx=0)
    memory = model.encode(src, src_mask)
    generated = [SOS_IDX]

    for _ in range(max_tokens):
        tgt = torch.tensor([generated], dtype=torch.long, device=device)
        logits = model.generator(model.decode(tgt, memory, src_mask, create_decoder_mask(tgt, pad_idx=0)))
        next_token = logits[:, -1, :].argmax(dim=-1).item()
        generated.append(next_token)
        if next_token == EOS_IDX:
            break
    return generated


@torch.no_grad()
def greedy_translate(model, sentence: str, src_vocab, tgt_vocab, device: torch.device, max_tokens: int = 64) -> str:
    """从 ``<sos>`` 开始逐 token 贪心解码，直到 ``<eos>`` 或长度上限。"""
    src_ids = src_vocab.encode(tokenize_en(sentence), add_sos_eos=True)
    generated = greedy_decode_ids(model, torch.tensor(src_ids, dtype=torch.long), device, max_tokens)

    return "".join(tgt_vocab.decode(generated))

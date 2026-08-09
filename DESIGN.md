# Transformer 中英翻译项目 —— 设计文档

> 目标读者：刚开始学习 Transformer 的初学者
> 目标任务：基于 `cmn.txt`（中英平行语料）从零实现一个 Transformer，完成中英翻译
> 阅读方式建议：先看「整体架构图」建立全局印象，再逐个文件细读，最后看「数据是怎么流动的」把所有文件串起来

---

## 一、整体项目结构回顾

```
transformer-python/
├── data/
│   └── cmn.txt
├── transformer/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── vocab.py
│   │   ├── tokenizer.py
│   │   └── dataset.py
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── embedding.py
│   │   ├── positional_encoding.py
│   │   ├── masks.py
│   │   ├── attention.py
│   │   ├── feed_forward.py
│   │   └── layer_norm.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   └── transformer.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── loss.py
│   │   ├── optimizer.py
│   │   └── trainer.py
│   ├── config.py
│   └── utils.py
├── scripts/
│   ├── train.py
│   ├── translate.py
│   └── evaluate.py
├── checkpoints/
└── tests/
```

整个项目可以看成**四条流水线**，一条接一条：

1. **数据流水线**（`transformer/data/`）：把 `cmn.txt` 这种纯文本，变成模型能吃的数字张量（tensor）
2. **模型零件**（`transformer/modules/`）：搭积木用的一个个小零件，比如"注意力"、"位置编码"
3. **模型组装**（`transformer/model/`）：把零件拼成完整的 Encoder-Decoder Transformer
4. **训练流水线**（`transformer/training/`）：拿数据喂给模型，算 loss，反向传播，更新参数

`scripts/` 里的文件不写具体逻辑，只负责"按顺序调用上面四条流水线"，是真正跑起来的入口。

下面逐个文件讲。

---

## 二、`transformer/data/` —— 数据流水线

这一层解决的问题是：**计算机不认识"我爱你"和"I love you"这样的文字，只认识数字**。所以要先把文字转换成数字序列，这个过程叫**分词（tokenize）**和**数值化（numericalize）**。

### 2.1 `vocab.py` —— 词表管理

**作用**：维护"字/词 ↔ 数字编号"的双向映射表。

**你可以把它想象成一本词典**：
- 中文词典：`{"我": 5, "爱": 12, "你": 8, ...}`
- 英文词典：`{"i": 3, "love": 20, "you": 7, ...}`

因为中英文是两种完全不同的语言，**通常需要两份独立的词表**（一份中文，一份英文）。

**核心内容**：
- 特殊符号，一定要预留：
  - `<pad>`：填充符（因为一个 batch 里句子长短不一，短的要补齐）
  - `<sos>`：句子开始（start of sentence）
  - `<eos>`：句子结束（end of sentence）
  - `<unk>`：未登录词（词表里没见过的词，比如生僻字）
- `build_vocab(sentences, min_freq)`：遍历所有句子，统计词频，词频低于 `min_freq` 的词不收录（防止词表过大、稀有词学不好）
- `word2idx` / `idx2word`：两个字典，互为反查
- `__len__`：返回词表大小，后面模型的 Embedding 层需要这个数字

**为什么要单独拆出来**：`tokenizer.py` 负责"怎么切词"，`vocab.py` 负责"切出来的词怎么变成数字"，这是两件独立的事——切词方式可能换（比如从"按字切"换成"按 BPE 切"），但词表的管理逻辑不用大改。拆开后两边互不影响，方便你后续做实验对比不同分词方式的效果。

### 2.2 `tokenizer.py` —— 分词器

**作用**：把一句话切成一个个"词元（token）"的列表。

例如：
- 英文："I love you" → `["i", "love", "you"]`（通常按空格切，可能还要处理标点）
- 中文："我爱你" → `["我", "爱", "你"]`（中文没有空格，初学阶段最简单的做法是**按字切**，即每个汉字算一个 token；进阶可以用 jieba 分词按词切）

**核心内容**：
- `Tokenizer` 类（或者简单函数），针对中英文可能需要两套逻辑：
  - `tokenize_en(sentence)`：英文分词（转小写、按空格拆分、去除/分离标点）
  - `tokenize_zh(sentence)`：中文分词（按字拆分，最简单可靠，初学建议先用这个）
- 输出统一是一个 Python 列表，比如 `["我", "爱", "你"]`

**注意**：这个文件里**不应该**出现"转数字"的逻辑，切词和编号是两个步骤，混在一起会导致代码难以复用和测试。

### 2.3 `dataset.py` —— 数据集与批处理

**作用**：这是数据流水线里最关键的一环，把「文本文件」最终变成「模型 forward 时能直接喂进去的张量」。

**具体要做的事情**，按顺序：

1. **读取 `cmn.txt`**：这个文件每一行通常是 `英文\t中文\t其他信息`（Tab 分隔），需要解析出中英文两列
2. **调用 `tokenizer.py`** 把每一行的中英文分别切成 token 列表
3. **调用 `vocab.py`** 建好中文词表和英文词表（如果还没建的话），并把 token 列表转成数字 id 列表，同时在句子首尾加上 `<sos>` 和 `<eos>`
   - 例："我爱你" → `["<sos>", "我", "爱", "你", "<eos>"]` → `[1, 5, 12, 8, 2]`
4. **实现 PyTorch 的 `Dataset` 类**：
   - `__len__`：数据集一共多少句对
   - `__getitem__(idx)`：返回第 `idx` 条数据的 `(源语言 id 序列, 目标语言 id 序列)`
5. **实现 `collate_fn`（拼 batch 的函数）**：因为一个 batch 里句子长度不一样，需要：
   - 找出这个 batch 里最长的句子长度
   - 把短句子用 `<pad>` 的 id 补齐到相同长度
   - 最终返回形状为 `(batch_size, seq_len)` 的张量
6. **构建 `DataLoader`**：PyTorch 自带的工具，自动打乱数据、按批取数据、调用 `collate_fn`

**这个文件的输出**（也就是它和外部世界打交道的接口）：
一个 `DataLoader` 对象，每次迭代吐出一个 batch，形如：
```python
{
  "src": tensor(shape=[batch_size, src_len]),   # 源语言(比如英文)的 id 序列
  "tgt": tensor(shape=[batch_size, tgt_len]),   # 目标语言(比如中文)的 id 序列
}
```

这就是**数据流水线的终点**，后面所有模块拿到的都是这种张量，不再关心"这原本是什么句子"。

---

## 三、`transformer/modules/` —— 模型基础零件

这一层是 Transformer 论文里各个公式的直接代码实现，每个文件都很独立，**建议每写完一个就单独测试一下输出的形状对不对**，这是最容易出 bug 又最难排查的部分。

### 3.1 `embedding.py` —— 词嵌入层

**作用**：把"词的编号（一个整数）"变成"一个有意义的向量（一串浮点数）"。

**为什么需要它**：数字 5（"我"）和数字 6（"你"）这两个编号本身没有任何数学关系，编号相邻不代表语义相近。Embedding 层本质是一张"查找表"，每个词对应一行可学习的向量，训练过程中，语义相近的词会自动学出相近的向量。

**核心内容**：
- 就是对 `nn.Embedding(vocab_size, d_model)` 的封装
- 论文里有个细节：**乘上 `sqrt(d_model)` 做缩放**，你实现时容易漏掉，建议注意一下
- 输入：`(batch_size, seq_len)` 的整数张量
- 输出：`(batch_size, seq_len, d_model)` 的浮点张量，`d_model` 是你设定的向量维度（比如 512）

### 3.2 `positional_encoding.py` —— 位置编码

**作用**：给每个位置的词向量"打上位置标签"。

**为什么需要它**：Transformer 的核心 Attention 机制本身不知道"词的顺序"（它是并行处理所有词的，不像 RNN 一个个按顺序读），"我爱你"和"你爱我"如果不加位置信息，Attention 层看到的是一样的东西。所以要额外注入位置信息。

**核心内容**：
- 用固定的正弦、余弦函数公式生成一张"位置编码表"，形状是 `(max_len, d_model)`
- 把这张表**加到** `embedding.py` 输出的词向量上（不是拼接，是相加）
- 因为是固定公式算出来的，不需要训练，所以通常写成 `register_buffer` 而不是可学习参数

**这一步做完之后**，模型看到的向量就同时包含了"这个词是什么意思"和"这个词在句子里第几个位置"两种信息。

### 3.3 `masks.py` —— 掩码生成

**作用**：告诉 Attention 机制"哪些位置不能看"。这是 Transformer 里最容易搞错、但又非常重要的部分，建议花时间理解清楚。

有两种 mask，作用完全不同：

**① Padding Mask（填充掩码）**
- 因为 `dataset.py` 里把短句子用 `<pad>` 补齐了，这些补出来的位置是没有意义的
- Padding Mask 的作用：让 Attention 计算时，直接忽略这些 `<pad>` 位置，不让它们影响真实词的计算
- Encoder 和 Decoder 都需要

**② Look-ahead Mask（前瞻掩码 / 因果掩码）**
- 只有 Decoder 需要
- 训练时，Decoder 是"一次性"看到整个目标句子的，但翻译是"从左到右一个个词生成"的过程，不能让模型在预测第 3 个词的时候"偷看"第 4、5 个词的答案
- 所以要生成一个"上三角"的掩码矩阵，强制第 `i` 个位置只能看到第 `1` 到第 `i` 个位置

**核心内容**：
- `create_padding_mask(seq, pad_idx)`：返回一个布尔张量，标记哪些位置是 `<pad>`
- `create_look_ahead_mask(size)`：返回一个上三角布尔矩阵
- Decoder 实际使用的是**两者的结合**（既不能看 pad，也不能看未来）

### 3.4 `attention.py` —— 注意力机制

**这是整个 Transformer 最核心的部分**，建议多花时间理解。

**核心思想一句话**：对句子里的每个词，去"查询（Query）"句子里所有词的"内容（Key/Value）"，算出"这个词应该更关注谁"的权重，再按权重把所有词的信息加权求和。

**核心内容**，通常包含两个东西：

**① Scaled Dot-Product Attention（缩放点积注意力）**
- 输入三个矩阵：`Q`（查询）、`K`（键）、`V`（值）
- 计算公式：`Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V`
- 用大白话解释每一步：
  1. `QK^T`：算出每个词和每个词之间的"相关性得分"
  2. `/ sqrt(d_k)`：缩放一下，防止得分太大导致 softmax 梯度消失
  3. 如果传入了 mask，在这一步把不该看的位置的得分设成一个很小的数（比如 -1e9），这样 softmax 之后这些位置的权重会接近 0
  4. `softmax(...)`：把得分变成"权重"（加起来等于 1）
  5. `... V`：按权重把所有词的 V 加权求和，得到这个词新的表示

**② Multi-Head Attention（多头注意力）**
- 不是只算一次上面的 Attention，而是把 `Q/K/V` 切成多个"头"（比如 8 个头），每个头独立算一次 Attention，再把结果拼接起来
- 为什么要这样做：让模型能同时从多个不同的"角度"去关注句子里的关系（比如一个头关注语法关系，另一个头关注语义相关性）

**这个文件会被用到三个地方**（这也是为什么要单独封装成通用模块）：
1. Encoder 内部的 **Self-Attention**（自己关注自己，Q/K/V 都来自源语言句子）
2. Decoder 内部的 **Masked Self-Attention**（自己关注自己，但加了 look-ahead mask）
3. Decoder 里的 **Cross-Attention**（Q 来自目标语言，K/V 来自 Encoder 的输出——这一步是翻译任务真正"对齐"两种语言的地方）

### 3.5 `feed_forward.py` —— 前馈神经网络

**作用**：每个词的向量经过 Attention 处理后，再单独过一个小型的两层全连接网络，做进一步的非线性变换。

**核心内容**：
- 结构很简单：`Linear(d_model, d_ff) → ReLU → Linear(d_ff, d_model)`
- `d_ff` 通常比 `d_model` 大很多（论文里 `d_model=512, d_ff=2048`），可以理解成"先把维度打开变大，让模型有更多空间做非线性变换，再压缩回原来的维度"
- 这个变换是"逐位置"独立进行的，句子里第 1 个词和第 5 个词的这一步计算互不影响

### 3.6 `layer_norm.py` —— 归一化与残差连接

**作用**：这是让深层网络能够训练稳定收敛的"稳定器"，论文里每个子层（Attention、FFN）后面都会接这个。

**核心内容**：
- **残差连接（Residual Connection）**：`输出 = 子层(输入) + 输入`。好处是梯度可以直接"抄近道"传回浅层，不容易梯度消失，训练更容易
- **Layer Normalization**：对每个词自己的向量做归一化（均值0方差1），让训练过程更稳定
- 通常封装成一个叫 `SublayerConnection` 的小类，把"子层 + 残差 + LayerNorm"这一整套操作打包，这样在 `encoder.py`/`decoder.py` 里写起来更简洁：
  ```python
  x = sublayer_connection(x, lambda x: self_attention(x, x, x, mask))
  ```

---

## 四、`transformer/model/` —— 模型组装

这一层把上面 `modules/` 里的小零件，按照 Transformer 的论文架构图拼起来。

### 4.1 `encoder.py` —— 编码器

**作用**：把源语言句子（比如英文）编码成一系列"富含上下文信息"的向量表示。

**结构**（从内到外）：
- **`EncoderLayer`（单层编码器）**：一层里包含两个子层：
  1. Multi-Head **Self**-Attention（+ 残差 + LayerNorm）
  2. Feed Forward（+ 残差 + LayerNorm）
- **`Encoder`（编码器整体）**：把 `EncoderLayer` **堆叠 N 层**（论文里 N=6），每层结构相同但参数不共享，前一层的输出是后一层的输入

**输入**：源语言的 embedding + 位置编码后的张量，形状 `(batch, src_len, d_model)`，以及 padding mask
**输出**：形状不变，还是 `(batch, src_len, d_model)`，但每个位置的向量现在已经"融合了整个源语言句子的上下文信息"

### 4.2 `decoder.py` —— 解码器

**作用**：结合 Encoder 的输出和已经生成的目标语言部分句子，一步步预测下一个词。

**结构**：
- **`DecoderLayer`（单层解码器）**：一层里包含**三个**子层（比 Encoder 多一个）：
  1. Masked Multi-Head **Self**-Attention（用 look-ahead mask，只能看自己前面的词）+ 残差 + LayerNorm
  2. Multi-Head **Cross**-Attention（Q 来自上一步的输出，K/V 来自 **Encoder 的输出**）+ 残差 + LayerNorm —— 这一步就是"翻译时去参考原文"
  3. Feed Forward + 残差 + LayerNorm
- **`Decoder`（解码器整体）**：同样堆叠 N 层

**输入**：目标语言的 embedding+位置编码张量、Encoder 的输出、look-ahead mask、padding mask
**输出**：`(batch, tgt_len, d_model)`

### 4.3 `transformer.py` —— 完整模型

**作用**：把 Encoder、Decoder、输入输出层拼装成一个完整的、能直接调用的模型类，这是训练脚本真正会 import 的类。

**核心内容**：
- `Transformer(nn.Module)` 类，`__init__` 里创建：
  - 源语言 `Embedding` + `PositionalEncoding`
  - 目标语言 `Embedding` + `PositionalEncoding`（中英文用**不同**的 embedding，因为是两种语言）
  - `Encoder`
  - `Decoder`
  - 最后一个 `Linear(d_model, tgt_vocab_size)`：把 Decoder 输出的向量，映射成"每个目标语言词的得分"，后面接 softmax 就能得到"下一个词是什么"的概率分布
- `forward(src, tgt, src_mask, tgt_mask)` 方法，按顺序调用：
  1. 源语言过 embedding + 位置编码
  2. 过 Encoder，得到 `encoder_output`
  3. 目标语言过 embedding + 位置编码
  4. 连同 `encoder_output` 一起过 Decoder
  5. 过最后的 Linear 层，输出 `(batch, tgt_len, tgt_vocab_size)` 的得分张量

**这个文件是整个 `modules/` + 大部分 `model/` 内容的"最终交付物"**——`scripts/train.py` 只需要 `from transformer.model.transformer import Transformer` 这一行就能拿到完整模型。

---

## 五、`transformer/training/` —— 训练流水线

### 5.1 `loss.py` —— 损失函数

**作用**：衡量模型预测的"下一个词概率分布"和"真实的下一个词"之间差多少。

**核心内容**：
- 基础做法是交叉熵损失（CrossEntropyLoss）
- **注意两个细节**（论文里的重点，初学者容易忽略）：
  1. 计算 loss 时要**忽略 `<pad>` 位置**（这些位置不是真实的词，不该参与 loss 计算），`nn.CrossEntropyLoss(ignore_index=pad_idx)` 可以直接做到
  2. 论文用了 **Label Smoothing**：不要求模型把正确答案的概率学到100%，而是留一点概率给其他词，防止模型过度自信、提高泛化能力。PyTorch 的 `CrossEntropyLoss` 自带 `label_smoothing` 参数，可以直接用，初学阶段不需要自己手写

### 5.2 `optimizer.py` —— 学习率调度

**作用**：论文里用的不是固定学习率，而是一个"先热身、后衰减"的调度策略（Warmup），这对训练稳定性影响很大，很多人复现效果不好就是漏了这一步。

**核心内容**：
- 底层优化器用 Adam
- 在 Adam 外面包一层学习率调度逻辑：
  - 训练刚开始的若干步（warmup steps），学习率从很小的值线性增长
  - 过了 warmup 阶段后，学习率按 `step_num^(-0.5)` 的规律衰减
- 通常封装成一个 `NoamOpt`（论文作者名 Noam 命名）类，包一层 `step()` 方法，每次调用既更新学习率又调用 Adam 的 `step()`

### 5.3 `trainer.py` —— 训练循环

**作用**：把"数据、模型、损失函数、优化器"这几样东西组织起来，真正执行"喂数据 → 前向传播 → 算 loss → 反向传播 → 更新参数"的循环，也就是整个项目"跑起来"的地方。

**核心内容**：
- `Trainer` 类，`__init__` 接收 model、DataLoader、loss 函数、optimizer、config（超参数）等
- `train_one_epoch()`：遍历一遍训练数据的完整逻辑：
  1. 从 `dataset.py` 产出的 DataLoader 里取一个 batch（`src`, `tgt`）
  2. 用 `masks.py` 生成对应的 padding mask 和 look-ahead mask
     - 这里有个重要细节：Decoder 的输入和"要预测的目标"要错开一位（Teacher Forcing）：
       - Decoder 输入：`tgt[:, :-1]`（去掉最后一个词）
       - 预测目标（算 loss 用）：`tgt[:, 1:]`（去掉第一个词，也就是 `<sos>`）
       - 这样模型在预测第 `i` 个词时，输入的是真实的前 `i-1` 个词，这就是"Teacher Forcing"训练方式
  3. 调用 `transformer.py` 里的模型做 forward，拿到预测得分
  4. 用 `loss.py` 算 loss
  5. `loss.backward()` 反向传播
  6. `optimizer.step()` 更新参数（这里的 optimizer 就是 `optimizer.py` 里包装过的）
  7. 打印/记录日志（比如每 N 个 batch 打印一次当前 loss）
- `evaluate()`：类似的流程，但不做反向传播，只算验证集上的 loss，用于监控是否过拟合
- `save_checkpoint()`：把模型参数存到 `checkpoints/` 目录

---

## 六、`transformer/config.py` 和 `utils.py`

### 6.1 `config.py`

**作用**：所有超参数集中在一个地方管理，避免散落在各个文件里改起来到处找。

建议包含：
```python
d_model = 512        # 词向量维度
num_heads = 8         # 多头注意力的头数
num_layers = 6         # Encoder/Decoder 各堆叠几层
d_ff = 2048         # 前馈网络中间层维度
dropout = 0.1
max_len = 100        # 位置编码支持的最大句子长度
batch_size = 32
warmup_steps = 4000
num_epochs = 20
pad_idx = 0         # 需要和 vocab.py 里的约定一致
```

**为什么重要**：几乎每个文件（`embedding.py` 需要 `d_model`，`attention.py` 需要 `num_heads`，`trainer.py` 需要 `batch_size`……）都要用到这些数字，写死在各处非常容易出现"改了一个地方忘了改另一个地方"导致的 bug。

### 6.2 `utils.py`

**作用**：放一些跟"业务逻辑"无关、但又到处都要用的小工具函数，比如：
- `set_seed(seed)`：固定随机种子，保证实验可复现
- `get_device()`：判断用 CPU 还是 GPU
- 简单的日志打印/计时函数

---

## 七、`scripts/` —— 真正跑起来的入口

这一层的文件**不写核心逻辑**，只做"组装调用"，好处是逻辑改动都在 `transformer/` 包里，入口脚本基本不用动。

### 7.1 `train.py`

大致逻辑（伪代码，帮助你理解调用顺序）：
```python
from transformer.config import Config
from transformer.data.dataset import get_dataloader
from transformer.model.transformer import Transformer
from transformer.training.trainer import Trainer

config = Config()
train_loader, src_vocab, tgt_vocab = get_dataloader("data/cmn.txt", config)
model = Transformer(len(src_vocab), len(tgt_vocab), config)
trainer = Trainer(model, train_loader, config)
trainer.train()
```

### 7.2 `translate.py`

**作用**：拿训练好的模型权重，输入一句新的英文（或中文），输出翻译结果。

这里会用到一个训练时不需要的技术叫 **贪心解码 / Beam Search**（因为推理时没有"标准答案"可以 Teacher Forcing 了，要模型自己一个词一个词生成，直到生成 `<eos>` 或达到最大长度）。这是初学阶段可以放到最后再实现的部分。

### 7.3 `evaluate.py`

**作用**：在验证集/测试集上跑一遍模型，计算 BLEU 等翻译质量指标，量化评估模型效果。

---

## 八、数据是怎么"流动"起来的（串联全流程）

用一次完整的训练迭代，把所有文件串一遍：

```
cmn.txt (原始文本)
   │
   ▼ [tokenizer.py 切词] + [vocab.py 转数字]
dataset.py 产出: src_ids, tgt_ids  (整数张量, 已 padding)
   │
   ▼
transformer.py::forward(src, tgt_input)
   │
   ├─▶ src 走: embedding.py → positional_encoding.py → encoder.py
   │        （用到 masks.py 生成的 padding mask）
   │        输出 encoder_output
   │
   └─▶ tgt_input 走: embedding.py → positional_encoding.py → decoder.py
            （用到 masks.py 生成的 look-ahead mask + padding mask）
            （decoder.py 内部会用 encoder_output 做 cross-attention）
            输出 decoder_output
   │
   ▼
Linear 层: decoder_output → 每个位置对目标词表的得分
   │
   ▼
loss.py: 得分 vs tgt_label(真实的下一个词) → 算出 loss
   │
   ▼
loss.backward() → optimizer.py(带 warmup 的 Adam) → 更新模型所有参数
```

而 `attention.py` / `feed_forward.py` / `layer_norm.py` 这些，是 `encoder.py` 和 `decoder.py` 内部反复调用的"零件"，不直接出现在这张主流程图里，但每一层都在用它们。

---

## 九、建议的实现顺序（附验收标准）

按这个顺序写，每一步都有一个简单的方法自查是否写对了：

| 步骤 | 文件 | 验收方法 |
|---|---|---|
| 1 | `vocab.py` | 手动传几句话进去，打印 `word2idx`，检查特殊符号在不在、编号对不对 |
| 2 | `tokenizer.py` | 打印切词结果，人工看着是否合理 |
| 3 | `dataset.py` | 从 DataLoader 里取一个 batch，打印 `src.shape` / `tgt.shape`，确认是 `(batch_size, seq_len)` |
| 4 | `embedding.py` + `positional_encoding.py` | 随便造一个 `(batch, seq_len)` 的整数张量传进去，检查输出形状是 `(batch, seq_len, d_model)` |
| 5 | `masks.py` | 打印生成的 mask 矩阵，人工检查上三角/pad位置对不对 |
| 6 | `attention.py` | 用随机张量测试，**重点检查加了 mask 之后，对应位置的权重是否接近 0** |
| 7 | `feed_forward.py` + `layer_norm.py` | 检查输入输出形状是否一致（这两个模块都不改变形状） |
| 8 | `encoder.py` | 单独测试：随机张量输入，输出形状 `(batch, src_len, d_model)` |
| 9 | `decoder.py` | 单独测试：配合一个假的 `encoder_output`，检查输出形状 |
| 10 | `transformer.py` | 完整跑一次 forward，用极小的假数据（比如词表 10 个词，2 条句子），确认整个流程不报错 |
| 11 | `loss.py` + `optimizer.py` | 用第 10 步的假数据跑几十步，观察 loss 是否在下降 |
| 12 | `trainer.py` + `scripts/train.py` | 先用 `cmn.txt` 里的一小部分数据（比如前 100 行）跑通全流程，loss 能下降，再上全量数据 |
| 13 | `scripts/translate.py` | 用训练好的模型翻译一句你熟悉的话，直觉判断翻得像不像样 |

**一个非常重要的调试建议**：Transformer 90% 的 bug 都是"张量形状不对"或者"mask 用错了方向"，强烈建议每写一个模块，都先用**极小的假数据**（比如 `batch_size=2, seq_len=5, d_model=8`）单独跑一下，打印 `.shape`，确认没问题了再往下一层拼，不要一次性把所有模块写完再联调，那样出了问题很难定位在哪一层。

---

需要我接下来帮你把这套目录结构在项目里实际创建出来（建好文件夹和空文件、写好每个文件的函数签名和 docstring 占位）吗？你确认后可以直接下载放到 `D:\UV\transformer-python` 里，然后按上面的顺序一步步填代码。

# Transformer 项目待办

本项目目标是实现一个可训练、可保存、可推理和可评估的英译中 Transformer。

## 当前状态

- [x] 数据读取、英文/中文分词、词表、`Dataset` 与 `DataLoader`
- [x] Embedding、位置编码、掩码、多头注意力、前馈网络与 LayerNorm
- [x] Encoder、Decoder 与 Transformer 模型组装
- [ ] 训练、保存、翻译、评估与自动化测试

## P0：先恢复模型可运行性

- [x] 修复 `transformer/model/transformer.py`：`forward()` 应返回 `logits`，而不是未定义的 `log`。
- [x] 用一个小型随机 batch 验证：模型前向传播返回 `(batch, tgt_len, tgt_vocab_size)`，且可以正常执行 `loss.backward()`。
- [x] 为模型构造增加基础参数校验，例如 `d_model % num_heads == 0`（注意力层已有校验）和合理的词表大小。

验收：可通过一段最小 Python 脚本完成 forward、交叉熵 loss 和 backward，且无异常或 NaN。

## P1：补齐训练闭环

- [x] 实现 `transformer/training/loss.py`：基于 `nn.CrossEntropyLoss(ignore_index=PAD_IDX)` 的目标语言 token loss。
- [x] 实现 `transformer/training/optimizer.py`：创建 Adam/AdamW 优化器；可选加入 Transformer 学习率 warmup 调度器。
- [x] 实现 `transformer/training/trainer.py`：
  - [x] Teacher forcing：输入 `tgt[:, :-1]`，监督目标为 `tgt[:, 1:]`。
  - [x] 创建源 padding mask 与目标 decoder mask。
  - [x] 训练 step：zero grad、forward、loss、backward、optimizer step。
  - [x] 验证 step：`model.eval()` 和 `torch.no_grad()`。
  - [x] 记录每 epoch 的 train/validation loss。
  - [x] 支持 device（CPU/CUDA）和随机种子。
- [x] 实现 `transformer/config.py`：集中放置数据路径、模型超参数、训练超参数和 checkpoint 路径。
- [x] 实现 `scripts/train.py`：串联数据集、模型、优化器和 Trainer，并保存 checkpoint。

验收：使用少量样本可以过拟合；使用完整语料可以完成至少一个 epoch，并在 `checkpoints/` 输出可恢复的模型。

## P2：实现推理与评估

- [x] 定义 checkpoint 格式：至少保存 `model_state_dict`、词表、模型配置、epoch 与验证损失。
- [x] 实现 checkpoint 加载与恢复训练能力。
- [x] 实现 `scripts/translate.py`：
  - [x] 输入英文句子，复用训练时的英文/中文词表。
  - [x] Encoder 仅运行一次。
  - [x] 从 `<sos>` 开始贪心解码，遇到 `<eos>` 或达到最大长度时停止。
  - [x] 将 token id 解码成中文文本并输出。
- [x] 实现 `scripts/evaluate.py`：在验证集上报告 token loss/perplexity；可选增加 BLEU。

验收：可从 checkpoint 翻译任意英文输入，并能在验证集输出稳定的评估指标。

## P3：测试与质量保障

- [ ] 为 tokenizer、vocab、dataset/collate_fn 编写单元测试。
- [ ] 测试三种 mask 的形状和语义（pad 不能被关注、未来 token 不能被关注）。
- [ ] 测试注意力和完整模型的输出形状、梯度回传与 padding 兼容性。
- [ ] 增加端到端 smoke test：小数据集训练 1--2 个 step，保存并加载 checkpoint，再完成一次翻译。
- [ ] 为公开函数补齐类型标注与简短文档；删除过长或与实现重复的行内注释。
- [ ] 在 README 中写明安装、训练、翻译、评估命令和预期目录结构。

验收：测试可重复运行；新环境用户按 README 能完成训练和单句翻译。

## 后续可选优化

- [x] 先随机切分训练/验证/测试集，再只按训练集建立词表，避免 token 泄漏。
- [x] 使用固定随机种子进行可复现的数据切分。
- [ ] 加入 batch/epoch 进度条、日志和最佳验证 loss checkpoint。
- [ ] 支持 beam search、长度惩罚与注意力可视化。
- [ ] 使用更合适的中文子词分词器（如 BPE/SentencePiece）并限制最大序列长度。
- [ ] 增加数据清洗、异常样本过滤与训练集/验证集/测试集三分。

## 建议实施顺序

1. 修复 `forward()` 返回值并做最小 forward/backward 验证。
2. 实现 loss、optimizer、trainer 与 `train.py`，先在很小的数据子集上过拟合。
3. 加入 checkpoint，再实现贪心翻译。
4. 完成评估脚本与测试，最后更新 README。

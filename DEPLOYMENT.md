# Linux 服务器训练指南

本项目已具备训练、checkpoint、恢复训练、翻译和验证评估功能，可以部署到 Linux 服务器进行较大规模实验。

## 1. 准备服务器

建议使用 Ubuntu 22.04+、Python 3.10--3.12 和 NVIDIA GPU。先确认 GPU 与驱动可见：

```bash
nvidia-smi
```

克隆项目并进入目录：

```bash
git clone https://github.com/Unsone/Transformer-Translation.git
cd Transformer-Translation
```

## 2. 创建环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

GPU 训练时，请按服务器驱动与 CUDA 版本选择 PyTorch 官方提供的安装命令。以下是 CUDA 12.1 的示例；如不匹配，请改用 PyTorch 官网对应命令：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

没有 GPU 时：

```bash
pip install -r requirements.txt
```

确认 PyTorch 能识别 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 3. 运行 notebook

```bash
jupyter lab --no-browser --ip=0.0.0.0 --port=8888
```

在本机使用 SSH 隧道访问：

```bash
ssh -L 8888:localhost:8888 <user>@<server>
```

然后在浏览器打开 `http://localhost:8888`，运行 `run_experiment.ipynb`。只应在受信任网络中暴露 Jupyter 服务；优先使用 SSH 隧道，不要直接开放 8888 端口。

## 4. 直接从终端训练（推荐用于长任务）

```bash
mkdir -p logs checkpoints
nohup .venv/bin/python scripts/train.py \
  --epochs 50 \
  --batch-size 128 \
  --d-model 256 \
  --num-heads 8 \
  --num-layers 4 \
  --d-ff 1024 \
  --checkpoint-path checkpoints/initial.pt \
  > logs/initial.log 2>&1 &
```

查看训练输出：

```bash
tail -f logs/initial.log
```

恢复训练：

```bash
.venv/bin/python scripts/train.py \
  --resume checkpoints/initial.pt \
  --epochs 20 \
  --checkpoint-path checkpoints/resumed.pt
```

## 5. 评估与翻译

```bash
.venv/bin/python scripts/evaluate.py checkpoints/initial.pt
.venv/bin/python scripts/translate.py checkpoints/initial.pt "How are you?"
```

## 注意事项

- `checkpoints/*.pt` 被 Git 忽略，不会自动上传到仓库；请单独备份重要 checkpoint。
- 训练集目前约 2.3 万句，适合验证实现和初步实验。要获得更好的翻译质量，建议加入更大、更干净的平行语料，并在 P3 阶段改进数据划分与分词。
- 训练脚本默认每个 epoch 会保存 `initial-epochN.pt`、更新 `initial.pt`（最新状态），并在验证损失改善时更新 `initial-best.pt`。可用 `--save-every N` 调整 epoch checkpoint 保存间隔，`--resume` 可从任一受信任 checkpoint 恢复。
- 每次训练还会写入与 checkpoint 同名的 `.log` 文件，记录 loss、学习率和每轮耗时；可选使用 `--warmup-steps` 与 `--max-grad-norm` 调整训练稳定性。

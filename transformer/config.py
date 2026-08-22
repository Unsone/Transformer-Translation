"""项目默认配置。训练脚本可通过命令行参数覆盖这些值。"""

from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    max_len: int = 256

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    data_path: Path = PROJECT_ROOT / "data" / "cmn.txt"
    checkpoint_path: Path = PROJECT_ROOT / "checkpoints" / "latest.pt"
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    min_freq: int = 1
    max_examples: int | None = None
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42

    def to_dict(self) -> dict:
        values = asdict(self)
        values["data_path"] = str(self.data_path)
        values["checkpoint_path"] = str(self.checkpoint_path)
        return values

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class TrainingConfig:
    num_epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    ckpt_path: str = ""
    num_workers: int = 0


def generate_config(yaml_file: str) -> TrainingConfig:

    # If the file doesn't exist, return default config.
    if not os.path.exists(yaml_file):
        return TrainingConfig()

    # Otherwise parse the file.
    kwargs: dict[str, Any] = {}
    with open(yaml_file, "r", encoding="utf-8") as f:
        for line in f:
            k, v = line.split(":")
            k = k.strip()
            v = v.strip("\n").strip()
            if k in ("max_epochs", "batch_size", "num_workers"):
                kwargs[k] = int(v)
            elif k in ("learning_rate", "weight_decay"):
                kwargs[k] = float(v)
            elif k == "betas":
                b1_str, b2_str = v.split(", ")
                b1_str = b1_str.strip()
                b2_str = b2_str.strip()
                kwargs[k] = (float(b1_str[1:]), float(b2_str[:-1]))
    if "training" in yaml_file.lower():
        return TrainingConfig(**kwargs)
    raise ValueError("File not recognised!")

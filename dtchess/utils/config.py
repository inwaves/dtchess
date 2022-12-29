import os
from dataclasses import dataclass
from typing import Any

INT_FIELDS = ("max_epochs", "batch_size", "num_workers", "checkpoint_every_n")
FLOAT_FIELDS = ("learning_rate", "weight_decay")
TUPLE_FIELDS = "betas"


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

            # Parse integers, floats and tuple args.
            if k in INT_FIELDS:
                kwargs[k] = int(v)
            elif k in FLOAT_FIELDS:
                kwargs[k] = float(v)
            elif k in TUPLE_FIELDS:
                fst_el, snd_el = v.split(", ")
                fst_el = fst_el.strip()
                snd_el = snd_el.strip()
                kwargs[k] = (float(fst_el[1:]), float(snd_el[:-1]))
            else:  # String fields are not parsed.
                kwargs[k] = v
    if "training" in yaml_file.lower():
        return TrainingConfig(**kwargs)
    raise ValueError("File not recognised!")

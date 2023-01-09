import time
import os
import platform
import xml.etree.ElementTree as ET
import torch as t
from argparse import ArgumentParser
from typing import Any, Callable, Tuple
import chess
import datasets
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dtchess.utils.config import TrainingConfig
from dtchess.utils.model import load_pretrained_tokeniser


def read_lines(filepath: str) -> list[Any]:
    lines = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            lines += [line.strip("\n")]

    return lines


def cuda_stats() -> str:
    if not t.cuda.is_available():
        return "No CUDA detected!"

    num_devices = t.cuda.device_count()
    to_gb = 1024**3
    stats: list[str] = ["\n#######\n"]
    for i in range(num_devices):
        current_stat = (
            f"Device: {t.cuda.get_device_name(i)}\n"
            f"Reserved/allocated/total (GB): {t.cuda.memory_reserved(i)/to_gb:.2f}"
            f"/{t.cuda.memory_allocated(i)/to_gb:.2f}"
            f"/{t.cuda.get_device_properties(i).total_memory/to_gb:.2f}\n"
            "########"
        )
        stats += [current_stat]

    return "\n".join(stats)


def extract_tag(input_string: str, tag_name: str) -> str | None:
    if tag_name not in input_string:
        raise ValueError("Tag not present in input string!")
    input_string = f"<root>{input_string}</root>"

    element = ET.fromstring(input_string).find(tag_name)

    return element.text if element is not None else element


def count_python_processes() -> int:
    cmd = "top -l 1" if "macOS" in platform.platform() else "top -bn1"
    cmd = f"{cmd} | grep python | wc -l"
    return int(os.popen(cmd, "r").read())


def timer(logger):
    def time_function(func) -> Callable[[Any, ...], Any]:
        def wrap_function(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            logger.info(
                f"Function {getattr(func, '__name__', func)} running on "
                f"process {os.getpid()} took {end-start:.4f}s."
            )
            return result

        return wrap_function

    return time_function


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def extract_filename(filepath: str) -> str:
    """Takes in a UNIX style filepath and returns the filename
    without an extension."""
    return filepath.split("/")[-1][:-4]


def parse_args() -> dict:
    parser = ArgumentParser()

    parser.add_argument("--input_filepath", help="Path to the PGN input file.")
    parser.add_argument(
        "--num_random_games",
        type=int,
        help="number of random games to generate",
        default=10000,
    )
    parser.add_argument("--model_name", help="Name of a model to load from wandb")
    parser.add_argument("--prompt_file", help="Filepath to prompts to complete.")
    parser.add_argument(
        "--generate_tokens",
        type=int,
        default=10,
        help="Number of tokens to generate for each prompt.",
    )
    argspace = parser.parse_args()
    return vars(argspace)


def training_setup(
    config: TrainingConfig,
) -> Tuple[AutoTokenizer, DataLoader,]:
    tokeniser = load_pretrained_tokeniser()
    train_dataloader = preprocess_data(tokeniser, config)

    return tokeniser, train_dataloader


def preprocess_data(tokeniser: AutoTokenizer, config: TrainingConfig) -> DataLoader:
    """Preprocesses data for the model."""

    dataset = datasets.load_dataset(config.dataset, streaming=True, split="train")

    input_ids = dataset.map(
        lambda seq: tokeniser(
            seq["text"],
            padding="max_length",
            max_length=tokeniser.model_max_len,
            truncation=True,
            return_tensors="pt",
        ),
        batched=True,
    )
    train_dl = DataLoader(
        input_ids, batch_size=config.batch_size, num_workers=config.num_shards
    )
    return train_dl


def board_to_sequence(board: chess.Board) -> str:
    return board.fen().split(" ")[0]


def dist_setup(rank, world_size) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def dist_cleanup() -> None:
    dist.destroy_process_group()

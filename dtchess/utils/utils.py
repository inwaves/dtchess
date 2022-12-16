import time
import os
import platform

from argparse import ArgumentParser
from typing import Any, Callable, Tuple
import chess
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Model, GPT2Tokenizer

import dtchess as dt
from dtchess.utils.config import TrainingConfig


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
                f"Function {getattr(func, '__name__', func)} running on process {os.getpid()} took {end-start:.4f}s."
            )
            return result

        return wrap_function

    return time_function


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel for p in model.parameters() if p.requires_grad)


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
    argspace = parser.parse_args()
    return vars(argspace)


def training_setup(
    config: TrainingConfig
) -> Tuple[
    GPT2Tokenizer,
    GPT2Model,
    optim.Adam,
    Tuple[DataLoader, DataLoader],
    nn.CrossEntropyLoss,
]:
    tokeniser, model = dt.models.gpt.create_model()
    optimiser = optim.Adam(model.parameters, lr=config.learning_rate)
    dataloaders = preprocess_data(tokeniser, config)
    loss_fn = nn.CrossEntropyLoss

    return tokeniser, model, optimiser, dataloaders, loss_fn


def preprocess_data(
    tokeniser: GPT2Tokenizer, config: TrainingConfig
) -> Tuple[DataLoader, DataLoader]:
    """Preprocesses data for the decision transformer."""

    # TODO: What needs to happen here:
    #   - data is loaded from the input file
    #   - data is encoded into input_ids
    #   - data is loaded into a dataloader and batched
    #   - the dataloader manages parallelism

    # input_ids = tokeniser.encode(input_sequences)
    # train_dl = DataLoader(train_ds, batch_size=config.batch_size)
    # test_dl = DataLoader(test_ds, batch_size=config.batch_size)
    # return train_dl, test_dl
    return None, None


def board_to_sequence(board: chess.Board) -> str:
    return board.fen().split(" ")[0]

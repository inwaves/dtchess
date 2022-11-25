import dtchess as dt
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
from transformers import GPT2Model, GPT2Tokenizer
from typing import Tuple
from argparse import ArgumentParser


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel for p in model.parameters() if p.requires_grad)


def parse_args() -> dict:
    parser = ArgumentParser()

    # Use this pattern to control args...
    # parser.add_argument()

    argspace = parser.parse_args()
    return vars(argspace)


def setup(args: dict) -> Tuple[GPT2Tokenizer, GPT2Model, optim.Adam, Tuple[DataLoader, DataLoader]]:
    tokeniser, model = dt.models.create_model()
    optimiser = optim.Adam(model.parameters, lr=args["lr"])
    dataloaders = preprocess_data(args)

    return tokeniser, model, optimiser, dataloaders


def preprocess_data(args: dict) -> Tuple[DataLoader, DataLoader]:
    pass

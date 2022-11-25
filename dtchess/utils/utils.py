import os
from argparse import ArgumentParser
from typing import Tuple

import chess.pgn as pgn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Model, GPT2Tokenizer

import dtchess as dt


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel for p in model.parameters() if p.requires_grad)


def parse_args() -> dict:
    parser = ArgumentParser()

    # TODO: Use this pattern to control args...
    # parser.add_argument()

    argspace = parser.parse_args()
    return vars(argspace)


def setup(args: dict) -> Tuple[
    GPT2Tokenizer, GPT2Model, optim.Adam, Tuple[DataLoader, DataLoader], nn.CrossEntropyLoss]:
    tokeniser, model = dt.models.create_model()
    optimiser = optim.Adam(model.parameters, lr=args["lr"])
    dataloaders = preprocess_data(tokeniser, args)
    loss_fn = nn.CrossEntropyLoss

    return tokeniser, model, optimiser, dataloaders, loss_fn


def preprocess_data(tokeniser: GPT2Tokenizer, args: dict) -> Tuple[DataLoader, DataLoader]:
    # TODO: how to preprocess .PGN data?
    print(os.getcwd())
    pgn_file = open("./dtchess/data/sample.pgn")
    game = pgn.read_game(pgn_file)
    all_nodes = []
    while game.next():
        next_node = game.next()
        all_nodes.append(next_node)
        game = next_node
    moves_and_evals = [(node.move.uci(), node.eval().relative.cp) for node in all_nodes]
    print(moves_and_evals)

    return None, None

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
    """Converts PGN data into sequences usable by our model, then wraps them in DataLoaders.
       This should allow the use of CLI args to switch between different modes of parsing the file:
        - convert to "[<ELO>|<OUTCOME>|<EVALS>] move1 move2 ..."
        - convert to "[<ELO>|<OUTCOME>|<EVALS>] board1 board2 ..."
    """
    pgn_file = open("./dtchess/data/sample.pgn")
    game = pgn.read_game(pgn_file)
    elo = game.headers["WhiteElo"]
    result= game.headers["Result"]
    all_nodes = []
    while game.next():
        next_node = game.next()
        all_nodes.append(next_node)
        game = next_node
    moves = [node.move.uci() for node in all_nodes]
    evals = [node.eval().relative.cp for node in all_nodes]
    white_total_loss = sum(evals[::2])
    black_total_loss = sum(evals[1::2])
    sequence = ""
    if args.sequence_type == "full":
        sequence = f"<ELO>{elo}</ELO> <RES>{result}</RES> <RET>{white_total_loss}</RET> {' '.join(moves)}"
    elif args.sequence_type == "elo":
        sequence = f"<ELO>{elo}</ELO> {' '.join(moves)}"
    elif args.sequence_type == "result":
        sequence = f"<RES>{result}</RES> {' '.join(moves)}"
        

    return None, None


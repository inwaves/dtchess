import time
import os
import platform

from argparse import ArgumentParser
from typing import Any, Callable, Tuple

import chess.pgn as pgn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Model, GPT2Tokenizer

import dtchess as dt


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

    parser.add_argument(
        "--input_filepath", required=True, help="Path to the PGN input file."
    )
    argspace = parser.parse_args()
    return vars(argspace)


def setup(
    args: dict,
) -> Tuple[
    GPT2Tokenizer,
    GPT2Model,
    optim.Adam,
    Tuple[DataLoader, DataLoader],
    nn.CrossEntropyLoss,
]:
    tokeniser, model = dt.models.create_model()
    optimiser = optim.Adam(model.parameters, lr=args["lr"])
    dataloaders = preprocess_data(tokeniser, args)
    loss_fn = nn.CrossEntropyLoss

    return tokeniser, model, optimiser, dataloaders, loss_fn


def process_game(game: pgn.Game, sequence_type: str) -> str:
    """Converts PGN data into sequences usable by our model, then wraps them in DataLoaders.
    Allows the use of CLI args to switch between different modes of parsing the file:
     - "[<ELO>|<RESULT>|<RETURN>] move1 move2 ..."
     - "[<ELO>|<RESULT>|<RETURN>] board1 board2 ..."
    """
    body, header = "", ""
    elo = game.headers["WhiteElo"]
    result = game.headers["Result"]

    # Play the game until the end, remembering moves, evals and board states.
    moves, evals, boards = [], [], []
    board = game.board()
    while game.next():
        # We don't remember the initial board state, it's always the same.
        move = game.variation(0).move
        board.push(game.mainline_moves()[0])
        boards.append(board.fen())
        moves.append(move.uci())

        # Not all game files have evals.
        if game.eval() and hasattr(game.eval().relative, "cp"):
            evals.append(game.eval().relative.cp)

        game = game.next()
    # TODO: Assert that moves, evals and boards are the correct shape.

    # A sequence comprises a header and a body. The header contains some
    # combination of ELO, game result and total return tokens. The body
    # is either a sequence of moves or boards, with or without the evals
    # of these from a chess engine.
    header_type, body_type = sequence_type.split(":")
    evals_present = len(evals) > 0
    if header_type == "full" and evals_present:
        # "<ELO>[ELO]</ELO> <RES>[RES]</RES> <RET>[RETURN]</RET> [MOVE1] [MOVE2] ..."
        white_total_loss = sum(evals[::2])
        header = f"<ELO>{elo}</ELO> <RES>{result}</RES> <RET>{white_total_loss}</RET>"
    elif header_type == "elo":
        # "<ELO>[ELO]</ELO> [MOVE1] [MOVE2] ..."
        header = f"<ELO>{elo}</ELO>"
    elif header_type == "result":
        # "<RES>[RESULT]</RES> [MOVE1] [MOVE2] ..."
        header = f"<RES>{result}</RES>"

    # Header can be blank if we want to feed in just the board/move with or without evals.
    # TODO: these should flag if evals are not present.
    elif body_type == "moves":
        # "[MOVE1] [MOVE2] ...".
        body = f"{' '.join(moves)}"
    elif body_type == "boards":
        # "[BOARD1] [BOARD2] ...
        body = f"{' '.join(boards)}"
    elif body_type == "evalmoves" and evals_present:
        # "[MOVE1] [EVAL1] [MOVE2] [EVAL2] ...".
        moves_evals_sequence = " ".join(list(zip(moves, evals)))
        body = f"{' '.join(moves_evals_sequence)}"
    elif sequence_type == "evalboards" and evals_present:
        # "[BOARD1] [EVAL1] [BOARD2] [EVAL2] ...".
        boards_evals_sequence = " ".join(list(zip(boards, evals)))
        body = f"{' '.join(boards_evals_sequence)}"

    return header + body


def preprocess_data(
    tokeniser: GPT2Tokenizer, args: dict
) -> Tuple[DataLoader, DataLoader]:
    """Preprocesses data for the decision transformer."""
    # TODO: Add logic here to crawl all the files.

    pgn_file = open("./dtchess/data/sample.pgn")
    while game := pgn.read_game(pgn_file) is not None:
        _ = process_game(game, args["sequence_type"])

    train_ds, test_ds = None, None
    # TODO: How to stream this data?
    # TODO: Add logic here to tokenise the sequences.
    train_dl = DataLoader(train_ds, batch_size=args["batch_size"])
    test_dl = DataLoader(test_ds, batch_size=args["batch_size"])
    return train_dl, test_dl

import time
from argparse import ArgumentParser
from typing import Tuple, Optional

import chess
import chess.pgn as pgn
import torch.nn as nn
import torch.optim as optim
from chess.pgn import ChildNode
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
    nodes: list[pgn.GameNode] = []
    boards: list[chess.Board] = []
    board = game.board()
    while game.next():
        # We don't remember the initial board state, it's always the same.
        board.push(game.variation(0).move)
        boards.append(board.copy())

        next_node: Optional[ChildNode] = game.next()
        nodes.append(next_node)
        game = next_node
    moves: list[str] = [node.move.uci() for node in nodes]
    evals: list[int] = [node.eval().relative.cp for node in nodes if node.eval() and hasattr(node.eval().relative, "cp")]
    # TODO: assert that moves, evals and boards are correct size.

    # A sequence comprises a header and a body.
    # The header contains some combination of ELO, game result and total return tokens.
    # The body is either a sequence of moves or boards, with or without the evals of these from a chess engine.
    header_type, body_type = sequence_type.split(":")
    if header_type == "full":
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
    if body_type == "moves":
        # "[MOVE1] [MOVE2] ...".
        body = f"{' '.join(moves)}"
    elif body_type == "boards":
        # "[BOARD1] [BOARD2] ...
        body = f"{' '.join(boards)}"
    elif body_type == "evalmoves":
        # "[MOVE1] [EVAL1] [MOVE2] [EVAL2] ...".
        moves_evals_sequence = ' '.join(list(zip(moves, evals)))
        body = f"{' '.join(moves_evals_sequence)}"
    elif body_type == "evalboards":
        # "[BOARD1] [EVAL1] [BOARD2] [EVAL2] ...".
        boards_evals_sequence = ' '.join(list(zip(boards, evals)))
        body = f"{' '.join(boards_evals_sequence)}"

    return header + body


def preprocess_data(tokeniser: GPT2Tokenizer, args: dict) -> Tuple[DataLoader, DataLoader]:
    """Preprocesses data for the decision transformer."""
    # TODO: Add logic here to crawl all the files.

    start = time.time()
    pgn_file = open("lichess_db_antichess_rated_2022-10.pgn")
    sequence_filepath = "./dtchess/data/sequences.txt"
    with open(sequence_filepath, "a+") as out:
        game: pgn.Game = pgn.read_game(pgn_file)
        while game is not None:
            sequence = process_game(game, args["sequence_type"])
            out.write(sequence + "\n")
            game = pgn.read_game(pgn_file)
    print(f"Writing sequences took: {time.time() - start}ms")
    # TODO: How to stream this data?
    # TODO: Add logic here to tokenise the sequences.
    # train_dl, test_dl = DataLoader(train_ds, batch_size=args["batch_size"]), DataLoader(test_ds, batch_size=args["batch_size"])
    return None, None

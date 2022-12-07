import chess.pgn as pgn
import multiprocessing as mp
import dill

from multiprocessing import Queue
from threading import Lock
from typing import NoReturn

import time
import io
import os

output_filepath = "./dtchess/data/sequences"
NUM_CORES = mp.cpu_count()
LINES_PER_GAME = 18


def read_games(input_filepath: str, game_queue: Queue, written: int, errs: int) -> None:
    # Fetch one game for each process but this one, and put it to the shared queue.
    with open(input_filepath, "r", encoding="utf-8") as pgnfile:
        ctr = 0
        lines = []
        for line in pgnfile:
            if ctr < LINES_PER_GAME:
                lines += [line]
                ctr += 1
            else:
                game = "".join(lines)
                game_queue.put(game)
                written += 1
                lines = []
                ctr = 0


def sequence_game(output_filepath: str, write_lock: Lock, game_queue: Queue) -> None:
    written = 0
    while not game_queue.empty():
        # This is a string representing a game.
        game_string = game_queue.get()
        try:
            game = pgn.read_game(io.StringIO(game_string))
        except ValueError:
            print(
                f"!!!!GAME DID NOT LOAD CORRECTLY!!!! Check the string:\n{game_string}"
            )
        elo = game.headers["WhiteElo"] if "WhiteElo" in game.headers else None
        result = game.headers["Result"] if "Result" in game.headers else None

        # Parse the GameNode object into moves, evals and boards.
        evals, boards = [], []
        board = game.board()
        try:
            while game.next():
                # We don't remember the initial board state, it's always the same.
                move = game.variation(0).move
                boards.append(board.fen())
                board.push(move)

                # Not all games have evals; if they do, record them.
                if game.eval() and hasattr(game.eval().relative, "cp"):
                    evals.append(game.eval().relative.cp)

                # Follow the game tree until the end node.
                game = game.next()
        except ValueError:
            continue

        # Use the moves, evals and boards to generate a sequence.
        if elo is not None and result is not None:
            header = f"<ELO>{elo}</ELO> <RES>{result}</RES>"
        else:
            header = ""
        if len(evals) > 0:  # i.e. if there are evals at all, use them.
            white_total_loss = sum(evals[::2])
            header = f"{header} <RET>{white_total_loss}</RET>"
            body = "||".join([f"{board}::{ev}" for (board, ev) in zip(boards, evals)])
        else:  # otherwise, just use the board states.
            body = f"{'||'.join(boards)}"

        # Append sequence to file.
        # write_lock.acquire()
        try:
            with open(f"{output_filepath}_{os.getpid()}.txt", "a+") as f:
                f.write(f"{header} {body}\n")
                written += 1
                # print(f"Worker proc: put game {game}")
        finally:
            pass
            # write_lock.release()
    # print(f"WP: no more to read after {written} games written.")


if __name__ == "__main__":
    print(f"{NUM_CORES=}")
    mp.set_start_method("fork")
    input_filepath = "./antichess2.pgn"
    output_filepath = f"./dtchess/data/sequences_{input_filepath[2:-4]}"
    written, errs = 0, 0

    # Spawn processes to read games from a PGN file and convert them to string sequences.
    write_lock = mp.Lock()
    game_queue: Queue = Queue()
    reader_process = mp.Process(
        target=read_games, args=(input_filepath, game_queue, written, errs)
    )
    sequencing_processes = [
        mp.Process(target=sequence_game, args=(output_filepath, write_lock, game_queue))
        for _ in range(NUM_CORES - 1)
    ]

    start = time.time()
    # Start all processes.
    reader_process.start()
    time.sleep(2)
    for process in sequencing_processes:
        process.start()

    reader_process.join()
    for process in sequencing_processes:
        process.join()

    print(f"Finished! Took {time.time() - start}s.")

import os
import sys
import random
import multiprocessing as mp
import chess  # type: ignore
import threading
from typing import List, Tuple
from utils import board_to_sequence, parse_args
from loguru import logger


NUM_CORES = mp.cpu_count()


def simulate_games(n: int, output_filepath: str, write_lock: threading.Lock) -> None:
    for i in range(n):
        game_seq, round_ct = one_game()
        write_lock.acquire()
        try:
            with open(output_filepath, "a+", encoding="utf-8") as f:
                f.write(game_seq)
                f.write("\n")
        finally:
            write_lock.release()
        logger.info(f"WP {os.getpid()} wrote game {i}.")


def one_game() -> Tuple[str, int]:
    board_sequences: List[str] = []
    board = chess.Board()
    round_ct: int = 0
    while board.outcome() is None:
        one_round(board)
        board_sequences += [board_to_sequence(board)]
        round_ct += 1

    game_sequence: str = "||".join(board_sequences)
    return game_sequence, round_ct


def one_round(board: chess.Board) -> None:
    if board.outcome() is not None:
        return
    lm = list(board.legal_moves)
    board.push(random.choice(lm))


def setup() -> list[mp.Process]:
    cl_args = parse_args()

    mp.set_start_method("fork")
    logpath = "./logs/random_moves.log"
    logger.add(sys.stderr, format="{time} {message}", enqueue=True, level="DEBUG")
    logger.add(
        logpath, format="{time} {message}", enqueue=True, retention=1, level="INFO"
    )
    logger.info(f"Writing to {logpath}.")
    output_filepath = "./data/random_games.txt"
    write_lock = mp.Lock()
    workers = [
        mp.Process(
            target=simulate_games,
            args=(cl_args["num_random_games"], output_filepath, write_lock),
        )
        for _ in range(NUM_CORES)
    ]
    return workers


if __name__ == "__main__":
    workers = setup()

    for i, worker in enumerate(workers):
        print(f"Starting worker {i}")
        worker.start()

    for worker in workers:
        worker.join()

    logger.info("Finished!")

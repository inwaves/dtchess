import os
import sys
import random
import multiprocessing as mp
import chess  # type: ignore
import threading
from typing import List, Tuple
from dtchess.utils.utils import board_to_sequence, parse_args
from loguru import logger


NUM_CORES = mp.cpu_count()

# Summary statistics for the distribution of ELO, returns
# and results from real chess games. These are mimicked when
# generating the random games so that it isn't immediately obvious
# that the values are made up.
SUM_STATS = {
    "elo_mean": 1658,
    "elo_variance": 375.6826,
    "return_mean": 560,
    "return_variance": 13630.9876,
    "white_win": 50530229,
    "black_win": 47251633,
    "tie": 4032083,
}
RESULTS = ["1-0", "0-1", "1/2-1/2"]


def simulate_games(n: int, output_filepath: str, write_lock: threading.Lock) -> None:
    for i in range(n):
        game_seq, round_ct = one_game()
        with write_lock:
            with open(output_filepath, "a+", encoding="utf-8") as f:
                f.write(game_seq)
                f.write("\n")
        logger.info(f"WP {os.getpid()} wrote game {i}.")


def one_game() -> Tuple[str, int]:
    elo = int(random.gauss(mu=SUM_STATS["elo_mean"], sigma=SUM_STATS["elo_variance"]))
    ret = int(
        random.gauss(mu=SUM_STATS["return_mean"], sigma=SUM_STATS["return_variance"])
    )
    result = random.choices(
        RESULTS, [SUM_STATS["white_win"], SUM_STATS["black_win"], SUM_STATS["tie"]]
    )[0]
    header = f"<ELO>{elo}</ELO><RET>{ret}</RET><RES>{result}</RES>"

    board_sequences: List[str] = []
    board = chess.Board()
    round_ct: int = 0
    while board.outcome() is None:
        one_round(board)
        board_sequences += [board_to_sequence(board)]
        round_ct += 1

    body: str = "||".join(board_sequences)
    game_sequence: str = f"{header} {body}"

    return game_sequence, round_ct


def one_round(board: chess.Board) -> None:
    if board.outcome() is not None:
        return
    lm = list(board.legal_moves)
    board.push(random.choice(lm))


def setup(cl_args: dict) -> list[mp.Process]:
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
    cl_args = parse_args()
    workers = setup(cl_args)

    for i, worker in enumerate(workers):
        print(f"Starting worker {i}")
        worker.start()

    for worker in workers:
        worker.join()

    logger.info(f"Finished generating {cl_args['num_random_games']} games.")

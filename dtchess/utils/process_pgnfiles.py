import time
import io
import os
import sys
import multiprocessing as mp
import chess.pgn as pgn
from loguru import logger

from multiprocessing import Queue
from threading import Lock
from utils import parse_args, extract_filename, time_decorator

NUM_CORES = mp.cpu_count()
MAX_LOG_SIZE = "2 GB"
MAX_ITEMS_IN_QUEUE = 10000

@time_decorator(logger)
def read_games(input_filepath: str, game_queue: Queue) -> None:
    games_processed = 0
    with open(input_filepath, "r", encoding="utf-8") as pgnfile:
        saw_newline = False
        lines = []
        for line in pgnfile:
            if not saw_newline:
                lines += [line]
                saw_newline = True if line == "\n" else 0
            else:
                lines += [line]
                game = "".join(lines)

                # Make sure there is room on the queue before putting.
                while game_queue.full():
                    logger.debug(f"Game queue full! Its size is: {game_queue.qsize()}")
                    time.sleep(0.001)

                # logger.info(f"RP {os.getpid()} just put game: {game}.")
                game_queue.put(game)
                games_processed += 1
                lines = []
                saw_newline = False

    logger.info(f"RP {os.getpid()} finished! Processed {games_processed} games.")


def sequence_game(output_filepath: str, write_lock: Lock, game_queue: Queue) -> None:
    num_games = 0
    total_elapsed: float = 0
    while not game_queue.empty():
        game_string = game_queue.get()
        try:
            game = pgn.read_game(io.StringIO(game_string))
        except ValueError:
            logger.debug(f"Game didn't load correctly. Check the string:\n{game_string}")
            continue

        if "Termination" in game.headers and game.headers["Termination"] == "Abandoned":
            continue

        elo = game.headers["WhiteElo"] if "WhiteElo" in game.headers else None
        result = game.headers["Result"] if "Result" in game.headers else None
        start = time.time()

        # Parse the GameNode object into moves, evals and boards.
        evals, boards = [], []
        board = game.board()
        try:
            while game.next():
                move = game.variation(0).move
                boards.append(
                    board.fen().split(" ")[0]
                )  # Only want the piece positions.
                board.push(move)

                # Not all games have evals; if they do, record them.
                if game.eval() and hasattr(game.eval().relative, "cp"):
                    evals.append(game.eval().relative.cp)

                # Follow the game tree until the end node.
                game = game.next()
        except ValueError:
            logger.debug("ValueError while parsing game. Skipping...")
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
        write_lock.acquire()
        try:
            with open(output_filepath, "a+") as f:
                f.write(f"{header} {body}\n")
                num_games += 1
                logger.info(f"WP {os.getpid()}: put game number {num_games}. {game_queue.qsize()} left in the queue.")
        finally:
            write_lock.release()
        total_elapsed += time.time() - start

    logger.info(
        f"WP {os.getpid()} processed {num_games} games, taking "
        f"{total_elapsed / num_games:.4f}s on average."
    )


def setup():
    logger.info(f"{NUM_CORES=}")
    mp.set_start_method("fork")

    args = parse_args()

    input_filepath = args["input_filepath"]
    output_filepath = f"./dtchess/data/sequences_{extract_filename(input_filepath)}.txt"
    logfile = f"./dtchess/logs/sequences_{extract_filename(input_filepath)}.log"
    logger.add(sys.stderr, format="{time} {message}", enqueue=True)
    logger.add(logfile, format="{time} {message}", enqueue=True, rotation=MAX_LOG_SIZE, retention=1)
    logger.info(f"Reading from {input_filepath}.\n Writing to: {output_filepath}")

    # Spawn processes to read games from a PGN file and
    # convert them to string sequences.
    write_lock = mp.Lock()
    game_queue: Queue = Queue()
    reader_process = mp.Process(target=read_games, args=(input_filepath, game_queue))
    sequencing_processes = [
        mp.Process(target=sequence_game, args=(output_filepath, write_lock, game_queue))
        for _ in range(NUM_CORES - 1)
    ]
    return reader_process, sequencing_processes


if __name__ == "__main__":
    reader_process, sequencing_processes = setup()
    start = time.time()
    # Start all processes.
    reader_process.start()
    time.sleep(2)
    for process in sequencing_processes:
        process.start()

    reader_process.join()
    for process in sequencing_processes:
        process.join()

    logger.info("Finished! Took {time.time() - start}s.")

import time
import io
import os
import queue
import sys
import multiprocessing as mp
import chess.pgn as pgn  # type: ignore
from loguru import logger  # type: ignore

from multiprocessing import Queue
from threading import Lock
from typing import List
from utils import extract_filename, parse_args, timer, extract_tag  # type: ignore

NUM_CORES = mp.cpu_count()
MAX_LOG_SIZE = "2 GB"
MAX_ITEMS_IN_QUEUE = 10000

def extract_distributions(filepath: str) -> None:
    elo_total, return_total = 0, 0
    failures = 0
    result_bincount = {"white_win": 0, "black_win": 0, "tie": 0}
    logger.info(f"Calculating average of ELOs and returns, counting results...")
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            # If present, extract ELO.
            try:
                elo_string = extract_tag(line, "ELO")
                elo_total += int(elo_string)
                elo_count += 1
            except ValueError:
                failures += 1

            # If present, extract returns.
            return_total += int(result_string)
            return_count += 1

            # If present, count the result outcomes.
            if result_string == "1/2-1/2":
                result_bincount["tie"] += 1
            elif result_string == "1-0":
                result_bincount["white_win"] += 1
            else:
                result_bincount["black_win"] += 1

        elo_mean = elo_total // elo_count
        return_mean = return_total // return_count
        logger.info("Done with average. Calculating deviations...")
        for line in f:
            pass
        logger.info("Done with deviations...")

    logger.info(f"{elo_mean=}, variance: {elo_variance=}")
    logger.info(f"{return_mean=}, variance: {return_variance=}")
    logger.info(f"{result_bincount=}")



@timer(logger)
def read_games(input_filepath: str, game_queue: Queue) -> None:
    games_processed: int = 0
    with open(input_filepath, "r", encoding="utf-8") as pgnfile:
        saw_newline: bool = False
        lines: List[str] = []
        for line in pgnfile:
            if not saw_newline:
                lines += [line]
                saw_newline = True if line == "\n" else False
            else:
                lines += [line]
                game = "".join(lines)

                # Make sure there is room in the queue before putting.
                try:
                    logger.info(
                        f"[RP {os.getpid()}] Put game number {games_processed}."
                    )
                    game_queue.put(game)
                    games_processed += 1
                except queue.Full:
                    logger.debug(f"Game queue full! Its size is: {game_queue.qsize()}")
                    time.sleep(0.001)
                finally:
                    lines = []
                    saw_newline = False

    # Finished writing all games from the input file.
    # Flag the end & flush all buffered data to pipe.
    for _ in range(NUM_CORES - 1):
        game_queue.put("DONE")
    game_queue.close()
    logger.info(f"RP {os.getpid()} finished! Processed {games_processed} games.")


@timer(logger)
def sequence_game(output_filepath: str, write_lock: Lock, game_queue: Queue) -> None:
    num_games: int = 0
    total_elapsed: float = 0
    game_string: str = ""
    while game_string != "DONE":
        game_string = game_queue.get()
        try:
            game: pgn.GameNode = pgn.read_game(io.StringIO(game_string))
        except ValueError:
            logger.debug(
                f"Game didn't load correctly. Check the string:\n{game_string}"
            )
            continue

        if "Termination" in game.headers and game.headers["Termination"] == "Abandoned":
            continue

        elo: str = game.headers["WhiteElo"] if "WhiteElo" in game.headers else None
        result: str = game.headers["Result"] if "Result" in game.headers else None
        start = time.time()

        # Parse the GameNode object into moves, evals and boards.
        evals, boards = [], []
        board: pgn.Board = game.board()
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
                logger.info(
                    f"[WP {os.getpid()}] Put game number {num_games}."
                    f" {game_queue.qsize()} left in the queue."
                    #    f" {count_python_processes()} processes are running."
                )
        finally:
            write_lock.release()
        total_elapsed += time.time() - start

    logger.info(
        f"[WP {os.getpid()}] Processed {num_games} games, taking "
        f"{total_elapsed / num_games:.4f}s on average."
    )


def setup():
    logger.info(f"{NUM_CORES=}")
    mp.set_start_method("fork")

    args = parse_args()

    input_filepath = args["input_filepath"]
    if input_filepath is None:
        raise ValueError("Must provide a valid input file!")
    input_filename = extract_filename(input_filepath)
    output_filepath = f"./dtchess/data/sequences_{input_filename}.txt"

    # Configure logger.
    logfile = f"./dtchess/logs/sequences_{input_filename}.log"
    logger.add(sys.stderr, format="{time} {message}", enqueue=True, level="DEBUG")
    logger.add(
        logfile,
        format="{time} {message}",
        enqueue=True,
        rotation=MAX_LOG_SIZE,
        retention=1,
        level="INFO",
    )
    logger.info(f"Reading from {input_filepath}.\n Writing to: {output_filepath}")

    # Spawn processes: one to read games from a PGN file
    # and all the others to convert them to string sequences.
    write_lock = mp.Lock()
    game_queue = Queue()
    reader_process = mp.Process(target=read_games, args=(input_filepath, game_queue))
    sequencing_processes = [
        mp.Process(target=sequence_game, args=(output_filepath, write_lock, game_queue))
        for _ in range(NUM_CORES - 1)
    ]
    return reader_process, sequencing_processes, game_queue


if __name__ == "__main__":
    reader_process, sequencing_processes, game_queue = setup()
    start = time.time()
    # Start all processes.
    reader_process.start()
    for process in sequencing_processes:
        process.start()

    reader_process.join()
    for process in sequencing_processes:
        process.join()

    logger.info(
        f"Finished! Took {time.time() - start}s. Queue size at finish: {game_queue.qsize()}, .full= {game_queue.full()}, .empty={game_queue.empty()}"
    )

import time
import io
import os
import multiprocessing as mp
import chess.pgn as pgn

from multiprocessing import Queue
from threading import Lock
from utils import parse_args

output_filepath = "./dtchess/data/sequences"
NUM_CORES = mp.cpu_count()


def read_games(input_filepath: str, game_queue: Queue, written: int) -> None:
    # Fetch one game for each process but this one,
    # and put it to the shared queue.
    start = time.time()
    with open(input_filepath, "r", encoding="utf-8") as pgnfile:
        # There are always newlines between games; but games also contain
        # a newline between the headers and the moves.
        newline_ctr = 0
        lines = []
        for line in pgnfile:
            if newline_ctr < 2:
                lines += [line]
                newline_ctr += 1 if line == "\n" else 0
            else:
                game = "".join(lines)

                # Make sure there is room on the queue before putting.
                while game_queue.full():
                    time.sleep(0.001)
                game_queue.put(game)
                written += 1
                lines = []
                newline_ctr = 0

    print(
        f"RP {os.getpid()} took {time.time() - start:.4f}s to process {written} games."
    )


def sequence_game(output_filepath: str, write_lock: Lock, game_queue: Queue) -> None:
    num_games = 0
    total_elapsed: int = 0  # Total time taken to process games.
    while game_queue.qsize() > 0:
        # print(f"WP {os.getpid()} getting game; ~{game_queue.qsize()} left in queue.")
        # This is a string representing a game.
        game_string = game_queue.get()
        try:
            game = pgn.read_game(io.StringIO(game_string))
        except ValueError:
            print(f"Game didn't load correctly. Check the string:\n{game_string}")
            continue

        if game.headers["Termination"] == "Abandoned":
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
            with open(f"{output_filepath}.txt", "a+") as f:
                f.write(f"{header} {body}\n")
                num_games += 1
                # print(f"Worker proc: put game {game}")
        finally:
            pass
            write_lock.release()
        total_elapsed += time.time() - start

    print(
        f"WP {os.getpid()} processed {num_games} games, taking "
        f"{total_elapsed / num_games:.4f}s on average."
    )


if __name__ == "__main__":
    print(f"{NUM_CORES=}")
    mp.set_start_method("fork")

    args = parse_args()

    input_filepath = args["input_filepath"]
    output_filepath = f"./dtchess/data/sequences_{input_filepath.split('/')[-1][:-4]}"
    logfile = f"{output_filepath.split('/')[-1][:-4]}_log.txt"
    print(f"Reading games from {input_filepath}")
    print(f"Writing sequences to: {output_filepath}")
    written, errs = 0, 0

    # Spawn processes to read games from a PGN file and
    # convert them to string sequences.
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

    with open(logfile, "a+", encoding="utf-8") as f:
        f.write(f"Finished! Took {time.time() - start}s.")

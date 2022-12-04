import chess.pgn as pgn
import multiprocessing as mp

output_filepath = "./dtchess/data/sequences"


def process_file(file, read_lock, write_lock):
    # This needs to be able to read the file; it shouldn't
    # lock read access, though. Once the game data is in this object,
    # all further processing doesn't touch the file.

    # TODO: It doesn't look like this is working correctly by reading one game and then releasing.
    read_lock.acquire()
    try:
        game = pgn.read_game(file)
    finally:
        read_lock.release()

    while game:
        elo = game.headers["WhiteElo"] if "WhiteElo" in game.headers else None
        result = game.headers["Result"] if "Result" in game.headers else None

        # Parse the GameNode object into moves, evals and boards.
        evals, boards = [], []
        board = game.board()
        while game.next():
            # We don't remember the initial board state, it's always the same.
            move = game.variation(0).move
            board.push(move)
            boards.append(board.fen())

            # Not all games have evals; if they do, record them.
            if game.eval() and hasattr(game.eval().relative, "cp"):
                evals.append(game.eval().relative.cp)

            # Follow the game tree until the end node.
            game = game.next()

        # Use the moves, evals and boards to generate a sequence.
        if elo is not None and result is not None:
            header = f"<ELO>{elo}</ELO> <RES>{result}</RES>"
        else:
            header = ""
        if len(evals) > 0:  # i.e. if there are evals at all, use them.
            white_total_loss = sum(evals[::2])
            header = f"{header} <RET>{white_total_loss}</RET>"
            body = '||'.join([f"{board}::{ev}" for (board, ev) in zip(boards, evals)])
        else:  # otherwise, just use the board states.
            body = f"{'||'.join(boards)}"

        # Append sequence to file.
        write_lock.acquire()
        try:
            with open(f"{output_filepath}_{file.name[2:-4]}.txt", "a+") as f:
                f.write(f"{header} {body}\n")
        finally:
            write_lock.release()

        # Move onto the next game.
        read_lock.acquire()
        try:
            game = pgn.read_game(file)
        finally:
            read_lock.release()


if __name__ == "__main__":
    mp.set_start_method("fork")
    filepath = "./antichess1.pgn"
    file = open(filepath, "r")
    # filepaths = ["../data/standard_10.pgn"]

    read_lock, write_lock = mp.Lock(), mp.Lock()
    processes = [mp.Process(target=process_file, args=(file, read_lock, write_lock)) for _ in range(mp.cpu_count())]
    for process in processes:
        process.start()

    # This needs to happen in a separate loop, otherwise the second process
    # only starts when the first one is done with its logic.
    for process in processes:
        process.join()

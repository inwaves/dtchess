import chess.pgn as pgn
import multiprocessing as mp

output_filepath = "./dtchess/data/sequences"

output_file = open(output_filepath, "a+")


def process_file(filepath):
    file = open(filepath, "r")
    game = pgn.read_game(file)

    while game:
        elo = game.headers["WhiteElo"]
        result = game.headers["Result"]

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
        header = f"<ELO>{elo}</ELO> <RES>{result}</RES>"
        if len(evals) > 0:  # i.e. if there are evals at all, use them.
            white_total_loss = sum(evals[::2])
            header = f"{header} <RET>{white_total_loss}</RET>"
            boards_evals_sequence = ' '.join([f"{board} {ev}" for (board, ev) in zip(boards, evals)])
            body = f"{' '.join(boards_evals_sequence)}"
        else:   # otherwise, just use the board states.
            body = f"{' '.join(boards)}"

        # Append sequence to file.
        with open(f"{output_filepath}_{filepath[2:-4]}.txt", "a+") as f:
            f.write(f"{header}||{body}")

        # Move onto the next game.
        game = pgn.read_game(file)


if __name__ == "__main__":
    mp.set_start_method("fork")
    filepaths = ["./antichess1.pgn", "./antichess2.pgn"]
    processes = [mp.Process(target=process_file, args=(filepath,)) for filepath in filepaths]
    for process in processes:
        process.start()

    # This needs to happen in a separate loop, otherwise the second process
    # only starts when the first one is done with its logic.
    for process in processes:
        process.join()

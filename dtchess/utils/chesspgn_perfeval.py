import chess.pgn as pgn
import time
import multiprocessing as mp

output_filepath = "./dtchess/data/sequences.txt"

output_file = open(output_filepath, "a+")


def process_file(file):
    game = pgn.read_game(file)
    ct, total_elapsed = 0, 0.0
    while game and ct < 1_000:
        start = time.time()
        boards, moves = [], []
        mainline_moves = game.mainline_moves()
        board = game.board()
        for move in mainline_moves:
            moves.append(move.uci())
            # board.push(move)
            # boards.append(board.fen())
        elapsed = time.time() - start
        ct += 1
        total_elapsed += elapsed
        print(f"Processed game {ct} in {elapsed}s")
        game = pgn.read_game(file)

    # print(boards)
    # print(moves)
    print(f"On average processing took: {total_elapsed / ct}")
    print(f"In total, processing {ct} games took {total_elapsed}")


if __name__ == "__main__":
    mp.set_start_method("fork")
    input_filepath = "./antichess1.pgn"
    input_filepath2 = "./antichess2.pgn"
    f1 = open(input_filepath, "r")
    f2 = open(input_filepath2, "r")
    p1 = mp.Process(target=process_file, args=(f1,))
    p2 = mp.Process(target=process_file, args=(f2,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    
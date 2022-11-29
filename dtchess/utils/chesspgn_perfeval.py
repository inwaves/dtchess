import chess.pgn as pgn
import time

input_filepath = "./lichess_db_antichess_rated_2022-10.pgn"
output_filepath = "./dtchess/data/sequences.txt"

input_file = open(input_filepath, "r")
output_file = open(output_filepath, "a+")

game = pgn.read_game(input_file)
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
    game = pgn.read_game(input_file)

# print(boards)
# print(moves)
print(f"On average processing took: {total_elapsed/ct}")
print(f"In total, processing {ct} games took {total_elapsed}")

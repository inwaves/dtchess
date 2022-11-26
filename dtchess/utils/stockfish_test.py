import os
import chess
import chess.engine as engine
import chess.pgn as pgn

stockfish = engine.SimpleEngine.popen_uci(r"/usr/local/bin/stockfish")
print(os.getcwd())
pgnfile = open("./data/sample.pgn")
game = pgn.read_game(pgnfile)

board = game.board()

info = stockfish.analyse(board, engine.Limit(depth=20))
print(f"Information: {info}")

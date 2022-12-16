import chess
from dtchess.utils.utils import extract_filename, board_to_sequence


def test_board_to_sequence() -> None:
    board = chess.Board()
    expected = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    assert board_to_sequence(board) == expected


def test_extract_filename() -> None:
    fpath = "./data/antichess1.pgn"
    assert extract_filename(fpath) == "antichess1"

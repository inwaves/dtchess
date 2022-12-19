import chess
import pytest
from dtchess.utils.utils import extract_filename, board_to_sequence, extract_tag

@pytest.mark.parametrize("input_string, tag_name, expected", [("<RET>1209</RET>", "<RET>", "1209"), ("<ELO>2100</ELO>", "ELO", "2100"), pytest.param("<ELO>1857</ELO>", "FOO", "1857", marks=pytest.mark.xfail)])
def test_extract_tag(input_string: str, tag_name: str, expected: str) -> None:
    assert extract_tag(input_string, tag_name) == expected

def test_board_to_sequence() -> None:
    board = chess.Board()
    expected = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    assert board_to_sequence(board) == expected


def test_extract_filename() -> None:
    fpath = "./data/antichess1.pgn"
    assert extract_filename(fpath) == "antichess1"

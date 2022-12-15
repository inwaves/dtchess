from dtchess.utils.utils import extract_filename


def test_extract_filename() -> None:
    fpath = "./data/antichess1.pgn"
    assert extract_filename(fpath) == "antichess1"

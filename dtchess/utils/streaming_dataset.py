from itertools import chain

from torch.utils.data import IterableDataset
import chess.pgn as pgn


class StreamingDataset(IterableDataset):
    def __init__(self, dirpath):
        super().__init__()
        self.dirpath = dirpath

    def parse_directory(self):
        pgnfiles = (open(file, "r") for file in self.dirpath)
        yield from pgnfiles

    def parse_pgn(self, pgnfile):
        yield from pgn.read_game(pgnfile)

    def game_to_sequence(self, game):
        sequence = None
        # process the game into a sequence as in utils.
        yield from sequence

    def __iter__(
        self,
    ):
        return chain(self.game_to_sequence(self.parse_pgn(self.parse_directory())))

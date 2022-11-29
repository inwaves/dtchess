from torch.utils.data import IterableDataset


class StreamingDataset(IterableDataset):

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def parse_file(self, file_path):
        with open(file_path, "r") as game_file:
            game = pgn.read_game(game_file)
            while game is not None:
                # TODO: I don't think I want args here.
                token_sequence = process_game(game, args["sequence_type"])
                yield from token_sequence

    def __iter__(self, ):
        return self.parse_file(self.file_path)

import random
import chess    # type: ignore
from typing import List, Tuple
from utils import board_to_sequence, timer
from loguru import logger


@timer(logger)
def one_game() -> Tuple[str, int]:
    board_sequences: List[str] = []
    board = chess.Board()
    round_ct: int = 0
    while board.outcome() is None:
        one_round(board)
        board_sequences += [board_to_sequence(board)]
        round_ct += 1

    game_sequence: str = "||".join(board_sequences)
    return game_sequence, round_ct


def one_round(board: chess.Board) -> None:
    if board.outcome() is not None:
        return
    lm = list(board.legal_moves)
    board.push(random.choice(lm))


if __name__ == "__main__":
    logger.add("./logs/random_moves.log", format="{time} {message}", enqueue=True)
    num_games: int = 5
    games = [one_game() for _ in range(num_games)]
    print(games[1])

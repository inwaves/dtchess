# DTchess

This library fine-tunes language models to play chess. The end goal is to make these models available for mechanistic interpretability, such that their internal computations can be reverse engineered. Ultimately, we'd like to understand what it looks like for language models to implement a particular functionality. For more information, see [Auditing games for high-level interpretability](https://www.lesswrong.com/posts/EbL5W5ccwfbqFiYBJ/auditing-games-for-high-level-interpretability-1).

Chess is a useful example because it's likely that the model applies some form of internal optimisation to play, perhaps something like an internal search over board states. There are other games and environments that have similar properties, but chess seems particularly tractable because there is lots of public data, and it is fairly easy to translate it into sequences of tokens.


# Datasets
Two datasets have been generated using this library, starting from the January 2021 games on [lichess](https://database.lichess.org/#standard_games). [dtchess-standard](https://huggingface.co/datasets/inwaves/dtchess-standard) was generated directly from the January PGN file. It contains ~100 million chess games, after filtering for normal termination conditions. [dtchess-random](https://huggingface.co/datasets/inwaves/dtchess-random) contains games with completely random moves, generated using the `python-chess` library. Moves are sampled uniformly at random from a list of legal moves at a given board state. There are currently 4.5 million such games. 

In the `utils` package you'll find utilities to process PGN files into token sequences that look like this:

    <ELO>1365</ELO> <RET>1640</RET> <RES>0-1</RES>rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR||rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR||rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR||rnbqkbnr/pp1ppppp/2p5/8/4P3/5N2/PPPP1PPP/RNBQKB1R||rnbqkbnr/pp2pppp/2p5/3p4/4P3/5N2/PPPP1PPP/RNBQKB1R||rnbqkbnr/pp2pppp/2p5/3P4/8/5N2/PPPP1PPP/RNBQKB1R||rnbqkbnr/pp2pppp/8/3p4/8/5N2/PPPP1PPP/RNBQKB1R||rnbqkbnr/pp2pppp/8/3p4/8/2N2N2/PPPP1PPP/R1BQKB1R||rn1qkbnr/pp2pppp/8/3p4/6b1/2N2N2/PPPP1PPP/R1BQKB1R||rn1qkbnr/pp2pppp/8/1B1p4/6b1/2N2N2/PPPP1PPP/R1BQK2R||r2qkbnr/pp2pppp/2n5/1B1p4/6b1/2N2N2/PPPP1PPP/R1BQK2R||r2qkbnr/pp2pppp/2n5/1B1p4/3P2b1/2N2N2/PPP2PPP/R1BQK2R||r2qkbnr/1p2pppp/p1n5/1B1p4/3P2b1/2N2N2/PPP2PPP/R1BQK2R||r2qkbnr/1p2pppp/p1B5/3p4/3P2b1/2N2N2/PPP2PPP/R1BQK2R||r2qkbnr/4pppp/p1p5/3p4/3P2b1/2N2N2/PPP2PPP/R1BQK2R||r2qkbnr/4pppp/p1p5/3p4/N2P2b1/5N2/PPP2PPP/R1BQK2R||r3kbnr/4pppp/p1p5/q2p4/N2P2b1/5N2/PPP2PPP/R1BQK2R||r3kbnr/4pppp/p1p5/q2p4/3P2b1/2N2N2/PPP2PPP/R1BQK2R||r3kbnr/5ppp/p1p1p3/q2p4/3P2b1/2N2N2/PPP2PPP/R1BQK2R||r3kbnr/5ppp/p1p1p3/q2p4/3P2b1/2NQ1N2/PPP2PPP/R1B1K2R||r3kb1r/5ppp/p1p1pn2/q2p4/3P2b1/2NQ1N2/PPP2PPP/R1B1K2R||r3kb1r/5ppp/p1p1pn2/q2pN3/3P2b1/2NQ4/PPP2PPP/R1B1K2R||r3kb1r/5ppp/p1p1pn2/q2pNb2/3P4/2NQ4/PPP2PPP/R1B1K2R||r3kb1r/5ppp/p1p1pn2/q2pNb2/3P4/2N3Q1/PPP2PPP/R1B1K2R||r3kb1r/5ppp/p1p1p3/q2pNb2/3Pn3/2N3Q1/PPP2PPP/R1B1K2R||r3kb1r/5ppp/p1p1p3/q2pNb2/3PnQ2/2N5/PPP2PPP/R1B1K2R||r3kb1r/5ppp/p1p1p3/q2pNb2/3P1Q2/2n5/PPP2PPP/R1B1K2R||r3kb1r/5ppp/p1p1p3/q2pNb2/3P1Q2/2n5/PPPB1PPP/R3K2R||r3k2r/5ppp/p1p1p3/q2pNb2/1b1P1Q2/2n5/PPPB1PPP/R3K2R||r3k2r/5ppp/p1N1p3/q2p1b2/1b1P1Q2/2n5/PPPB1PPP/R3K2R||r3k2r/5ppp/p1N1p3/1q1p1b2/1b1P1Q2/2n5/PPPB1PPP/R3K2R||r3k2r/5ppp/p3p3/1q1p1b2/1N1P1Q2/2n5/PPPB1PPP/R3K2R

These comprise a short header which contains:

- the ELO score of the White player 
- the accumulated return of the White player. This is derived from a set of evaluations provided by [Stockfish](https://stockfishchess.org/) for each more. Stockfish gives you a [centipawn loss](https://lichess.org/faq#acpl) for each evaluated move, which expresses how much worse that move is compared to the best move it could come up with given a particular search depth. Not all games have evals, so it is not uncommon to see game sequences without the <RET> tag.
- the result: in the example above, the result is a Black win. Ties are also possible, and are denoted with 1/2-1/2.

The bulk of the string is a sequence of game states, expressed as a slightly modified [FEN (Forsyth-Edwards Notation)](https://www.chess.com/terms/fen-chess) of the board. They contain just the description of the board squares -- 8 rows divided by a slash. Capitalised letters are White pieces, while lowercase ones are Black pieces. Board states are divided by a double pipe symbol (||). For example, the board state: `r3kb1r/5ppp/p1p1pn2/q2pN3/3P2b1/2NQ4/PPP2PPP/R1B1K2R` says that:
- on the first row, there is a rook, three spaces, the black king, a bishop, a space and a rook; 
- on the second row, there are 5 blank spaces, followed by 3 black pawns;
- on the third row, there is a pawn, a space, another pawn and space, then a pawn, knight and two spaces;
- on the fourth row, there is a black queen, two spaces, a pawn, a white kinght and three spaces;
- etc.

It is possible to visualise these FEN states using chess.com's [analysis](https://www.chess.com/analysis) module.

# Models
TODO

from dtchess.utils.utils import parse_args, setup

MAIN = __name__ == "__main__"

if MAIN:
    args = parse_args()
    tokeniser, model, optimiser, dataloaders = setup(args)

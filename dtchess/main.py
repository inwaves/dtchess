from typing import Tuple

import torch as t
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Model, GPT2Tokenizer
from dtchess.models.decision_transformer import create_model
from dtchess.utils.utils import parse_args, training_setup

MAIN = __name__ == "__main__"
device = "cuda" if t.cuda.is_available() else "cpu"


# TODO: add wandb tracking


def train(
    tokeniser: GPT2Tokenizer,
    model: GPT2Model,
    optimiser: optim.Adam,
    dataloaders: Tuple[DataLoader, DataLoader],
    loss_fn: nn.CrossEntropyLoss,
    args: dict,
):
    train_dl, test_dl = dataloaders

    # Writing a generic training loop for now, update later.
    for _ in range(args["num_epochs"]):
        for X, y in enumerate(tqdm(train_dl)):
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = loss_fn(preds, y)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    return model


if MAIN:
    args = parse_args()
    # tokeniser, model, optimiser, dataloaders, loss_fn = training_setup(args)
    # trained_model = train(tokeniser, model, optimiser, dataloaders, loss_fn, args)
    tokeniser, model = create_model()

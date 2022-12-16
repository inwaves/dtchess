from typing import Tuple

import torch as t
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Model, GPT2Tokenizer
from dtchess.models.gpt import create_model
from dtchess.utils.utils import training_setup
from dtchess.utils.config import config_from_file, TrainingConfig, ModelConfig

MAIN = __name__ == "__main__"
device = "cuda" if t.cuda.is_available() else "cpu"


# TODO: add wandb tracking


def train(
    tokeniser: GPT2Tokenizer,
    model: GPT2Model,
    optimiser: optim.Adam,
    dataloaders: Tuple[DataLoader, DataLoader],
    loss_fn: nn.CrossEntropyLoss,
    config: TrainingConfig,
):
    train_dl, test_dl = dataloaders

    # Writing a generic training loop for now, update later.
    for _ in range(config.num_epochs):
        for (input_ids, _) in enumerate(tqdm(train_dl)):
            input_ids = input_ids.to(device)

            # TODO: implement causal masking
            model_inputs = None
            preds = model(model_inputs)
            loss = loss_fn(preds, input_ids)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    return model


if MAIN:
    train_config: TrainingConfig = config_from_file("./dtchess/config.yaml")
    model_config: ModelConfig = config_from_file("./dtchess/model_config.yaml")
    tokeniser, model, optimiser, dataloaders, loss_fn = training_setup(train_config)
    # trained_model = train(tokeniser, model, optimiser, dataloaders, loss_fn, train_config)
    tokeniser, model = create_model(model_config)

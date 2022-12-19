import torch as t
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Model, GPT2Tokenizer
from dtchess.utils.utils import training_setup
from dtchess.utils.config import generate_config, TrainingConfig

MAIN = __name__ == "__main__"
device = "cuda" if t.cuda.is_available() else "cpu"


# TODO: add wandb tracking


def train(
    tokeniser: GPT2Tokenizer,
    model: GPT2Model,
    optimiser: optim.Adam,
    train_dataloader: DataLoader,
    loss_fn: nn.CrossEntropyLoss,
    config: TrainingConfig,
):
    # Writing a generic training loop for now, update later.
    for _ in range(config.num_epochs):
        for (input_ids, _) in enumerate(tqdm(train_dataloader)):
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
    train_config: TrainingConfig = generate_config("./dtchess/config.yaml")
    tokeniser, model, optimiser, train_dataloader, loss_fn = training_setup(
        train_config
    )
    print(tokeniser, model)
    # trained_model = train(tokeniser, model, optimiser, train_dataloaders, loss_fn, train_config)

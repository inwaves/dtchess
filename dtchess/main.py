import dataclasses
import torch as t
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Model, GPT2Tokenizer
from dtchess.utils.utils import training_setup
from dtchess.utils.config import generate_config, TrainingConfig
import wandb


MAIN = __name__ == "__main__"
device = "cuda" if t.cuda.is_available() else "cpu"


def train(
    tokeniser: GPT2Tokenizer,
    model: GPT2Model,
    optimiser: optim.Adam,
    train_dataloader: DataLoader,
    loss_fn: nn.CrossEntropyLoss,
    config: TrainingConfig,
):
    wandb.init(project="dtchess", config=dataclasses.asdict(config))
    wandb.watch(model, log_freq=config.log_every_n)
    print(config.batch_size)

    model.train()
    for current_epoch in range(config.num_epochs):
        for (_i, tokenised_sequences) in enumerate(tqdm(train_dataloader)):
            input_ids = tokenised_sequences["input_ids"].squeeze().to(device)

            # TODO: implement causal masking
            model_inputs = input_ids

            preds = model(model_inputs, labels=model_inputs)
            loss = preds.loss

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            del preds
            t.cuda.empty_cache()

            if current_epoch % config.log_every_n == 0:
                wandb.log({"loss": loss})

        if current_epoch % config.checkpoint_every_n == 0:
            model_path = f"./models/gpt2-{wandb.run.id}.pt"
            t.save(model_path)
            model_artifact = wandb.Artifact(f"gpt2-{wandb.run.id}", type="model")
            model_artifact.add_file(model_path, f"gpt2-{wandb.run.id}.pt")
            wandb.log_artifact(model_artifact)
    return model


if MAIN:
    train_config: TrainingConfig = generate_config("./dtchess/config.yaml")
    tokeniser, model, optimiser, train_dataloader, loss_fn = training_setup(
        train_config
    )
    print(train_config)
    trained_model = train(
        tokeniser, model, optimiser, train_dataloader, loss_fn, train_config
    )

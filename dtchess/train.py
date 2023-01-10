import dataclasses
import torch as t
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Literal
from dtchess.utils.utils import training_setup, dist_setup, dist_cleanup
from dtchess.utils.config import generate_config, TrainingConfig
from dtchess.utils.model import load_pretrained_model
import wandb  # type: ignore
from wandb import Artifact


MAIN = __name__ == "__main__"
device: Literal["cuda", "cpu"] = "cuda" if t.cuda.is_available() else "cpu"


def train(
    rank: int,
    world_size: int,
    tokeniser: AutoTokenizer,
    # train_dataloader: DataLoader,
    config: TrainingConfig,
) -> AutoModelForCausalLM:

    print(f"{rank=}, {world_size=}")

    dist_setup(rank, world_size)
    train_dataloader = None

    # Instantiate a replica of the model.
    model: AutoModelForCausalLM = load_pretrained_model(tokeniser=tokeniser).to(rank)
    ddp_model: AutoModelForCausalLM = DDP(model, device_ids=[rank])
    ddp_model.train()

    optimiser = optim.Adam(
        ddp_model.parameters(), lr=config.learning_rate, betas=config.betas
    )

    # Start watching the training process.
    wandb.init(project="dtchess", config=dataclasses.asdict(config))
    wandb.watch(ddp_model, log_freq=config.log_every_n)

    for _ in range(config.num_epochs):  # Probably only one epoch.
        for (i, tokenised_sequences) in enumerate(tqdm(train_dataloader)):
            input_ids = tokenised_sequences["input_ids"].squeeze().to(device)

            preds = ddp_model(input_ids, labels=input_ids)
            loss = preds.loss

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if i > 0 and i % config.checkpoint_every_n == 0:
                model_path: str = f"{config.ckpt_path}/gpt2-{wandb.run.id}.pt"
                t.save(ddp_model, model_path)
                model_artifact: Artifact = Artifact(
                    f"gpt2-{wandb.run.id}", type="model"
                )
                model_artifact.add_file(model_path)
                wandb.log_artifact(model_artifact)

    dist_cleanup()

    return ddp_model


if MAIN:
    mp.set_start_method("spawn")

    train_config: TrainingConfig = generate_config("./dtchess/config.yaml")
    tokeniser, train_dataloader = training_setup(train_config)
    world_size = train_config.num_shards

    mp.spawn(
        train,
        args=(world_size, tokeniser, train_config),
        nprocs=1,  # Running this on a single node with multiple GPUs.
        join=True,
    )

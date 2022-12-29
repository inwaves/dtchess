import torch as t
import datasets
from torch.utils.data import DataLoader
from dtchess.models.gpt import create_model

device = "cuda" if t.cuda.is_available() else "cpu"


def prep():
    tokeniser, model = create_model("gpt2")
    dataset = datasets.load_dataset(
        "inwaves/dtchess-standard", streaming=True, split="train"
    )

    input_ids = dataset.map(
        lambda seq: tokeniser(
            seq["text"],
            padding="max_length",
            max_length=model.wpe.num_embeddings,
            truncation=True,
            return_tensors="pt",
        ),
        batched=True,
    )
    train_dl = DataLoader(input_ids, batch_size=32)

    return train_dl, tokeniser, model

import torch as t  # type: ignore
import torch.nn as nn  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
import wandb    # type: ignore


def load_pretrained_model(
    model_type: str = "gpt2",
) -> AutoModelForCausalLM:
    """Loads a pre-trained transformer and the corresponding tokeniser from huggingface."""

    device = "cuda" if t.cuda.is_available() else "cpu"
    tokeniser = AutoTokenizer.from_pretrained(model_type)
    model = AutoModelForCausalLM.from_pretrained(model_type)

    # Adding a padding token and updating the vocab size.
    tokeniser.add_special_tokens({"pad_token": "[PAD]"})
    model.transformer.wte = nn.Embedding(
        tokeniser.vocab_size + 1, model.transformer.wte.embedding_dim
    )
    model.lm_head = nn.Linear(
        model.lm_head.in_features, tokeniser.vocab_size + 1, bias=False
    )
    model = model.to(device)

    return model

def load_pretrained_tokeniser(
    tokeniser_type: str = "gpt2",
) -> AutoTokenizer:



def load_model_checkpoint(model_name: str) -> AutoModelForCausalLM:
    """Loads a model checkpoint from wandb."""

    wandb.init(project="dtchess")
    artifact = wandb.use_artifact(f"{model_name}:latest")
    artifact = artifact.download()

    device = "cuda" if t.cuda.is_available() else "cpu"
    model = t.load(f"{artifact}/{model_name}.pt", map_location=device)

    return model

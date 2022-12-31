import torch as t  # type: ignore
import torch.nn as nn  # type: ignore
from transformers import AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel  # type: ignore
import wandb


def create_model(
    model_type: str = "gpt2",
) -> tuple[GPT2LMHeadModel, GPT2Tokenizer]:

    # gpt2 is 124m parameters, gpt2-medium is 355m.
    device = "cuda" if t.cuda.is_available() else "cpu"
    tokeniser = GPT2Tokenizer.from_pretrained(model_type)
    tokeniser.add_special_tokens({"pad_token": "[PAD]"})
    # model = GPT2LMHeadModel.from_pretrained(model_type)
    model = AutoModelForCausalLM.from_pretrained(model_type)

    # Modifying token embedding since we added a new token type...
    model.transformer.wte = nn.Embedding(
        tokeniser.vocab_size + 1, model.transformer.wte.embedding_dim
    )
    model.lm_head = nn.Linear(
        model.lm_head.in_features, tokeniser.vocab_size + 1, bias=False
    )
    model = model.to(device)

    return model, tokeniser


def load_model(model_name: str) -> GPT2LMHeadModel:
    wandb.init(project="dtchess")
    artifact = wandb.use_artifact(f"{model_name}:latest")
    artifact = artifact.download()

    device = "cuda" if t.cuda.is_available() else "cpu"
    model = t.load(f"{artifact}/{model_name}.pt", map_location=device)

    return model

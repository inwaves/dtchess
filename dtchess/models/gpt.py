import torch as t  # type: ignore
import torch.nn as nn  # type: ignore
from transformers import GPT2Tokenizer, GPT2Model  # type: ignore


def create_model(model_type: str = "gpt2-medium") -> tuple[GPT2Model, GPT2Tokenizer]:
    # gpt2 is 1.5B parameters, gpt2-medium is 355m.
    device = "cuda" if t.cuda.is_available() else "cpu"
    tokeniser = GPT2Tokenizer.from_pretrained(model_type)
    tokeniser.add_special_tokens({"pad_token": "[PAD]"})
    model = GPT2Model.from_pretrained(model_type)

    # Modifying token embedding since we added a new token type...
    model.wte = nn.Embedding(tokeniser.vocab_size + 1, tokeniser.model_max_length)
    model = model.to(device)

    return tokeniser, model

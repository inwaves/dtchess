import torch as t  # type: ignore
from transformers import GPT2Tokenizer, GPT2Model  # type: ignore


def create_model(model_type: str = "gpt2-medium") -> tuple[GPT2Model, GPT2Tokenizer]:
    # gpt2 is 1.5B parameters, gpt2-medium is 355m.
    device = "cuda" if t.cuda.is_available() else "cpu"
    tokeniser = GPT2Tokenizer.from_pretrained(model_type)
    model = GPT2Model.from_pretrained(model_type)
    model = model.to(device)

    return tokeniser, model

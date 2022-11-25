from dtchess.utils.utils import count_parameters
from transformers import GPT2Tokenizer, GPT2Model
import torch as t


def create_model():
    # gpt2 is 1.5B parameters, gpt2-medium is 355m.
    device = "cuda" if t.cuda.is_available() else "cpu"
    tokeniser = GPT2Tokenizer.from_pretrained("gpt2-medium")
    model = GPT2Model.from_pretrained("gpt2-medium")

    model = model.to(device)
    print(model)

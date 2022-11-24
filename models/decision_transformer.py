from transformers import GPT2Tokenizer, GPT2Model
import torch as t


if __name__ == "__main__":
    # gpt2 is 1.5B parameters, gpt2-medium is 355m.
    device = "cuda" if t.cuda.is_available() else "cpu"
    tokeniser = GPT2Tokenizer.from_pretrained("gpt2-medium")
    model = GPT2Model.from_pretrained("gpt2-medium")

    # Prediction: this isn't going to fit on my device.
    model = model.to(device)

    modely
    print(model)

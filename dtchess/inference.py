import torch as t
from dtchess.utils.utils import parse_args, read_lines
from dtchess.utils.model import load_pretrained_model, load_model_checkpoint


if __name__ == "__main__":
    cl_args = parse_args()
    model = load_model_checkpoint(cl_args["model_name"])
    _, tokeniser = load_pretrained_model()
    tokeniser.padding_side = "left"

    prompts = read_lines(cl_args["prompt_file"])
    prompts = tokeniser(prompts, return_tensors="pt", padding=True, truncation=True)

    with t.inference_mode():
        generated_tokens = model.generate(
            **prompts, max_new_tokens=cl_args["generate_tokens"]
        )

    generated_text = tokeniser.batch_decode(generated_tokens, skip_special_tokens=True)

    print(generated_text)

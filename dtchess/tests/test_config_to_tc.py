from dtchess.utils.config import generate_config, TrainingConfig


def test_generate_config() -> None:
    yaml_file = "./dtchess/tests/test_training_config.yaml"
    tc = generate_config(yaml_file)

    assert tc == TrainingConfig(batch_size=128, betas=(0.9, 0.1)), tc

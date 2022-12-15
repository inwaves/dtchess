from dtchess.utils.config import config_from_file, TrainingConfig


def test_config_from_file() -> None:
    yaml_file = "./dtchess/tests/test_config.yaml"
    tc = config_from_file(yaml_file)

    assert tc == TrainingConfig(batch_size=128, betas=(0.9, 0.1)), tc

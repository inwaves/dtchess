from utils import extract_tag
from loguru import logger
import sys
import math


def extract_distributions(filepaths: list[str]) -> None:
    """Gathers the mean and variance of the ELO, return and result distributions.
    Takes in as argument a list of filepaths, each file containing sequences of games.
    """
    elo_total, return_total = 0, 0
    elo_count, return_count = 0, 0
    failures = {"elo": 0, "return": 0, "result": 0}
    result_bincount = {"white_win": 0, "black_win": 0, "tie": 0}
    logger.info("Calculating average of ELOs and returns, counting results...")
    for filepath in filepaths:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                # If present, extract ELO.
                try:
                    elo_string = extract_tag(line, "ELO")
                    elo_total += int(elo_string)
                    elo_count += 1
                except ValueError:
                    failures["elo"] += 1

                # If present, extract returns.
                try:
                    return_string = extract_tag(line, "RET")
                    return_total += int(return_string)
                    return_count += 1
                except ValueError:
                    failures["return"] += 1

                # If present, count the result outcomes.
                try:
                    result_string = extract_tag(line, "RES")
                    if result_string == "1/2-1/2":
                        result_bincount["tie"] += 1
                    elif result_string == "1-0":
                        result_bincount["white_win"] += 1
                    else:
                        result_bincount["black_win"] += 1
                except ValueError:
                    failures["result"] += 1
        logger.info(f"Finished processing average for file {filepath}")

    elo_mean = elo_total // elo_count
    return_mean = return_total // return_count
    logger.info(f"{failures=}")
    failures = {"elo": 0, "return": 0}  # Reset to count deviation errors.
    logger.info("Done with average. Calculating deviations...")

    sum_elo_deviations, sum_return_deviations = 0, 0
    for filepath in filepaths:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                # Calculate ELO deviations.
                try:
                    elo_string = extract_tag(line, "ELO")
                    sum_elo_deviations += int(elo_string) - elo_mean
                except ValueError:
                    failures["elo"] += 1

                # Calculate return deviations.
                try:
                    return_string = extract_tag(line, "RET")
                    sum_return_deviations += int(return_string) - return_mean
                except ValueError:
                    failures["return"] += 1
    elo_variance = math.sqrt(sum_elo_deviations / elo_count)
    return_variance = math.sqrt(sum_return_deviations / return_count)

    logger.info("Done with deviations for file {filepath}")
    logger.info(f"{elo_mean=}, variance: {elo_variance=}")
    logger.info(f"{return_mean=}, variance: {return_variance=}")
    logger.info(f"{result_bincount=}")


def setup():
    logfile = "./dtchess/logs/distribution_log.log"
    logger.add(sys.stderr, format="{time} {message}", level="DEBUG")
    logger.add(
        logfile,
        format="{time} {message}",
        enqueue=True,
        rotation="2 GB",
        retention=1,
        level="INFO",
    )


if __name__ == "__main__":
    filename = "sequences.txt"
    setup()
    logger.info("Setup done!")
    extract_distributions(filename)

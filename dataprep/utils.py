import json
import logging
import pathlib
import sys

from typing import Union


def load_json(path: Union[str, pathlib.Path]):
    """
    Loads JSON file

    :param path: path to desired file

    :returns: content of JSON file
    """
    with open(str(path), "r") as f:
        return json.load(f)


def save_json(obj, path: Union[str, pathlib.Path]) -> None:
    """
    Save obj as JSON filae

    :param obj: object to save

    :param path: path to save
    """
    with open(str(path), 'w') as f:
        json.dump(obj, f)


def configure_logger(logger: logging.Logger, level: int):
    """
    Configure logger

    :param logger: logger to configure
    :param level: logger info level

    :return: configured logger
    """
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", style="%"
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)

    return logger
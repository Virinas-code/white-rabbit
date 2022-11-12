#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

Training alogrithm config.
"""
import json
import os
from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {"iterations": 10}
WHITE_RABBIT_ETC: str = os.path.expanduser("~/.local/etc/white-rabbit")


def load_config() -> dict[str, Any]:
    """
    Loads config from JSON file.

    File is stored in ~/.local/etc/white-rabbit/train-config.json.

    :return dict[str, Any]: Config.
    """
    try:
        with open(
            WHITE_RABBIT_ETC + "/train-config.json", encoding="utf-8"
        ) as file:
            return json.load(file)
    except FileNotFoundError:
        os.mkdir(WHITE_RABBIT_ETC)
        with open(
            WHITE_RABBIT_ETC + "/train-config.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(DEFAULT_CONFIG, file)
        return DEFAULT_CONFIG


def prompt_config(self):
    """
    Configure training.

    Prompts settings.
    """
    self.config["iterations"] = self.rich.prompt("Iterations", int)

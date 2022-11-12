#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

Base training object.
"""
import json
import os
import sys

from typing import Any

from .rich_cli import RichCLI

DEFAULT_CONFIG: dict[str, Any] = {"iterations": 10}
WHITE_RABBIT_ETC: str = os.path.expanduser("~/.local/etc/white-rabbit")


class Trainer:
    """Base object for training."""

    def __init__(self):
        """
        Initialize object.

        TODO: Complete this
        """
        self.rich: RichCLI = RichCLI()
        """Base logger using :py:mod:`rich`."""
        self.config: dict[str, Any] = self.load_config()

    def load_config(self) -> dict[str, Any]:
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

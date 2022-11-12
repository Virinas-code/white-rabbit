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
                "~/.local/etc/white-rabbit/train-config.json", encoding="utf-8"
            ) as file:
                return json.load(file)
        except FileNotFoundError:
            os.mkdir("~/.local/etc/white-rabbit")
            with open(
                "~/.local/etc/white-rabbit/train-config.json",
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
        iterations: int = self.rich.prompt("Iterations", int)

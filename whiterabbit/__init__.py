#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

Main functions.
"""
import os

from .neural_network.training import Trainer


def train_start(config: bool = False) -> None:
    """
    Train target.

    :param bool config: Wether to reset config or not.
    """
    trainer: Trainer = Trainer()
    if config:
        trainer.prompt_conf()


def train_cleanup() -> None:
    """Cleanup train config."""
    os.remove("~/.local/etc/white-rabbit/train-config.json")

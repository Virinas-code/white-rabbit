#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

Main functions.
"""
import os

from .neural_network.training import Trainer


def train() -> None:
    """Train target."""
    trainer: Trainer = Trainer()
    print(trainer)


def train_cleanup() -> None:
    """Cleanup train config."""
    os.remove("~/.local/etc/white-rabbit/train-config.json")

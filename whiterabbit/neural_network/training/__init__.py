#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

Base training object.
"""
from typing import Any, Callable

from .config import load_config, prompt_config
from .rich_cli import RichCLI


class Trainer:
    """Base object for training."""

    def __init__(self):
        """
        Initialize object.

        TODO: Complete this
        """
        self.rich: RichCLI = RichCLI()
        """Base logger using :py:mod:`rich`."""
        self.config: dict[str, Any] = self.load_conf()

    load_conf: staticmethod = staticmethod(load_config)
    prompt_conf: Callable = prompt_config

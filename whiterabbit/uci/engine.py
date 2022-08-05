#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

UCI engine link.
"""
import chess

from .options import Option, SpinOption
from ..engine.evaluation import Evaluation


class Engine:
    """Engine wrapper for UCI."""

    def __init__(self):
        """
        Initialize engine wrapper.

        Starts shared variables management.
        """
        self.options: dict[str, Option] = {
            "Hash": SpinOption("Hash", 32, 4096, 1)
        }
        """Engine options."""
        self.transpositions: dict[int, Evaluation] = {}
        """Transpositions table."""
        self.position: chess.Board = chess.Board()
        """Current position."""

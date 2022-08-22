#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

UCI engine link.
"""
from multiprocessing import Process
from typing import Optional

import chess

from whiterabbit import engine

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
            "Hash": SpinOption("Hash", 32, 4096, 1),
            "MultiPV": SpinOption("MultiPV", 1, 500, 1),
        }
        """Engine options."""
        self.transpositions: dict[int, Evaluation] = {}
        """Transpositions table."""
        self.position: chess.Board = chess.Board()
        """Current position."""
        self.process: Optional[Process] = None
        """Current search process."""
        self.engine: engine.Engine = engine.Engine()

    def search(self, **kwargs) -> None:
        """
        Starts searching.

        Starts a process for engine search.
        """
        self.stop()
        self.process = Process(
            target=self.engine.search,
            args=[self.position, self.options],
            kwargs=kwargs,
        )
        self.process.start()

    def stop(self) -> None:
        """
        Stop search process if running.

        Should be runned when command `quit` received.
        """
        if isinstance(self.process, Process):
            self.process.kill()

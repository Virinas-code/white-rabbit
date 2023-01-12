#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

Main engine.
"""
import chess

from ..neural_network import NeuralNetwork
from .evaluation import Evaluation


class Engine:
    """Engine class."""

    def __init__(self):
        """
        Initialize engine.

        :param NeuralNetwork neural_network: Neural network to use.
        """
        self.neural_network: NeuralNetwork = NeuralNetwork.load(
            "best_network.npz"
        )

    def search(self, position: chess.Board, *, movetime: int) -> Evaluation:
        
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

Main engine.
"""
from ..neural_network import NeuralNetwork


class Engine:
    """Engine class."""

    def __init__(self, neural_network: NeuralNetwork):
        """
        Initialize engine.

        :param NeuralNetwork neural_network: Neural network to use.
        """
        self.neural_network: NeuralNetwork = neural_network
    
    def search(self, position)

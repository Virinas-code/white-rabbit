#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

Neural network iterator.
"""
from typing import Iterator

import numpy


def network_iter(network) -> Iterator[numpy.ndarray]:
    """
    Iter through a network.

    :param NeuralNetwork network: Network to get repr of.
    :return str: Network repr.
    """
    all_matrices: list[numpy.ndarray] = [
        network.matrices_left,
        network.matrices_right,
        network.scalar_matrices,
        network.reduce_matrices,
        network.biases,
        network.correction,
    ]
    return (matrix for matrix in all_matrices)

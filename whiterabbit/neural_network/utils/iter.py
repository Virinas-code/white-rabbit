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
        *network.matrices_left,
        *network.matrices_right,
        *network.scalar_matrices.values(),
        *network.reduce_matrices.values(),
        *network.biases,
        *network.correction.values(),
    ]
    return (matrix for matrix in all_matrices)

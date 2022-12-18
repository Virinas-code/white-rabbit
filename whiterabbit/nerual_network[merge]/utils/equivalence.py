#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

Neural network equivalence.
"""
import numpy as np


def networks_equal(network1, network2) -> bool:
    """
    Check if two networks are the same.

    :param NeuralNetwork network1: First network to compare.
    :param NeuralNetwork network2: Second network to compare.
    :return bool: True if the two networks are the same.
    """
    return (
        all(
            [
                np.allclose(x, y)
                for x, y in zip(network1.matrices_left, network2.matrices_left)
            ]
        )
        and all(
            [
                np.allclose(x, y)
                for x, y in zip(
                    network1.matrices_right, network2.matrices_right
                )
            ]
        )
        and all(
            [
                np.allclose(x, y)
                for x, y in zip(
                    list(network1.scalar_matrices.values()),
                    list(network2.scalar_matrices.values()),
                )
            ]
        )
        and all(
            [
                np.allclose(x, y)
                for x, y in zip(
                    list(network1.reduce_matrices.values()),
                    list(network2.reduce_matrices.values()),
                )
            ]
        )
        and all(
            [
                np.allclose(x, y)
                for x, y in zip(network1.biases, network2.biases)
            ]
        )
        and all(
            [
                np.allclose(x, y)
                for x, y in zip(
                    list(network1.correction.values()),
                    list(network2.correction.values()),
                )
            ]
        )
    )

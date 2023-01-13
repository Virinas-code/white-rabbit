#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

Neural network random generation.
"""
import numpy as np

HIDDEN_LAYERS: int = 8


def random_method(cls, maximum: int = 255):
    """
    Generate a random network.

    Fully random network.
    """
    matrices_left: list[np.ndarray] = []
    matrices_right: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for _ in range(HIDDEN_LAYERS + 2):
        matrices_left.append(
            np.random.randint(0, maximum, (8, 8, 12, 12)).astype(np.uint8)
        )
        matrices_right.append(
            np.random.randint(0, maximum, (8, 8, 12, 12)).astype(np.uint8)
        )
        biases.append(
            np.random.randint(0, maximum, (8, 8, 12, 12)).astype(np.uint8)
        )
    scalar_matrices: dict[str, np.ndarray] = {
        "R-Gi": np.random.randint(0, maximum, (8, 8, 1, 12)).astype(np.uint8),
        "R-Di": np.random.randint(0, maximum, (8, 8, 12, 1)).astype(np.uint8),
        "R-Ge": np.random.randint(0, maximum, (1, 8, 1, 1)).astype(np.uint8),
        "R-De": np.random.randint(0, maximum, (8, 1, 1, 1)).astype(np.uint8),
    }
    reduce_matrices: dict[str, np.ndarray] = {
        "RM-G": np.random.randint(0, maximum, (16, 96)).astype(np.uint8),
        "RM-D": np.random.randint(0, maximum, (96, 14)).astype(np.uint8),
    }
    correction: dict[str, np.ndarray] = {
        "R-G": np.random.randint(0, maximum, (8, 8, 12, 12)).astype(np.uint8),
        "R-D": np.random.randint(0, maximum, (8, 8, 12, 12)).astype(np.uint8),
    }
    return cls(
        matrices_left,
        matrices_right,
        list(scalar_matrices.values()),
        list(reduce_matrices.values()),
        biases,
        tuple(correction.values()),
    )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit Chess Engine.

Training algorithm functions.
"""
import numpy as np

from ..neural_network import HIDDEN_LAYERS, NeuralNetwork


def gen_direction_matrices(self) -> None:
    """
    Generate direction matrices.

    Matrices are stored in self.direction_matrices.
    """
    matrices_left: list[np.ndarray] = []
    matrices_right: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for _ in range(HIDDEN_LAYERS + 2):
        matrices_left.append(
            np.random.randint(0, 255, (8, 8, 12, 12)).astype(np.uint8)
        )
        matrices_right.append(
            np.random.randint(0, 255, (8, 8, 12, 12)).astype(np.uint8)
        )
        biases.append(
            np.random.randint(0, 255, (8, 8, 12, 12)).astype(np.uint8)
        )
    scalar_matrices: dict[str, np.ndarray] = {
        "R-Gi": np.random.randint(0, 255, (8, 8, 1, 12)).astype(np.uint8),
        "R-Di": np.random.randint(0, 255, (8, 8, 12, 1)).astype(np.uint8),
        "R-Ge": np.random.randint(0, 255, (1, 8, 1, 1)).astype(np.uint8),
        "R-De": np.random.randint(0, 255, (8, 1, 1, 1)).astype(np.uint8),
    }
    reduce_matrices: dict[str, np.ndarray] = {
        "RM-G": np.random.randint(0, 255, (16, 96)).astype(np.uint8),
        "RM-D": np.random.randint(0, 255, (96, 14)).astype(np.uint8),
    }
    correction: dict[str, np.ndarray] = {
        "R-G": np.random.randint(0, 255, (8, 8, 12, 12)).astype(np.uint8),
        "R-D": np.random.randint(0, 255, (8, 8, 12, 12)).astype(np.uint8),
    }
    self.direction_matrices = (
        {
            "matrices_left": matrices_left,
            "matrices_right": matrices_right,
            "biases": biases,
        },
        {
            "scalar_matrices": scalar_matrices,
            "reduce_matrices": reduce_matrices,
            "correction": correction,
        },
    )


def gen_mutated_network(self, mutation: int) -> NeuralNetwork:
    """
    Generate a mutated network.

    :param int mutation: Network index.
    :return NeuralNetwork: Mutated neural network.
    """
    matrices_left: list[np.ndarray] = []
    matrices_right: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for layer in range(HIDDEN_LAYERS + 2):
        matrices_left.append(
            self.first_network.matrices_left[layer]
            + mutation * self.direction_matrices[0]["matrices_left"][layer]
        )
        matrices_right.append(
            self.first_network.matrices_left[layer]
            + mutation * self.direction_matrices[0]["matrices_left"][layer]
        )
        biases.append(
            self.first_network.matrices_left[layer]
            + mutation * self.direction_matrices[0]["matrices_left"][layer]
        )
    scalar_matrices: dict[str, np.ndarray] = {}
    reduce_matrices: dict[str, np.ndarray] = {}
    correction: dict[str, np.ndarray] = {}
    for key, value in self.direction_matrices[1]["scalar_matrices"].items():
        scalar_matrices[key] = (
            self.first_network.scalar_matrices[key] + mutation * value
        )
    for key, value in self.direction_matrices[1]["reduce_matrices"].items():
        reduce_matrices[key] = (
            self.first_network.reduce_matrices[key] + mutation * value
        )
    for key, value in self.direction_matrices[1]["correction"].items():
        correction[key] = self.first_network.correction[key] + mutation * value
    return NeuralNetwork(
        matrices_left,
        matrices_right,
        list(scalar_matrices.values()),
        list(reduce_matrices.values()),
        biases,
        (correction["R-G"], correction["R-D"]),
    )

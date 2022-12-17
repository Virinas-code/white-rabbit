#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

Base training object.
"""
from typing import Union

import numpy as np

from .. import NeuralNetwork

HIDDEN_LAYERS: int = 8  # Amount of hidden layers


class Trainer:
    """Base object for training."""

    def __init__(self):
        """
        Initialize object.

        TODO: Complete this
        """
        self.first_network: NeuralNetwork = NeuralNetwork.random()

    def main_loop(self) -> None:
        """
        Main training loop.

        TODO: Stop
        """
        while True:
            self.train_step()

    def train_step(self) -> None:
        """
        Training iteration.

        Train networks 1 time.
        """
        self.generate_dir_matrices()

    def generate_dir_matrices(
        self,
    ) -> list[Union[list[np.ndarray], dict[str, np.ndarray]]]:
        """
        Generate direction matrices.
        """
        matrices_left: list[np.ndarray] = []
        matrices_right: list[np.ndarray] = []
        biases: list[np.ndarray] = []
        for layer in range(HIDDEN_LAYERS + 2):
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
        return cls(
            matrices_left,
            matrices_right,
            list(scalar_matrices.values()),
            list(reduce_matrices.values()),
            biases,
            tuple(correction.values()),
        )

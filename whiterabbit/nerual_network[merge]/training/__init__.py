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
        direction_matrices: list[
            list[np.ndarray] | dict[str, np.ndarray]
        ] = self.generate_dir_matrices()
        self.n_loop(direction_matrices)

    def generate_dir_matrices(
        self,
    ) -> dict[str, list[np.ndarray] | dict[str, np.ndarray]]:
        """
        Generate direction matrices.
        """
        returned: dict[str, list[np.ndarray] | dict[str, np.ndarray]] = {}
        returned["matrices_left"] = []
        returned["matrices_right"] = []
        returned["biases"] = []
        for _ in range(HIDDEN_LAYERS + 2):
            returned["matrices_left"].append(
                np.random.randint(0, 255, (8, 8, 12, 12)).astype(np.uint8)
            )
            returned["matrices_right"].append(
                np.random.randint(0, 255, (8, 8, 12, 12)).astype(np.uint8)
            )
            returned["biases"].append(
                np.random.randint(0, 255, (8, 8, 12, 12)).astype(np.uint8)
            )
        returned["scalar_matrices"] = {
            "R-Gi": np.random.randint(0, 255, (8, 8, 1, 12)).astype(np.uint8),
            "R-Di": np.random.randint(0, 255, (8, 8, 12, 1)).astype(np.uint8),
            "R-Ge": np.random.randint(0, 255, (1, 8, 1, 1)).astype(np.uint8),
            "R-De": np.random.randint(0, 255, (8, 1, 1, 1)).astype(np.uint8),
        }
        returned["reduce_matrices"] = {
            "RM-G": np.random.randint(0, 255, (16, 96)).astype(np.uint8),
            "RM-D": np.random.randint(0, 255, (96, 14)).astype(np.uint8),
        }
        returned["correction"] = {
            "R-G": np.random.randint(0, 255, (8, 8, 12, 12)).astype(np.uint8),
            "R-D": np.random.randint(0, 255, (8, 8, 12, 12)).astype(np.uint8),
        }
        return returned

    def n_loop(self, dir_matrices: dict[str, list[np.ndarray] | dict[str, np.ndarray]]) -> None:
        """
        Loop all directions.

        :param dict[str, list[np.ndarray] | dict[str, np.ndarray]] dir_matrice: Direction matrices.
        """
        for network in range(0, 256):
            self.generate_network(network, dir_matrices)
    
    def generate_network(self, network: int, dir_matrices: dict[str, list[np.ndarray] | dict[str, np.ndarray]]) -> NeuralNetwork:
        """
        Generate a mutated network.

        :param int network: Network ID.
        :param list[list[np.ndarray]  |  dict[str, np.ndarray]] dir_matrices: Direction matrices
        :return NeuralNetwork: Mutated neural network.
        """
        mutated_matrices_left: list[np.ndarray] = []
        for index, matrix_left in enumerate(self.first_network.matrices_left):
            mutated_matrices_left.append(matrix_left + network * dir_matrices["matrices_left"][index])
        return NeuralNetwork(
            
        )
        
,
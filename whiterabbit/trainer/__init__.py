#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit Chess Engine.

Base training algorithm (v2).
"""
import sys
from typing import TypeAlias

import numpy as np

from ..neural_network import HIDDEN_LAYERS, NeuralNetwork
from .cli import TrainerCLI
from .config import NETWORKS_INDEXES_PLAYING

NetworkID: TypeAlias = int
"""A network hash."""
Score: TypeAlias = int
"""Score of a network."""


class Trainer:
    """Main class for training."""

    def __init__(self):
        """
        Initialize training.

        TODO: Complete doc.
        """
        self.mutated_networks: list[NeuralNetwork] = []
        """List of all the mutated networks created from the first network."""
        self.first_network: NeuralNetwork = self._gen_first_network()
        """First network."""
        self.scores: dict[NetworkID, Score] = {}
        """Map between hashes and scores."""

        self.direction_matrices: tuple[
            dict[str, list[np.ndarray]], dict[str, dict[str, np.ndarray]]
        ] = ({}, {})

        self.cli: TrainerCLI = TrainerCLI()
        """CLI object."""

    def _gen_first_network(self) -> NeuralNetwork:
        """
        Generate first network.

        Loads it from data/training/best-network.npz.
        If -r in argv, then uses a random one.

        :return NeuralNetwork: First neural network.
        """
        if "-r" in sys.argv:
            return NeuralNetwork.random()
        return NeuralNetwork.load("data/training/best-network.npz")

    def main_loop(self) -> None:
        """
        Training main loop.

        Does nothing.
        """
        training_iterations: int = 0

        while True:
            training_iterations += 1
            self.cli.training_iteration(training_iterations)

            self.train()

    def train(self) -> None:
        """
        Training main loop iteration.

        Calls all subfunctions.
        """
        self.generate_direction_matrices()
        self.generate_networks()
        self.game_loop()
        self.fetch_results()

    def generate_direction_matrices(self) -> None:
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

    def generate_networks(self) -> None:
        """
        Generate mutated networks.

        Networks are stored in self.mutated_networks.
        They are mutations of self.first_network.
        """
        self.cli.init_network_gen()
        self.mutated_networks = []
        for _ in range(len(NETWORKS_INDEXES_PLAYING) - 1):
            self.cli.network_gen_iteration()
            self.mutated_networks.append(self.generate_mutated_network())
        self.cli.end_network_gen()
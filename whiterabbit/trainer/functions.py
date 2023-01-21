#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit Chess Engine.

Training algorithm functions.
"""
import chess
import numpy as np

from .config import DEPTHS, DIR_PROB, POSITIONS
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
            np.random.choice((0, 1), (8, 8, 12, 12), p=DIR_PROB).astype(
                np.uint8
            )
        )
        matrices_right.append(
            np.random.choice((0, 1), (8, 8, 12, 12), p=DIR_PROB).astype(
                np.uint8
            )
        )
        biases.append(
            np.random.choice((0, 1), (8, 8, 12, 12), p=DIR_PROB).astype(
                np.uint8
            )
        )
    scalar_matrices: dict[str, np.ndarray] = {
        "R-Gi": np.random.choice((0, 1), (8, 8, 1, 12), p=DIR_PROB).astype(
            np.uint8
        ),
        "R-Di": np.random.choice((0, 1), (8, 8, 12, 1), p=DIR_PROB).astype(
            np.uint8
        ),
        "R-Ge": np.random.choice((0, 1), (1, 8, 1, 1), p=DIR_PROB).astype(
            np.uint8
        ),
        "R-De": np.random.choice((0, 1), (8, 1, 1, 1), p=DIR_PROB).astype(
            np.uint8
        ),
    }
    reduce_matrices: dict[str, np.ndarray] = {
        "RM-G": np.random.choice((0, 1), (16, 96), p=DIR_PROB).astype(
            np.uint8
        ),
        "RM-D": np.random.choice((0, 1), (96, 14), p=DIR_PROB).astype(
            np.uint8
        ),
    }
    correction: dict[str, np.ndarray] = {
        "R-G": np.random.choice((0, 1), (8, 8, 12, 12), p=DIR_PROB).astype(
            np.uint8
        ),
        "R-D": np.random.choice((0, 1), (8, 8, 12, 12), p=DIR_PROB).astype(
            np.uint8
        ),
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


def func_play_game(self, first_index: int, second_index: int) -> None:
    """
    Play games between two networks.

    :param int first_index: First network index.
    :param int second_index: Second network index.
    """
    first_network: NeuralNetwork = self.mutated_networks[first_index]
    second_network: NeuralNetwork = self.mutated_networks[second_index]
    self.cli.init_game()
    for depth in DEPTHS:
        self.core_play_game(first_network, second_network, depth)
    self.cli.end_game()


def func_core_play_game(
    self,
    first_network: NeuralNetwork,
    second_network: NeuralNetwork,
    depth: int,
) -> None:
    """
    Play a game between two networks.

    :param NeuralNetwork first_network: First network.
    :param NeuralNetwork second_network: Second network.
    :param int depth: Depth to play at.
    """
    for position in POSITIONS:
        first_network.new_game()
        second_network.new_game()
        game: chess.Board = chess.Board(position)
        while not game.is_game_over(claim_draw=True):
            if game.turn is chess.WHITE:
                game.push(first_network.search(game, depth))
            else:
                game.push(second_network.search(game, depth))
        first_network.game_end()
        second_network.game_end()
        outcome: chess.Outcome = game.outcome(claim_draw=True)  # type: ignore
        if outcome.result() == "1/2-1/2":  # Draw, +1*depth to each
            self.scores[hash(first_network)] = (
                self.scores.get(hash(first_network), 0) + depth * 1
            )
            self.scores[hash(second_network)] = (
                self.scores.get(hash(second_network), 0) + depth * 1
            )
        else:  # Win, +3*depth to winner
            winner_hash = hash(first_network)
            if outcome.winner is chess.BLACK:
                winner_hash = hash(second_network)
            self.scores[winner_hash] = (
                self.scores.get(winner_hash, 0) + depth * 3
            )
        self.cli.game_iteration()

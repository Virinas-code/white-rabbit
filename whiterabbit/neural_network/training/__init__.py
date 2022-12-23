#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

Base training object.
"""
import datetime
import os

import chess
import chess.pgn
import numpy as np
import rich.progress as progress

from .. import NeuralNetwork

HIDDEN_LAYERS: int = 8  # Amount of hidden layers

DirectionMatrices: type = tuple[
    dict[str, list[np.ndarray]], dict[str, dict[str, np.ndarray]]
]


class Trainer:
    """Base object for training."""

    def __init__(self):
        """
        Initialize object.

        TODO: Complete this
        """
        self.first_network: NeuralNetwork = NeuralNetwork.random()
        self.progress: progress.Progress = progress.Progress()
        self.progress.start()
        """self.play_task: progress.TaskID = self.progress.add_task(
            "[bold red] Playing games...", total=255 * 255
        )
        self.depth_task: progress.TaskID = self.progress.add_task(
            "[cyan] Loading...", total=None
        )"""

    def main_loop(self) -> None:
        """
        Main training loop.

        TODO: Stop
        """
        main_progress: progress.TaskID = self.progress.add_task(
            "[bold green] Training...", total=None
        )
        while True:
            self.progress.update(main_progress)
            self.train_step()

    def train_step(self) -> None:
        """
        Training iteration.

        Train networks 1 time.
        """
        dir_matrices: DirectionMatrices = self.generate_dir_matrices()
        self.mutations_loop(dir_matrices)

    def generate_dir_matrices(
        self,
    ) -> DirectionMatrices:
        """
        Generate direction matrices.

        :return tuple[dict[str, list[np.ndarray]],
            dict[str, dict[str, np.ndarray]]]: Direction matrices.
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
        return (
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

    def mutations_loop(self, direction_matrices: DirectionMatrices) -> None:
        """
        Iteration main loop.

        Generates all mutated networks.

        :param DirectionMatrices direction_matrices: Direction matrices.
        """
        mutated_networks: list[NeuralNetwork] = []  # Rj
        generate_networks_progress: progress.TaskID = self.progress.add_task(
            "[magenta] Generating networks", total=256
        )
        for network in range(0, 256):
            self.progress.update(generate_networks_progress, advance=1)
            mutated_networks.append(
                self.generate_network(network, direction_matrices)
            )
        self.progress.remove_task(generate_networks_progress)
        self.games_loop(mutated_networks)

    def generate_network(
        self, mutation: int, direction_matrices: DirectionMatrices
    ) -> NeuralNetwork:
        """
        Generate a mutated network.

        :param int mutation: Network ID.
        :param DirectionMatrices direction_matrices: Direction matrices.
        :return NeuralNetwork: Mutated neural network.
        """
        matrices_left: list[np.ndarray] = []
        matrices_right: list[np.ndarray] = []
        biases: list[np.ndarray] = []
        for layer in range(HIDDEN_LAYERS + 2):
            matrices_left.append(
                self.first_network.matrices_left[layer]
                + mutation * direction_matrices[0]["matrices_left"][layer]
            )
            matrices_right.append(
                self.first_network.matrices_left[layer]
                + mutation * direction_matrices[0]["matrices_left"][layer]
            )
            biases.append(
                self.first_network.matrices_left[layer]
                + mutation * direction_matrices[0]["matrices_left"][layer]
            )
        scalar_matrices: dict[str, np.ndarray] = {}
        reduce_matrices: dict[str, np.ndarray] = {}
        correction: dict[str, np.ndarray] = {}
        for key, value in direction_matrices[1]["scalar_matrices"].items():
            scalar_matrices[key] = (
                self.first_network.scalar_matrices[key] + mutation * value
            )
        for key, value in direction_matrices[1]["reduce_matrices"].items():
            reduce_matrices[key] = (
                self.first_network.reduce_matrices[key] + mutation * value
            )
        for key, value in direction_matrices[1]["correction"].items():
            correction[key] = (
                self.first_network.correction[key] + mutation * value
            )
        return NeuralNetwork(
            matrices_left,
            matrices_right,
            list(scalar_matrices.values()),
            list(reduce_matrices.values()),
            biases,
            tuple(correction.values()),
        )

    def games_loop(self, mutated_networks: list[NeuralNetwork]) -> None:
        """
        Play games between original and mutated networks.

        :param NeuralNetwork mutated_network: Mutated networks.
        """
        games_progress: progress.TaskID = self.progress.add_task(
            "[bold red] Playing games", total=256
        )
        for first_network in range(256):
            self.progress.update(
                games_progress,
                description=f"[bold red] Matchmaking {id(mutated_networks[first_network])}",
            )
            second_progress: progress.TaskID = self.progress.add_task(
                "[red] Playing games", total=256
            )
            for second_network in range(256):
                self.progress.update(
                    second_progress,
                    description=f"[red] Playing vs {id(mutated_networks[second_network])}",
                )
                self.play_game(
                    mutated_networks[first_network],
                    mutated_networks[second_network],
                    first_network * second_network,
                )
                self.progress.update(second_progress, advance=1)
            self.progress.remove_task(second_progress)
            self.progress.update(games_progress, advance=1)
        self.progress.remove_task(games_progress)

    def play_game(
        self,
        first_network: NeuralNetwork,
        second_network: NeuralNetwork,
        round: int,
    ) -> None:
        """
        Play a game between two networks.

        :param NeuralNetwork first_network: First network.
        :param NeuralNetwork second_network: Second network.
        """
        depth_progress: progress.TaskID = self.progress.add_task(
            "[cyan] Testing networks", total=3
        )
        for depth in range(1, 4):
            first_network.new_game()
            second_network.new_game()
            game: chess.Board = chess.Board()
            while not game.is_game_over(claim_draw=True):
                if game.turn is chess.WHITE:
                    game.push(first_network.search(game, depth))
                else:
                    game.push(second_network.search(game, depth))
            self.save_game(
                game, (id(first_network), id(second_network)), (round, depth)
            )
            self.progress.update(depth_progress, advance=1)
        self.progress.remove_task(depth_progress)

    def save_game(
        self,
        game: chess.Board,
        networks_id: tuple[int, int],
        round: tuple[int, int],
    ) -> None:
        """
        Save game.

        :param chess.Board game: Game to save.
        :param tuple[int, int] networks_id: Networks IDs.
        """
        game_pgn: chess.pgn.Game = chess.pgn.Game.from_board(game)
        date: datetime.datetime = datetime.datetime.now()
        game_pgn.headers = chess.pgn.Headers(
            Event="White Rabbit training",
            White=f"White Rabbit #{networks_id[0]}",
            WhiteTitle="BOT",
            Black=f"White Rabbit #{networks_id[1]}",
            BlackTitle="BOT",
            Date=f"{date.year}.{date.month}.{date.day}",
            Termination="normal",
            Site=os.uname().nodename,
            Round=str(round[0]),
            Board=str(round[1]),
        )
        file_name = f"games/{str(date)}-{networks_id[0]}_{networks_id[1]}-{round[0]}_{round[1]}.pgn"
        with open(
            file_name,
            "w",
            encoding="utf-8",
        ) as file:
            file.write(str(game_pgn))
        if game.result(claim_draw=True) == "1-0":
            self.progress.console.log(
                f"[bold green] {networks_id[0]} won vs {networks_id[1]}"
            )
        elif game.result(claim_draw=True) == "1-0":
            self.progress.console.log(
                f"[bold red] {networks_id[0]} lost vs {networks_id[1]}"
            )
        else:
            self.progress.console.log(
                f"[bold blue] {networks_id[0]} drawed vs {networks_id[1]}"
            )

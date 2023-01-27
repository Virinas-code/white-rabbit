#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit Chess Engine.

Base training algorithm (v2).
"""
from __future__ import annotations

import hashlib
import json
import sys
from typing import Callable, Literal, Union, TypeAlias

from filelock import FileLock
import numpy as np

from ..neural_network import NeuralNetwork
from .cli import TrainerCLI
from .config import DEPTHS, NETWORKS_INDEXES_PLAYING, RANDOM_MAXIMUM
from .functions import (
    func_core_play_game,
    func_play_game,
    gen_direction_matrices,
    gen_mutated_network,
)
from .lock import func_acquire_lock
from .stats import func_update_stats

NetworkID: TypeAlias = int
"""A network hash."""
Score: TypeAlias = int
"""Score of a network."""
NetworkSource: TypeAlias = Union[
    Literal["Random"], Literal["Mutation"], Literal["First"]
]


class Trainer:
    """Main class for training."""

    def __init__(self):
        """
        Initialize training.

        TODO: Complete doc.
        """
        self.cli: TrainerCLI = TrainerCLI()
        """CLI object."""

        self.mutated_networks: list[NeuralNetwork] = []
        """List of all the mutated networks created from the first network."""
        self.first_network: NeuralNetwork = self._gen_first_network()
        """First network."""
        self.scores: dict[NetworkID, Score] = {}
        """Map between hashes and scores."""

        self.direction_matrices: tuple[
            dict[str, list[np.ndarray]], dict[str, dict[str, np.ndarray]]
        ] = ({}, {})

        # Red / green
        self.previous_winner: int = hash(self.first_network)

        # Stats
        self.networks_sources: dict[NetworkID, NetworkSource] = {}
        self.stats: dict[NetworkSource, int] = {
            "Random": 0,
            "Mutation": 0,
            "First": 0,
        }
        self.stats_graph: dict[str, list[float]] = {
            "med": [],
            "mea": [],
            "ete": [],
        }

        # Lock file
        self.lock: FileLock = FileLock("data/training/training.lock")

    generate_direction_matrices: Callable = gen_direction_matrices
    generate_mutated_network: Callable = gen_mutated_network
    play_game: Callable = func_play_game
    core_play_game: Callable = func_core_play_game

    acquire_lock: Callable = func_acquire_lock

    update_stats: Callable = func_update_stats

    def _load_stats(self) -> dict[NetworkSource, int]:
        """
        Load stats.

        :return dict[NetworkSource, int]: Loaded stats.
        """
        with open(
            "data/training/statistics.json", "r", encoding="utf-8"
        ) as file:
            data: dict[str, dict[NetworkSource, int]] = json.load(file)
        with open("whiterabbit/trainer/config.py", "rb") as file:
            checksum: str = hashlib.sha256(file.read()).hexdigest()
        if checksum in data:
            return data[checksum]
        return {
            "Random": 0,
            "Mutation": 0,
            "First": 0,
        }

    def _save_stats(self) -> None:
        """
        Save stats.

        Saved into data/training/statistics.json.
        """
        with open("whiterabbit/trainer/config.py", "rb") as file:
            checksum: str = hashlib.sha256(file.read()).hexdigest()
        with open(
            "data/training/statistics.json", "r+", encoding="utf-8"
        ) as file:
            data: dict[str, dict[NetworkSource, int]] = json.load(file)
            if checksum in data:
                for source, score in self.stats.items():
                    data[checksum][source] += score
            else:
                data[checksum] = {}
                for source, score in self.stats.items():
                    data[checksum][source] = score
            file.seek(0)
            json.dump(data, file)

    def _gen_first_network(self) -> NeuralNetwork:
        """
        Generate first network.

        Loads it from data/training/best-network.npz.
        If -r in argv, then uses a random one.

        :return NeuralNetwork: First neural network.
        """
        if "-r" in sys.argv:
            return NeuralNetwork.random(RANDOM_MAXIMUM)
        loaded_network: NeuralNetwork = NeuralNetwork.load(
            "data/training/best-network.npz"
        )
        self.cli.print(
            "[bold magenta]Loaded network "
            + f"[not bold magenta on gray23] #{hash(loaded_network)} "
        )
        return loaded_network

    def _calc_mut_stat(self) -> float:
        muts_amount: int = len(NETWORKS_INDEXES_PLAYING)
        muts_amount -= NETWORKS_INDEXES_PLAYING.count(0)
        muts_amount -= NETWORKS_INDEXES_PLAYING.count(256)
        return (
            self.stats["Random"] / sum(self.stats.values()) * 100 / muts_amount
        )

    def main_loop(self) -> None:
        """
        Training main loop.

        Call this to start training.
        """
        training_iterations: int = 0

        infos: str = (
            f"({len(NETWORKS_INDEXES_PLAYING)} networks playing, "
            + f"{len(DEPTHS)} depths)"
        )
        self.cli.print(
            f"[bold cyan]Starting training session [not bold]{infos}"
        )

        self.acquire_lock()

        try:
            while True:
                training_iterations += 1
                self.cli.training_iteration(training_iterations)

                self.train()

                self.update_stats()

                self.scores = {}  # reset scores
                self.networks_sources = {}  # reset sources
        except KeyboardInterrupt:
            previous_winner_id: NetworkID = hash(self.previous_winner)
            self.cli.print(
                "[bold cyan]Ending training session "
                + f"[not bold]({training_iterations} iterations)"
            )
            self.cli.print("[bold yellow]Statistics:")
            self._save_stats()
            if not sum(self.stats.values()) > 0:
                self.cli.print("[yellow] No statistics available.")
                self.cli.clear()
                sys.exit(-1)
            self.cli.print(
                "[yellow] - Random: "
                + f"{self.stats['Random'] / sum(self.stats.values()) * 100}%"
            )
            self.cli.print(
                "[yellow] - Mutation: " + f"{self._calc_mut_stat()}%"
            )
            self.cli.print(
                "[yellow] - First: "
                + f"{self.stats['First'] / sum(self.stats.values()) * 100}%"
            )
            self.cli.print(
                "[bold magenta]Last saved network "
                + f"[not bold magenta on gray23] #{previous_winner_id} "
            )
            self.cli.clear()
            self.lock.release()
            sys.exit(0)

    def train(self) -> None:
        """
        Training main loop iteration.

        Calls all subfunctions.
        """
        self.generate_direction_matrices()
        self.generate_networks()
        self.game_loop()
        self.fetch_results()

    def generate_networks(self) -> None:
        """
        Generate mutated networks.

        Networks are stored in self.mutated_networks.
        They are mutations of self.first_network.
        """
        self.cli.init_network_gen()
        mutated_networks: list[NeuralNetwork] = [
            self.first_network,
            *([NeuralNetwork.random()] * 256),
        ]
        self.networks_sources[hash(self.first_network)] = "First"
        for mutation_index in NETWORKS_INDEXES_PLAYING:
            self.cli.network_gen_iteration()
            if mutation_index != 0:
                mutated_networks[
                    mutation_index
                ] = self.generate_mutated_network(mutation_index)
                self.networks_sources[
                    hash(mutated_networks[mutation_index])
                ] = "Mutation"
        mutated_networks[256] = NeuralNetwork.random(RANDOM_MAXIMUM)
        self.networks_sources[hash(mutated_networks[256])] = "Random"
        self.mutated_networks = mutated_networks
        self.cli.end_network_gen()

    def game_loop(self) -> None:
        """
        Play games between networks.

        Matchmakes all networks.
        """
        for first_network_index in NETWORKS_INDEXES_PLAYING:
            self.cli.play_name(
                hash(self.mutated_networks[first_network_index])
            )
            for second_network_index in NETWORKS_INDEXES_PLAYING:
                if first_network_index != second_network_index:
                    self.cli.match_name(
                        hash(self.mutated_networks[second_network_index])
                    )
                    self.play_game(first_network_index, second_network_index)
                    self.cli.match_iteration()
            self.cli.play_iteration()

    def fetch_results(self) -> None:
        """
        Save best network and print some infos.

        Prints each score set.
        """
        string: str = ""
        for network, score in self.scores.items():
            string += f"#{network}: {score} / "
        string = string[:-3]
        best_network_id: NetworkID = max(
            self.scores, key=self.scores.get  # type: ignore
        )
        best_network: NeuralNetwork = NeuralNetwork.random()
        for mutated_network in self.mutated_networks:
            if hash(mutated_network) == best_network_id:
                best_network = mutated_network
        best_network.save("data/training/best-network.npz")
        self.first_network = best_network
        colors: dict[NetworkSource, str] = {
            "Random": "red",
            "Mutation": "blue",
            "First": "green",
        }
        color: str = colors[self.networks_sources[best_network_id]]
        self.stats[self.networks_sources[best_network_id]] += 1
        infos: str = (
            f"[bold {color}] • Saved [{color} on gray23] #{hash(best_network)}"
            + f"[{self.networks_sources[best_network_id].upper()}]"
            + f" [{color} on default] "
            + f"[not bold]({string})"
        )
        self.cli.print(infos)

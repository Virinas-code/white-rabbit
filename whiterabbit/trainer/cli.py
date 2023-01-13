# -*- coding: utf-8 -*-
"""
White Rabbit Chess Engine.

CLI for training algorithm.
"""
from rich import progress

from .config import DEPTHS, NETWORKS_INDEXES_PLAYING


class TrainerCLI:
    """CLI for training."""

    def __init__(self):
        """
        Initializes all progress bars.

        TODO: Update.
        """
        self.progress: progress.Progress = progress.Progress(
            progress.SpinnerColumn(),
            *progress.Progress.get_default_columns(),
            progress.TimeElapsedColumn(),
        )
        self.progress.start()

        # Progress bars
        self.main_progress: progress.TaskID = self._main_progress()
        self.general_progress: progress.TaskID = self._general_progress()
        self.games_progress: progress.TaskID = self._games_progress()
        self.second_progress: progress.TaskID = self._second_progress()
        self.depth_progress: progress.TaskID = self._depth_progress()
        self.generate_networks_progress: progress.TaskID = (
            self._generate_networks_progress()
        )

    def _main_progress(self) -> progress.TaskID:
        return self.progress.add_task("[bold green] Training...", total=None)

    def _general_progress(self) -> progress.TaskID:
        return self.progress.add_task(
            "[green] Training completion",
            total=len(NETWORKS_INDEXES_PLAYING) * len(DEPTHS),
        )

    def _games_progress(self) -> progress.TaskID:
        return self.progress.add_task(
            "[bold red] Playing games", total=len(NETWORKS_INDEXES_PLAYING)
        )

    def _second_progress(self) -> progress.TaskID:
        return self.progress.add_task(
            "[red] Playing games", total=len(NETWORKS_INDEXES_PLAYING) - 1
        )  # -1 for play vs itself

    def _depth_progress(self) -> progress.TaskID:
        return self.progress.add_task(
            "[bold blue] Testing networks", total=len(DEPTHS)
        )

    def _generate_networks_progress(self) -> progress.TaskID:
        return self.progress.add_task(
            "[blue] Generating networks",
            total=len(NETWORKS_INDEXES_PLAYING) - 1,
        )  # -1 for first network

    def training_iteration(self, iteration: int) -> None:
        """
        Reset progresses after training.

        Resets Training completion, Matchmaking and Playing games.
        Updates Training.
        """
        self.progress.update(
            self.main_progress,
            description=f"[bold green] Training [italic]#{iteration}",
        )
        self.progress.remove_task(self.general_progress)
        self.progress.remove_task(self.games_progress)
        self.progress.remove_task(self.second_progress)
        self.general_progress = self._general_progress()
        self.games_progress = self._games_progress()
        self.second_progress = self._second_progress()

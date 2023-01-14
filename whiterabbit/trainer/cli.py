# -*- coding: utf-8 -*-
"""
White Rabbit Chess Engine.

CLI for training algorithm.
"""
from typing import Any

from rich import progress

from .config import DEPTHS, NETWORKS_INDEXES_PLAYING


class TrainerCLI:
    """CLI for training."""

    generate_networks_progress: progress.TaskID
    depth_progress: progress.TaskID

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

    def _main_progress(self) -> progress.TaskID:
        return self.progress.add_task("[bold green] Training...", total=None)

    def _general_progress(self) -> progress.TaskID:
        return self.progress.add_task(
            "[green] Training completion",
            total=len(NETWORKS_INDEXES_PLAYING)
            * (len(NETWORKS_INDEXES_PLAYING) - 1)
            * len(DEPTHS),
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
        description: str = (
            "[bold green] Training " + f"[green on gray23] n°{iteration} "
        )
        self.progress.update(
            self.main_progress,
            description=description,
        )
        self.progress.remove_task(self.general_progress)
        self.progress.remove_task(self.games_progress)
        self.progress.remove_task(self.second_progress)
        self.general_progress = self._general_progress()
        self.games_progress = self._games_progress()
        self.second_progress = self._second_progress()

    def play_name(self, name: int) -> None:
        """
        Update first progress name.

        :param int name: First network hash.
        """
        self.progress.update(
            self.games_progress,
            description=f"[bold red] Matchmaking [red on grey23] #{name} ",
        )

    def match_name(self, name: int) -> None:
        """
        Update second progress name.

        :param int name: Second network hash.
        """
        self.progress.update(
            self.second_progress,
            description=f"[red] Playing vs [red on grey23] #{name} ",
        )

    def play_iteration(self) -> None:
        """
        Called after a matchmake.

        Advances first progress.
        """
        self.progress.update(self.games_progress, advance=1)
        self.progress.remove_task(self.second_progress)
        self.second_progress = self._second_progress()

    def match_iteration(self) -> None:
        """
        Called after a match.

        Advances second progress.
        """
        self.progress.update(self.second_progress, advance=1)

    def init_network_gen(self) -> None:
        """
        Initialize network generation.

        Adds the network generation task.
        """
        self.generate_networks_progress = self._generate_networks_progress()

    def network_gen_iteration(self) -> None:
        """
        Update network initialization progress bar.

        Just adds 1.
        """
        self.progress.update(self.generate_networks_progress, advance=1)

    def end_network_gen(self) -> None:
        """
        End of network generation.

        Removes progress bar.
        """
        self.progress.remove_task(self.generate_networks_progress)

    def init_game(self) -> None:
        """
        Initialize games playing.

        Adds the Testing networks task.
        """
        self.depth_progress = self._depth_progress()

    def game_iteration(self) -> None:
        """
        Update games playing progress bar.

        Adds 1 to it.
        """
        self.progress.update(self.depth_progress, advance=1)
        self.progress.update(self.general_progress, advance=1)

    def end_game(self) -> None:
        """
        End of games playing.

        Removes progress bar.
        """
        self.progress.remove_task(self.depth_progress)

    def print(self, text: str) -> None:
        """
        Print some text in console.

        :param str text: Text to print.
        """
        self.progress.console.print(text)

    def log(self, text: Any) -> None:
        """
        Log some text in console.

        :param str text: Text to log.
        """
        self.progress.console.log(text)

    def clear(self) -> None:
        """
        Clear progress bars.

        Used at end of program.
        """
        try:
            self.progress.remove_task(self.main_progress)
            self.progress.remove_task(self.general_progress)
            self.progress.remove_task(self.games_progress)
            self.progress.remove_task(self.second_progress)
            self.progress.remove_task(self.depth_progress)
            self.progress.remove_task(self.generate_networks_progress)
        except KeyError:
            pass
        self.progress.stop()

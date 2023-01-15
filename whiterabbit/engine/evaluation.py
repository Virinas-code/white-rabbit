#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

Engine evaluation.
"""
import chess


class Evaluation:
    """An engine evaluation of a position."""

    def __init__(
        self,
        depth: int,
        time: int,
        nodes: int,
        best_moves: list[chess.Move],
        score: tuple[int, int],
        hash_full: int,
        nps: int,
        tbhits: int,
        cpuload: int,
    ):
        """
        An engine evaluation.

        :param int depth: Search depth.
        :param int time: Time
        :param int nodes: Number of nodes searched.
        :param list[chess.Move] pv: PVs.
        :param tuple[int, int] score: Centipawns and mate score.
        :param int hash_full: Permill of hash filled.
        :param int nps: Nodes per second.
        :param int tbhits: Tablebase hits.
        :param int cpuload: Permill of CPU used.
        """
        self.depth: int = depth
        """Search depth."""
        self.time: int = time
        """Time"""
        self.nodes: int = nodes
        """Number of nodes searched."""
        self.best_moves: list[chess.Move] = best_moves
        """PVs."""
        self.score: tuple[int, int] = score
        """Centipawns and mate score."""
        self.hash_full: int = hash_full
        """Permill of hash filled."""
        self.nps: int = nps
        """Nodes per second."""
        self.tbhits: int = tbhits
        """Tablebase hits."""
        self.cpuload: int = cpuload
        """Permill of CPU used."""

    def info(self, multi_pv: int) -> None:
        """
        Print UCI `info` string.

        ..note::
            This doesn't use the `UCI` class, because syncing all was too hard.

        :param int multi_pv: MultiPV value.
        """
        for multipv, move in enumerate(self.best_moves):
            if multipv < multi_pv:
                score: str = f"cp {self.score[0]}"
                if self.score[1] > 0:
                    score = f"mate {self.score[1]}"
                print(
                    "info",
                    "depth",
                    self.depth,
                    "seldepth",
                    self.depth,
                    "time",
                    self.time,
                    "nodes",
                    self.nodes,
                    "pv",
                    move.uci(),
                    "multipv",
                    multipv + 1,
                    "score",
                    score,
                    "hashfull",
                    self.hash_full,
                    "nps",
                    self.nps,
                    "tbhits",
                    self.tbhits,
                    "cpuload",
                    self.cpuload,
                )

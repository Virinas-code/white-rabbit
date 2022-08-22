#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

Main engine.
"""
import random
import sys
from time import sleep

import chess
import numpy

from .evaluation import Evaluation


class Engine:
    """The main engine."""

    def search(
        self,
        board: chess.Board,
        *,
        max_depth: int = 0,
        max_nodes: int = sys.maxsize,
        move_time: int = sys.maxsize
    ) -> tuple[chess.Move, chess.Move]:
        """
        Search in position.

        :param chess.Board board: Board to search best move.
        :param int max_depth: Maximum allowed depth.
        :param int max_nodes: Maximum number of nodes.
        :param int move_time: Time to move.
        :return tuple[chess.Move, chess.Move]
        """
        hidden_layers: list[numpy.ndarray] = []
        maximum_depth: int = sys.maxsize
        nodes_error: int = 0
        if max_nodes:
            moves: int = len(list(board.legal_moves))
            maximum_depth = int(max_nodes / moves)
            nodes_error = max_nodes % moves
        elif max_depth:
            maximum_depth = max_depth
        evaluation: Evaluation = Evaluation(0, 0, 0, [], (0, 0), 0, 0, 0, 0)
        for depth in range(maximum_depth):
            evaluation = self.iteration(board, move_time)
            evaluation.info()
        if nodes_error:
            evaluation.nodes = max_nodes + nodes_error
            evaluation.info()

    def iteration(self, board: chess.Board, move_time: int) -> Evaluation:
        """
        Run an iteration.

        :param chess.Board board: Current board.
        :param int move_time: Maximum move time.
        :return Evaluation: Position evaluation.
        """
        multipv: list[chess.Move] = list(board.legal_moves)
        random.shuffle(multipv)
        sleep(3)
        return Evaluation(
            1,
            1,
            1,
            multipv,
            (0, 0),
            0,
            sys.maxsize,
            0,
            0,
        )

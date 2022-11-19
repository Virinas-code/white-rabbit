#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

Main engine.
"""
import math
import random
import sys
from time import sleep

import chess
import chess.engine
import numpy

from ..uci.options import Option, SpinOption
from ..neural_network import HIDDEN_LAYERS, NeuralNetwork

from .evaluation import Evaluation


def search(
    self,
    board: chess.Board,
    options: dict[str, Option],
    limit: chess.engine.Limit,
) -> chess.engine.PlayResult:
    # Time left and increment
    my_clock_time: float = float("inf")
    if board.turn and limit.white_clock:
        my_clock_time = limit.white_clock
    elif not board.turn and limit.black_clock:
        my_clock_time = limit.black_clock
    my_increment: float = 0
    if board.turn and limit.white_inc:
        my_increment = limit.white_inc
    elif not board.turn and limit.black_inc:
        my_increment = limit.black_inc

    # Move play time
    move_play_time: float = sfloat("inf")
    if limit.time:
        move_play_time = limit.time
    else:
        move_play_time = my_increment + my_clock_time * (1 / 5)

    # Search depth
    target_depth: int = sys.maxsize
    if limit.depth:
        target_depth = limit.depth
    elif limit.nodes:
        target_depth = math.floor(limit.nodes / HIDDEN_LAYERS)

    good_moves: list[chess.Move] = self.neural_network.search(
        board, target_depth
    )

    return chess.engine.PlayResult(good_moves[0], None)


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

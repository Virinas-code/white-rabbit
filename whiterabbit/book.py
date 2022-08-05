#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

Book and Syzygy moves.
"""
from typing import Callable
import chess.polyglot
import chess.syzygy
from chess import Board, Move


class Book:
    """Engine's own book and syzygy."""

    def __init__(self, syzygy_path: str, book_path: str) -> None:
        """
        Initialize book.

        Loads table and book.

        :param str syzygy_path: Path to Syzygy tables.
        :param str book_path: Path to book BIN file.
        """
        self.book_path: str = book_path
        """Path to book BIN file."""
        self.syzygy_path: str = syzygy_path
        """Path to Syzygy tables."""
        self.book: chess.polyglot.MemoryMappedReader = (
            chess.polyglot.open_reader(self.book_path)
        )
        """Book python-chess object."""
        self.syzygy_tables: chess.syzygy.Tablebase = (
            chess.syzygy.open_tablebase(syzygy_path)
        )
        """Syzygy tables python-chess object."""

    def is_book_position(self, position: Board) -> bool:
        """
        Check if a position is present in book.

        :param Board position: Python-chess board to check.
        :return bool: Wether the position is in the book or not.
        """
        return bool(self.book.get(position))

    def is_syzygy_position(self, position: Board) -> bool:
        """
        Check if a position is present in Syzygy tables.

        :param Board position: Python-chess board to check.
        :return bool: Wether the position is in the Syzygy tables or not.
        """
        return bool(self.syzygy_tables.get_dtz(position))

    def best_book_move(self, position: Board) -> Move:
        """
        Get the best move from book in a position.

        :param Board position: Python-chess board to get best move.
        :return Move: The best move in position.
        :raises IndexError: If the move isn't found in book.
            Use :meth:`is_book_position` to check if the move is in book.
        """
        return self.book.weighted_choice(position).move

    def _best_syzygy_move(
        self, position: Board, function: Callable[[Board], int]
    ) -> Move:
        """
        Get the best move from Syzygy tables in a position.

        Returned move is the best WHite's move.

        :param Board position: Python-chess board to get best move.
        :param Callable[[Board], int] function: Syzgy function to use.
            Should be probe_wdl() or probe_dtz().
        :return Move: The best move in position.
        """
        best_move: Move = Move.null()
        best_wdl: int = -3
        for move in position.legal_moves:
            test_board: Board = position.copy()
            test_board.push(move)
            if function(test_board) > best_wdl:
                best_move = move
        return best_move

    def best_syzygy_move(self, position: Board) -> Move:
        """
        Get the best move from Syzygy tables in a position.

        Returned move is the best WHite's move.

        :param Board position: Python-chess board to get best move.
        :return Move: The best move in position.
        :raises IndexError: If the move isn't found in Syzygy tables.
            Use :meth:`is_book_position` to check if the move is in book.
        """
        if self.syzygy_tables.get_wdl(position):
            return self._best_syzygy_move(
                position, self.syzygy_tables.probe_wdl
            )
        if self.syzygy_tables.get_dtz(position):
            return self._best_syzygy_move(
                position, self.syzygy_tables.probe_dtz
            )
        raise IndexError(
            f"No moves found in Syzygy tables in position {position.fen()}"
        )

    def is_table_position(self, position: Board) -> bool:
        """
        Check if a position is present in book or Syzygy tables.

        :param Board position: Python-chess board to check.
        :return bool: Wether the position is in the book or
            in the Syzgy tables or not.
        """
        return self.is_book_position(position) or self.is_syzygy_position(
            position
        )

    def best_table_move(self, position: Board) -> Move:
        """
        Get the best move from book or Syzygy tables in a position.

        :param Board position: Python-chess board to get best move.
        :return Move: The best move in position.
        :raises IndexError: If the move isn't found in book or Syzgy tables.
            Use :meth:`is_book_position` to check if the move is in book.
        """
        try:
            return self.best_book_move(position)
        except IndexError:
            try:
                return self.best_syzygy_move(position)
            except IndexError as exception:
                raise IndexError(
                    f"Position not found in tables {position.fen()}"
                ) from exception

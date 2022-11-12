#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

Neural network object.
"""
from types import LambdaType
from typing import Callable
import chess
import numpy as np
from .utils.save import load_method, save_method

HIDDEN_LAYERS: int = 16  # Amount of hidden layers
RTS_DIFF: int = 12
PIECES_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


class NeuralNetwork:
    """Base neural network object."""

    def __init__(
        self,
        matrices_left: list[np.ndarray],
        matrices_right: list[np.ndarray],
        scalar_matrices: list[np.ndarray],
        reduce_matrices: list[np.ndarray],
        expand_matrices: list[np.ndarray],
        biases: list[np.ndarray],
        correction: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """
        Create a new network.

        :param list[np.ndarray] matrices: Weights
        :param list[np.ndarray] scalar_matrices: Reduce to scalar matrices.
        :param list[np.ndarray] expand_matrices: Expansion matrices.
        :param list[np.ndarray] reduce_matrices: Reduction matrices.
        :param list[np.ndarray] biases: Biases matrices.
        :param tuple[np.ndarray, np.ndarray] correction: Correction matrice for scalar reduction.
        """
        self.matrices_left: list[np.ndarray] = matrices_left
        self.matrices_right: list[np.ndarray] = matrices_right
        self.scalar_matrices: dict[str, np.ndarray] = {
            "R-Gi": scalar_matrices[0],
            "R-Di": scalar_matrices[1],
            "R-Ge": scalar_matrices[2],
            "R-De": scalar_matrices[3],
        }
        """Reduce to scalar matrices."""
        self.reduce_matrices: dict[str, np.ndarray] = {
            "RM-G": reduce_matrices[0],
            "RM-D": reduce_matrices[1],
        }
        self.expand_matrices: dict[str, np.ndarray] = {
            "EX-G": expand_matrices[0],
            "EX-D": expand_matrices[1],
        }
        self.biases: list[np.ndarray] = biases
        self.correction: dict[str, np.ndarray] = {
            "R-G": correction[0],
            "R-D": correction[1],
        }

    save: Callable = save_method
    load: classmethod = classmethod(load_method)

    @classmethod
    def random(cls):
        """
        Generate a random network.

        Fully random network.
        """
        matrices_left: list[np.ndarray] = []
        matrices_right: list[np.ndarray] = []
        biases: list[np.ndarray] = []
        for layer in range(HIDDEN_LAYERS + 2):
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
        expand_matrices: dict[str, np.ndarray] = {
            "EX-G": np.random.randint(0, 255, (16, 96)).astype(np.uint8),
            "EX-D": np.random.randint(0, 255, (96, 14)).astype(np.uint8),
        }
        correction: dict[str, np.ndarray] = {
            "R-G": np.random.randint(0, 255, (8, 8, 12, 12)).astype(np.uint8),
            "R-D": np.random.randint(0, 255, (8, 8, 12, 12)).astype(np.uint8),
        }
        return cls(
            matrices_left,
            matrices_right,
            list(scalar_matrices.values()),
            list(reduce_matrices.values()),
            list(expand_matrices.values()),
            biases,
            tuple(correction.values()),
        )

    def search(self, board: chess.Board, depth: int) -> list[chess.Move]:
        """
        Search best moves in a position.

        Uses Neural Network.

        :param chess.Board board: Actual position.
        :param int depth: Search depth.
        :return list[chess.Move]: Good moves in the position.
        """
        input_layer: np.ndarray = self.generate_inputs(board)
        last_hidden_layer: np.ndarray = self.calculate(input_layer, depth)
        return self.output(board, last_hidden_layer)

    @staticmethod
    def inputs_last_line(board: chess.Board) -> list[np.uint8]:
        """
        Last line of input block.

        :param chess.Board board: Actual position.
        :return list[np.uint8]: Last line content.
        """
        last_line: list[np.uint8] = []
        if board.has_kingside_castling_rights(chess.WHITE):
            last_line.append(np.uint8(127))
        else:
            last_line.append(np.uint8(0))
        if board.has_kingside_castling_rights(chess.BLACK):
            last_line.append(np.uint8(127))
        else:
            last_line.append(np.uint8(0))
        if board.has_queenside_castling_rights(chess.WHITE):
            last_line.append(np.uint8(127))
        else:
            last_line.append(np.uint8(0))
        if board.has_queenside_castling_rights(chess.BLACK):
            last_line.append(np.uint8(127))
        else:
            last_line.append(np.uint8(0))
        ep_values: list[np.uint8] = 8 * [np.uint8(0)]
        if board.ep_square:
            ep_values[chess.square_file(board.ep_square)] = np.uint8(127)
        last_line.extend(ep_values)
        return last_line

    def generate_inputs(self, board: chess.Board) -> np.ndarray:
        """
        Generate inputs.

        :param chess.Board board: Actual position.
        :return np.ndarray: Input layer.
        """
        pieces: dict[chess.Square, chess.Piece] = board.piece_map()
        input_layer: list[list] = np.zeros((8, 8)).tolist()
        for rank in range(8):
            for file in range(8):
                piece_value: list[np.uint8] = [np.uint8(0)] * 12
                if pieces.get(rank * 8 + file, False):
                    piece: chess.Piece = pieces[rank * 8 + file]
                    piece_index: int = (
                        PIECES_VALUES[piece.piece_type]
                        if piece.color is chess.WHITE
                        else PIECES_VALUES[piece.piece_type] + 6
                    )
                    piece_value[piece_index] = np.uint8(127)

                input_block: list[list[np.uint8]] = 11 * [piece_value]
                input_block.append(self.inputs_last_line(board))

                input_layer[rank][file] = input_block

        return np.array(input_layer)

    def calculate(
        self, input_layer: np.ndarray, iterations: int
    ) -> np.ndarray:
        """
        Start calculating.

        :param np.ndarray input_layer: Input layer.
        :param int iterations: Amount of iterations (depth-like).
        :return np.ndarray: Last hidden layer.
        """
        e_layer: np.ndarray = input_layer  # First calculated layer
        # TODO: Pre-init
        for iteration in range(iterations):
            hidden_layer1: np.ndarray = (
                self.matrices_left[0] @ e_layer @ self.matrices_right[0]
                + self.biases[0]
            )
            previous_hidden_layer: np.ndarray = hidden_layer1
            hidden_layer: np.ndarray = np.empty((1, 1))
            previous_rts: int = int(
                (
                    self.scalar_matrices["R-Ge"]
                    @ (
                        self.scalar_matrices["R-Gi"]
                        @ hidden_layer1
                        @ self.scalar_matrices["R-Di"]
                    )
                    @ self.scalar_matrices["R-De"]
                )[0][0]
            )
            for layer_index in range(HIDDEN_LAYERS):
                hidden_layer = (
                    self.matrices_left[layer_index + 1]
                    @ previous_hidden_layer
                    @ self.matrices_right[layer_index + 1]
                )
                current_rts: int = int(
                    (
                        self.scalar_matrices["R-Ge"]
                        @ (
                            self.scalar_matrices["R-Gi"]
                            @ hidden_layer
                            @ self.scalar_matrices["R-Di"]
                        )
                        @ self.scalar_matrices["R-De"]
                    )[0][0]
                )
                correction_r: np.ndarray = (
                    self.correction["R-G"]
                    @ hidden_layer
                    @ self.correction["R-D"]
                )
                if (
                    current_rts >= previous_rts
                    and current_rts - previous_rts >= RTS_DIFF
                ):
                    self.matrices_left[layer_index + 2] -= correction_r
                    self.matrices_right[layer_index + 2] -= correction_r
                if (
                    previous_rts >= current_rts
                    and previous_rts - current_rts >= RTS_DIFF
                ):
                    self.matrices_left[layer_index + 2] += correction_r
                    self.matrices_right[layer_index + 2] += correction_r
            e_layer = hidden_layer
        e_layer = e_layer.reshape(96, 96)
        print(e_layer.shape)
        print(
            self.expand_matrices["EX-G"].shape,
            self.expand_matrices["EX-D"].shape,
        )
        extended_matrix: np.ndarray = (
            self.expand_matrices["EX-G"]
            @ e_layer
            @ self.expand_matrices["EX-D"]
        )
        output_layer: np.ndarray = (
            self.reduce_matrices["RM-G"]
            @ extended_matrix
            @ self.reduce_matrices["RM-D"]
        )
        return output_layer

    def output(
        self, board: chess.Board, output_layer: np.ndarray
    ) -> set[chess.Move]:
        """
        Parse output layer to get best move.

        :param chess.Board board: Current position.
        :param np.ndarray output_layer: Output layer from the NN.
        :return set[chess.Move]: Good moves in the position (unordered).
        """
        legal_moves: list[chess.Move] = list(board.legal_moves)
        piece_map: dict[chess.Square, chess.Piece] = board.piece_map()

        def parse_move(line: np.ndarray) -> chess.Move:
            bool_to_int: Callable[[np.ndarray], int] = (
                lambda a: (1 if a[0] else 0)
                + (2 if a[1] else 0)
                + (4 if a[2] else 0)
            )
            from_rank: int = bool_to_int(line[0:3])
            to_rank: int = bool_to_int(line[3:6])
            from_file: int = bool_to_int(line[6:9])
            to_file: int = bool_to_int(line[9:12])
            promotion: int = chess.PieceType(bool_to_int(line[12:15]) + 1)
            return chess.Move(
                chess.square(from_file, from_rank),
                chess.square(to_file, to_rank),
                promotion=promotion,
            )

        def is_legal(move: chess.Move) -> bool:
            for legal_move in legal_moves:
                if (
                    legal_move.from_square == move.from_square
                    and legal_move.to_square == move.to_square
                ):
                    if (
                        piece_map[legal_move.from_square].piece_type
                        == chess.PAWN
                        and chess.square_rank(legal_move.to_square)
                        in (
                            0,
                            7,
                        )
                        and move in legal_moves
                    ):
                        return True
                    else:
                        move.promotion = None
                        return True
            return False

        output: np.ndarray = output_layer > 127
        good_moves: list[chess.Move] = []
        for line in output:
            move: chess.Move = parse_move(line)
            if is_legal(move):
                good_moves.append(move)
        return good_moves

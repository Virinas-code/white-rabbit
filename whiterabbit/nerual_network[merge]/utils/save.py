#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

Neural network save and load.
"""
import numpy as np


def save_method(self, file: str) -> None:
    """
    Save to file.

    :param str file: File path.
    """
    save_data: dict[str, np.ndarray] = {
        **self.scalar_matrices,
        **self.reduce_matrices,
        **self.correction,
    }
    for index, matrix in enumerate(self.matrices_left):
        save_data[f"M-G{index}"] = matrix
    for index, matrix in enumerate(self.matrices_right):
        save_data[f"M-D{index}"] = matrix
    for index, matrix in enumerate(self.biases):
        save_data[f"B{index}"] = matrix
    np.savez(file, **save_data)


def load_method(cls, file: str):
    """
    Load network from file.

    :param str file: File path to load.
    """
    loaded_file = np.load(file)
    matrices_left: list[np.ndarray] = []
    matrices_right: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for matrix in loaded_file:
        if matrix[0:3] == "M-G":
            matrices_left.append(loaded_file[matrix].astype(np.int8))
        elif matrix[0:3] == "M-D":
            matrices_right.append(loaded_file[matrix].astype(np.int8))
        elif matrix[0] == "B":
            biases.append(loaded_file[matrix].astype(np.int8))
    scalar_matrices = [
        loaded_file["R-Gi"].astype(np.int8),
        loaded_file["R-Di"].astype(np.int8),
        loaded_file["R-Ge"].astype(np.int8),
        loaded_file["R-De"].astype(np.int8),
    ]
    reduce_matrices: list[np.ndarray] = [
        loaded_file["RM-G"].astype(np.int8),
        loaded_file["RM-D"].astype(np.int8),
    ]
    correction: tuple[np.ndarray, np.ndarray] = (
        loaded_file["R-G"].astype(np.int8),
        loaded_file["R-D"].astype(np.int8),
    )
    return cls(
        matrices_left,
        matrices_right,
        scalar_matrices,
        reduce_matrices,
        biases,
        correction,
    )

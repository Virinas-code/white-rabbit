#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

Neural network hash.
"""
import numpy as np


def hash_arrays(arrays: list[np.ndarray]) -> tuple[int]:
    """
    Hash a list of numpy arrays.

    :param list[np.ndarray] arrays: List of numpy arrays.
    :return tuple[int]: Tuple of hashes.
    """
    hashes: list[int] = []
    for array in arrays:
        hashes.append(hash(array.tobytes()))
    return tuple(hashes)


def hash_arrays_dict(arrays_dict: dict[str, np.ndarray]) -> tuple[int]:
    """
    Hash a dict of numpy arrays.

    :param dict[str, np.ndarray] arrays_dict: Dict of numpy arrays.
    :return tuple[int]: Tuple of hashes.
    """
    hashes: list[int] = []
    for key, array in arrays_dict.items():
        hashes.append(hash((key, array.tobytes())))
    return tuple(hashes)


def network_hash(network) -> int:
    """
    Get a network hash.

    :param NeuralNetwork network: Network to get hash of.
    :return int: Network hash.
    """
    return hash(
        (
            hash_arrays(network.matrices_left),
            hash_arrays(network.matrices_right),
            hash_arrays_dict(network.scalar_matrices),
            hash_arrays_dict(network.reduce_matrices),
            hash_arrays(network.biases),
            hash_arrays_dict(network.correction),
        )
    )

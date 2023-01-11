#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

Neural network hash.
"""
from hashlib import md5

import numpy as np


def calc_hash(data: bytes) -> int:
    """
    Compute hash of data.

    :param bytes data: Data to compute hash of.
    :return int: Hashed data.
    """
    return int(md5(data).hexdigest(), 16)


def list_hash(data: list[int]) -> int:
    """
    Compute hash of a list of hashes.

    :param tuple[int] data: List of hashes.
    :return int: Hashed data.
    """
    string: str = ""
    for element in data:
        string += str(element)
    return calc_hash(string.encode())


def hash_arrays(arrays: list[np.ndarray]) -> int:
    """
    Hash a list of numpy arrays.

    :param list[np.ndarray] arrays: List of numpy arrays.
    :return tuple[int]: Hash.
    """
    hashes: list[int] = []
    for array in arrays:
        hashes.append(calc_hash(array.tobytes()))
    return list_hash(hashes)


def hash_arrays_dict(arrays_dict: dict[str, np.ndarray]) -> int:
    """
    Hash a dict of numpy arrays.

    :param dict[str, np.ndarray] arrays_dict: Dict of numpy arrays.
    :return int: Hash.
    """
    hashes: list[int] = []
    for key, array in arrays_dict.items():
        hashes.append(calc_hash(key.encode() + array.tobytes()))
    return list_hash(hashes)


def network_hash(network) -> int:
    """
    Get a network hash.

    :param NeuralNetwork network: Network to get hash of.
    :return int: Network hash.
    """
    return list_hash(
        [
            hash_arrays(network.matrices_left),
            hash_arrays(network.matrices_right),
            hash_arrays_dict(network.scalar_matrices),
            hash_arrays_dict(network.reduce_matrices),
            hash_arrays(network.biases),
            hash_arrays_dict(network.correction),
        ]
    )

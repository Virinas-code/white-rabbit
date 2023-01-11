#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

Neural network hash.
"""
from hashlib import md5


def calc_hash(data: bytes) -> int:
    """
    Compute hash of data.

    :param bytes data: Data to compute hash of.
    :return int: Hashed data.
    """
    return int(md5(data).hexdigest(), 16)


def network_hash(network) -> int:
    """
    Get a network hash.

    :param NeuralNetwork network: Network to get hash of.
    :return int: Network hash.
    """
    all_bytes: bytes = b""
    for array in network:
        all_bytes += array.tobytes()
    return calc_hash(all_bytes)

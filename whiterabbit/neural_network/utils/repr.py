#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

Neural network repr.
"""


def network_repr(network) -> str:
    """
    Get a network repr.

    :param NeuralNetwork network: Network to get repr of.
    :return str: Network repr.
    """
    return f"<NeuralNetwork object @{hash(network)}>"

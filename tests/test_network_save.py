#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

Networks save / load tests.
"""
from whiterabbit.neural_network import NeuralNetwork

network1: NeuralNetwork = NeuralNetwork.random()
network2: NeuralNetwork = NeuralNetwork.random()


def test_save():
    """
    Test network save.

    Saves network to tests/test_network_save.npz.
    """
    network1.save("tests/test_network_save.npz")


def test_load():
    """
    Test network load.

    Network is loaded from file tests/test_network_save.npz.
    """
    global network2
    network2 = NeuralNetwork.load("tests/test_network_save.npz")


def test_same():
    """
    Test if networks are the same.

    See test_save() and test_load().
    """
    assert hash(network1) == hash(network2)

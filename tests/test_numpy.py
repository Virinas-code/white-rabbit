#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

Speed tests for nummpy.
"""
import numpy as np


def matmul_speed(dtype: str, iterations: int):
    """
    Make boolean matrices multiplication.

    Simple benchmark.

    :param str dtype: Type of the matrix.
    :param int iterations: Number of iterations.
    """
    matrix1: np.ndarray = np.empty((12, 12, 8, 8))
    matrix2: np.ndarray = np.empty((12, 12, 8, 8))
    if dtype == "bool":
        matrix1 = np.array(
            np.random.choice(
                a=[False, True], size=(12, 12, 8, 8), p=[0.75, 0.25]
            ),
            dtype=dtype,
        )
        matrix2 = np.array(
            np.random.choice(
                a=[False, True], size=(12, 12, 8, 8), p=[0.75, 0.25]
            ),
            dtype=dtype,
        )
    else:
        try:
            min_value = np.iinfo(np.dtype(dtype)).min
            max_value = np.iinfo(np.dtype(dtype)).max
        except ValueError:
            min_value = np.finfo(np.dtype(dtype)).min
            max_value = np.finfo(np.dtype(dtype)).max
        print(max_value - min_value)
        matrix1 = np.random.uniform(
            min_value, max_value, size=(12, 12, 8, 8)
        ).astype(dtype)
        matrix2 = np.random.uniform(
            min_value, max_value, size=(12, 12, 8, 8)
        ).astype(dtype)
    for loop in range(iterations):
        matrix1 = matrix1 @ matrix2


def test_bool_matmul(benchmark):
    """
    Test speed of matmul on bool matrices.

    dtype: bool
    """
    benchmark(matmul_speed, "bool", 10000)


def test_uint8_matmul(benchmark):
    """
    Test speed of matmul on integers matrices.

    dtype: uint8
    """
    benchmark(matmul_speed, "uint8", 10000)


def test_int8_matmul(benchmark):
    """
    Test speed of matmul on integers matrices.

    dtype: int8
    """
    benchmark(matmul_speed, "int8", 10000)


def test_uint16_matmul(benchmark):
    """
    Test speed of matmul on bool matrices.

    dtype: uint16
    """
    benchmark(matmul_speed, "uint16", 10000)


def test_int16_matmul(benchmark):
    """
    Test speed of matmul on bool matrices.

    dtype: int16
    """
    benchmark(matmul_speed, "int16", 10000)


def test_float16_matmul(benchmark):
    """
    Test speed of matmul on bool matrices.

    dtype: float16
    """
    benchmark(matmul_speed, "float16", 10000)

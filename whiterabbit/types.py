# -*- coding: utf-8 -*-
"""
White Rabbit Chess Engine.

Global type hints.
"""
import typing
from typing import NewType, Type


class _Object:
    """An empty object for method type."""

    def method(self) -> None:
        """
        A method.

        :return None: Nothing.
        """
        return None


MethodType: Type = type(_Object().method)
Method = NewType("Method", MethodType)
typing.Method

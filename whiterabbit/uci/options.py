#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

UCI options.
"""
from typing import Any, Callable


TYPES: dict[str, str] = {
    "CheckOption": "check",
    "SpinOption": "spin",
    "ComboOption": "combo",
    "ButtonOption": "button",
    "StringOption": "string",
}


class Option:
    """An UCI option."""

    def __init__(self, name: str):
        """
        Initialize option.

        :param str name: Name of the option.
        """
        self.name: str = name
        """Option name."""
        self.value: Any = None
        """Option value."""

    def set(self, value: str) -> bool:
        """
        Set option value.

        Base class, used for compatibility.

        :param str value: New option value.
        :return bool: Wether the value was accepted or not.
        """
        return False

    def get_string(self) -> str:
        """
        Return default value as string.

        Aslo includes min and max for spin type.

        :return str: Current value.
        """
        return ""

    def option(self) -> str:
        """
        Get UCI option command string.

        :return str: String to print.
        """
        return (
            "option name "
            + self.name
            + " type "
            + TYPES[self.__class__.__name__]
            + (" default " + self.get_string())
            if self.get_string()
            else ""
        )


class CheckOption(Option):
    """UCI option type `bool`."""

    def __init__(self, name: str, default: bool):
        """
        Initialize option.

        :param str name: Name of the option.
        :param bool default: Default value of the option.
        """
        super().__init__(name)
        self.default: bool = default
        """Default value."""
        self.value: bool = default
        """Option value."""

    def set(self, value: str) -> bool:
        """
        Set option value.

        :param str value: New option value.
        :return bool: Wether the value was accepted or not.
        """
        if value == "true":
            self.value = True
            return True
        if value == "false":
            self.value = False
            return True
        return False

    def get_string(self) -> str:
        """
        Return default value as string.

        Should be `true`or `false`.

        :return str: Current value.
        """
        return "true" if self.default else "false"


class SpinOption(Option):
    """UCI option type `spin`."""

    def __init__(
        self, name: str, default: int, max_value: int, min_value: int
    ):
        """
        Initialize option.

        :param str name: Name of the option.
        :param int default: Default value of the option.
        :param int max: Maximum value.
        :param int min: Minimumu value.
        """
        super().__init__(name)
        self.default: int = default
        """Default value."""
        self.value: int = default
        """Option value."""
        self.max_value: int = max_value
        """Max value."""
        self.min_value: int = min_value
        """Min value."""

    def set(self, value: str) -> bool:
        """
        Set option value.

        :param str value: New option value.
        :return bool: Wether the value was accepted or not.
        """
        if not value.isnumeric():
            return False
        int_value: int = int(value)
        if self.max_value > int_value > self.min_value:
            self.value = int_value
            return True
        return False

    def get_string(self) -> str:
        """
        Return default value as string.

        Should be `true`or `false`.

        :return str: Current value.
        """
        return str(self.default)


class ComboOption(Option):
    """UCI option type `combo`."""

    def __init__(self, name: str, default: str, *args: str):
        """
        Initialize option.

        :param str name: Name of the option.
        :param str default: Default value of the option.
        :param srt args: Possible values.
        """
        super().__init__(name)
        self.default: str = default
        """Default value."""
        self.value: str = default
        """Option value."""
        self.values: list[str] = [*args]
        """Possible value."""

    def set(self, value: str) -> bool:
        """
        Set option value.

        :param str value: New option value.
        :return bool: Wether the value was accepted or not.
        """
        if value in self.values:
            self.value = value
            return True
        return False

    def get_string(self) -> str:
        """
        Return default value as string.

        Should be `true`or `false`.

        :return str: Current value.
        """
        return self.default


class ButtonOption(Option):
    """UCI option type `button`."""

    def __init__(self, name: str, default: Callable):
        """
        Initialize option.

        :param str name: Name of the option.
        :param Callable default: Function to call.
        """
        super().__init__(name)
        self.function: Callable = default
        """Function to call."""

    def set(self, value: str) -> bool:
        """
        Run function.

        :param str value: Used for compatibility, should be empty.
        :return bool: Wether the value was accepted or not.
        """
        if value:
            return False
        self.function()
        return True

    def get_string(self) -> str:
        """
        Return default value as string.

        Should be `true`or `false`.

        :return str: Current value.
        """
        return ""


class StringOption(Option):
    """UCI option type `string`."""

    def __init__(self, name: str, default: str):
        """
        Initialize option.

        :param str name: Name of the option.
        :param str default: Default value of the option.
        """
        super().__init__(name)
        self.default: str = default
        """Default value."""
        self.value: str = default
        """Option value."""

    def set(self, value: str) -> bool:
        """
        Set option value.

        :param str value: New option value.
        :return bool: Wether the value was accepted or not.
        """
        self.value = value
        return True

    def get_string(self) -> str:
        """
        Return default value as string.

        Should be `true`or `false`.

        :return str: Current value.
        """
        return self.default

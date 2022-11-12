#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

CLI Logger to display rich text.
"""
from typing import Any

from rich.console import Console
from rich.traceback import install

install(show_locals=True)


class RichCLI:
    """CLI object."""

    def __init__(self):
        """
        Initialize object.

        TODO: Complete this
        """
        self.console: Console = Console()

    def prompt(self, name: str, to_type: type) -> Any:
        """
        Prompt something.

        :param str name: Name of input.
        :param type to_type: Destination type.
        :return Any: User answer.
        """
        while True:
            answer: str = self.console.input(
                f"[magenta bold]{name}: [/magenta  bold]"
            )
            try:
                answer = to_type(answer)
                break
            except (ValueError, TypeError):
                self.console.print(
                    "[bold red]Please enter a valid value.[/bold red]"
                )
        return answer

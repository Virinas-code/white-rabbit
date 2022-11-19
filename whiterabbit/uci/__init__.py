#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

UCI main loop.
"""
import importlib.metadata
from multiprocessing.managers import BaseManager
import threading
from typing import Optional

from .commands import Commands
from .engine import Engine


class UCI:
    """White Rabbit's UCI."""

    def __init__(self) -> None:
        """
        Initialize UCI.

        This doesn't starts main loop.
        """
        BaseManager.register("Engine", Engine)
        self.manager: BaseManager = BaseManager()
        """Shared variables manager."""
        self.commands_parser: Commands = Commands()  # type: ignore
        """Commands parser."""
        self.about()

    def about(self) -> None:
        """
        Prints engine infos.

        Prints name and author.
        """
        version: str = importlib.metadata.version("white-rabbit")
        self.commands_parser.uci_id("name", "White Rabbit " + version)
        self.commands_parser.uci_id("author", "Virinas-code and others")

    def mainloop(self) -> None:
        """
        Start UCI;

        Runs the mainloop.
        """
        while True:
            command_string: str = input()
            thread: threading.Thread = threading.Thread(
                target=self.parse, args=[command_string]
            )
            thread.start()
            if command_string == "quit":
                break

    def parse(self, command: str) -> None:
        """
        Parse a command.

        :param str command: Command to parse.
        """
        parsed: list[str] = command.strip().split()
        keyword: str = parsed[0] if parsed else ""
        arguments: list[str] = parsed[1:] if len(parsed) > 1 else []
        if keyword == "uci":
            self.commands_parser.uci()
        elif keyword == "debug":
            self.commands_parser.debug(*arguments)
        elif keyword == "isready":
            self.commands_parser.isready()
        elif keyword == "setoption":
            self.commands_parser.setoption(*arguments)
        elif keyword == "register":
            self.commands_parser.register()
        elif keyword == "ucinewgame":
            self.commands_parser.ucinewgame()
        elif keyword == "position":
            self.commands_parser.position(*arguments)
        elif keyword == "go":
            self.commands_parser.uci_go(*arguments)
        elif keyword == "stop":
            self.commands_parser.stop()
        elif keyword == "ponderhit":
            self.commands_parser.ponderhit()
        elif keyword == "quit":
            self.commands_parser.quit()
        elif keyword == "":
            pass
        else:
            self.commands_parser.send(f"Unknown command: {' '.join(parsed)}")

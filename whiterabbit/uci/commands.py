#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

UCI commands parser.
"""
import chess
from .options import ButtonOption, Option
from .engine import Engine


class Commands:
    """White Rabbit's UCI commands."""

    def __init__(self, engine: Engine):
        """
        Initialize command parser.

        Define a few technical arguments.
        """
        self.debug_mode: bool = False
        """Debug mode status."""
        self.engine: Engine = engine
        """Main engine."""

    def send(self, *args: str) -> None:
        """
        Send data.

        :param str args: Message to send.
        """
        print(*args)

    def uci(self) -> None:
        """
        UCI `uci` command.

        Show informations about engine and options list.
        """
        self.uci_id("name", "White Rabbit")
        self.uci_id("author", "Virinas-code and ZeBox")
        self.send()
        for option in self.engine.options.values():
            self.option(option)
        self.send()
        self.uciok()

    def debug(self, *args: str) -> None:
        """
        UCI `debug` command.

        Enable or disable debug mode.

        :param str args: Command arguments. Should be length of one
            and should be "on" or "off".
        """
        if args:
            self.debug_mode = args[0] == "on"

    def isready(self) -> None:
        """
        UCI `isready` command.

        Check if the engine is responding.
        """
        self.readyok()

    def setoption(self, *args: str) -> None:
        """
        UCI `setoption` command.

        Set the value of an option.

        :param str args: Command arguments.
        """
        if len(args) > 1 and args[0] == "name":
            option_name: str = args[1]
            if len(args) > 3 and args[3] == "value":
                option_value: str = " ".join(args[4:])
                if option_name in self.engine.options:
                    self.engine.options[option_name].set(option_value)
            else:
                if isinstance(
                    self.engine.options.get(option_name), ButtonOption
                ):
                    self.engine.options[option_name].set("")

    def register(self) -> None:
        """
        UCI `register` command.

        Not enabled on White Rabbit.
        """
        self.send("This engine is open-source, sorry.")

    def ucinewgame(self) -> None:
        """
        UCI `ucinewgame` command.

        Start a new game.
        """
        self.engine.transpositions.clear()

    def position(self, *args: str) -> None:
        """
        UCI `position` command.

        Set the current position.

        :param str args: Command arguments.
        """
        if args:
            if args[0] == "fen" and len(args) > 6:
                self.engine.position = chess.Board(" ".join(args[1:7]))
            elif args[0] == "startpos":
                self.engine.position = chess.Board()
            else:
                try:
                    for move in args:
                        self.engine.position.push(chess.Move.from_uci(move))
                except ValueError:
                    pass

    def uci_id(self, data: str, value: str) -> None:
        """
        UCI `id` command.

        Show informations about engine.

        :param str data: Type of the information, should be "name" or "author".
        :param str value: Value of the information.
        """
        self.send("id", data, value)

    def uciok(self) -> None:
        """
        UCI `uciok` command.

        Tell the GUI that the engine is responding.
        """
        self.send("uciok")

    def readyok(self) -> None:
        """
        UCI `readyok` command.

        Tell the GUI that the engine is responding.
        """
        self.send("readyok")

    def option(self, option: Option) -> None:
        """
        UCI `option` command.

        Sho informations about an option.

        :param Option option: Option object to display.
        """
        self.send(option.option())

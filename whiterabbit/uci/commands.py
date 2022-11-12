#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.
 Flib1301
 
UCI commands parser.
"""
import sys

import chess
from .options import ButtonOption, Option
from .engine import Engine


class Commands:
    """White Rabbit's UCI commands."""

    def __init__(self):
        """
        Initialize command parser.

        Define a few technical arguments.
        """
        self.debug_mode: bool = False
        """Debug mode status."""
        self.engine: Engine = Engine()
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
            if len(args) > 3 and args[2] == "value":
                option_value: str = " ".join(args[3:])
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

    def uci_go(self, *args: str) -> None:
        """
        UCI `go` command.

        Start calculating.

        :param str args: Command arguments.
        """
        search_moves: list[chess.Move] = []
        skip_count: int = 0  # Skip arguments because they are used
        move_time: int = 0
        mode: str = "infinite"
        for index, arg in enumerate(args):
            if skip_count == 0:
                if arg == "searchmoves":
                    search_move_index: int = 0  # Index to find in string
                    while True:
                        search_move_index += 1
                        try:
                            search_moves.append(
                                chess.Move.from_uci(
                                    args[index + search_move_index]
                                )
                            )  # Add next arguments
                            skip_count += 1  # Skip argument
                        except ValueError:  # The value is not a valid UCI
                            break
                elif arg == "movetime":
                    move_time = int(args[index + 1])
                    mode = "movetime"
                    skip_count = 1
            else:
                skip_count -= 1
        for move in search_moves:
            self.engine.position.push(move)
        if mode == "infinite":
            self.engine.search(max_depth=float("inf"))
        elif mode == "movetime":
            print("## MOVETIME", file=sys.stderr)
            self.engine.search(move_time=move_time)

    def stop(self) -> None:
        """
        UCI `stop` command.

        Stop engine thread.
        """
        self.engine.stop()  # TODO: bestmove

    def quit(self) -> None:
        """
        UCI `quit` command.

        Stop engine thread.
        """
        self.engine.stop()

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

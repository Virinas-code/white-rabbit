#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

UCI interface.
"""
from __future__ import annotations

import copy
import sys
import time
from typing import Optional, Union

import chess

from .engine import Engine


class UCI:
    """Base class for White Rabbit UCI."""

    def __init__(self):
        """Initialize UCI."""
        self.name: str = "White Rabbit [Indev]"
        """Engine name and version."""
        self.authors: str = "See CREDITS.txt for more info"
        """Engine authors."""
        self.options: dict[str, Union[int, bool, str]] = {
            "Hash": 16,
            "NeuralNetwork": "true",
            "OwnBook": "false",
            "SyzygyOnline": "false",
            "SyzygyPath": "",
            "HashPath": "",
        }
        self.debug_mode = False
        self.board = chess.Board()
        self.positionned = False
        self.engine = Engine()
        print(self.name)
        print(self.authors)

    def run(self):
        """
        Run UCI input.

        This will run a forever loop until quit command received.
        """
        inner = ""
        while inner != "quit":
            inner = input()
            self.uci_parse(inner)

    def uci_parse(self, string: str) -> None:
        """
        Parse UCI command.

        :param str string: UCI command to parse.
        """
        self.info("Received " + string)
        args = [value for value in string.split(" ") if value != ""]
        if string != "":
            command = args[0]
        else:
            command = ""
        if command == "uci":
            self.uci()
        elif command == "isready":
            print("readyok")
        elif command == "debug" and len(args) > 1:
            self.debug(args[1])
        elif command == "quit":
            pass
        elif command == "register":
            print("No support for register")
        elif command == "ucinewgame":
            self.new_game()
        elif command == "go" and len(args) > 1:
            self.go(args[1:])
        elif command == "position" and len(args) > 1:
            self.position(args[1:])
        elif command == "setoption" and len(args) > 1:
            self.set_option(args[1:])
        elif command == "crocrodile.display" and self.debug_mode is True:
            print(self.board)
        elif command == "crocrodile.bruh" and self.debug_mode is True:
            print("Yes BRUH.")
            print("https://lichess.org/83hsKBy2/black#2")
        elif len(args) == 0:
            pass
        elif len(args) == 1:
            print(f"Unknown command: {string} with no arguments")
        else:
            print(f"Unknown command: {string}")

    def info(self, msg: str) -> None:
        """
        Print debug information.

        :param str msg: Message to display.
        """
        if self.debug_mode:
            print("info string", msg)

    def uci(self) -> None:
        """
        Uci UCI command.

        Prints name and options.
        """
        print(f"id name {self.name}")
        print(f"id author {self.authors}")
        print()
        print("option name Hash type spin default 16 min 0 max 65536")
        print("option name NeuralNetwork type check default true")
        print("option name OwnBook type check default false")
        print("option name SyzygyOnline type check default false")
        print("option name SyzygyPath type string default <empty>")
        print("option name HashPath type string default <empty>")
        print("uciok")

    def debug(self, boolean: str) -> None:
        """
        Debug UCI command.

        Enable or disable debug mode.

        :param str boolean: 'on' or 'off'
        """
        if boolean == "on":
            self.debug_mode = True
        elif boolean == "off":
            self.debug_mode = False
        else:
            print(f"Unknown debug mode: {boolean}")

    def set_option(self, args) -> None:
        """
        Setoption UCI command.

        Configure Crocrodile.
        """
        if len(args) > 3 and args[0] == "name" and args[2] == "value":
            if args[1] in self.options:
                self.options[args[1]] = " ".join(args[3:])
                self.engine.hashlimit = int(self.options["Hash"])
                if self.options["NeuralNetwork"] == "true":
                    self.engine.use_nn = True
                else:
                    self.engine.use_nn = False
                if self.options["OwnBook"] == "true":
                    self.engine.own_book = True
                else:
                    self.engine.own_book = False
                if self.options["SyzygyOnline"] == "true":
                    self.engine.syzygy_online = True
                else:
                    self.engine.syzygy_online = False
                if self.options["SyzygyPath"] != "":
                    self.engine.syzygy_tb: chess.syzygy.Table = (
                        chess.syzygy.open_tablebase(self.options["SyzygyPath"])
                    )
                if self.options["HashPath"] != "":
                    self.engine.hashpath = self.options["HashPath"]
                if args[1] == "HashPath":
                    self.engine.tb_update()
            else:
                print("Unknow option:", args[1])
        else:
            print("Unknow syntax: setoption", " ".join(args))

    def go(self, args: list) -> None:
        """
        Go UCI command.

        Start calculating.
        """
        depth: Optional[int] = 256
        wtime: Optional[int] = None
        btime: Optional[int] = None
        movetime: bool | int = False
        for indice, element in enumerate(args):
            if element == "depth":
                try:
                    depth = int(args[indice + 1])
                except ValueError:
                    print("Invalid depth.", file=sys.stderr)
            if element == "wtime":
                try:
                    wtime = int(args[indice + 1])
                except ValueError:
                    print("Invalid wtime.", file=sys.stderr)
            if element == "btime":
                try:
                    btime = int(args[indice + 1])
                except ValueError:
                    print("Invalid btime.", file=sys.stderr)
            if element == "movetime":
                try:
                    movetime = int(args[indice + 1])
                except ValueError:
                    print("Invalid movetime.", file=sys.stderr)
        if depth != 256:
            if self.board.turn and wtime:
                limit = ((wtime / 1000) / 40) + time.time()
            elif (not self.board.turn) and btime:
                limit = ((btime / 1000) / 40) + time.time()
            elif movetime:
                limit = movetime / 1000
            else:
                limit = float("inf")
        else:
            limit = float("inf")
        print(limit)
        evaluation, best_move = self.engine.search(
            self.board, 1, self.board.turn, float("inf")
        )
        last_best_move = copy.copy(best_move.uci())
        for search_depth in range(2, depth + 1):
            evaluation, best_move = self.engine.search(
                self.board, search_depth, self.board.turn, limit
            )
            if evaluation == float("inf"):
                break
            else:
                last_best_move = copy.copy(best_move.uci())
        print(f"bestmove {last_best_move}")

    def position(self, args: list[str]) -> None:
        """
        Position UCI command.

        Change current position.
        """
        next_arg = 0
        if args[0] == "startpos":
            self.board = chess.Board()
            self.positionned = True
            next_arg = 1
        elif args[0] == "fen" and len(args) > 6:
            self.board = chess.Board(" ".join(args[1:7]))
            self.positionned = True
            next_arg = 7
        else:
            print("Unknow syntax: position", " ".join(args))
        if next_arg and len(args) > next_arg + 1:
            self.info(args[next_arg])
            self.info(args[next_arg + 1 :])
            for uci_move in args[next_arg + 1 :]:
                try:
                    self.board.push(chess.Move.from_uci(uci_move))
                except ValueError:
                    print("Unknow UCI move:", uci_move)

    def new_game(self):
        self.board = chess.Board()
        self.positionned = False


if __name__ == "__main__":
    uci = UCI()
    uci.run()

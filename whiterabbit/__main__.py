#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

Launcher, CLI coming soon.
"""
from .uci import UCI


def main() -> None:
    """
    Main function.

    Starts the UCI mainloop.
    """
    uci: UCI = UCI()
    uci.mainloop()


if __name__ == "__main__":
    main()

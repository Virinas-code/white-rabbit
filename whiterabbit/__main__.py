#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

Launcher, CLI coming soon.
"""
from .uci import UCI


if __name__ == "__main__":
    uci: UCI = UCI()
    uci.mainloop()

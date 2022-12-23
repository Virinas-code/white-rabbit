#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

Start training.
"""
from . import Trainer

if __name__ == "__main__":
    trainer: Trainer = Trainer()
    trainer.main_loop()

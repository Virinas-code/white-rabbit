# -*- coding: utf-8 -*-
"""
White Rabbit Chess Engine.

Start training.
"""
from . import Trainer


if __name__ == "__main__":
    trainer: Trainer = Trainer()
    trainer.main_loop()
